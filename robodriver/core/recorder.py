import threading
import time
from dataclasses import dataclass
from typing import Optional

import logging_mp
from deepdiff import DeepDiff
from lerobot.datasets.pipeline_features import (
    aggregate_pipeline_dataset_features,
    create_initial_features,
)
from lerobot.datasets.utils import build_dataset_frame, combine_feature_dicts
from lerobot.processor import make_default_processors
from lerobot.robots import Robot
from lerobot.teleoperators import Teleoperator
from lerobot.utils.constants import ACTION, OBS_STR

from robodriver.dataset.dorobot_dataset import *
from robodriver.robots.configs import RobotConfig

# from operating_platform.dataset.functions import get_features_from_robot_new
from robodriver.robots.daemon import Daemon
from robodriver.robots.utils import busy_wait
from robodriver.utils.data_file import (
    delete_dataid_json,
    get_data_duration,
    get_data_size,
    update_common_record_json,
    update_dataid_json,
    validate_session,
)

logger = logging_mp.get_logger(__name__)


def sanity_check_dataset_robot_compatibility(
    dataset: DoRobotDataset, robot: RobotConfig, fps: int, use_videos: bool
) -> None:
    fields = [
        ("robot_type", dataset.meta.robot_type, robot.robot_type),
        ("fps", dataset.fps, fps),
        # ("features", dataset.features, get_features_from_robot(robot, use_videos)),
    ]

    mismatches = []
    for field, dataset_value, present_value in fields:
        diff = DeepDiff(
            dataset_value, present_value, exclude_regex_paths=[r".*\['info'\]$"]
        )
        if diff:
            mismatches.append(f"{field}: expected {present_value}, got {dataset_value}")

    if mismatches:
        raise ValueError(
            "Dataset metadata compatibility check failed with mismatches:\n"
            + "\n".join(mismatches)
        )


@dataclass
class RecordConfig:
    # Dataset identifier. By convention it should match '{hf_username}/{dataset_name}' (e.g. `lerobot/test`).
    repo_id: str
    # A short but accurate description of the task performed during the recording (e.g. "Pick the Lego block and drop it in the box on the right.")
    single_task: str = "TEST: no task description. Example: Pick apple."
    # Root directory where the dataset will be stored (e.g. 'dataset/path').
    root: str | Path | None = None
    # Limit the frames per second. By default, uses the policy fps.
    fps: int = 30

    # Encode frames in the dataset into video
    video: bool = False  # 这是实际控制use_videos的地方

    # Upload dataset to Hugging Face hub.
    push_to_hub: bool = False

    # Upload on private repository on the Hugging Face hub.
    private: bool = False

    # Add tags to your dataset on the hub.
    tags: list[str] | None = None

    # Number of subprocesses handling the saving of frames as PNG. Set to 0 to use threads only;
    # set to ≥1 to use subprocesses, each using threads to write images. The best number of processes
    # and threads depends on your system. We recommend 4 threads per camera with 0 processes.
    # If fps is unstable, adjust the thread count. If still unstable, try using 1 or more subprocesses.
    num_image_writer_processes: int = 0
    # Number of threads writing the frames as png images on disk, per camera.
    # Too many threads might cause unstable teleoperation fps due to main thread being blocked.
    # Not enough threads might cause low camera fps.
    num_image_writer_threads_per_camera: int = 4

    # Resume recording on an existing dataset.
    resume: bool = False

    record_cmd = None


class Record:
    def __init__(
        self,
        fps: int,
        robot: Robot,
        teleop: Optional[Teleoperator],
        daemon: Daemon,
        record_cfg: RecordConfig,
        record_cmd: dict,
    ):
        self.robot = robot
        self.daemon = daemon
        self.record_cfg = record_cfg
        self.fps = fps
        self.record_cmd = record_cmd
        self.last_record_episode_index = 0
        self.record_complete = False
        self.save_data = None
        self._last_record_observation_sequence = None
        self._last_record_action_sequence = None
        self._record_sample_count = 0
        self._reused_observation_count = 0
        self._reused_action_count = 0
        self.frame_count = 0
        self.record_error: Exception | None = None
        self._astribot_joints = robot.robot_type.startswith("astribott1_aio_") and teleop is None
        self._last_joint_counts: dict[str, int] = {}
        self._joint_cache_reuse: dict[str, int] = {}
        self._rate_start = time.monotonic()

        self.record_cfg.record_cmd = record_cmd

        print(f"in Record init record_cmd: {self.record_cmd}")

        teleop_action_processor, robot_action_processor, robot_observation_processor = (
            make_default_processors()
        )

        action_features = (
            teleop.action_features if teleop is not None else robot.action_features
        )

        dataset_features = combine_feature_dicts(
            aggregate_pipeline_dataset_features(
                pipeline=teleop_action_processor,
                initial_features=create_initial_features(action=action_features),
                use_videos=record_cfg.video,
            ),
            aggregate_pipeline_dataset_features(
                pipeline=robot_observation_processor,
                initial_features=create_initial_features(
                    observation=robot.observation_features
                ),
                use_videos=record_cfg.video,
            ),
        )

        logger.info(f"Dataset features: {dataset_features}")

        if self.record_cfg.resume:
            self.dataset = DoRobotDataset(
                record_cfg.repo_id,
                root=record_cfg.root,
            )
            if len(robot.cameras) > 0:
                self.dataset.start_image_writer(
                    num_processes=record_cfg.num_image_writer_processes,
                    num_threads=record_cfg.num_image_writer_threads_per_camera
                    * len(robot.cameras),
                )
            if len(robot.microphones) > 0:
                self.dataset.start_audio_writer(
                    microphones=robot.microphones,
                )
            sanity_check_dataset_robot_compatibility(
                self.dataset, robot, record_cfg.fps, record_cfg.video
            )
        else:
            self.dataset = DoRobotDataset.create(
                record_cfg.repo_id,
                record_cfg.fps,
                root=record_cfg.root,
                robot=robot,
                features=dataset_features,
                use_videos=record_cfg.video,
                use_audios=len(robot.microphones) > 0,
                image_writer_processes=record_cfg.num_image_writer_processes,
                image_writer_threads=record_cfg.num_image_writer_threads_per_camera
                * len(robot.cameras),
            )

        self.thread = threading.Thread(target=self.process, daemon=True)
        self.running = True

    def start(self):
        self._rate_start = time.monotonic()
        self.thread.start()
        self.running = True

    def process(self):
        try:
            while self.running:
                if self.dataset is None:
                    break
                start_loop_t = time.perf_counter()

                observation, action, obs_seq, action_seq = self.daemon.get_record_snapshot()
                if self._astribot_joints and observation is not None and action is not None:
                    joint_observation, joint_action, joint_counts = self.robot.get_record_joints()
                    observation.update(joint_observation)
                    action.update(joint_action)
                    for part, count in joint_counts.items():
                        if self._last_joint_counts.get(part) == count:
                            self._joint_cache_reuse[part] = self._joint_cache_reuse.get(part, 0) + 1
                        self._last_joint_counts[part] = count
                if self._last_record_observation_sequence == obs_seq:
                    self._reused_observation_count += 1
                if self._last_record_action_sequence == action_seq:
                    self._reused_action_count += 1
                self._last_record_observation_sequence = obs_seq
                self._last_record_action_sequence = action_seq
                self._record_sample_count += 1
                if self._record_sample_count % 200 == 0:
                    elapsed = time.monotonic() - self._rate_start
                    logger.info(
                        "[录制采样] 目标=%sHz, 当前约=%.1fHz; %d次读取: daemon图像/观测缓存复用=%d, daemon动作缓存复用=%d, ROS关节无新消息=%s",
                        self.fps,
                        self._record_sample_count / elapsed if elapsed > 0 else 0.0,
                        self._record_sample_count,
                        self._reused_observation_count,
                        self._reused_action_count,
                        dict(self._joint_cache_reuse) if self._astribot_joints else "未单独采样",
                    )
                    self._record_sample_count = 0
                    self._reused_observation_count = 0
                    self._reused_action_count = 0
                    self._joint_cache_reuse.clear()
                    self._rate_start = time.monotonic()

                # logger.info(f"observation: {observation} \n action:{action}")

                if self.dataset is not None:
                    observation_frame = build_dataset_frame(
                        self.dataset.features, observation, prefix=OBS_STR
                    )
                    action_frame = build_dataset_frame(
                        self.dataset.features, action, prefix=ACTION
                    )

                frame = {
                    **observation_frame,
                    **action_frame,
                    "task": self.record_cfg.single_task,
                }
                self.dataset.add_frame(frame)
                self.frame_count += 1

                dt_s = time.perf_counter() - start_loop_t

                if self.fps is not None:
                    busy_wait(1 / self.fps - dt_s)
        except Exception as exc:
            self.record_error = exc
            self.running = False
            logger.exception("录制线程写帧失败（已保存原始异常）")

    def stop(self):
        self.running = False
        if self.thread.ident is not None:
            self.thread.join()
        if self.dataset is not None:
            self.dataset.stop_audio_writer()

    def save(self) -> dict:
        if self.thread.is_alive():
            raise RuntimeError("录制线程尚未停止，不能保存 episode")
        if self.record_error is not None:
            raise RuntimeError(
                f"录制线程写帧失败: {type(self.record_error).__name__}: {self.record_error}"
            ) from self.record_error
        if self.frame_count == 0:
            raise ValueError("本次采集没有成功写入任何帧，请确认采集已启动及相机/action 数据就绪")
        print("will save_episode")

        episode_index = self.dataset.save_episode()

        print("save_episode succcess, episode_index:", episode_index)

        print(f"in Record stop record_cmd: {self.record_cmd}")

        update_dataid_json(self.record_cfg.root, episode_index, self.record_cmd)
        if episode_index == 0 and self.dataset.meta.total_episodes == 1:
            update_common_record_json(self.record_cfg.root, self.record_cmd)

        print("update_dataid_json succcess")

        if self.record_cfg.push_to_hub:
            self.dataset.push_to_hub(
                tags=self.record_cfg.tags, private=self.record_cfg.private
            )

        file_size = get_data_size(self.record_cfg.root, self.record_cmd)
        file_duration = get_data_duration(self.record_cfg.root, self.record_cmd)

        print("get_data_size succcess, file_size:", file_size)

        validate_result = validate_session(
            self.record_cfg.root,
            "episode_{episode_index:06d}".format(episode_index=episode_index),
        )
        print(f"Data validate complete, result:{validate_result}")

        verification = validate_result["verification"]

        data = {
            "file_message": {
                "file_name": self.record_cfg.repo_id,
                "file_local_path": str(self.record_cfg.root),
                "file_size": str(file_size),
                "file_duration": str(file_duration),
            },
            "verification": {
                "file_integrity": verification["file_integrity"],
                "camera_frame_rate": verification["camera_frame_rate"],
                "action_frame_rate": verification["action_frame_rate"],
                "file_integrity_comment": verification["file_integrity_comment"],
                "camera_frame_rate_comment": verification["camera_frame_rate_comment"],
                "action_frame_rate_comment": verification["action_frame_rate_comment"],
            },
        }

        self.record_complete = True
        self.last_record_episode_index = episode_index

        self.save_data = data

    def discard(self):
        if self.record_complete == True:
            delete_dataid_json(
                self.record_cfg.root, self.last_record_episode_index, self.record_cmd
            )
            self.dataset.remove_episode(self.last_record_episode_index)
        else:
            self.dataset.clear_episode_buffer()
