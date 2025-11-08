import time
import threading
import logging_mp

from deepdiff import DeepDiff
from dataclasses import dataclass

from operating_platform.robot.robots.configs import RobotConfig
from operating_platform.robot.robots.utils import Robot, busy_wait, safe_disconnect, make_robot_from_config

from operating_platform.dataset.dorobot_dataset import *
from operating_platform.robot.daemon import Daemon
from operating_platform.utils import parser
from operating_platform.utils.utils import has_method, log_say, get_current_git_branch, git_branch_log, get_container_ip_from_hosts

from operating_platform.utils.constants import DOROBOT_DATASET
from operating_platform.utils.data_file import (
    get_data_duration, 
    get_data_size ,
    update_dataid_json,
    update_common_record_json,
    delete_dataid_json,
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
        diff = DeepDiff(dataset_value, present_value, exclude_regex_paths=[r".*\['info'\]$"])
        if diff:
            mismatches.append(f"{field}: expected {present_value}, got {dataset_value}")

    if mismatches:
        raise ValueError(
            "Dataset metadata compatibility check failed with mismatches:\n" + "\n".join(mismatches)
        )


@dataclass
class RecordConfig():
    # Dataset identifier. By convention it should match '{hf_username}/{dataset_name}' (e.g. `lerobot/test`).
    repo_id: str
    # A short but accurate description of the task performed during the recording (e.g. "Pick the Lego block and drop it in the box on the right.")
    single_task: str = "TEST: no task description. Example: Pick apple."
    # Root directory where the dataset will be stored (e.g. 'dataset/path').
    root: str | Path | None = None
    # Limit the frames per second. By default, uses the policy fps.
    fps: int = 30

    # Encode frames in the dataset into video
    video: bool = False # 这是实际控制use_videos的地方

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
    def __init__(self, fps: int, robot: Robot, daemon: Daemon, record_cfg: RecordConfig, record_cmd: dict):
        self.robot = robot
        self.daemon = daemon
        self.record_cfg = record_cfg
        self.fps = fps
        self.record_cmd = record_cmd
        self.last_record_episode_index = 0
        self.record_complete = False
        self.save_data = None

        self.record_cfg.record_cmd = record_cmd

        print(f"in Record init record_cmd: {self.record_cmd}")

        if self.record_cfg.resume:
            self.dataset = DoRobotDataset(
                record_cfg.repo_id,
                root=record_cfg.root,
            )
            if len(robot.cameras) > 0:
                self.dataset.start_image_writer(
                    num_processes=record_cfg.num_image_writer_processes,
                    num_threads=record_cfg.num_image_writer_threads_per_camera * len(robot.cameras),
                )
            if len(robot.microphones) > 0:
                self.dataset.start_audio_writer(
                    microphones=robot.microphones,
                )
            sanity_check_dataset_robot_compatibility(self.dataset, robot, record_cfg.fps, record_cfg.video)
        else:
            self.dataset = DoRobotDataset.create(
                record_cfg.repo_id,
                record_cfg.fps,
                root=record_cfg.root,
                robot=robot,
                use_videos=record_cfg.video,
                use_audios=len(robot.microphones) > 0,
                image_writer_processes=record_cfg.num_image_writer_processes,
                image_writer_threads=record_cfg.num_image_writer_threads_per_camera * len(robot.cameras),
            )

        self.thread = threading.Thread(target=self.process, daemon=True)
        self.running = True

    def start(self):
        self.thread.start()
        self.running = True

    def process(self):
        while self.running:
            if self.dataset is not None:
                start_loop_t = time.perf_counter()

                observation = self.daemon.get_observation()
                action = self.daemon.get_obs_action()
                
                frame = {**observation, **action, "task": self.record_cfg.single_task}
                self.dataset.add_frame(frame)

                dt_s = time.perf_counter() - start_loop_t

                if self.fps is not None:
                    busy_wait(1 / self.fps - dt_s)


    def stop(self):
        if self.running == True:
            self.running = False
            self.thread.join()
            self.dataset.stop_audio_writer()


    def save(self) -> dict:
        print("will save_episode")

        episode_index = self.dataset.save_episode()

        print("save_episode succcess, episode_index:", episode_index)

        print(f"in Record stop record_cmd: {self.record_cmd}")

        update_dataid_json(self.record_cfg.root, episode_index,  self.record_cmd)
        if episode_index == 0 and self.dataset.meta.total_episodes == 1:
            update_common_record_json(self.record_cfg.root, self.record_cmd)
        
        print("update_dataid_json succcess")

        if self.record_cfg.push_to_hub:
            self.dataset.push_to_hub(tags=self.record_cfg.tags, private=self.record_cfg.private)

        file_size = get_data_size(self.record_cfg.root, self.record_cmd)
        file_duration = get_data_duration(self.record_cfg.root, self.record_cmd)

        print("get_data_size succcess, file_size:", file_size)

        validate_result = validate_session(self.record_cfg.root, "episode_{episode_index:06d}".format(episode_index = episode_index))
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
            }
        }

        self.record_complete = True
        self.last_record_episode_index = episode_index

        self.save_data = data

    def discard(self):
        if self.record_complete == True:
            delete_dataid_json(self.record_cfg.root, self.last_record_episode_index, self.record_cmd)
            self.dataset.remove_episode(self.last_record_episode_index)
        else:
            self.dataset.clear_episode_buffer()
