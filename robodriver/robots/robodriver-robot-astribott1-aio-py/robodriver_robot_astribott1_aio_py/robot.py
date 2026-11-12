import atexit
import time
from functools import cached_property
from typing import Any, Dict, List

import logging_mp
import numpy as np

from lerobot.cameras import make_cameras_from_configs
from lerobot.robots.robot import Robot
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from .config import AstribotT1AioPyRobotConfig
from .node import AstribotT1AioPyNode


logger = logging_mp.get_logger(__name__)
# command_forward 在停止遥操作后可能不再发布；超过此窗口按当前 obs 记录。
ACTION_STALE_S = 0.2


class AstribotT1AioPyRobot(Robot):
    config_class = AstribotT1AioPyRobotConfig
    name = "astribott1_aio_py"

    def __init__(self, config: AstribotT1AioPyRobotConfig):
        super().__init__(config)
        self.config = config
        self.robot_type = config.type
        self.use_videos = config.use_videos
        self.microphones = config.microphones

        self.leader_motors = config.leader_motors
        self.follower_motors = config.follower_motors
        self.cameras = make_cameras_from_configs(config.cameras)
        self.connect_excluded_cameras: List[str] = []

        # 通信节点（start() 在 connect() 时调用）
        self.robot_node = AstribotT1AioPyNode(
            record_parts=config.record_parts,
            camera_name_map=config.camera_name_map,
            camera_crops=config.camera_crops,
            ip=config.ip,
            node_name=config.node_name,
            ros_domain_id=config.ros_domain_id,
            ros_wait_first_frame_s=config.ros_wait_first_frame_s,
            camera_timeout=config.camera_timeout,
            control_mode=config.control_mode,
            takeover_priority=config.takeover_priority,
            takeover_sub_priority=config.takeover_sub_priority,
            force_takeover=config.force_takeover,
            control_parts=config.control_parts,
        )

        self.connected = False
        self.logs: Dict[str, Any] = {}
        self._obs_call_count = 0
        self._rate_start = time.monotonic()
        self._last_callback_counts: Dict[str, int] = {}
        self._last_action_callback_counts: Dict[str, int] = {}
        self._last_read_ts: Dict[str, float] = {}
        self._new_reads: Dict[str, int] = {}

    # ========= features =========

    @property
    def _follower_motors_ft(self) -> Dict[str, type]:
        """follower 关节特征：按 record_parts 过滤后的关节名生成。"""
        return {
            f"follower_{joint_name}.pos": float
            for comp_name, joints in self.follower_motors.items()
            for joint_name in joints.keys()
            if comp_name in self.config.record_parts
        }

    @property
    def _leader_motors_ft(self) -> Dict[str, type]:
        return {
            f"leader_{joint_name}.pos": float
            for comp_name, joints in self.leader_motors.items()
            for joint_name in joints.keys()
            if comp_name in self.config.record_parts
        }

    @property
    def _cameras_ft(self) -> Dict[str, tuple]:
        return {
            cam: (
                self.config.cameras[cam].height,
                self.config.cameras[cam].width,
                3,
            )
            for cam in self.cameras
            if cam in self.config.camera_name_map
        }

    @cached_property
    def observation_features(self) -> Dict[str, Any]:
        return {**self._follower_motors_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> Dict[str, Any]:
        return self._leader_motors_ft

    @property
    def is_connected(self) -> bool:
        return self.connected

    # ========= connect / disconnect =========

    def connect(self, calibrate: bool = True) -> None:
        if self.connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        self.robot_node.start()

        # 等待相机图像就绪
        timeout = self.config.camera_timeout
        start_time = time.perf_counter()
        while time.perf_counter() - start_time < timeout:
            if all(
                name in self.robot_node.recv_images
                for name in self.config.camera_name_map
                if name not in self.connect_excluded_cameras
            ):
                break
            time.sleep(0.1)

        missing = [
            name
            for name in self.config.camera_name_map
            if name not in self.robot_node.recv_images
            and name not in self.connect_excluded_cameras
        ]
        if missing:
            logger.warning(f"连接超时，以下相机未收到首帧: {missing}，继续启动")

        # 只统计连接完成后的回调，避免 SDK/相机启动时间稀释第一段频率。
        with self.robot_node.lock:
            self._last_callback_counts = dict(self.robot_node.recv_follower_count)
            self._last_action_callback_counts = dict(self.robot_node.recv_leader_count)
            self._last_read_ts = dict(self.robot_node.recv_follower_ts)
        self._rate_start = time.monotonic()
        self._obs_call_count = 0
        self._new_reads.clear()
        self.connected = True
        atexit.register(self.disconnect)
        logger.info(f"{self} connected")

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    def disconnect(self) -> None:
        if not self.connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        self.robot_node.stop()
        self.connected = False
        logger.info(f"{self} disconnected")

    def __del__(self):
        if getattr(self, "connected", False):
            self.disconnect()

    # ========= obs / action =========

    def get_observation(self) -> Dict[str, Any]:
        if not self.connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        start = time.perf_counter()
        obs_dict: Dict[str, Any] = {}

        node = self.robot_node
        with node.lock:
            follower_snapshot = {k: list(v) for k, v in node.recv_follower.items()}
            ts_snapshot = dict(node.recv_follower_ts)
            callback_counts = dict(node.recv_follower_count)
            action_callback_counts = dict(node.recv_leader_count)

        # 回调写入计数与采集读到的新帧数分别统计；缓存覆盖的中间帧不算采集帧。
        for part, ts in ts_snapshot.items():
            if ts != self._last_read_ts.get(part):
                self._new_reads[part] = self._new_reads.get(part, 0) + 1
            self._last_read_ts[part] = ts
        self._obs_call_count += 1
        if self._obs_call_count >= 200:
            now = time.monotonic()
            elapsed = now - self._rate_start
            if elapsed > 0:
                callback_hz = {
                    part: round((count - self._last_callback_counts.get(part, 0)) / elapsed, 1)
                    for part, count in callback_counts.items()
                }
                read_hz = {
                    part: round(count / elapsed, 1) for part, count in self._new_reads.items()
                }
                action_callback_hz = {
                    part: round((count - self._last_action_callback_counts.get(part, 0)) / elapsed, 1)
                    for part, count in action_callback_counts.items()
                }
                logger.info(
                    "[关节频率] obs回调=%s Hz; action回调=%s Hz; 采集轮询=%.1f Hz; obs读到新帧=%s Hz",
                    callback_hz, action_callback_hz, self._obs_call_count / elapsed, read_hz,
                )
            self._last_callback_counts = callback_counts
            self._last_action_callback_counts = action_callback_counts
            self._new_reads.clear()
            self._obs_call_count = 0
            self._rate_start = now

        for comp_name, joints in self.follower_motors.items():
            if comp_name not in self.config.record_parts:
                continue
            vec = follower_snapshot.get(comp_name, [])
            for idx, joint_name in enumerate(joints.keys()):
                if idx >= len(vec):
                    break
                obs_dict[f"follower_{joint_name}.pos"] = float(vec[idx])

        self.logs["read_joints_dt_s"] = time.perf_counter() - start

        start = time.perf_counter()
        with node.lock:
            images_snapshot = {k: v for k, v in node.recv_images.items()}

        for image_key in self.config.camera_name_map:
            frame = images_snapshot.get(image_key)
            if frame is None:
                logger.warning(f"相机 {image_key} 本帧无数据")
                continue
            obs_dict[image_key] = frame
        self.logs["read_cameras_dt_s"] = time.perf_counter() - start

        return obs_dict

    def get_action(self) -> Dict[str, Any]:
        if not self.connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        start = time.perf_counter()
        node = self.robot_node
        with node.lock:
            leader_snapshot = {k: list(v) for k, v in node.recv_leader.items()}
            follower_snapshot = {k: list(v) for k, v in node.recv_follower.items()}
            action_ts = dict(node.recv_leader_ts)
        act_dict = self._action_from_snapshot(leader_snapshot, follower_snapshot, action_ts)
        self.logs["read_action_dt_s"] = time.perf_counter() - start
        return act_dict

    def _action_from_snapshot(
        self,
        leader_snapshot: Dict[str, List[float]],
        follower_snapshot: Dict[str, List[float]],
        action_ts: Dict[str, float],
    ) -> Dict[str, float]:
        """Use recent commands, falling back to measured joints when missing or stale."""
        act_dict: Dict[str, float] = {}
        now = time.monotonic()
        for comp_name, joints in self.leader_motors.items():
            if comp_name not in self.config.record_parts:
                continue
            vec = leader_snapshot.get(comp_name, [])
            if now - action_ts.get(comp_name, float('-inf')) > ACTION_STALE_S:
                vec = []
            obs = follower_snapshot.get(comp_name, [])
            for idx, joint_name in enumerate(joints.keys()):
                if idx < len(vec):
                    value = vec[idx]
                elif idx < len(obs):
                    value = obs[idx]
                else:
                    raise RuntimeError(f"{comp_name} 的关节 {joint_name} 无 command_forward 和 obs 数据")
                act_dict[f"leader_{joint_name}.pos"] = float(value)
        return act_dict

    def get_record_joints(self) -> tuple[Dict[str, float], Dict[str, float], Dict[str, int]]:
        """Read ROS joint caches at the recorder's fps, independently of camera sampling."""
        if not self.connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        node = self.robot_node
        with node.lock:
            follower_snapshot = {k: list(v) for k, v in node.recv_follower.items()}
            leader_snapshot = {k: list(v) for k, v in node.recv_leader.items()}
            action_ts = dict(node.recv_leader_ts)
            obs_counts = dict(node.recv_follower_count)

        observation: Dict[str, float] = {}
        for part, joints in self.follower_motors.items():
            if part not in self.config.record_parts:
                continue
            positions = follower_snapshot.get(part, [])
            for index, name in enumerate(joints):
                if index >= len(positions):
                    raise RuntimeError(f"{part} 的关节 {name} 无 obs 数据")
                observation[f"follower_{name}.pos"] = float(positions[index])

        action = self._action_from_snapshot(leader_snapshot, follower_snapshot, action_ts)
        return observation, action, obs_counts

    # ========= send_action =========

    def send_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """下发关节位置指令（仅对 control_parts 中的部件生效）。"""
        if not self.is_connected:
            raise DeviceNotConnectedError(
                f"{self} is not connected. You need to run `robot.connect()`."
            )

        parts = [p for p in self.config.record_parts if p in self.config.control_parts]
        grouped: Dict[str, List[float]] = {part: [] for part in parts}

        for comp_name, joints in self.leader_motors.items():
            if comp_name not in parts:
                continue
            for joint_name in joints.keys():
                key = f"leader_{joint_name}.pos"
                if key not in action:
                    raise KeyError(f"action 缺少 {key}")
                grouped[comp_name].append(float(action[key]))

        if not parts:
            return {}

        self.robot_node.send_joints(
            grouped=grouped,
            control_parts=parts,
            freq=self.config.freq,
        )

        return {name.replace("leader_", ""): value for name, value in action.items()}

