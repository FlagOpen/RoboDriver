import atexit
import time
from functools import cached_property
from typing import Any, Dict, List

import logging_mp
import numpy as np

from lerobot.cameras import make_cameras_from_configs
from lerobot.robots.robot import Robot
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from .config import AstribotT1AioPyRobotConfig, AstribotT1AioSharpaPyRobotConfig
from .node import AstribotT1AioPyNode
from .sharpa_hand import SharpaHands


logger = logging_mp.get_logger(__name__)


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

        # recv_follower 由 ROS spin 线程的 _on_obs 回调持续更新，直接读缓存
        node = self.robot_node
        with node.lock:
            follower_snapshot = {k: list(v) for k, v in node.recv_follower.items()}

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
        act_dict: Dict[str, Any] = {}

        # recv_leader 由 _on_action 回调更新；静止时无 command_forward，vec 为空
        node = self.robot_node
        with node.lock:
            leader_snapshot = {k: list(v) for k, v in node.recv_leader.items()}

        for comp_name, joints in self.leader_motors.items():
            if comp_name not in self.config.record_parts:
                continue
            vec = leader_snapshot.get(comp_name, [])
            for idx, joint_name in enumerate(joints.keys()):
                if idx >= len(vec):
                    break
                act_dict[f"leader_{joint_name}.pos"] = float(vec[idx])

        self.logs["read_action_dt_s"] = time.perf_counter() - start
        return act_dict

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


class AstribotT1AioSharpaPyRobot(AstribotT1AioPyRobot):
    """AstriBot T1 AIO + Sharpa Wave 灵巧手组合。

    手臂/AstriBot 部分完全复用父类；灵巧手通过厂商 SDK 以以太网接入，
    每只手 22 个关节按 `hand_<side>_joint1..22` 记录。
    """

    config_class = AstribotT1AioSharpaPyRobotConfig
    name = "astribott1_aio_sharpa_py"

    def __init__(self, config: AstribotT1AioSharpaPyRobotConfig):
        super().__init__(config)
        self.config = config
        self._sharpa_hands = SharpaHands(config.sharpa_hands)

    @property
    def observation_features(self) -> Dict[str, Any]:
        features = dict(super().observation_features)
        features.update(self._sharpa_hands.follower_features())
        return features

    @property
    def action_features(self) -> Dict[str, Any]:
        features = dict(super().action_features)
        features.update(self._sharpa_hands.leader_features())
        return features

    def connect(self, calibrate: bool = True) -> None:
        super().connect(calibrate=calibrate)
        try:
            self._sharpa_hands.connect()
        except Exception:
            self.disconnect()
            raise

    def disconnect(self) -> None:
        try:
            self._sharpa_hands.disconnect()
        finally:
            super().disconnect()

    def get_observation(self) -> Dict[str, Any]:
        observation = super().get_observation()
        if self._sharpa_hands.connected:
            observation.update(self._sharpa_hands.get_observation())
        return observation

    def get_action(self) -> Dict[str, Any]:
        action = super().get_action()
        if self._sharpa_hands.connected:
            action.update(self._sharpa_hands.get_action())
        return action

    def send_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        body_action = {
            key: value for key, value in action.items() if not key.startswith("leader_hand_")
        }
        hand_action = {
            key: value for key, value in action.items() if key.startswith("leader_hand_")
        }

        sent = super().send_action(body_action) if body_action else {}
        if hand_action:
            sent.update(self._sharpa_hands.send_action(hand_action))
        return sent

