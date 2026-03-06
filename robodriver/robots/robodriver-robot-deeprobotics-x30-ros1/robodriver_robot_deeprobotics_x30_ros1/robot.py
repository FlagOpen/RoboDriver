
import time
from functools import cached_property
from typing import Any

import numpy as np
import logging_mp

from lerobot.cameras import make_cameras_from_configs
from lerobot.utils.errors import DeviceNotConnectedError, DeviceAlreadyConnectedError
from lerobot.robots.robot import Robot

from .config import DeeproboticsX30Ros1RobotConfig
from .node import DeeproboticsX30Ros1Node


logger = logging_mp.get_logger(__name__)


class DeeproboticsX30Ros1Robot(Robot):
    config_class = DeeproboticsX30Ros1RobotConfig
    name = "deeprobotics_x30_ros1"

    def __init__(self, config: DeeproboticsX30Ros1RobotConfig):
        super().__init__(config)
        self.config = config
        self.robot_type = self.config.type
        self.use_videos = self.config.use_videos
        self.microphones = self.config.microphones

        self.leader_motors = config.leader_motors
        self.follower_motors = config.follower_motors
        self.cameras = make_cameras_from_configs(self.config.cameras)

        self.connect_excluded_cameras = ["image_pika_pose"]

        self.robot_ros2_node = DeeproboticsX30Ros1Node()
        self.robot_ros2_node.start()

        self.connected = False
        self.logs = {}

    # ========= features =========

    @property
    def _follower_motors_ft(self) -> dict[str, type]:
        return {
            f"follower_{joint_name}.pos": float
            for comp_name, joints in self.follower_motors.items()
            for joint_name in joints.keys()
        }
    
    @property
    def _leader_motors_ft(self) -> dict[str, type]:
        return {
            f"leader_{joint_name}.pos": float
            for comp_name, joints in self.leader_motors.items()
            for joint_name in joints.keys()
        }

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {
            cam: (
                self.config.cameras[cam].height,
                self.config.cameras[cam].width,
                3,
            )
            for cam in self.cameras
        }

    @cached_property
    def observation_features(self) -> dict[str, Any]:
        return {**self._follower_motors_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, Any]:
        return self._leader_motors_ft

    @property
    def is_connected(self) -> bool:
        return self.connected

    # ========= connect / disconnect =========

    def connect(self):
        timeout = 20
        start_time = time.perf_counter()

        if self.connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        node = self.robot_ros2_node

        conditions = [
            # 摄像头图像
            (
                lambda: all(
                    name in node.recv_images
                    for name in self.cameras
                    if name not in self.connect_excluded_cameras
                ),
                lambda: [
                    name
                    for name in self.cameras
                    if name not in node.recv_images
                    and name not in self.connect_excluded_cameras
                ],
                "等待摄像头图像超时",
            ),
            # 主臂
            (
                lambda: all(
                    any(name in key for key in node.recv_leader)
                    for name in self.leader_motors
                ),
                lambda: [
                    name
                    for name in self.leader_motors
                    if not any(name in key for key in node.recv_leader)
                ],
                "等待主臂数据超时",
            ),
            # 从臂
            (
                lambda: all(
                    any(name in key for key in node.recv_follower)
                    for name in self.follower_motors
                ),
                lambda: [
                    name
                    for name in self.follower_motors
                    if not any(name in key for key in node.recv_follower)
                ],
                "等待从臂数据超时",
            ),
        ]

        completed = [False] * len(conditions)

        while True:            
            for i, (cond, _get_missing, _msg) in enumerate(conditions):
                if not completed[i] and cond():
                    completed[i] = True

            if all(completed):
                break

            if time.perf_counter() - start_time > timeout:
                failed_messages = []
                for i, (cond, get_missing, base_msg) in enumerate(conditions):
                    if completed[i]:
                        continue

                    missing = get_missing()
                    if cond() or not missing:
                        completed[i] = True
                        continue

                    if i == 0:
                        received = [
                            name
                            for name in self.cameras
                            if name not in missing
                        ]
                    elif i == 1:
                        received = [
                            name
                            for name in self.leader_motors
                            if name not in missing
                        ]
                    else:
                        received = [
                            name
                            for name in self.follower_motors
                            if name not in missing
                        ]

                    msg = (
                        f"{base_msg}: 未收到 [{', '.join(missing)}]; "
                        f"已收到 [{', '.join(received)}]"
                    )
                    failed_messages.append(msg)

                if not failed_messages:
                    break

                raise TimeoutError(
                    f"连接超时，未满足的条件: {{'; '.join(failed_messages)}}"
                )

            time.sleep(0.01)

        # 成功日志
        success_messages = []

        if conditions[0][0]():
            cam_received = [
                name
                for name in self.cameras
                if name in node.recv_images
                and name not in self.connect_excluded_cameras
            ]
            success_messages.append(f"摄像头: {', '.join(cam_received)}")

        if conditions[1][0]():
            leader_received = [
                name
                for name in self.leader_motors
                if any(name in key for key in node.recv_leader)
            ]
            success_messages.append(f"主臂数据: {', '.join(leader_received)}")

        if conditions[2][0]():
            follower_received = [
                name
                for name in self.follower_motors
                if any(name in key for key in node.recv_follower)
            ]
            success_messages.append(f"从臂数据: {', '.join(follower_received)}")

        log_message = "\n[连接成功] 所有设备已就绪:\n"
        log_message += "\n".join(f"  - {msg}" for msg in success_messages)
        log_message += f"\n  总耗时: {time.perf_counter() - start_time:.2f} 秒\n"
        logger.info(log_message)

        self.connected = True

    def disconnect(self):
        if not self.connected:
            raise DeviceNotConnectedError()
        self.connected = False

    def __del__(self):
        if getattr(self, "connected", False):
            self.disconnect()

    # ========= calibrate / configure =========

    def calibrate(self):
        pass

    def configure(self):
        pass

    @property
    def is_calibrated(self):
        return True

    # ========= obs / action =========

    def get_observation(self) -> dict[str, Any]:
        if not self.connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        start = time.perf_counter()
        obs_dict: dict[str, Any] = {}

        node = self.robot_ros2_node

        # follower joints
        for comp_name, joints in self.follower_motors.items():
            state_dict = node.recv_follower.get(comp_name)
            if state_dict:
                position = state_dict.get("position", [])
                name = state_dict.get("name", [])
                
                for joint_name, pos in zip(name, position):
                    obs_dict[f"follower_{joint_name}.pos"] = float(pos)
        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read follower state: {dt_ms:.1f} ms")

        # camera images
        for cam_key, _cam in self.cameras.items():
            start = time.perf_counter()
            for name, val in node.recv_images.items():
                if cam_key == name or cam_key in name:
                    obs_dict[cam_key] = val
                    break
            dt_ms = (time.perf_counter() - start) * 1e3
            logger.debug(f"{self} read {cam_key}: {dt_ms:.1f} ms")

        return obs_dict
    
    def get_action(self) -> dict[str, Any]:
        if not self.connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        start = time.perf_counter()
        act_dict: dict[str, Any] = {}

        node = self.robot_ros2_node

        for comp_name, joints in self.leader_motors.items():
            state_dict = node.recv_leader.get(comp_name)
            if state_dict:
                position = state_dict.get("position", [])
                name = state_dict.get("name", [])
                
                for joint_name, pos in zip(name, position):
                    act_dict[f"leader_{joint_name}.pos"] = float(pos)

        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read action: {dt_ms:.1f} ms")

        return act_dict

    # ========= send_action =========

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        """The provided action is expected to be a vector."""
        if not self.is_connected:
            raise DeviceNotConnectedError(
                f"{self} is not connected. You need to run `robot.connect()`."
            )

        goal_joint = [val for _key, val in action.items()]
        goal_joint_numpy = np.array(goal_joint, dtype=np.float32)

        # 当前 ZMQ Node 没有实现下行控制，如果以后需要，可以在 node 中加一个 ZMQ PUB
        if hasattr(self.robot_ros2_node, "ros2_send"):
            self.robot_ros2_node.ros2_send("action_joint", goal_joint_numpy)

        return {f"{arm_motor}.pos": val for arm_motor, val in action.items()}
