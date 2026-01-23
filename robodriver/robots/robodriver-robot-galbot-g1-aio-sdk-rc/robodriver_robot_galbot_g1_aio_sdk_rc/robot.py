import time
import logging_mp
import numpy as np

from functools import cached_property
from typing import Any

from lerobot.cameras import make_cameras_from_configs
from lerobot.utils.errors import DeviceNotConnectedError, DeviceAlreadyConnectedError
from lerobot.robots.robot import Robot

from .config import GalbotG1AIOSDKRCRobotConfig
from .node import GalbotG1AIOSDKRCRobotNode


logger = logging_mp.get_logger(__name__)


class GalbotG1AIOSDKRCRobot(Robot):
    config_class = GalbotG1AIOSDKRCRobotConfig
    name = "galbot_g1_aio_sdk_rc"

    def __init__(self, config: GalbotG1AIOSDKRCRobotConfig):
        super().__init__(config)
        self.config = config
        self.robot_type = self.config.type
        self.use_videos = self.config.use_videos
        self.microphones = self.config.microphones

        self.leader_motors = config.leader_motors
        self.follower_motors = config.follower_motors
        self.cameras = make_cameras_from_configs(self.config.cameras)

        self.connect_excluded_cameras = ["image_pika_pose"]

        self.robot_node = GalbotG1AIOSDKRCRobotNode("127.0.0.1")
        self.robot_node.start()

        self.connected = False
        self.logs = {}

    # ========= features =========
    @property
    def _follower_motors_ft(self) -> dict[str, type]:
        return {
            f"follower_{motor}.pos": float
            for motor in self.follower_motors
        }
    
    @property
    def _leader_motors_ft(self) -> dict[str, type]:
        return {
            f"leader_{motor}.pos": float
            for motor in self.leader_motors
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
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._follower_motors_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return self._leader_motors_ft

    # ========= connect / disconnect =========
    @property
    def is_connected(self) -> bool:
        return self.connected
    
    def connect(self):
        timeout = 20
        start_time = time.perf_counter()

        if self.connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        # 约定：node 里有 recv_images / recv_follower / recv_leader
        conditions = [
            # 摄像头图像
            (
                lambda: all(
                    name in self.robot_node.recv_images
                    for name in self.cameras
                    if name not in self.connect_excluded_cameras
                ),
                lambda: [
                    name
                    for name in self.cameras
                    if name not in self.robot_node.recv_images
                    and name not in self.connect_excluded_cameras
                ],
                "等待摄像头图像超时",
            ),
            (
                lambda: len(self.robot_node.recv_follower_arm_right) > 0,
                lambda: [] if len(self.robot_node.recv_follower_arm_right) > 0 else ["recv_follower_joint_right"],
                "等待右臂关节角度超时",
            ),
            (
                lambda: len(self.robot_node.recv_follower_arm_left) > 0,
                lambda: [] if len(self.robot_node.recv_follower_arm_left) > 0 else ["recv_follower_joint_left"],
                "等待左臂关节角度超时",
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
                if name in self.robot_node.recv_images
                and name not in self.connect_excluded_cameras
            ]
            success_messages.append(f"摄像头: {', '.join(cam_received)}")

        if conditions[1][0]():
            success_messages.append(f"右臂关节角度: 已接收 ({len(self.robot_node.recv_follower_arm_right)}个数据点)")


        if conditions[2][0]():
            success_messages.append(f"左臂关节角度: 已接收 ({len(self.robot_node.recv_follower_arm_left)}个数据点)")


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

        for i, motor in enumerate(self.follower_motors):
            if "arm" in motor and "right" in motor:
                obs_dict[f"follower_{motor}.pos"] = self.robot_node.recv_follower_arm_right[i-1]
            elif "gripper" in motor and "right" in motor:
                obs_dict[f"follower_{motor}.pos"] = self.robot_node.recv_follower_gripper_right[0]

            elif "arm" in motor and "left" in motor:
                obs_dict[f"follower_{motor}.pos"] = self.robot_node.recv_follower_arm_left[i-9]
            elif "gripper" in motor and "left" in motor:
                obs_dict[f"follower_{motor}.pos"] = self.robot_node.recv_follower_gripper_left[0]

            elif "leg" in motor:
                obs_dict[f"follower_{motor}.pos"] = self.robot_node.recv_follower_leg[i-17]
            elif "head" in motor:
                obs_dict[f"follower_{motor}.pos"] = self.robot_node.recv_follower_head[i-22]
            elif "chassis" in motor:
                obs_dict[f"follower_{motor}.pos"] = self.robot_node.recv_follower_chassis[i-24]
            

        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read follower state: {dt_ms:.1f} ms")

        for cam_key, _cam in self.cameras.items():
            start = time.perf_counter()
            for name, val in self.robot_node.recv_images.items():
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

        for i, motor in enumerate(self.follower_motors):
            if "arm" in motor and "right" in motor:
                act_dict[f"leader_{motor}.pos"] = self.robot_node.recv_follower_arm_right[i-1]
            elif "gripper" in motor and "right" in motor:
                act_dict[f"leader_{motor}.pos"] = self.robot_node.recv_follower_gripper_right[0]

            elif "arm" in motor and "left" in motor:
                act_dict[f"leader_{motor}.pos"] = self.robot_node.recv_follower_arm_left[i-9]
            elif "gripper" in motor and "left" in motor:
                act_dict[f"leader_{motor}.pos"] = self.robot_node.recv_follower_gripper_left[0]

            elif "leg" in motor:
                act_dict[f"leader_{motor}.pos"] = self.robot_node.recv_follower_leg[i-17]
            elif "head" in motor:
                act_dict[f"leader_{motor}.pos"] = self.robot_node.recv_follower_head[i-22]

            elif "chassis" in motor:
                act_dict[f"leader_{motor}.pos"] = self.robot_node.recv_follower_chassis[i-24]

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

        # goal_joint = [val for _key, val in action.items()]
        # goal_joint_numpy = np.array(goal_joint, dtype=np.float32)

        # Extract motor names from keys like 'leader_elbow.pos' -> 'elbow'
        cleaned_action = {}
        for key, value in action.items():
            if key.startswith("leader_") and key.endswith(".pos"):
                motor = key[len("leader_"):-len(".pos")]
                cleaned_action[motor] = value
            else:
                raise ValueError(f"Unexpected action key format: {key}. Expected 'leader_{{motor}}.pos'.")

        # Send the cleaned action to the ROS 2 node
        # self.robot_ros2_node.ros2_send(cleaned_action)

        return {f"{arm_motor}.pos": val for arm_motor, val in action.items()}

