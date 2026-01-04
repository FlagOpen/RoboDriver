import threading
import time
from typing import Any, Tuple

import logging_mp
import numpy as np
import torch
from lerobot.cameras import make_cameras_from_configs
from lerobot.robots.robot import Robot
from lerobot.utils.errors import DeviceNotConnectedError, DeviceAlreadyConnectedError
from functools import cached_property

from .config import AgilexAlohaAIODoraRobotConfig
from .status import AgilexAlohaAIODoraRobotStatus
from .node import AgilexAlohaAIODoraRobotNode


logger = logging_mp.get_logger(__name__)


class AgilexAlohaAIODoraRobot(Robot):
    config_class = AgilexAlohaAIODoraRobotConfig
    name = "agilex_aloha_aio_dora"

    def __init__(self, config: AgilexAlohaAIODoraRobotConfig):
        super().__init__(config)
        self.config = config
        self.robot_type = self.config.type
        self.use_videos = self.config.use_videos
        self.microphones = self.config.microphones

        self.right_leader_motors = config.right_leader_motors
        self.left_leader_motors = config.left_leader_motors
        self.right_follower_motors = config.right_follower_motors
        self.left_follower_motors = config.left_follower_motors
        self.cameras = make_cameras_from_configs(self.config.cameras)

        self.connect_excluded_cameras = ["image_pika_pose"]

        self.status = AgilexAlohaAIODoraRobotStatus()
        self.robot_dora_node = AgilexAlohaAIODoraRobotNode()
        self.robot_dora_node.start()

        self.connected = False
        self.logs = {}

    @property
    def _right_leader_motors_ft(self) -> dict[str, type]:
        return {f"right_leader_{motor}.pos": float for motor in self.right_leader_motors}

    @property
    def _left_leader_motors_ft(self) -> dict[str, type]:
        return {f"left_leader_{motor}.pos": float for motor in self.left_leader_motors}

    @property
    def _right_follower_motors_ft(self) -> dict[str, type]:
        return {f"right_leader_{motor}.pos": float for motor in self.right_follower_motors}

    @property
    def _left_follower_motors_ft(self) -> dict[str, type]:
        return {f"left_leader_{motor}.pos": float for motor in self.left_follower_motors}

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {
            cam: (self.config.cameras[cam].height, self.config.cameras[cam].width, 3) for cam in self.cameras
        }

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._right_follower_motors_ft, **self._left_follower_motors_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return {**self._right_leader_motors_ft, **self._left_leader_motors_ft}
    
    @property
    def is_connected(self) -> bool:
        return self.connected
    
    def connect(self):
        timeout = 20  # 统一的超时时间（秒）
        start_time = time.perf_counter()

        if self.connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        # 定义所有需要等待的条件及其错误信息
        conditions = [
            (
                lambda: all(
                    name in self.robot_dora_node.recv_images
                    for name in self.cameras
                    if name not in self.connect_excluded_cameras
                ),
                lambda: [name for name in self.cameras if name not in self.robot_dora_node.recv_images],
                "等待摄像头图像超时",
            ),
            (
                lambda: len(self.robot_dora_node.recv_leader_joint_right) > 0,
                lambda: [] if len(self.robot_dora_node.recv_leader_joint_right) > 0 else ["recv_leader_joint_right"],
                "等待右臂关节角度超时",
            ),
            (
                lambda: len(self.robot_dora_node.recv_leader_joint_left) > 0,
                lambda: [] if len(self.robot_dora_node.recv_leader_joint_left) > 0 else ["recv_leader_joint_left"],
                "等待左臂关节角度超时",
            ),
        ]

        # 跟踪每个条件是否已完成
        completed = [False] * len(conditions)

        while True:
            # 检查每个未完成的条件
            for i in range(len(conditions)):
                if not completed[i]:
                    condition_func = conditions[i][0]
                    if condition_func():
                        completed[i] = True

            # 如果所有条件都已完成，退出循环
            if all(completed):
                break

            # 检查是否超时
            if time.perf_counter() - start_time > timeout:
                failed_messages = []
                for i in range(len(completed)):
                    if not completed[i]:
                        condition_func, get_missing, base_msg = conditions[i]
                        missing = get_missing()

                        # 重新检查条件是否满足（可能刚好在最后一次检查后满足）
                        if condition_func():
                            completed[i] = True
                            continue

                        # 如果没有 missing，也视为满足
                        if not missing:
                            completed[i] = True
                            continue

                        # 构造错误信息
                        if i == 0:  # 摄像头条件
                            received = [
                                name for name in self.cameras if name not in missing
                            ]
                            msg = f"{base_msg}: 未收到 [{', '.join(missing)}]; 已收到 [{', '.join(received)}]"
                        else:  # 关节角度条件
                            # 对于关节角度条件，missing要么是空列表，要么是["recv_leader_joint_right"]或["recv_leader_joint_left"]
                            if i == 1:  # 右臂关节角度
                                data_source = self.robot_dora_node.recv_leader_joint_right
                            else:  # 左臂关节角度
                                data_source = self.robot_dora_node.recv_leader_joint_left
                            
                            received_count = len(data_source)
                            msg = f"{base_msg}: 未收到数据; 已收到 {received_count} 个数据点"

                        failed_messages.append(msg)

                # 如果所有条件都已完成，break
                if not failed_messages:
                    break

                # 抛出超时异常
                raise TimeoutError(
                    f"连接超时，未满足的条件: {'; '.join(failed_messages)}"
                )

            # 减少 CPU 占用
            time.sleep(0.01)

        # ===== 新增成功打印逻辑 =====
        success_messages = []
        # 摄像头连接状态
        if conditions[0][0]():
            cam_received = [
                name
                for name in self.cameras
                if name in self.robot_dora_node.recv_images and name not in self.connect_excluded_cameras
            ]
            success_messages.append(f"摄像头: {', '.join(cam_received)}")

        # 右臂关节角度状态
        if conditions[1][0]():
            # 检查recv_leader_joint_right字典中实际接收到的键
            success_messages.append(f"右臂关节角度: 已接收 ({len(self.robot_dora_node.recv_leader_joint_right)}个数据点)")

        # 左臂关节角度状态
        if conditions[2][0]():
            # 检查recv_leader_joint_left字典中实际接收到的键
            success_messages.append(f"左臂关节角度: 已接收 ({len(self.robot_dora_node.recv_leader_joint_left)}个数据点)")

        log_message = "\n[连接成功] 所有设备已就绪:\n"
        log_message += "\n".join(f"  - {msg}" for msg in success_messages)
        log_message += f"\n  总耗时: {time.perf_counter() - start_time:.2f} 秒\n"
        logger.info(log_message)
        # ===========================

        for i in range(self.status.specifications.camera.number):
            self.status.specifications.camera.information[i].is_connect = True
        for i in range(self.status.specifications.arm.number):
            self.status.specifications.arm.information[i].is_connect = True

        self.connected = True

    @property
    def is_calibrated(self) -> bool:
        """Whether the robot is currently calibrated or not. Should be always `True` if not applicable"""
        return True

    def calibrate(self) -> None:
        """
        Calibrate the robot if applicable. If not, this should be a no-op.

        This method should collect any necessary data (e.g., motor offsets) and update the
        :pyattr:`calibration` dictionary accordingly.
        """
        pass

    def configure(self) -> None:
        """
        Apply any one-time or runtime configuration to the robot.
        This may include setting motor parameters, control modes, or initial state.
        """
        pass

    def get_observation(self) -> dict[str, Any]:
        if not self.connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        for key in self.robot_dora_node.recv_images_status:
            self.robot_dora_node.recv_images_status[key] = max(0, self.robot_dora_node.recv_images_status[key] - 1)
        self.robot_dora_node.recv_follower_joint_right_status = max(0, self.robot_dora_node.recv_follower_joint_right_status - 1)
        self.robot_dora_node.recv_follower_joint_left_status = max(0, self.robot_dora_node.recv_follower_joint_left_status - 1)

        # Read arm position
        start = time.perf_counter()
        obs_dict = {}
        
        # Add right arm positions
        for i, motor in enumerate(self.right_follower_motors):
            if "joint" in motor:
                obs_dict[f"right_follower_{motor}.pos"] = self.robot_dora_node.recv_follower_joint_right[i]
            if "pose" in motor:
                obs_dict[f"right_follower_{motor}.pos"] = self.robot_dora_node.recv_follower_endpose_right[i-6]
            if "gripper" in motor:
                obs_dict[f"right_follower_{motor}.pos"] = self.robot_dora_node.recv_follower_joint_right[i-6]
                
        
        # Add left arm positions
        for i, motor in enumerate(self.left_follower_motors):
            if "joint" in motor:
                obs_dict[f"left_follower_{motor}.pos"] = self.robot_dora_node.recv_follower_joint_left[i]
            if "pose" in motor:
                obs_dict[f"left_follower_{motor}.pos"] = self.robot_dora_node.recv_follower_endpose_left[i-6]
            if "gripper" in motor:
                obs_dict[f"left_follower_{motor}.pos"] = self.robot_dora_node.recv_follower_joint_left[i-6]
                
        
        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read state: {dt_ms:.1f} ms")

        # Capture images from cameras
        for cam_key, _cam in self.cameras.items():
            start = time.perf_counter()
            
            for name, val in self.robot_dora_node.recv_images.items():
                if cam_key in name:
                    obs_dict[cam_key] = val
            
            dt_ms = (time.perf_counter() - start) * 1e3
            logger.debug(f"{self} read {cam_key}: {dt_ms:.1f} ms")

        return obs_dict
    
    def get_action(self) -> Tuple[dict[str, Any], dict[str, Any]]:
        if not self.connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")


        self.robot_dora_node.recv_leader_joint_right_status = max(0, self.robot_dora_node.recv_leader_joint_right_status - 1)
        self.robot_dora_node.recv_leader_joint_left_status = max(0, self.robot_dora_node.recv_leader_joint_left_status - 1)
        
        # Read arm position
        start = time.perf_counter()
        act_dict = {}


        # Add right arm positions
        for i, motor in enumerate(self.right_leader_motors):
            if "joint" in motor:
                act_dict[f"right_leader_{motor}.pos"] = self.robot_dora_node.recv_leader_joint_right[i]
            if "pose" in motor:
                act_dict[f"right_leader_{motor}.pos"] = self.robot_dora_node.recv_follower_endpose_right[i-6]
            if "gripper" in motor:
                act_dict[f"right_leader_{motor}.pos"] = self.robot_dora_node.recv_leader_joint_right[i-6]
                
        
        # Add left arm positions
        for i, motor in enumerate(self.left_leader_motors):
            if "joint" in motor:
                act_dict[f"left_leader_{motor}.pos"] = self.robot_dora_node.recv_leader_joint_left[i]
            if "pose" in motor:
                act_dict[f"left_leader_{motor}.pos"] = self.robot_dora_node.recv_follower_endpose_left[i-6]
            if "gripper" in motor:
                act_dict[f"left_leader_{motor}.pos"] = self.robot_dora_node.recv_leader_joint_left[i-6]
        
        # # Add right arm positions
        # for name, val in self.robot_dora_node.recv_leader_joint_right.items():
        #     for motor in self.right_leader_motors:
        #         if motor in name:
        #             act_dict[f"right_leader_{motor}.pos"] = val
        
        # # Add left arm positions
        # for name, val in self.robot_dora_node.recv_leader_joint_left.items():
        #     for motor in self.left_leader_motors:
        #         if motor in name:
        #             act_dict[f"left_leader_{motor}.pos"] = val
        
        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read action: {dt_ms:.1f} ms")

        return act_dict

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        """The provided action is expected to be a vector."""
        if not self.is_connected:
            raise DeviceNotConnectedError(
                f"{self} is not connected. You need to run `robot.connect()`."
            )
        
        # Separate right and left arm actions
        right_arm_action = []
        left_arm_action = []
        
        for key, val in action.items():
            if "right_leader" in key:
                right_arm_action.append(val)
            elif "left_leader" in key:
                left_arm_action.append(val)
        
        # Send right arm action
        if right_arm_action:
            goal_joint_numpy = np.array(right_arm_action, dtype=np.float32)
            self.robot_dora_node.dora_send(f"action_joint_right", goal_joint_numpy)
        
        # Send left arm action
        if left_arm_action:
            goal_joint_numpy = np.array(left_arm_action, dtype=np.float32)
            self.robot_dora_node.dora_send(f"action_joint_left", goal_joint_numpy)
        
        return {f"{arm_motor}.pos": val for arm_motor, val in action.items()}

    def update_status(self) -> str:
        for i in range(self.status.specifications.camera.number):
            match_name = self.status.specifications.camera.information[i].name
            for name in self.robot_dora_node.recv_images_status:
                if match_name in name:
                    self.status.specifications.camera.information[i].is_connect = (
                        True if self.robot_dora_node.recv_images_status[name] > 0 else False
                    )

        self.status.specifications.arm.information[0].is_connect = (
            True if self.robot_dora_node.recv_leader_joint_right_status > 0 else False
        )
        self.status.specifications.arm.information[1].is_connect = (
            True if self.robot_dora_node.recv_leader_joint_left_status > 0 else False
        )

        return self.status.to_json()

    def disconnect(self):
        if not self.is_connected:
            raise DeviceNotConnectedError(
                "Agilex Aloha is not connected. You need to run `robot.connect()` before disconnecting."
            )

        self.robot_dora_node.running = False
        self.robot_dora_node.stop()

        self.connected = False

    def __del__(self):
        if getattr(self, "is_connected", False):
            self.disconnect()
