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

from .config import SimFrankaAIODoraRobotConfig
# from .status import SimFrankaAIODoraRobotStatus
from .node import SimFrankaAIODoraRobotNode


logger = logging_mp.getLogger(__name__)


class SimFrankaAIODoraRobot(Robot):
    config_class = SimFrankaAIODoraRobotConfig
    name = "genesis_franka_aio_dora"

    def __init__(self, config: SimFrankaAIODoraRobotConfig):
        super().__init__(config)
        self.config = config
        self.robot_type = self.config.type
        self.use_videos = self.config.use_videos
        self.microphones = self.config.microphones

        self.leader_motors = config.leader_motors
        self.follower_motors = config.follower_motors
        self.cameras = make_cameras_from_configs(self.config.cameras)

        self.connect_excluded_cameras = ["image_pika_pose"]

        # self.status = SO101AIODoraRobotStatus()
        self.robot_dora_node = SimFrankaAIODoraRobotNode()
        self.robot_dora_node.start()

        self.connected = False
        self.logs = {}

    @property
    def _leader_motors_ft(self) -> dict[str, type]:
        return {f"leader_{motor}.pos": float for motor in self.leader_motors}

    @property
    def _follower_motors_ft(self) -> dict[str, type]:
        return {f"follower_{motor}.pos": float for motor in self.follower_motors}

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {
            cam: (self.config.cameras[cam].height, self.config.cameras[cam].width, 3) for cam in self.cameras
        }

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._follower_motors_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return self._leader_motors_ft
    
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
            # (
            #     lambda: all(
            #         any(name in key for key in self.robot_dora_node.recv_joint_leader)
            #         for name in self.leader_motors
            #     ),
            #     lambda: [
            #         name
            #         for name in self.leader_motors
            #         if not any(name in key for key in self.robot_dora_node.recv_joint_leader)
            #     ],
            #     "等待主臂关节角度超时",
            # ),
            (
                lambda: all(
                    any(name in key for key in self.robot_dora_node.recv_joint_follower)
                    for name in self.follower_motors
                ),
                lambda: [
                    name
                    for name in self.follower_motors
                    if not any(name in key for key in self.robot_dora_node.recv_joint_follower)
                ],
                "等待从臂关节角度超时",
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

                        # 计算已接收的项
                        if i == 0:
                            received = [
                                name for name in self.cameras if name not in missing
                            ]
                        else:
                            received = [
                                name
                                for name in self.follower_motors
                                if name not in missing
                            ]

                        # 构造错误信息
                        msg = f"{base_msg}: 未收到 [{', '.join(missing)}]; 已收到 [{', '.join(received)}]"
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

        # # 主臂数据状态
        # arm_data_types = [
        #     "主臂关节角度",
        # ]
        # for i, data_type in enumerate(arm_data_types, 1):
        #     if conditions[i][0]():
        #         arm_received = [
        #             name
        #             for name in self.leader_motors
        #             if any(name in key for key in (self.robot_dora_node.recv_joint_leader,)[i - 1])
        #         ]
        #         success_messages.append(f"{data_type}: {', '.join(arm_received)}")

        # 从臂数据状态
        arm_data_types = [
            "从臂关节角度",
        ]
        for i, data_type in enumerate(arm_data_types, 1):
            if conditions[i][0]():
                arm_received = [
                    name
                    for name in self.follower_motors
                    if any(name in key for key in (self.robot_dora_node.recv_joint_follower,)[i - 1])
                ]
                success_messages.append(f"{data_type}: {', '.join(arm_received)}")

        log_message = "\n[连接成功] 所有设备已就绪:\n"
        log_message += "\n".join(f"  - {msg}" for msg in success_messages)
        log_message += f"\n  总耗时: {time.perf_counter() - start_time:.2f} 秒\n"
        logger.info(log_message)
        # ===========================

        # for i in range(self.status.specifications.camera.number):
        #     self.status.specifications.camera.information[i].is_connect = True
        # for i in range(self.status.specifications.arm.number):
        #     self.status.specifications.arm.information[i].is_connect = True

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

        # for key in self.robot_dora_node.recv_images_status:
        #     self.robot_dora_node.recv_images_status[key] = max(0, self.robot_dora_node.recv_images_status[key] - 1)
        # for key in self.robot_dora_node.recv_joint_follower_status:
        #     self.robot_dora_node.recv_joint_follower_status[key] = max(0, self.robot_dora_node.recv_joint_follower_status[key] - 1)

        # Read arm position
        start = time.perf_counter()
        obs_dict = {
            f"follower_{motor}.pos": val 
            for name, val in self.robot_dora_node.recv_joint_follower.items() 
                for motor in self.follower_motors
                    if motor in name
        }
        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read state: {dt_ms:.1f} ms")

        # Capture images from cameras
        for cam_key, _cam in self.cameras.items():
            
            start = time.perf_counter()

            # obs_dict[cam_key] = cam.async_read()
            for name, val in self.robot_dora_node.recv_images.items():
                if cam_key in name:
                    obs_dict[cam_key] = val
            
            # self.robot_dora_node.recv_images[name]
            
            dt_ms = (time.perf_counter() - start) * 1e3
            logger.debug(f"{self} read {cam_key}: {dt_ms:.1f} ms")

        return obs_dict
    
    def get_action(self) -> Tuple[dict[str, Any], dict[str, Any]]:
        if not self.connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        # for key in self.robot_dora_node.recv_joint_leader_status:
        #     self.robot_dora_node.recv_joint_leader_status[key] = max(0, self.robot_dora_node.recv_joint_leader_status[key] - 1)
        
        # Read arm position
        start = time.perf_counter()
        act_dict = {
            f"leader_{motor}.pos": val 
            for name, val in self.robot_dora_node.recv_joint_follower.items() 
                for motor in self.leader_motors
                    if motor in name
        }
        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read action: {dt_ms:.1f} ms")

        return act_dict

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        """The provided action is expected to be a vector."""
        if not self.is_connected:
            raise DeviceNotConnectedError(
                f"{self} is not connected. You need to run `robot.connect()`."
            )
        
        goal_joint = [ val for _key, val in action.items()]

        goal_joint_numpy = np.array(goal_joint, dtype=np.float32)
        # goal_gripper_numpy = np.array([t.item() for t in goal_gripper], dtype=np.float32)
        # position = np.concatenate([goal_joint_numpy, goal_gripper_numpy], axis=0)

        # logger.debug(f"action: {action}, goal_joint:{goal_joint}, goal_joint_numpy:{goal_joint_numpy}")
        self.robot_dora_node.dora_send(f"action_joint", goal_joint_numpy)
        
        return {f"{arm_motor}.pos": val for arm_motor, val in action.items()}

    # def update_status(self) -> str:
    #     for i in range(self.status.specifications.camera.number):
    #         match_name = self.status.specifications.camera.information[i].name
    #         for name in self.robot_dora_node.recv_images_status:
    #             if match_name in name:
    #                 self.status.specifications.camera.information[i].is_connect = (
    #                     True if self.robot_dora_node.recv_images_status[name] > 0 else False
    #                 )

    #     for i in range(self.status.specifications.arm.number):
    #         match_name = self.status.specifications.arm.information[i].name
    #         for name in self.robot_dora_node.recv_joint_leader_status:
    #             if match_name in name:
    #                 self.status.specifications.arm.information[i].is_connect = (
    #                     True if self.robot_dora_node.recv_joint_leader_status[name] > 0 else False
    #                 )

    #     for i in range(self.status.specifications.arm.number):
    #         match_name = self.status.specifications.arm.information[i].name
    #         for name in self.robot_dora_node.recv_joint_follower_status:
    #             if match_name in name:
    #                 self.status.specifications.arm.information[i].is_connect = (
    #                     True if self.robot_dora_node.recv_joint_follower_status[name] > 0 else False
    #                 )

    #     return self.status.to_json()

    def disconnect(self):
        if not self.is_connected:
            raise DeviceNotConnectedError(
                "Aloha is not connected. You need to run `robot.connect()` before disconnecting."
            )

        self.robot_dora_node.running = False
        self.robot_dora_node.stop()

        self.connected = False

    def __del__(self):
        if getattr(self, "is_connected", False):
            self.disconnect()
