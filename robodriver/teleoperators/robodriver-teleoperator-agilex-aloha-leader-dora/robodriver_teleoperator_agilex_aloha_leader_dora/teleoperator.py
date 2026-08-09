import threading
import time
from typing import Any, Tuple

import logging_mp
import numpy as np
import torch
from lerobot.cameras import make_cameras_from_configs
from lerobot.teleoperators.teleoperator import Teleoperator
from lerobot.utils.errors import DeviceNotConnectedError, DeviceAlreadyConnectedError
from functools import cached_property

from .config import AgilexAlohaLeaderDoraTeleoperatorConfig
from .status import AgilexAlohaLeaderDoraTeleoperatorStatus
from .node import AgilexAlohaLeaderDoraTeleoperatorNode


logger = logging_mp.get_logger(__name__)


class AgilexAlohaLeaderDoraTeleoperator(Teleoperator):
    config_class = AgilexAlohaLeaderDoraTeleoperatorConfig
    name = "agilex_aloha_aio_dora"

    def __init__(self, config: AgilexAlohaLeaderDoraTeleoperatorConfig):
        super().__init__(config)
        self.config = config
        self.robot_type = self.config.type
        self.use_videos = self.config.use_videos
        self.microphones = self.config.microphones

        self.leader_motors = config.leader_motors
        self.follower_motors = config.follower_motors
        self.cameras = make_cameras_from_configs(self.config.cameras)

        self.connect_excluded_cameras = ["image_pika_pose"]

        self.status = AgilexAlohaLeaderDoraTeleoperatorStatus()
        self.teleoperator_dora_node = AgilexAlohaLeaderDoraTeleoperatorNode()
        self.teleoperator_dora_node.start()

        self.connected = False
        self.logs = {}

    @property
    def _leader_motors_ft(self) -> dict[str, type]:
        return {f"leader_{motor}.pos": float for motor in self.leader_motors}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return {**self._leader_motors_ft}
    
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
                lambda: len(self.robot_dora_node.recv_leader_joint_right) > 0,
                lambda: [] if len(self.robot_dora_node.recv_leader_joint_right) > 0 else ["recv_leader_joint_right"],
                "等待主臂右臂关节角度超时",
            ),
            (
                lambda: len(self.robot_dora_node.recv_leader_joint_left) > 0,
                lambda: [] if len(self.robot_dora_node.recv_leader_joint_left) > 0 else ["recv_leader_joint_left"],
                "等待主臂左臂关节角度超时",
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

                        if i == 0:  # 右臂关节角度
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
        # 右臂关节角度状态
        if conditions[0][0]():
            # 检查recv_leader_joint_right字典中实际接收到的键
            success_messages.append(f"右臂关节角度: 已接收 ({len(self.robot_dora_node.recv_leader_joint_right)}个数据点)")

        # 左臂关节角度状态
        if conditions[1][0]():
            # 检查recv_leader_joint_left字典中实际接收到的键
            success_messages.append(f"左臂关节角度: 已接收 ({len(self.robot_dora_node.recv_leader_joint_left)}个数据点)")

        log_message = "\n[连接成功] 所有设备已就绪:\n"
        log_message += "\n".join(f"  - {msg}" for msg in success_messages)
        log_message += f"\n  总耗时: {time.perf_counter() - start_time:.2f} 秒\n"
        logger.info(log_message)
        # ===========================

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

    
    def get_action(self) -> Tuple[dict[str, Any], dict[str, Any]]:
        if not self.connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")


        self.robot_dora_node.recv_leader_joint_right_status = max(0, self.robot_dora_node.recv_leader_joint_right_status - 1)
        self.robot_dora_node.recv_leader_joint_left_status = max(0, self.robot_dora_node.recv_leader_joint_left_status - 1)
        
        # Read arm position
        start = time.perf_counter()
        act_dict = {}


        # Add right arm positions
        for i, motor in enumerate(self.leader_motors):
            if "joint" in motor and "right" in motor:
                act_dict[f"leader_{motor}.pos"] = self.robot_dora_node.recv_leader_joint_right[i]
            if "gripper" in motor and "right" in motor:
                act_dict[f"leader_{motor}.pos"] = self.robot_dora_node.recv_leader_joint_right[i]

            if "joint" in motor and "left" in motor:
                act_dict[f"leader_{motor}.pos"] = self.robot_dora_node.recv_leader_joint_left[i-7]
            if "gripper" in motor and "left" in motor:
                act_dict[f"leader_{motor}.pos"] = self.robot_dora_node.recv_leader_joint_left[i-7]
        
        
        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read action: {dt_ms:.1f} ms")

        return act_dict

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
