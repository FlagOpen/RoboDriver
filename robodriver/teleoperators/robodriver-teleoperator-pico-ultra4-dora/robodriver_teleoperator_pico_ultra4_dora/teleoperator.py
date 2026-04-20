import time
import logging_mp
import numpy as np
from typing import Any, Dict

from lerobot.utils.errors import DeviceNotConnectedError, DeviceAlreadyConnectedError
from lerobot.teleoperators.teleoperator import Teleoperator

from .config import PicoUltra4DoraTeleoperatorConfig
from .status import PicoUltra4DoraTeleoperatorStatus
from .node import PicoUltra4DoraTeleoperatorNode


logger = logging_mp.getLogger(__name__)


class PicoUltra4DoraTeleoperator(Teleoperator):
    """
    Pico Ultra4 VR 遥操器

    使用 Pico Ultra4 VR 控制器进行末端位姿控制，
    通过 IK 求解转换为关节角度，控制 Piper 机械臂
    """

    config_class = PicoUltra4DoraTeleoperatorConfig
    name = "pico_ultra4_dora"

    def __init__(self, config: PicoUltra4DoraTeleoperatorConfig):
        super().__init__(config)
        self.config = config
        self.teleoperator_type = self.config.type

        # 状态
        self.status = PicoUltra4DoraTeleoperatorStatus()

        # 创建节点配置
        node_config = {
            "vr_controller": self.config.vr_controller,
            "control_trigger": self.config.control_trigger,
            "gripper_trigger": self.config.gripper_trigger,
            "end_effector_link": self.config.end_effector_link,
            "scale_factor": self.config.scale_factor,
            "joint_names": self.config.joint_names,
        }

        # 创建 Dora 节点
        self.teleoperator_node = PicoUltra4DoraTeleoperatorNode(
            robot_urdf_path=self.config.robot_urdf_path,
            config=node_config
        )

        self.connected = False
        self.logs = {}

    @property
    def action_features(self) -> dict[str, type]:
        """定义动作特征"""
        features = {f"{joint}.pos": float for joint in self.config.joint_names}
        features["gripper.pos"] = float
        return features

    @property
    def feedback_features(self) -> dict[str, type]:
        """定义反馈特征（暂不支持）"""
        return {}

    @property
    def is_connected(self) -> bool:
        return self.connected

    def connect(self):
        """连接 VR 控制器"""
        if self.connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        logger.info("Connecting Pico Ultra4 VR controller...")

        # 启动节点
        self.teleoperator_node.start()

        # 等待 VR 数据就绪
        timeout = 10
        start_time = time.perf_counter()

        while True:
            # 检查是否有 VR 数据
            action = self.teleoperator_node.get_action()
            if action is not None:
                logger.info("VR controller data received")
                break

            if time.perf_counter() - start_time > timeout:
                raise TimeoutError("Failed to connect to VR controller within timeout")

            time.sleep(0.1)

        # 更新状态
        self.status.specifications.arm.information[0].is_connect = True
        self.connected = True

        logger.info(f"{self} connected successfully")

    @property
    def is_calibrated(self) -> bool:
        """VR 控制器不需要校准"""
        return True

    def calibrate(self) -> None:
        """VR 控制器不需要校准"""
        pass

    def configure(self) -> None:
        """配置遥操器"""
        pass

    def get_action(self) -> dict[str, float]:
        """
        获取当前动作

        Returns:
            关节角度字典
        """
        if not self.connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        start = time.perf_counter()

        # 从节点获取动作
        action = self.teleoperator_node.get_action()

        if action is None:
            # 如果未激活控制，返回零动作
            action = {f"{joint}.pos": 0.0 for joint in self.config.joint_names}
            action["gripper.pos"] = 0.0

        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} get action: {dt_ms:.1f} ms")

        return action

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        """发送反馈（暂不支持）"""
        if not self.connected:
            raise DeviceNotConnectedError(
                f"{self} is not connected. You need to run `teleoperator.connect()`."
            )

        logger.warning(f"{self}: send_feedback() not implemented.")

    def update_status(self) -> str:
        """更新状态"""
        # 检查 VR 控制器连接状态
        action = self.teleoperator_node.get_action()
        self.status.specifications.arm.information[0].is_connect = action is not None

        return self.status.to_json()

    def disconnect(self):
        """断开连接"""
        if not self.connected:
            raise DeviceNotConnectedError(
                "Teleoperator is not connected. You need to run `teleoperator.connect()` before disconnecting."
            )

        self.teleoperator_node.stop()
        self.connected = False
        logger.info(f"{self} disconnected.")

    def __del__(self):
        if getattr(self, "connected", False):
            self.disconnect()
