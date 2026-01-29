import time
import torch
import logging_mp
import numpy as np
import threading

from typing import Any

from lerobot.utils.errors import DeviceNotConnectedError, DeviceAlreadyConnectedError
from lerobot.teleoperators.teleoperator import Teleoperator


from .config import AutoTaskTeleoperatorConfig
from .node import  AutoTaskTeleoperatorNode


logger = logging_mp.get_logger(__name__)


class AutoTaskTeleoperator(Teleoperator):
    config_class = AutoTaskTeleoperatorConfig
    name = "autotask"

    def __init__(self, config: AutoTaskTeleoperatorConfig):
        super().__init__(config)
        self.config = config
        self.teleoperator_type = self.config.type

        self.actuators = config.actuators # 大道至简。所有的action，不管是左右手臂，还是头部等等，能动的都算做actuators。


        self.teleoperator_node = AutoTaskTeleoperatorNode()
        # self.teleoperator_node.start()

        self.connected = False
        self.logs = {}

    @property
    def action_features(self) -> dict[str, type]:
        return {f"leader_{actuator}.pos": float for actuator in self.actuators}
    
    @property
    def feedback_features(self) -> dict[str, type]:
        return {}
    
    @property
    def is_connected(self) -> bool:
        return self.connected
    
    def connect(self):
        if self.connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        self.connected = True

    @property
    def is_calibrated(self) -> bool:
        """Whether the teleoperator is currently calibrated or not. Should be always `True` if not applicable"""
        return True

    def calibrate(self) -> None:
        """
        Calibrate the teleoperator if applicable. If not, this should be a no-op.

        This method should collect any necessary data (e.g., motor offsets) and update the
        :pyattr:`calibration` dictionary accordingly.
        """
        pass

    def configure(self) -> None:
        """
        Apply any one-time or runtime configuration to the teleoperator.
        This may include setting motor parameters, control modes, or initial state.
        """
        pass
    
    def get_action(self) -> dict[str, float]:
        if not self.connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        action = {}
        for i, actuator in enumerate(self.actuators):
            action[f"leader_{actuator}.pos"] = float(self.teleoperator_node.recv_data[i])

        return action

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        if not self.connected:
            raise DeviceNotConnectedError(
                f"{self} is not connected. You need to run `robot.connect()`."
            )
        
        logger.critical(f"{self}: send_feedback() not implemented.")
        raise NotImplementedError

    def disconnect(self):
        if not self.connected:
            raise DeviceNotConnectedError(
                "Teleoperator is not connected. You need to run `teleoperator.connect()` before disconnecting."
            )

        if hasattr(self, "teleoperator_node"):
            self.teleoperator_node.destroy()

        self.connected = False
        logger.info(f"{self} is not connected.")

    def __del__(self):
        if getattr(self, "connected", False):
            self.disconnect()
