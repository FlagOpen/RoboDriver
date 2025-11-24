import platform
import time
from typing import cast

import logging_mp
from lerobot.robots import Robot, RobotConfig

from operating_platform.robots.statuses import RobotStatus
from operating_platform.utils.import_utils import make_device_from_device_class

logger = logging_mp.get_logger(__name__)


def busy_wait(seconds):
    if platform.system() == "Darwin":
        # On Mac, `time.sleep` is not accurate and we need to use this while loop trick,
        # but it consumes CPU cycles.
        # TODO(rcadene): find an alternative: from python 11, time.sleep is precise
        end_time = time.perf_counter() + seconds
        while time.perf_counter() < end_time:
            pass
    else:
        # On Linux time.sleep is accurate
        if seconds > 0:
            time.sleep(seconds)


def safe_disconnect(func):
    # TODO(aliberts): Allow to pass custom exceptions
    # (e.g. ThreadServiceExit, KeyboardInterrupt, SystemExit, UnpluggedError, DynamixelCommError)
    def wrapper(robot, *args, **kwargs):
        try:
            return func(robot, *args, **kwargs)
        except Exception as e:
            if robot.is_connected:
                robot.disconnect()
            raise e

    return wrapper


# class RobotDeviceNotConnectedError(Exception):
#     """Exception raised when the robot device is not connected."""

#     def __init__(
#         self, message="This robot device is not connected. Try calling `robot_device.connect()` first."
#     ):
#         self.message = message
#         super().__init__(self.message)


# class RobotDeviceAlreadyConnectedError(Exception):
#     """Exception raised when the robot device is already connected."""

#     def __init__(
#         self,
#         message="This robot device is already connected. Try not calling `robot_device.connect()` twice.",
#     ):
#         self.message = message
#         super().__init__(self.message)


# def get_arm_id(name, arm_type):
#     """Returns the string identifier of a robot arm. For instance, for a bimanual manipulator
#     like Aloha, it could be left_follower, right_follower, left_leader, or right_leader.
#     """
#     return f"{name}_{arm_type}"


# class Robot(Protocol):
#     # TODO(rcadene, aliberts): Add unit test checking the protocol is implemented in the corresponding classes
#     robot_type: str
#     features: dict

#     def connect(self): ...
#     def run_calibration(self): ...
#     def teleop_step(self, record_data=False): ...
#     def capture_observation(self): ...
#     def send_action(self, action): ...
#     def disconnect(self): ...
#     def update_status(self): ...  # 声明但无实现


def make_robot_from_config(config: RobotConfig) -> Robot:
    logger.info("In make_robot_from_config")

    if config.type == "adora":
        from operating_platform.robot.robots.adora_v1.manipulator import (
            AdoraManipulator,
        )

        logger.info("In AdoraRobotConfig")
        return AdoraManipulator(config)

    elif config.type == "aloha_v1":
        from operating_platform.robot.robots.aloha_v1.src.manipulator import (
            AlohaManipulator,
        )

        logger.info("In AlohaManipulator")
        return AlohaManipulator(config)

    else:
        try:
            return cast(Robot, make_device_from_device_class(config))
        except Exception as e:
            logger.critical(f"Can't create robot with config {config}")
            raise ValueError(f"Error creating robot with config {config}: {e}") from e


def safe_update_status(robot: Robot) -> str:
    if hasattr(robot, "update_status"):
        robot.update_status()
    else:
        return RobotStatus().to_json()
