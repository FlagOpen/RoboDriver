from typing import Protocol

import platform
import time

from operating_platform.robot.robots.configs import RobotConfig
from operating_platform.robot.robots import (  # noqa: F401
    so101_v1,
    galbot_g1,
    leju_kuavo4p,
)

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


class RobotDeviceNotConnectedError(Exception):
    """Exception raised when the robot device is not connected."""

    def __init__(
        self, message="This robot device is not connected. Try calling `robot_device.connect()` first."
    ):
        self.message = message
        super().__init__(self.message)


class RobotDeviceAlreadyConnectedError(Exception):
    """Exception raised when the robot device is already connected."""

    def __init__(
        self,
        message="This robot device is already connected. Try not calling `robot_device.connect()` twice.",
    ):
        self.message = message
        super().__init__(self.message)


def get_arm_id(name, arm_type):
    """Returns the string identifier of a robot arm. For instance, for a bimanual manipulator
    like Aloha, it could be left_follower, right_follower, left_leader, or right_leader.
    """
    return f"{name}_{arm_type}"


class Robot(Protocol):
    # TODO(rcadene, aliberts): Add unit test checking the protocol is implemented in the corresponding classes
    robot_type: str
    features: dict

    def connect(self): ...
    def run_calibration(self): ...
    def teleop_step(self, record_data=False): ...
    def capture_observation(self): ...
    def send_action(self, action): ...
    def disconnect(self): ...


def make_robot_from_config(config: RobotConfig):
    print("In make_robot_from_config")

    if config.type == "adora":
        from operating_platform.robot.robots.adora_v1.manipulator import AdoraManipulator
        print("In AdoraRobotConfig")
        return AdoraManipulator(config)
    
    elif config.type == "aloha":
        from operating_platform.robot.robots.aloha_v1.manipulator import AlohaManipulator
        print("In AlohaManipulator")
        return AlohaManipulator(config)
    
    elif config.type == "pika":
        from operating_platform.robot.robots.pika_v1.manipulator import PikaV1Manipulator
        print("In PikaV1Manipulator")
        return PikaV1Manipulator(config)
    
    elif config.type == "so101":
        from operating_platform.robot.robots.so101_v1.src.manipulator import SO101Manipulator
        print("In SO101Manipulator")
        return SO101Manipulator(config)

    elif config.type == "realman":
        from operating_platform.robot.robots.realman_v1.manipulator import RealmanManipulator
        print("In RealmanRobotConfig")
        return RealmanManipulator(config)
    
    elif config.type == "dexhand":
        from operating_platform.robot.robots.dexterous_hand_v1.manipulator import DexterousHandManipulator
        print("In DexterousHandMotorsBusConfig")
        return DexterousHandManipulator(config)
    
    elif config.type == "galaxea":
        from operating_platform.robot.robots.galaxea_v1.manipulator import GALAXEAManipulator
        print("In GALAXEARobotConfig")
        return GALAXEAManipulator(config)
    
    elif config.type == "galbot_g1":
        from operating_platform.robot.robots.galbot_g1.manipulator import GalbotG1Manipulator
        print("In GalbotG1RobotConfig")
        return GalbotG1Manipulator(config)
    
    elif config.type == "leju_kuavo4p":
        from operating_platform.robot.robots.leju_kuavo4p.manipulator import LejuKuavo4pManipulator
        print("In LejuKuavo4pRobotConfig")
        return LejuKuavo4pManipulator(config)
    
    else:
        print("Not match robot")
        raise ValueError(f"Robot type is not available.")
