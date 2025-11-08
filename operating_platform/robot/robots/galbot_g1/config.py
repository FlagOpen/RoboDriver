import abc
from dataclasses import dataclass, field
from typing import Sequence, Dict

import draccus

from operating_platform.robot.robots.com_configs.cameras import (
    CameraConfig,
    OpenCVCameraConfig,
    DDSCameraConfig,
)

from operating_platform.robot.robots.com_configs.motors import (
    FeetechMotorsBusConfig,
    MotorsBusConfig,
    DDSMotorsBusConfig,
)

from operating_platform.robot.robots.configs import RobotConfig, ManipulatorRobotConfig, DDSManipulatorRobotConfig


@RobotConfig.register_subclass("galbot_g1")
@dataclass
class GalbotG1RobotConfig(DDSManipulatorRobotConfig):
    leader_motors: Dict[str, MotorsBusConfig] = field(
        default_factory = lambda: {
            # "right_arm": DDSMotorsBusConfig(
            #     topic="/dev/ttyACM0",
            #     motors={
            #         # name: (index, model)
            #         "joint_shoulder_pan": [1, "sts3215"],
            #         "joint_shoulder_lift": [2, "sts3215"],
            #         "joint_elbow_flex": [3, "sts3215"],
            #         "joint_wrist_flex": [4, "sts3215"],
            #         "joint_wrist_roll": [5, "sts3215"],
            #         "joint_gripper": [6, "sts3215"],
            #     },
            # ),
        }
    )

    follower_motors: Dict[str, MotorsBusConfig] = field(
        default_factory = lambda: {
            "right_arm": DDSMotorsBusConfig(
                topic="singorix/wbcs/sensor",
                group="right_arm",
                motors={
                    "joint_1": [1, "galbot_motor"],
                    "joint_2": [2, "galbot_motor"],
                    "joint_3": [3, "galbot_motor"],
                    "joint_4": [4, "galbot_motor"],
                    "joint_5": [5, "galbot_motor"],
                    "joint_6": [6, "galbot_motor"],
                    "joint_7": [7, "galbot_motor"],
                },
            ),
            "left_arm": DDSMotorsBusConfig(
                topic="singorix/wbcs/sensor",
                group="left_arm",
                motors={
                    "joint_1": [1, "galbot_motor"],
                    "joint_2": [2, "galbot_motor"],
                    "joint_3": [3, "galbot_motor"],
                    "joint_4": [4, "galbot_motor"],
                    "joint_5": [5, "galbot_motor"],
                    "joint_6": [6, "galbot_motor"],
                    "joint_7": [7, "galbot_motor"],
                },
            ),
            "right_gripper": DDSMotorsBusConfig(
                topic="singorix/wbcs/sensor",
                group="right_gripper",
                motors={
                    "joint_1": [1, "galbot_motor"],
                },
            ),
            "left_gripper": DDSMotorsBusConfig(
                topic="singorix/wbcs/sensor",
                group="left_gripper",
                motors={
                    "joint_1": [1, "galbot_motor"],
                },
            ),
            "leg": DDSMotorsBusConfig(
                topic="singorix/wbcs/sensor",
                group="leg",
                motors={
                    "joint_1": [1, "galbot_motor"],
                    "joint_2": [2, "galbot_motor"],
                    "joint_3": [3, "galbot_motor"],
                    "joint_4": [4, "galbot_motor"],
                    "joint_5": [5, "galbot_motor"],
                },
            ),
            "head": DDSMotorsBusConfig(
                topic="singorix/wbcs/sensor",
                group="head",
                motors={
                    "joint_1": [1, "galbot_motor"],
                    "joint_2": [2, "galbot_motor"],
                },
            ),
        }
    )

    cameras: Dict[str, CameraConfig] = field(
        default_factory=lambda: {
            "image_top_right": DDSCameraConfig(
                topic="/front_head_camera/right_color/image_raw",
                camera_index=0,
                fps=30,
                width=640,
                height=480,
            ),
            "image_top_left": DDSCameraConfig(
                topic="/front_head_camera/left_color/image_raw",
                camera_index=1,
                fps=30,
                width=640,
                height=480,
            ),
            "image_wrist_right": DDSCameraConfig(
                topic="/right_arm_camera/color/image_raw",
                camera_index=2,
                fps=30,
                width=640,
                height=480,
            ),
            "image_wrist_left": DDSCameraConfig(
                topic="/left_arm_camera/color/image_raw",
                camera_index=3,
                fps=30,
                width=640,
                height=480,
            ),
        }
    )

    use_videos: bool = False

    microphones: Dict[str, int] = field(
        default_factory=lambda: {
            # "audio_right": 2,
            # "audio_left": 4,
        }
    )