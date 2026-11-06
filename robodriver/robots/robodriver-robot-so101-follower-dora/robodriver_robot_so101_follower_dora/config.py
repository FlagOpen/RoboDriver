import abc
from dataclasses import dataclass, field
from typing import Sequence, Dict

import draccus

# from operating_platform.robot.robots.com_configs.cameras import (
#     CameraConfig,
#     OpenCVCameraConfig,
# )

# from operating_platform.robot.robots.com_configs.motors import (
#     FeetechMotorsBusConfig,
#     MotorsBusConfig,
# )

# from operating_platform.robot.robots.configs import RobotConfig, ManipulatorRobotConfig

from lerobot.robots.config import RobotConfig

from lerobot.cameras import CameraConfig
from lerobot.cameras.configs import ColorMode
from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.motors import Motor, MotorNormMode


@RobotConfig.register_subclass("so101_follower_dora")
@dataclass
class SO101FollowerDoraRobotConfig(RobotConfig):
    use_degrees = True
    norm_mode_body = MotorNormMode.DEGREES if use_degrees else MotorNormMode.RANGE_M100_100
 
    motors: Dict[str, Motor] = field(
        default_factory=lambda norm_mode_body=norm_mode_body: {
            "shoulder_pan": Motor(1, "sts3215", norm_mode_body),
            "shoulder_lift": Motor(2, "sts3215", norm_mode_body),
            "elbow_flex": Motor(3, "sts3215", norm_mode_body),
            "wrist_flex": Motor(4, "sts3215", norm_mode_body),
            "wrist_roll": Motor(5, "sts3215", norm_mode_body),
            "gripper": Motor(6, "sts3215", MotorNormMode.RANGE_0_100),
        }
    )

    reset_pose: Dict[str, float] = field(
        default_factory=lambda: {
            "shoulder_pan.pos": 2.154,
            "shoulder_lift.pos": -103.385,
            "elbow_flex.pos": 89.802,
            "wrist_flex.pos": 74.857,
            "wrist_roll.pos": -2.242,
            "gripper.pos": 0.784,
        }
    )

    cameras: Dict[str, CameraConfig] = field(
        default_factory=lambda: {
            "image_top_rgb": OpenCVCameraConfig(
                index_or_path=4,
                fps=30,
                width=640,
                height=480,
            ),
            "image_wrist": OpenCVCameraConfig(
                index_or_path=6,
                fps=30,
                width=640,
                height=480,
            ),
            "image_top_dep": OpenCVCameraConfig(
                index_or_path=2,
                fps=30,
                width=640,
                height=480,
            ),
        }
    )

    use_videos: bool = False

    microphones: Dict[str, int] = field(
        default_factory=lambda: {
        }
    )
    
    # super().__post_init__()