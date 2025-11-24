from dataclasses import dataclass, field
from typing import Dict

from lerobot.cameras import CameraConfig
from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.robots.config import RobotConfig

# from operating_platform.robot.robots.com_configs.cameras import (
#     CameraConfig,
#     OpenCVCameraConfig,
# )

# from operating_platform.robot.robots.com_configs.motors import (
#     FeetechMotorsBusConfig,
#     MotorsBusConfig,
# )

# from operating_platform.robot.robots.configs import RobotConfig, ManipulatorRobotConfig


@RobotConfig.register_subclass("so101_aio_dora")
@dataclass
class SO101AIODoraRobotConfig(RobotConfig):
    leader_arms: Dict[str, MotorsBusConfig] = field(
        default_factory=lambda: {
            "main": FeetechMotorsBusConfig(
                port="/dev/ttyACM0",
                motors={
                    "joint_shoulder_pan": [1, "sts3215"],
                    "joint_shoulder_lift": [2, "sts3215"],
                    "joint_elbow_flex": [3, "sts3215"],
                    "joint_wrist_flex": [4, "sts3215"],
                    "joint_wrist_roll": [5, "sts3215"],
                    "joint_gripper": [6, "sts3215"],
                },
            ),
        }
    )

    follower_arms: Dict[str, MotorsBusConfig] = field(
        default_factory=lambda: {
            "main": FeetechMotorsBusConfig(
                port="/dev/ttyACM1",
                motors={
                    "joint_shoulder_pan": [1, "sts3215"],
                    "joint_shoulder_lift": [2, "sts3215"],
                    "joint_elbow_flex": [3, "sts3215"],
                    "joint_wrist_flex": [4, "sts3215"],
                    "joint_wrist_roll": [5, "sts3215"],
                    "joint_gripper": [6, "sts3215"],
                },
            ),
        }
    )

    cameras: Dict[str, CameraConfig] = field(
        default_factory=lambda: {
            "image_top": OpenCVCameraConfig(
                camera_index=0,
                fps=30,
                width=640,
                height=480,
            ),
            "image_wrist": OpenCVCameraConfig(
                camera_index=1,
                fps=30,
                width=640,
                height=480,
            ),
            "image_top_dep": OpenCVCameraConfig(
                camera_index=2,
                fps=30,
                width=640,
                height=480,
            ),
        }
    )

    use_videos: bool = False

    microphones: Dict[str, int] = field(default_factory=lambda: {})

    super().__post_init__()
