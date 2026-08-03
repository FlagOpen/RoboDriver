from typing import Dict
from dataclasses import dataclass, field

from lerobot.robots.config import RobotConfig
from lerobot.cameras import CameraConfig
from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.motors import Motor, MotorNormMode


@RobotConfig.register_subclass("dobot-nova2-ros2")
@dataclass
class DobotNova2Ros2RobotConfig(RobotConfig):
    use_degrees = True
    norm_mode_body = (
        MotorNormMode.DEGREES if use_degrees else MotorNormMode.RANGE_M100_100
    )

    leader_motors: Dict[str, Motor] = field(
        default_factory=lambda norm_mode_body=norm_mode_body: {
            "leader_arms":{
            "joint1-l": Motor(1, "robot_motor", norm_mode_body),
            "joint2-l": Motor(2, "robot_motor", norm_mode_body),
            "joint3-l": Motor(3, "robot_motor", norm_mode_body),
            "joint4-l": Motor(4, "robot_motor", norm_mode_body),
            "joint5-l": Motor(5, "robot_motor", norm_mode_body),
            "joint6-l": Motor(6, "robot_motor", norm_mode_body),
            "gripper1-l": Motor(7, "robot_motor", norm_mode_body),
        }
        }
    )

    follower_motors: Dict[str, Motor] = field(
        default_factory=lambda norm_mode_body=norm_mode_body: {
            "follower_arms":{
            "joint1-r": Motor(1, "robot_motor", norm_mode_body),
            "joint2-r": Motor(2, "robot_motor", norm_mode_body),
            "joint3-r": Motor(3, "robot_motor", norm_mode_body),
            "joint4-r": Motor(4, "robot_motor", norm_mode_body),
            "joint5-r": Motor(5, "robot_motor", norm_mode_body),
            "joint6-r": Motor(6, "robot_motor", norm_mode_body),
            "gripper1-r": Motor(7, "robot_motor", norm_mode_body),
        }
        }
    )

    cameras: Dict[str, CameraConfig] = field(
        default_factory=lambda: {
            "image_top": OpenCVCameraConfig(
                index_or_path=1,
                fps=30,
                width=848,
                height=480,
            ),
            "image_wrist_left": OpenCVCameraConfig(
                index_or_path=2,
                fps=30,
                width=848,
                height=480,
            ),
            "image_wrist_right": OpenCVCameraConfig(
                index_or_path=3,
                fps=30,
                width=848,
                height=480,
            ),
        }
    )

    use_videos: bool = True

    microphones: Dict[str, int] = field(default_factory=lambda: {})

