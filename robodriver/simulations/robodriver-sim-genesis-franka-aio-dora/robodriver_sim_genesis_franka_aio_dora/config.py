from typing import Dict
from dataclasses import dataclass, field

from lerobot.robots.config import RobotConfig
from lerobot.cameras import CameraConfig
from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.motors import Motor, MotorNormMode


@RobotConfig.register_subclass("sim_genesis_franka_aio_dora")
@dataclass
class SimFrankaAIODoraRobotConfig(RobotConfig):
    use_degrees = False
    norm_mode_body = (
        MotorNormMode.DEGREES if use_degrees else MotorNormMode.RANGE_M100_100
    )

    leader_motors: Dict[str, Motor] = field(
        default_factory=lambda norm_mode_body=norm_mode_body: {
            "arm_joint1": Motor(1, "franka", norm_mode_body),
            "arm_joint2": Motor(2, "franka", norm_mode_body),
            "arm_joint3": Motor(3, "franka", norm_mode_body),
            "arm_joint4": Motor(4, "franka", norm_mode_body),
            "arm_joint5": Motor(5, "franka", norm_mode_body),
            "arm_joint6": Motor(6, "franka", norm_mode_body),
            "arm_joint7": Motor(7, "franka", norm_mode_body),
            "gripper_joint1": Motor(8, "franka", MotorNormMode.RANGE_0_100),
            "gripper_joint2": Motor(9, "franka", MotorNormMode.RANGE_0_100),
        }
    )

    follower_motors: Dict[str, Motor] = field(
        default_factory=lambda norm_mode_body=norm_mode_body: {
            "arm_joint1": Motor(1, "franka", norm_mode_body),
            "arm_joint2": Motor(2, "franka", norm_mode_body),
            "arm_joint3": Motor(3, "franka", norm_mode_body),
            "arm_joint4": Motor(4, "franka", norm_mode_body),
            "arm_joint5": Motor(5, "franka", norm_mode_body),
            "arm_joint6": Motor(6, "franka", norm_mode_body),
            "arm_joint7": Motor(7, "franka", norm_mode_body),
            "gripper_joint1": Motor(8, "franka", MotorNormMode.RANGE_0_100),
            "gripper_joint2": Motor(9, "franka", MotorNormMode.RANGE_0_100),
        }
    )

    cameras: Dict[str, CameraConfig] = field(
        default_factory=lambda: {
            "image_top": OpenCVCameraConfig(
                index_or_path=1,
                fps=30,
                width=1280,
                height=960,
            ),
            "image_wrist": OpenCVCameraConfig(
                index_or_path=2,
                fps=30,
                width=1280,
                height=960,
            ),
        }
    )

    use_videos: bool = False

    microphones: Dict[str, int] = field(default_factory=lambda: {})

