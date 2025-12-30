from typing import Dict
from dataclasses import dataclass, field

from lerobot.robots.config import RobotConfig
from lerobot.cameras import CameraConfig
from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.motors import Motor, MotorNormMode


@RobotConfig.register_subclass("galaxealite-follower-ros1")
@dataclass
class GALAXEALITEFollowerRos1RobotConfig(RobotConfig):
    use_degrees = True
    norm_mode_body = (
        MotorNormMode.DEGREES if use_degrees else MotorNormMode.RANGE_M100_100
    )
    follower_motors: Dict[str, Motor] = field(
        default_factory=lambda norm_mode_body=norm_mode_body: {
            "follower_arms":{
            "left_arm_joint_1_rad": Motor(1, "sts3215", norm_mode_body),
            "left_arm_joint_2_rad": Motor(2, "sts3215", norm_mode_body),
            "left_arm_joint_3_rad": Motor(3, "sts3215", norm_mode_body),
            "left_arm_joint_4_rad": Motor(4, "sts3215", norm_mode_body),
            "left_arm_joint_5_rad": Motor(5, "sts3215", norm_mode_body),
            "left_arm_joint_6_rad": Motor(6, "sts3215", norm_mode_body),
            "left_gripper_degree_mm": Motor(7, "sts3215", MotorNormMode.RANGE_0_100),
            "right_arm_joint_1_rad": Motor(8, "sts3215", norm_mode_body),
            "right_arm_joint_2_rad": Motor(9, "sts3215", norm_mode_body),
            "right_arm_joint_3_rad": Motor(10, "sts3215", norm_mode_body),
            "right_arm_joint_4_rad": Motor(11, "sts3215", norm_mode_body),
            "right_arm_joint_5_rad": Motor(12, "sts3215", norm_mode_body),
            "right_arm_joint_6_rad": Motor(13, "sts3215", norm_mode_body),
            "right_gripper_degree_mm": Motor(14, "sts3215", MotorNormMode.RANGE_0_100),
            "torso_joint_1":Motor(15, "sts3215", norm_mode_body),
            "torso_joint_2":Motor(16, "sts3215", norm_mode_body),
            "torso_joint_3":Motor(17, "sts3215", norm_mode_body),
        }
        }
    )

    cameras: Dict[str, CameraConfig] = field(
        default_factory=lambda: {
            "image_top_left": OpenCVCameraConfig(
                index_or_path=1,
                fps=30,
                width=1280,
                height=720,
            ),
            "image_top_right": OpenCVCameraConfig(
                index_or_path=2,
                fps=30,
                width=1280,
                height=720,  
            ),
            "image_wrist_left": OpenCVCameraConfig(
                index_or_path=3,
                fps=30,
                width=1280,
                height=720,
            ),
            "image_wrist_right": OpenCVCameraConfig(
                index_or_path=4,
                fps=30,
                width=1280,
                height=720,
            ),
        }
    )

    use_videos: bool = False

    microphones: Dict[str, int] = field(default_factory=lambda: {})

