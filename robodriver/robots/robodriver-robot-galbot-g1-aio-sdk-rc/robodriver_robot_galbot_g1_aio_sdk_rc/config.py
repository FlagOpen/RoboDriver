from typing import Dict
from dataclasses import dataclass, field

from lerobot.robots.config import RobotConfig
from lerobot.cameras import CameraConfig
from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.motors import Motor, MotorNormMode


@RobotConfig.register_subclass("galbot_g1_aio_sdk_rc")
@dataclass
class GalbotG1AIOSDKRCRobotConfig(RobotConfig):
    use_degrees = True
    norm_mode_body = (
        MotorNormMode.DEGREES if use_degrees else MotorNormMode.RANGE_M100_100
    )

    leader_motors: Dict[str, Dict[str, Motor]] = field(
        default_factory = lambda norm_mode_body=norm_mode_body: {
            "right_arm_joint1": Motor(1, "galbot-motor", norm_mode_body),
            "right_arm_joint2": Motor(2, "galbot-motor", norm_mode_body),
            "right_arm_joint3": Motor(3, "galbot-motor", norm_mode_body),
            "right_arm_joint4": Motor(4, "galbot-motor", norm_mode_body),
            "right_arm_joint5": Motor(5, "galbot-motor", norm_mode_body),
            "right_arm_joint6": Motor(6, "galbot-motor", norm_mode_body),
            "right_arm_joint7": Motor(7, "galbot-motor", norm_mode_body),
            "right_gripper": Motor(8, "galbot-gripper", MotorNormMode.RANGE_0_100),

            "left_arm_joint1": Motor(9, "galbot-motor", norm_mode_body),
            "left_arm_joint2": Motor(10, "galbot-motor", norm_mode_body),
            "left_arm_joint3": Motor(11, "galbot-motor", norm_mode_body),
            "left_arm_joint4": Motor(12, "galbot-motor", norm_mode_body),
            "left_arm_joint5": Motor(13, "galbot-motor", norm_mode_body),
            "left_arm_joint6": Motor(14, "galbot-motor", norm_mode_body),
            "left_arm_joint7": Motor(15, "galbot-motor", norm_mode_body),
            "left_gripper": Motor(16, "galbot-gripper", MotorNormMode.RANGE_0_100),

            "leg_joint1": Motor(17, "galbot-motor", norm_mode_body),
            "leg_joint2": Motor(18, "galbot-motor", norm_mode_body),
            "leg_joint3": Motor(19, "galbot-motor", norm_mode_body),
            "leg_joint4": Motor(20, "galbot-motor", norm_mode_body),
            "leg_joint5": Motor(21, "galbot-motor", norm_mode_body),

            "head_joint1": Motor(22, "galbot-motor", norm_mode_body),
            "head_joint2": Motor(23, "galbot-motor", norm_mode_body),

            "chassis_pos1": Motor(24, "galbot-motor", norm_mode_body),
            "chassis_pos2": Motor(25, "galbot-motor", norm_mode_body),
            "chassis_pos3": Motor(26, "galbot-motor", norm_mode_body),
            "chassis_pos4": Motor(27, "galbot-motor", norm_mode_body),

            "chassis_vel1": Motor(28, "galbot-motor", norm_mode_body),
            "chassis_vel2": Motor(29, "galbot-motor", norm_mode_body),
            "chassis_vel3": Motor(30, "galbot-motor", norm_mode_body),
            "chassis_vel4": Motor(31, "galbot-motor", norm_mode_body),
        }
    )

    follower_motors: Dict[str, Dict[str, Motor]] = field(
        default_factory = lambda norm_mode_body=norm_mode_body: {
            "right_arm_joint1": Motor(1, "galbot-motor", norm_mode_body),
            "right_arm_joint2": Motor(2, "galbot-motor", norm_mode_body),
            "right_arm_joint3": Motor(3, "galbot-motor", norm_mode_body),
            "right_arm_joint4": Motor(4, "galbot-motor", norm_mode_body),
            "right_arm_joint5": Motor(5, "galbot-motor", norm_mode_body),
            "right_arm_joint6": Motor(6, "galbot-motor", norm_mode_body),
            "right_arm_joint7": Motor(7, "galbot-motor", norm_mode_body),
            "right_gripper": Motor(8, "galbot-gripper", MotorNormMode.RANGE_0_100),

            "left_arm_joint1": Motor(9, "galbot-motor", norm_mode_body),
            "left_arm_joint2": Motor(10, "galbot-motor", norm_mode_body),
            "left_arm_joint3": Motor(11, "galbot-motor", norm_mode_body),
            "left_arm_joint4": Motor(12, "galbot-motor", norm_mode_body),
            "left_arm_joint5": Motor(13, "galbot-motor", norm_mode_body),
            "left_arm_joint6": Motor(14, "galbot-motor", norm_mode_body),
            "left_arm_joint7": Motor(15, "galbot-motor", norm_mode_body),
            "left_gripper": Motor(16, "galbot-gripper", MotorNormMode.RANGE_0_100),

            "leg_joint1": Motor(17, "galbot-motor", norm_mode_body),
            "leg_joint2": Motor(18, "galbot-motor", norm_mode_body),
            "leg_joint3": Motor(19, "galbot-motor", norm_mode_body),
            "leg_joint4": Motor(20, "galbot-motor", norm_mode_body),
            "leg_joint5": Motor(21, "galbot-motor", norm_mode_body),

            "head_joint1": Motor(22, "galbot-motor", norm_mode_body),
            "head_joint2": Motor(23, "galbot-motor", norm_mode_body),
            
            "chassis_pos1": Motor(24, "galbot-motor", norm_mode_body),
            "chassis_pos2": Motor(25, "galbot-motor", norm_mode_body),
            "chassis_pos3": Motor(26, "galbot-motor", norm_mode_body),
            "chassis_pos4": Motor(27, "galbot-motor", norm_mode_body),

            "chassis_vel1": Motor(28, "galbot-motor", norm_mode_body),
            "chassis_vel2": Motor(29, "galbot-motor", norm_mode_body),
            "chassis_vel3": Motor(30, "galbot-motor", norm_mode_body),
            "chassis_vel4": Motor(31, "galbot-motor", norm_mode_body),
        }
    )

    cameras: Dict[str, CameraConfig] = field(
        default_factory=lambda: {
            "image_head_right": OpenCVCameraConfig(
                index_or_path=1,
                fps=30,
                width=640,
                height=480,
            ),
            "image_head_left": OpenCVCameraConfig(
                index_or_path=2,
                fps=30,
                width=640,
                height=480,
            ),
            "image_arm_right": OpenCVCameraConfig(
                index_or_path=3,
                fps=30,
                width=640,
                height=480,
            ),
            "image_arm_left": OpenCVCameraConfig(
                index_or_path=4,
                fps=30,
                width=640,
                height=480,
            ),
        }
    )

    use_videos: bool = False

    microphones: Dict[str, int] = field(
        default_factory=lambda: {}
    )

