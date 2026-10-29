from typing import Dict
from dataclasses import dataclass, field

from lerobot.robots.config import RobotConfig
from lerobot.cameras import CameraConfig
from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.motors import Motor, MotorNormMode


@dataclass
class Actuator:
    id: int


@RobotConfig.register_subclass("agilex_aloha_eepose_mix")
@dataclass
class AgilexAlohaEEposeMixRobotConfig(RobotConfig):
    use_degrees = True
    norm_mode_body = (
        MotorNormMode.DEGREES if use_degrees else MotorNormMode.RANGE_M100_100
    )

    motors: Dict[str, Motor] = field(
        default_factory=lambda norm_mode_body=norm_mode_body: {
            "joint1_right": Motor(1, "piper-motor", norm_mode_body),
            "joint2_right": Motor(2, "piper-motor", norm_mode_body),
            "joint3_right": Motor(3, "piper-motor", norm_mode_body),
            "joint4_right": Motor(4, "piper-motor", norm_mode_body),
            "joint5_right": Motor(5, "piper-motor", norm_mode_body),
            "joint6_right": Motor(6, "piper-motor", norm_mode_body),
            "gripper_right": Motor(7, "piper-gripper", MotorNormMode.RANGE_0_100),

            "joint1_left": Motor(8, "piper-motor", norm_mode_body),
            "joint2_left": Motor(9, "piper-motor", norm_mode_body),
            "joint3_left": Motor(10, "piper-motor", norm_mode_body),
            "joint4_left": Motor(11, "piper-motor", norm_mode_body),
            "joint5_left": Motor(12, "piper-motor", norm_mode_body),
            "joint6_left": Motor(13, "piper-motor", norm_mode_body),
            "gripper_left": Motor(14, "piper-gripper", MotorNormMode.RANGE_0_100),

            "left_arm_pos_x_m": Motor(15, "sts3215", norm_mode_body),
            "left_arm_pos_y_m": Motor(16, "sts3215", norm_mode_body),
            "left_arm_pos_z_m": Motor(17, "sts3215", norm_mode_body),
            "left_arm_quat_w": Motor(18, "sts3215", norm_mode_body),
            "left_arm_quat_x": Motor(19, "sts3215", norm_mode_body),
            "left_arm_quat_y": Motor(20, "sts3215", norm_mode_body),
            "left_arm_quat_z": Motor(21, "sts3215", norm_mode_body),
            "right_arm_pos_x_m": Motor(22, "sts3215", norm_mode_body),
            "right_arm_pos_y_m": Motor(23, "sts3215", norm_mode_body),
            "right_arm_pos_z_m": Motor(24, "sts3215", norm_mode_body),
            "right_arm_quat_w": Motor(25, "sts3215", norm_mode_body),
            "right_arm_quat_x": Motor(26, "sts3215", norm_mode_body),
            "right_arm_quat_y": Motor(27, "sts3215", norm_mode_body),
            "right_arm_quat_z": Motor(28, "sts3215", norm_mode_body),
        }
    )

    actuators: Dict[str, Actuator] = field(
        default_factory=lambda: {
            "left_arm_pos_x_m": Actuator(1),
            "left_arm_pos_y_m": Actuator(2),
            "left_arm_pos_z_m": Actuator(3),
            "left_arm_quat_w": Actuator(4),
            "left_arm_quat_x": Actuator(5),
            "left_arm_quat_y": Actuator(6),
            "left_arm_quat_z": Actuator(7),

            "right_arm_pos_x_m": Actuator(8),
            "right_arm_pos_y_m": Actuator(9),
            "right_arm_pos_z_m": Actuator(10),
            "right_arm_quat_w": Actuator(11),
            "right_arm_quat_x": Actuator(12),
            "right_arm_quat_y": Actuator(13),
            "right_arm_quat_z": Actuator(14),

            "left_gripper_percent": Actuator(15),
            "right_gripper_percent": Actuator(16),
        }
    )

    cameras: Dict[str, CameraConfig] = field(
        default_factory=lambda: {
            "image_head": OpenCVCameraConfig(
                index_or_path=1,
                fps=30,
                width=640,
                height=480,
            ),
            "image_right": OpenCVCameraConfig(
                index_or_path=2,
                fps=30,
                width=640,
                height=480,
            ),
            "image_left": OpenCVCameraConfig(
                index_or_path=3,
                fps=30,
                width=640,
                height=480,
            ),
        }
    )

    use_videos: bool = False

    microphones: Dict[str, int] = field(default_factory=lambda: {
        # "audio_right": 2,
        # "audio_left": 4,
    })

    # Additional configuration for CAN bus ports
    can_right_port: str = "can_right"
    can_left_port: str = "can_left"
