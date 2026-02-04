from typing import Dict
from dataclasses import dataclass, field

from lerobot.robots.config import RobotConfig
from lerobot.cameras import CameraConfig
from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.motors import Motor, MotorNormMode


@dataclass
class Actuator:
    id: int

@RobotConfig.register_subclass("galaxea-lite-eepose-ros2")
@dataclass
class GalaxeaLiteEEposeROS2RobotConfig(RobotConfig):
    use_degrees = True
    norm_mode_body = (
        MotorNormMode.DEGREES if use_degrees else MotorNormMode.RANGE_M100_100
    )

    motors: Dict[str, Motor] = field(
        default_factory=lambda norm_mode_body=norm_mode_body: {
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
                width=640,
                height=360,
            ),
            "image_wrist_right": OpenCVCameraConfig(
                index_or_path=4,
                fps=30,
                width=640,
                height=360,
            ),
        }
    )

    use_videos: bool = False

    microphones: Dict[str, int] = field(default_factory=lambda: {})

