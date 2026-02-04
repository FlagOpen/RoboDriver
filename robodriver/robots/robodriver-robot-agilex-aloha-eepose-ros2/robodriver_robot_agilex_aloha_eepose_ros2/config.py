from typing import Dict
from dataclasses import dataclass, field

from lerobot.robots.config import RobotConfig
from lerobot.cameras import CameraConfig
from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.motors import Motor, MotorNormMode


@dataclass
class Actuator:
    id: int


@RobotConfig.register_subclass("agilex_aloha_eepose_ros2")
@dataclass
class AgilexAlohaEEposeROS2RobotConfig(RobotConfig):
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
            "pose_right_pos_x": Motor(8, "piper-pose", norm_mode_body),
            "pose_right_pos_y": Motor(9, "piper-pose", norm_mode_body),
            "pose_right_pos_z": Motor(10, "piper-pose", norm_mode_body),
            "pose_right_ori_w": Motor(11, "piper-pose", norm_mode_body),
            "pose_right_ori_x": Motor(12, "piper-pose", norm_mode_body),
            "pose_right_ori_y": Motor(13, "piper-pose", norm_mode_body),
            "pose_right_ori_z": Motor(14, "piper-pose", norm_mode_body),

            "joint1_left": Motor(15, "piper-motor", norm_mode_body),
            "joint2_left": Motor(16, "piper-motor", norm_mode_body),
            "joint3_left": Motor(17, "piper-motor", norm_mode_body),
            "joint4_left": Motor(18, "piper-motor", norm_mode_body),
            "joint5_left": Motor(19, "piper-motor", norm_mode_body),
            "joint6_left": Motor(20, "piper-motor", norm_mode_body),
            "gripper_left": Motor(21, "piper-gripper", MotorNormMode.RANGE_0_100),
            "pose_left_pos_x": Motor(22, "piper-pose", norm_mode_body),
            "pose_left_pos_y": Motor(23, "piper-pose", norm_mode_body),
            "pose_left_pos_z": Motor(24, "piper-pose", norm_mode_body),
            "pose_left_ori_w": Motor(25, "piper-pose", norm_mode_body),
            "pose_left_ori_x": Motor(26, "piper-pose", norm_mode_body),
            "pose_left_ori_y": Motor(27, "piper-pose", norm_mode_body),
            "pose_left_ori_z": Motor(28, "piper-pose", norm_mode_body),
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
            "image_top": OpenCVCameraConfig(
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
