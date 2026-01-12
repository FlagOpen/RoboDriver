from typing import Dict
from dataclasses import dataclass, field

from lerobot.robots.config import RobotConfig
from lerobot.cameras import CameraConfig
from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.motors import Motor, MotorNormMode


@RobotConfig.register_subclass("agilex_aloha_aio_dora")
@dataclass
class AgilexAlohaAIODoraRobotConfig(RobotConfig):
    use_degrees = True
    norm_mode_body = (
        MotorNormMode.DEGREES if use_degrees else MotorNormMode.RANGE_M100_100
    )

    leader_motors: Dict[str, Motor] = field(
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
        }
    )

    follower_motors: Dict[str, Motor] = field(
        default_factory=lambda norm_mode_body=norm_mode_body: {
            "joint1_right": Motor(1, "piper-motor", norm_mode_body),
            "joint2_right": Motor(2, "piper-motor", norm_mode_body),
            "joint3_right": Motor(3, "piper-motor", norm_mode_body),
            "joint4_right": Motor(4, "piper-motor", norm_mode_body),
            "joint5_right": Motor(5, "piper-motor", norm_mode_body),
            "joint6_right": Motor(6, "piper-motor", norm_mode_body),
            "gripper_right": Motor(7, "piper-gripper", MotorNormMode.RANGE_0_100),
            "pose_right_x": Motor(8, "piper-pose", norm_mode_body),
            "pose_right_y": Motor(9, "piper-pose", norm_mode_body),
            "pose_right_z": Motor(10, "piper-pose", norm_mode_body),
            "pose_right_rx": Motor(11, "piper-pose", norm_mode_body),
            "pose_right_ry": Motor(12, "piper-pose", norm_mode_body),
            "pose_right_rz": Motor(13, "piper-pose", norm_mode_body),

            "joint1_left": Motor(14, "piper-motor", norm_mode_body),
            "joint2_left": Motor(15, "piper-motor", norm_mode_body),
            "joint3_left": Motor(16, "piper-motor", norm_mode_body),
            "joint4_left": Motor(17, "piper-motor", norm_mode_body),
            "joint5_left": Motor(18, "piper-motor", norm_mode_body),
            "joint6_left": Motor(19, "piper-motor", norm_mode_body),
            "gripper_left": Motor(20, "piper-gripper", MotorNormMode.RANGE_0_100),
            "pose_left_x": Motor(21, "piper-pose", norm_mode_body),
            "pose_left_y": Motor(22, "piper-pose", norm_mode_body),
            "pose_left_z": Motor(23, "piper-pose", norm_mode_body),
            "pose_left_rx": Motor(24, "piper-pose", norm_mode_body),
            "pose_left_ry": Motor(25, "piper-pose", norm_mode_body),
            "pose_left_rz": Motor(26, "piper-pose", norm_mode_body),
        }
    )

    # Cameras configuration - using Orbbec cameras (as per dataflow.yml)
    cameras: Dict[str, CameraConfig] = field(
        default_factory=lambda: {
            "image_top": OpenCVCameraConfig(
                index_or_path=1,  # Camera index for top camera
                fps=30,
                width=640,
                height=480,
            ),
            "image_right": OpenCVCameraConfig(
                index_or_path=2,  # Camera index for right camera
                fps=30,
                width=640,
                height=480,
            ),
            "image_left": OpenCVCameraConfig(
                index_or_path=3,  # Camera index for left camera
                fps=30,
                width=640,
                height=480,
            ),
            # Depth cameras are handled by dora-camera-orbbec-v1 nodes
            # "image_depth_top": OpenCVCameraConfig(...),
            # "image_depth_right": OpenCVCameraConfig(...),
            # "image_depth_left": OpenCVCameraConfig(...),
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
