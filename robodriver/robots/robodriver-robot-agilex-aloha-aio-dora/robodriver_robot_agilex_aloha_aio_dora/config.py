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

    arm_need_connect=[
        "leader_jointstate_right",
        "leader_jointstate_left",
        "follower_jointstate_right",
        "follower_jointstate_left",
        "follower_endpose_right",
        "follower_endpose_left",
    ]

    # Right arm (leader) motors configuration for Piper arm
    right_leader_motors: Dict[str, Motor] = field(
        default_factory=lambda norm_mode_body=norm_mode_body: {
            "joint_1": Motor(1, "piper-motor", norm_mode_body),
            "joint_2": Motor(2, "piper-motor", norm_mode_body),
            "joint_3": Motor(3, "piper-motor", norm_mode_body),
            "joint_4": Motor(4, "piper-motor", norm_mode_body),
            "joint_5": Motor(5, "piper-motor", norm_mode_body),
            "joint_6": Motor(6, "piper-motor", norm_mode_body),
            "pose_x": Motor(7, "piper-pose", norm_mode_body),
            "pose_y": Motor(8, "piper-pose", norm_mode_body),
            "pose_z": Motor(9, "piper-pose", norm_mode_body),
            "pose_rx": Motor(10, "piper-pose", norm_mode_body),
            "pose_ry": Motor(11, "piper-pose", norm_mode_body),
            "pose_rz": Motor(12, "piper-pose", norm_mode_body),
            "gripper": Motor(13, "piper-gripper", MotorNormMode.RANGE_0_100),
        }
    )

    # Left arm (leader) motors configuration for Piper arm
    left_leader_motors: Dict[str, Motor] = field(
        default_factory=lambda norm_mode_body=norm_mode_body: {
            "joint_1": Motor(1, "piper-motor", norm_mode_body),
            "joint_2": Motor(2, "piper-motor", norm_mode_body),
            "joint_3": Motor(3, "piper-motor", norm_mode_body),
            "joint_4": Motor(4, "piper-motor", norm_mode_body),
            "joint_5": Motor(5, "piper-motor", norm_mode_body),
            "joint_6": Motor(6, "piper-motor", norm_mode_body),
            "pose_x": Motor(7, "piper-pose", norm_mode_body),
            "pose_y": Motor(8, "piper-pose", norm_mode_body),
            "pose_z": Motor(9, "piper-pose", norm_mode_body),
            "pose_rx": Motor(10, "piper-pose", norm_mode_body),
            "pose_ry": Motor(11, "piper-pose", norm_mode_body),
            "pose_rz": Motor(12, "piper-pose", norm_mode_body),
            "gripper": Motor(13, "piper-gripper", MotorNormMode.RANGE_0_100),
        }
    )

    # Right arm (follower) motors configuration for Piper arm
    right_follower_motors: Dict[str, Motor] = field(
        default_factory=lambda norm_mode_body=norm_mode_body: {
            "joint_1": Motor(1, "piper-motor", norm_mode_body),
            "joint_2": Motor(2, "piper-motor", norm_mode_body),
            "joint_3": Motor(3, "piper-motor", norm_mode_body),
            "joint_4": Motor(4, "piper-motor", norm_mode_body),
            "joint_5": Motor(5, "piper-motor", norm_mode_body),
            "joint_6": Motor(6, "piper-motor", norm_mode_body),
            "pose_x": Motor(7, "piper-pose", norm_mode_body),
            "pose_y": Motor(8, "piper-pose", norm_mode_body),
            "pose_z": Motor(9, "piper-pose", norm_mode_body),
            "pose_rx": Motor(10, "piper-pose", norm_mode_body),
            "pose_ry": Motor(11, "piper-pose", norm_mode_body),
            "pose_rz": Motor(12, "piper-pose", norm_mode_body),
            "gripper": Motor(13, "piper-gripper", MotorNormMode.RANGE_0_100),
        }
    )

    # Left arm (follower) motors configuration for Piper arm
    left_follower_motors: Dict[str, Motor] = field(
        default_factory=lambda norm_mode_body=norm_mode_body: {
            "joint_1": Motor(1, "piper-motor", norm_mode_body),
            "joint_2": Motor(2, "piper-motor", norm_mode_body),
            "joint_3": Motor(3, "piper-motor", norm_mode_body),
            "joint_4": Motor(4, "piper-motor", norm_mode_body),
            "joint_5": Motor(5, "piper-motor", norm_mode_body),
            "joint_6": Motor(6, "piper-motor", norm_mode_body),
            "pose_x": Motor(7, "piper-pose", norm_mode_body),
            "pose_y": Motor(8, "piper-pose", norm_mode_body),
            "pose_z": Motor(9, "piper-pose", norm_mode_body),
            "pose_rx": Motor(10, "piper-pose", norm_mode_body),
            "pose_ry": Motor(11, "piper-pose", norm_mode_body),
            "pose_rz": Motor(12, "piper-pose", norm_mode_body),
            "gripper": Motor(13, "piper-gripper", MotorNormMode.RANGE_0_100),
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
