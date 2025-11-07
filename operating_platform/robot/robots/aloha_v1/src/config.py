import abc
from dataclasses import dataclass, field
from typing import Sequence

import draccus

from operating_platform.robot.robots.com_configs.cameras import (
    CameraConfig,
    OpenCVCameraConfig,
)

from operating_platform.robot.robots.com_configs.motors import (
    PiperMotorsBusConfig,
)

from operating_platform.robot.robots.configs import RobotConfig


@RobotConfig.register_subclass("aloha")
@dataclass
class AlohaRobotConfig(RobotConfig):
    right_leader_arm = PiperMotorsBusConfig(
        port="can_right",
        motors={
            "joint_1": [1,  "piper-motor"],
            "joint_2": [2,  "piper-motor"],
            "joint_3": [3,  "piper-motor"],
            "joint_4": [4,  "piper-motor"],
            "joint_5": [5,  "piper-motor"],
            "joint_6": [6,  "piper-motor"],
            "pose_x":  [7,  "piper-pose"],
            "pose_y":  [8,  "piper-pose"],
            "pose_z":  [9,  "piper-pose"],
            "pose_rx": [10, "piper-pose"],
            "pose_ry": [11, "piper-pose"],
            "pose_rz": [12, "piper-pose"],
            "gripper": [13, "piper-gripper"],
        },
    )

    left_leader_arm = PiperMotorsBusConfig(
        port="can_left",
        motors={
            "joint_1": [1,  "piper-motor"],
            "joint_2": [2,  "piper-motor"],
            "joint_3": [3,  "piper-motor"],
            "joint_4": [4,  "piper-motor"],
            "joint_5": [5,  "piper-motor"],
            "joint_6": [6,  "piper-motor"],
            "pose_x":  [7,  "piper-pose"],
            "pose_y":  [8,  "piper-pose"],
            "pose_z":  [9,  "piper-pose"],
            "pose_rx": [10, "piper-pose"],
            "pose_ry": [11, "piper-pose"],
            "pose_rz": [12, "piper-pose"],
            "gripper": [13, "piper-gripper"],
        },
    )

    cameras: dict[str, CameraConfig] = field(
        default_factory=lambda: {
            "image_top": OpenCVCameraConfig(
                camera_index=1,
                fps=30,
                width=640,
                height=480,
            ),
            "image_right": OpenCVCameraConfig(
                camera_index=2,
                fps=30,
                width=640,
                height=480,
            ),
            "image_left": OpenCVCameraConfig(
                camera_index=3,
                fps=30,
                width=640,
                height=480,
            ),
            # "image_depth_top": OpenCVCameraConfig(
            #     camera_index=4,
            #     fps=30,
            #     width=640,
            #     height=400,
            # ),
            # "image_depth_right": OpenCVCameraConfig(
            #     camera_index=5,
            #     fps=30,
            #     width=640,
            #     height=400,
            # ),
            # "image_depth_left": OpenCVCameraConfig(
            #     camera_index=6,
            #     fps=30,
            #     width=640,
            #     height=400,
            # ),
        }
    )

    microphones: dict[str, int] = field(
        default_factory=lambda: {
            # "audio_right": 2,
            # "audio_left": 4,
        }
    )

    use_videos: bool = False
