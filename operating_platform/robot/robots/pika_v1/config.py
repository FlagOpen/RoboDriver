# from __future__ import annotations
import abc
from dataclasses import dataclass, field
from typing import Sequence

import draccus

from operating_platform.robot.robots.com_configs.cameras import (
    CameraConfig,
    OpenCVCameraConfig,
    DDSCameraConfig,
)

from operating_platform.robot.robots.com_configs.motors import (
    FeetechMotorsBusConfig,
    MotorsBusConfig,
    DDSMotorsBusConfig,
    PikaMotorsBusConfig,
)

from operating_platform.robot.robots.configs import RobotConfig, ManipulatorRobotConfig, DDSManipulatorRobotConfig
from typing import Union, Optional, List, Dict


@RobotConfig.register_subclass("pika_v1")
@dataclass
class PikaV1RobotConfig(RobotConfig):
    right_leader_arm = PikaMotorsBusConfig(
        port="right",
        motors={
            "pose_x":           [1, "pika-pose"],
            "pose_y":           [2, "pika-pose"],
            "pose_z":           [3, "pika-pose"],
            "rotation_quat_x":  [4, "pika-pose"],
            "rotation_quat_y":  [5, "pika-pose"],
            "rotation_quat_z":  [6, "pika-pose"],
            "rotation_quat_w":  [7, "pika-pose"],
            "gripper":          [8, "pika-gripper"],
        },
    )

    left_leader_arm = PikaMotorsBusConfig(
        port="left",
        motors={
            "pose_x":           [1, "pika-pose"],
            "pose_y":           [2, "pika-pose"],
            "pose_z":           [3, "pika-pose"],
            "rotation_quat_x":  [4, "pika-pose"],
            "rotation_quat_y":  [5, "pika-pose"],
            "rotation_quat_z":  [6, "pika-pose"],
            "rotation_quat_w":  [7, "pika-pose"],
            "gripper":          [8, "pika-gripper"],
        },
    )

    cameras: dict[str, CameraConfig] = field(
        default_factory=lambda: {
            "image_top": OpenCVCameraConfig(
                camera_index=0,
                fps=30,
                width=640,
                height=480,
            ),
            "image_right": OpenCVCameraConfig(
                camera_index=1,
                fps=30,
                width=640,
                height=480,
            ),
            "image_left": OpenCVCameraConfig(
                camera_index=2,
                fps=30,
                width=640,
                height=480,
            ),
            "image_right_tac_r": OpenCVCameraConfig(
                camera_index=3,
                fps=30,
                width=640,
                height=480,
            ),
            "image_left_tac_l": OpenCVCameraConfig(
                camera_index=4,
                fps=30,
                width=640,
                height=480,
            ),
            "image_pika_pose": OpenCVCameraConfig(
                camera_index=5,
                fps=30,
                width=640,
                height=480,
            ),
        }
    )

    use_videos: bool = False

    microphones: dict[str, int] = field(
        default_factory=lambda: {
            "audio_right": 0,
            "audio_left": 1,
        }
    )
