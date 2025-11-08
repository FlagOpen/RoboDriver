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
)

from operating_platform.robot.robots.configs import RobotConfig, ManipulatorRobotConfig, DDSManipulatorRobotConfig
from typing import Union, Optional, List, Dict


@RobotConfig.register_subclass("leju_kuavo4p")
@dataclass
class LejuKuavo4pRobotConfig(DDSManipulatorRobotConfig):
    leader_motors: Dict[str, MotorsBusConfig] = field(
        default_factory = lambda: {
            # "right_arm": DDSMotorsBusConfig(
            #     topic="/dev/ttyACM0",
            #     motors={
            #         # name: (index, model)
            #         "joint_shoulder_pan": [1, "sts3215"],
            #         "joint_shoulder_lift": [2, "sts3215"],
            #         "joint_elbow_flex": [3, "sts3215"],
            #         "joint_wrist_flex": [4, "sts3215"],
            #         "joint_wrist_roll": [5, "sts3215"],
            #         "joint_gripper": [6, "sts3215"],
            #     },
            # ),
        }
    )

    follower_motors: Dict[str, MotorsBusConfig] = field(
        default_factory = lambda: {
            "right_arm": DDSMotorsBusConfig(
                topic="/sensors_data_raw",
                group="right_arm",
                motors={
                    "r_arm_pitch": [1, "leju_arm_motor"],
                    "r_arm_roll": [2, "leju_arm_motor"],
                    "r_arm_yaw": [3, "leju_arm_motor"],
                    "r_forearm_pitch": [4, "leju_arm_motor"],
                    "r_hand_yaw": [5, "leju_arm_motor"],
                    "r_hand_pitch": [6, "leju_arm_motor"],
                    "r_hand_roll": [7, "leju_arm_motor"],
                },
            ),
            "left_arm": DDSMotorsBusConfig(
                topic="/sensors_data_raw",
                group="left_arm",
                motors={
                    "l_arm_pitch": [1, "leju_arm_motor"],
                    "l_arm_roll": [2, "leju_arm_motor"],
                    "l_arm_yaw": [3, "leju_arm_motor"],
                    "l_forearm_pitch": [4, "leju_arm_motor"],
                    "l_hand_yaw": [5, "leju_arm_motor"],
                    "l_hand_pitch": [6, "leju_arm_motor"],
                    "l_hand_roll": [7, "leju_arm_motor"],
                },
            ),
            "head": DDSMotorsBusConfig(
                topic="/sensors_data_raw",
                group="head",
                motors={
                    "head_yaw": [1, "leju_head_motor"],
                    "head_pitch": [2, "leju_head_motor"],
                },
            ),
            "right_dexhand": DDSMotorsBusConfig(
                topic="/dexhand/state",
                group="right_dexhand",
                motors={
                    "r_thumb": [1, "dexhand_motor"],
                    "r_thumb_aux": [2, "dexhand_motor"],
                    "r_index": [3, "dexhand_motor"],
                    "r_middle": [4, "dexhand_motor"],
                    "r_ring": [5, "dexhand_motor"],
                    "r_pinky": [6, "dexhand_motor"],
                },
            ),
            "left_dexhand": DDSMotorsBusConfig(
                topic="/dexhand/state",
                group="left_dexhand",
                motors={
                    "l_thumb": [1, "dexhand_motor"],
                    "l_thumb_aux": [2, "dexhand_motor"],
                    "l_index": [3, "dexhand_motor"],
                    "l_middle": [4, "dexhand_motor"],
                    "l_ring": [5, "dexhand_motor"],
                    "l_pinky": [6, "dexhand_motor"],
                },
            ),
        }
    )

    cameras: Dict[str, CameraConfig] = field(
        default_factory=lambda: {
            "image_top": DDSCameraConfig(
                topic="/camera/color/image_raw",
                camera_index=0,
                fps=30,
                width=640,
                height=480,
            ),
            # "image_top_depth": DDSCameraConfig(
            #     topic="/camera/depth/image_rect_raw",
            #     camera_index=1,
            #     fps=30,
            #     width=640,
            #     height=480,
            # ),
        }
    )

    use_videos: bool = False

    microphones: Dict[str, int] = field(
        default_factory=lambda: {
            # "audio_right": 2,
            # "audio_left": 4,
        }
    )
