# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import abc
from dataclasses import dataclass, field
from typing import Sequence

import draccus

from operating_platform.robot.robots.com_configs.cameras import (
    CameraConfig,
    IntelRealSenseCameraConfig,
    OpenCVCameraConfig,
)

from operating_platform.robot.robots.com_configs.motors import (
    PikaMotorsBusConfig,
    PiperMotorsBusConfig,
    DynamixelMotorsBusConfig,
    FeetechMotorsBusConfig,
    MotorsBusConfig,
)


@dataclass
class RobotConfig(draccus.ChoiceRegistry, abc.ABC):
    @property
    def type(self) -> str:
        return self.get_choice_name(self.__class__)


# TODO(rcadene, aliberts): remove ManipulatorRobotConfig abstraction
@dataclass
class ManipulatorRobotConfig(RobotConfig):
    leader_arms: dict[str, MotorsBusConfig] = field(default_factory=lambda: {})
    follower_arms: dict[str, MotorsBusConfig] = field(default_factory=lambda: {})
    cameras: dict[str, CameraConfig] = field(default_factory=lambda: {})

    # Optionally limit the magnitude of the relative positional target vector for safety purposes.
    # Set this to a positive scalar to have the same value for all motors, or a list that is the same length
    # as the number of motors in your follower arms (assumes all follower arms have the same number of
    # motors).
    max_relative_target: list[float] | float | None = None

    # Optionally set the leader arm in torque mode with the gripper motor set to this angle. This makes it
    # possible to squeeze the gripper and have it spring back to an open position on its own. If None, the
    # gripper is not put in torque mode.
    gripper_open_degree: float | None = None

    mock: bool = False

    def __post_init__(self):
        if self.mock:
            for arm in self.leader_arms.values():
                if not arm.mock:
                    arm.mock = True
            for arm in self.follower_arms.values():
                if not arm.mock:
                    arm.mock = True
            for cam in self.cameras.values():
                if not cam.mock:
                    cam.mock = True

        if self.max_relative_target is not None and isinstance(self.max_relative_target, Sequence):
            for name in self.follower_arms:
                if len(self.follower_arms[name].motors) != len(self.max_relative_target):
                    raise ValueError(
                        f"len(max_relative_target)={len(self.max_relative_target)} but the follower arm with name {name} has "
                        f"{len(self.follower_arms[name].motors)} motors. Please make sure that the "
                        f"`max_relative_target` list has as many parameters as there are motors per arm. "
                        "Note: This feature does not yet work with robots where different follower arms have "
                        "different numbers of motors."
                    )


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

    use_videos: bool = True


@RobotConfig.register_subclass("adora")
@dataclass
class AdoraRobotConfig(RobotConfig):
    ip = "192.168.1.19"

    start_pose = [-90.0, 90.0, 90.0, -90.0, 0.0, 0.0, 0.0]
    joint_p_limit = [169.0, 102.0, 169.0, 52.0, 169.0, 117.0, 169.0]
    joint_n_limit = [-169.0, -102.0, -169.0, -167.0, -169.0, -87.0, -169.0]

    leader_arms: dict[str, MotorsBusConfig] = field(
        default_factory=lambda: {
            "main": DynamixelMotorsBusConfig(
                port="/dev/ttyUSB0",
                motors={
                    "shoulder_pan": [1, "xl330-m288"],
                    "shoulder_lift": [2, "xl330-m288"],
                    "elbow_flex": [3, "xl330-m288"],
                    "wrist_flex": [4, "xl330-m288"],
                    "wrist_roll": [5, "xl330-m288"],
                    "wrist_1": [6, "xl330-m288"],
                    "weist_2": [7, "xl330-m288"],
                    "gripper": [8, "xl330-m288"],
                },
            ),
        }
    )

    cameras: dict[str, CameraConfig] = field(
        default_factory=lambda: {
            "top": OpenCVCameraConfig(
                camera_index=6,
                fps=30,
                width=640,
                height=480,
            ),
            "wrist": OpenCVCameraConfig(
                camera_index=14,
                fps=30,
                width=640,
                height=480,
            ),
        }
    )


@RobotConfig.register_subclass("adora_dual")
@dataclass
class AdoraDualRobotConfig(RobotConfig):

    right_arm_config = {}
    right_arm_config['usb_port'] = "/dev/ttyUSB0"
    right_arm_config['ip'] = "192.168.1.19"
    right_arm_config['fps'] = 30
    right_arm_config['calibration_dir'] = ".cache/calibration/adora_dual_right"
    right_arm_config['start_pose'] = [90.0, 90.0, -90.0, -90.0, 0.0, 0.0, 0.0]
    right_arm_config['joint_p_limit'] = [169.0, 102.0, 169.0, 52.0, 169.0, 117.0, 169.0]
    right_arm_config['joint_n_limit'] = [-169.0, -102.0, -169.0, -167.0, -169.0, -87.0, -169.0]
    right_leader_arm = DynamixelMotorsBusConfig(
        port=right_arm_config['usb_port'],
        motors={
            "joint_1": [1, "xl330-m288"],
            "joint_2": [2, "xl330-m288"],
            "joint_3": [3, "xl330-m288"],
            "joint_4": [4, "xl330-m288"],
            "joint_5": [5, "xl330-m288"],
            "joint_6": [6, "xl330-m288"],
            "joint_7": [7, "xl330-m288"],
            "gripper": [8, "xl330-m288"],
        },
    )

    left_arm_config = {}
    left_arm_config['usb_port'] = "/dev/ttyUSB1"
    left_arm_config['ip'] = "192.168.1.20"
    left_arm_config['fps'] = 30
    left_arm_config['calibration_dir'] = ".cache/calibration/adora_dual_left"
    left_arm_config['start_pose'] = [-90.0, 90.0, 90.0, -90.0, 0.0, 0.0, 0.0]
    left_arm_config['joint_p_limit'] = [169.0, 102.0, 169.0, 52.0, 169.0, 117.0, 169.0]
    left_arm_config['joint_n_limit'] = [-169.0, -102.0, -169.0, -167.0, -169.0, -87.0, -169.0]
    left_leader_arm = DynamixelMotorsBusConfig(
        port=left_arm_config['usb_port'],
        motors={
            "joint_1": [1, "xl330-m288"],
            "joint_2": [2, "xl330-m288"],
            "joint_3": [3, "xl330-m288"],
            "joint_4": [4, "xl330-m288"],
            "joint_5": [5, "xl330-m288"],
            "joint_6": [6, "xl330-m288"],
            "joint_7": [7, "xl330-m288"],
            "gripper": [8, "xl330-m288"],
        },
    )

    cameras: dict[str, CameraConfig] = field(
        default_factory=lambda: {
            "top": OpenCVCameraConfig(
                camera_index=14,
                fps=30,
                width=640,
                height=480,
            ),
            "left_wrist": OpenCVCameraConfig(
                camera_index=6,
                fps=30,
                width=640,
                height=480,
            ),
            "right_wrist": OpenCVCameraConfig(
                camera_index=22,
                fps=30,
                width=640,
                height=480,
            ),
        }
    )

    mock: bool = False


@RobotConfig.register_subclass("realman")
@dataclass
class RealmanRobotConfig(RobotConfig):

    right_arm_config = {}
    # right_arm_config['usb_port'] = "/dev/ttyUSB0"
    right_arm_config['ip'] = "169.254.128.19"
    right_arm_config['fps'] = 30
    # right_arm_config['calibration_dir'] = ".cache/calibration/adora_dual_right"
    # right_arm_config['start_pose'] = [90.0, 90.0, -90.0, -90.0, 0.0, 0.0, 0.0]
    # right_arm_config['joint_p_limit'] = [169.0, 102.0, 169.0, 52.0, 169.0, 117.0, 169.0]
    # right_arm_config['joint_n_limit'] = [-169.0, -102.0, -169.0, -167.0, -169.0, -87.0, -169.0]
    # right_leader_arm = DynamixelMotorsBusConfig(
    #     port=right_arm_config['usb_port'],
    #     motors={
    #         "joint_1": [1, "xl330-m288"],
    #         "joint_2": [2, "xl330-m288"],
    #         "joint_3": [3, "xl330-m288"],
    #         "joint_4": [4, "xl330-m288"],
    #         "joint_5": [5, "xl330-m288"],
    #         "joint_6": [6, "xl330-m288"],
    #         "joint_7": [7, "xl330-m288"],
    #         "gripper": [8, "xl330-m288"],
    #     },
    # )

    left_arm_config = {}
    # left_arm_config['usb_port'] = "/dev/ttyUSB1"
    left_arm_config['ip'] = "169.254.128.18"
    left_arm_config['fps'] = 30
    # left_arm_config['calibration_dir'] = ".cache/calibration/adora_dual_left"
    # left_arm_config['start_pose'] = [-90.0, 90.0, 90.0, -90.0, 0.0, 0.0, 0.0]
    # left_arm_config['joint_p_limit'] = [169.0, 102.0, 169.0, 52.0, 169.0, 117.0, 169.0]
    # left_arm_config['joint_n_limit'] = [-169.0, -102.0, -169.0, -167.0, -169.0, -87.0, -169.0]
    # left_leader_arm = DynamixelMotorsBusConfig(
    #     port=left_arm_config['usb_port'],
    #     motors={
    #         "joint_1": [1, "xl330-m288"],
    #         "joint_2": [2, "xl330-m288"],
    #         "joint_3": [3, "xl330-m288"],
    #         "joint_4": [4, "xl330-m288"],
    #         "joint_5": [5, "xl330-m288"],
    #         "joint_6": [6, "xl330-m288"],
    #         "joint_7": [7, "xl330-m288"],
    #         "gripper": [8, "xl330-m288"],
    #     },
    # )

    cameras: dict[str, CameraConfig] = field(
        default_factory=lambda: {
            "top": OpenCVCameraConfig(
                camera_index=20,
                fps=30,
                width=640,
                height=480,
            ),
            "left_wrist": OpenCVCameraConfig(
                camera_index=4,
                fps=30,
                width=640,
                height=480,
            ),
            "right_wrist": OpenCVCameraConfig(
                camera_index=22,
                fps=30,
                width=640,
                height=480,
            ),
        }
    )

    mock: bool = False


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

    left_leader_arm = PiperMotorsBusConfig(
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
            # "image_right_fisheye": OpenCVCameraConfig(
            #     camera_index=3,
            #     fps=30,
            #     width=640,
            #     height=480,
            # ),
            # "image_left_fisheye": OpenCVCameraConfig(
            #     camera_index=4,
            #     fps=30,
            #     width=640,
            #     height=480,
            # ),
            "image_right_tac_r": OpenCVCameraConfig(
                camera_index=5,
                fps=30,
                width=640,
                height=480,
            ),
            # "image_right_tac_l": OpenCVCameraConfig(
            #     camera_index=6,
            #     fps=30,
            #     width=640,
            #     height=480,
            # ),
            # "image_left_tac_r": OpenCVCameraConfig(
            #     camera_index=7,
            #     fps=30,
            #     width=640,
            #     height=480,
            # ),
            "image_left_tac_l": OpenCVCameraConfig(
                camera_index=8,
                fps=30,
                width=640,
                height=480,
            ),
            "image_pika_pose": OpenCVCameraConfig(
                camera_index=9,
                fps=30,
                width=640,
                height=480,
            ),
        }
    )

    microphones: dict[str, int] = field(
        default_factory=lambda: {
            "audio_right": 2,
            "audio_left": 4,
        }
    )

