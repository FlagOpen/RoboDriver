from typing import Dict
from dataclasses import dataclass, field

from lerobot.robots.config import RobotConfig
from lerobot.cameras import CameraConfig
from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.motors import Motor, MotorNormMode


@RobotConfig.register_subclass("deeprobotics_x30_ros1")
@dataclass
class DeeproboticsX30Ros1RobotConfig(RobotConfig):
    use_degrees = True
    norm_mode_body = (
        MotorNormMode.DEGREES if use_degrees else MotorNormMode.RANGE_M100_100
    )

    # 按组件分组：{ comp_id: { joint_name: Motor, ... }, ... }
    leader_motors: Dict[str, Dict[str, Motor]] = field(
        default_factory=lambda norm_mode_body=norm_mode_body: {
            "leader_joints": {
                # 前左腿 (front left)
                "fl_hipx": Motor(1, "robot_motor", norm_mode_body),
                "fl_hipy": Motor(2, "robot_motor", norm_mode_body),
                "fl_knee": Motor(3, "robot_motor", norm_mode_body),
                # 前右腿 (front right)
                "fr_hipx": Motor(4, "robot_motor", norm_mode_body),
                "fr_hipy": Motor(5, "robot_motor", norm_mode_body),
                "fr_knee": Motor(6, "robot_motor", norm_mode_body),
                # 后左腿 (hind left)
                "hl_hipx": Motor(7, "robot_motor", norm_mode_body),
                "hl_hipy": Motor(8, "robot_motor", norm_mode_body),
                "hl_knee": Motor(9, "robot_motor", norm_mode_body),
                # 后右腿 (hind right)               
                "hr_hipx": Motor(10, "robot_motor", norm_mode_body),
                "hr_hipy": Motor(11, "robot_motor", norm_mode_body),
                "hr_knee": Motor(11, "robot_motor", norm_mode_body),
            },
        }
    )

    follower_motors: Dict[str, Dict[str, Motor]] = field(
        default_factory=lambda norm_mode_body=norm_mode_body: {
            "follower_joint": {
                # 前左腿 (front left)
                "fl_hipx": Motor(1, "robot_motor", norm_mode_body),
                "fl_hipy": Motor(2, "robot_motor", norm_mode_body),
                "fl_knee": Motor(3, "robot_motor", norm_mode_body),
                # 前右腿 (front right)
                "fr_hipx": Motor(4, "robot_motor", norm_mode_body),
                "fr_hipy": Motor(5, "robot_motor", norm_mode_body),
                "fr_knee": Motor(6, "robot_motor", norm_mode_body),
                # 后左腿 (hind left)
                "hl_hipx": Motor(7, "robot_motor", norm_mode_body),
                "hl_hipy": Motor(8, "robot_motor", norm_mode_body),
                "hl_knee": Motor(9, "robot_motor", norm_mode_body),
                # 后右腿 (hind right)               
                "hr_hipx": Motor(10, "robot_motor", norm_mode_body),
                "hr_hipy": Motor(11, "robot_motor", norm_mode_body),
                "hr_knee": Motor(11, "robot_motor", norm_mode_body),
            },
        }
    )

    cameras: Dict[str, CameraConfig] = field(
        default_factory=lambda: {
            "image_top": OpenCVCameraConfig(index_or_path=0, fps=30, width=640, height=480),
        }
    )

    use_videos: bool = True

    microphones: Dict[str, int] = field(default_factory=lambda: {}
    )
