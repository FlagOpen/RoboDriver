from typing import Dict
from dataclasses import dataclass, field

from lerobot.robots.config import RobotConfig
from lerobot.cameras import CameraConfig
from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.motors import Motor, MotorNormMode


@RobotConfig.register_subclass("leju-kuavo-teleop-ros1")
@dataclass
class LEJUKuavoRos1Config(RobotConfig):
    use_degrees = True
    norm_mode_body = (
        MotorNormMode.DEGREES if use_degrees else MotorNormMode.RANGE_M100_100
    )

   

    follower_motors: Dict[str, Dict[str, Motor]] = field(
        default_factory=lambda norm_mode_body=norm_mode_body: {
            "right_arm": {
                "joint_0": Motor(1, "robot_motor", norm_mode_body),
                "joint_1": Motor(2, "robot_motor", norm_mode_body),
                "joint_2": Motor(3, "robot_motor", norm_mode_body),
                "joint_3": Motor(4, "robot_motor", norm_mode_body),
                "joint_4": Motor(5, "robot_motor", norm_mode_body),
                "joint_5": Motor(6, "robot_motor", norm_mode_body),
                "joint_6": Motor(7, "robot_motor", norm_mode_body),
            },
            "left_arm": {
                "joint_0": Motor(1, "robot_motor", norm_mode_body),
                "joint_1": Motor(2, "robot_motor", norm_mode_body),
                "joint_2": Motor(3, "robot_motor", norm_mode_body),
                "joint_3": Motor(4, "robot_motor", norm_mode_body),
                "joint_4": Motor(5, "robot_motor", norm_mode_body),
                "joint_5": Motor(6, "robot_motor", norm_mode_body),
                "joint_6": Motor(7, "robot_motor", norm_mode_body),
            },
            "right_dexhand": {
                "r_thumb": Motor(1, "robot_motor", norm_mode_body),
                "r_thumb_aux": Motor(2, "robot_motor", norm_mode_body),
                "r_index": Motor(3, "robot_motor", norm_mode_body),
                "r_middle": Motor(4, "robot_motor", norm_mode_body),
                "r_ring": Motor(5, "robot_motor", norm_mode_body),
                "r_pinky": Motor(6, "robot_motor", norm_mode_body),
            },
            "left_dexhand": {
                "l_thumb": Motor(1, "robot_motor", norm_mode_body),
                "l_thumb_aux": Motor(2, "robot_motor", norm_mode_body),
                "l_index": Motor(3, "robot_motor", norm_mode_body),
                "l_middle": Motor(4, "robot_motor", norm_mode_body), 
                "l_ring": Motor(5, "robot_motor", norm_mode_body),
                "l_pinky": Motor(6, "robot_motor", norm_mode_body),   
            },
            "head":{
                "head_yaw": Motor(1, "robot_motor", norm_mode_body),
                "head_pitch": Motor(2, "robot_motor", norm_mode_body),
            }
        }
    )

    cameras: Dict[str, CameraConfig] = field(
        default_factory=lambda: {
            "image_top": OpenCVCameraConfig(index_or_path=0, fps=30, width=424, height=240),
            "image_wrist_left": OpenCVCameraConfig(index_or_path=1, fps=30, width=848, height=480),
            "image_wrist_right": OpenCVCameraConfig(index_or_path=2, fps=30, width=848, height=480),
        }
    )

    use_videos: bool = True

    microphones: Dict[str, int] = field(default_factory=lambda: {}
    )
