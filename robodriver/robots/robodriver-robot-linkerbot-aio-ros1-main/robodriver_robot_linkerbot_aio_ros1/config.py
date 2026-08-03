from typing import Dict
from dataclasses import dataclass, field

from lerobot.robots.config import RobotConfig
from lerobot.cameras import CameraConfig
from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.motors import Motor, MotorNormMode


@RobotConfig.register_subclass("linkerbot_aio_ros1")
@dataclass
class LinkerBotAioRos1RobotConfig(RobotConfig):
    use_degrees = True
    norm_mode_body = (
        MotorNormMode.DEGREES if use_degrees else MotorNormMode.RANGE_M100_100
    )


    follower_motors: Dict[str, Dict[str, Motor]] = field(
        default_factory=lambda norm_mode_body=norm_mode_body: {
            "follower_left_joints": {
                "joint_1": Motor(1, "robot_motor", norm_mode_body),
                "joint_2": Motor(2, "robot_motor", norm_mode_body),
                "joint_3": Motor(3, "robot_motor", norm_mode_body),
                "joint_4": Motor(4, "robot_motor", norm_mode_body),
                "joint_5": Motor(5, "robot_motor", norm_mode_body),
                "joint_6": Motor(6, "robot_motor", norm_mode_body),
                "joint_7": Motor(7, "robot_motor", norm_mode_body),
            },
            "follower_right_joints": {
                "joint_1": Motor(1, "robot_motor", norm_mode_body),
                "joint_2": Motor(2, "robot_motor", norm_mode_body),
                "joint_3": Motor(3, "robot_motor", norm_mode_body),
                "joint_4": Motor(4, "robot_motor", norm_mode_body),
                "joint_5": Motor(5, "robot_motor", norm_mode_body),
                "joint_6": Motor(6, "robot_motor", norm_mode_body),
                "joint_7": Motor(7, "robot_motor", norm_mode_body),
            },
            "follower_left_hand": {
                "joint_1": Motor(1, "robot_motor", norm_mode_body),
                "joint_2": Motor(2, "robot_motor", norm_mode_body),
                "joint_3": Motor(3, "robot_motor", norm_mode_body),
                "joint_4": Motor(4, "robot_motor", norm_mode_body),
                "joint_5": Motor(5, "robot_motor", norm_mode_body),
                "joint_6": Motor(6, "robot_motor", norm_mode_body),
                "joint_7": Motor(7, "robot_motor", norm_mode_body),
                "joint_8": Motor(8, "robot_motor", norm_mode_body),
                "joint_9": Motor(9, "robot_motor", norm_mode_body),
                "joint_10": Motor(10, "robot_motor", norm_mode_body),
            },
            "follower_right_hand": {
                "joint_1": Motor(1, "robot_motor", norm_mode_body),
                "joint_2": Motor(2, "robot_motor", norm_mode_body),
                "joint_3": Motor(3, "robot_motor", norm_mode_body),
                "joint_4": Motor(4, "robot_motor", norm_mode_body),
                "joint_5": Motor(5, "robot_motor", norm_mode_body),
                "joint_6": Motor(6, "robot_motor", norm_mode_body),
                "joint_7": Motor(7, "robot_motor", norm_mode_body),
                "joint_8": Motor(8, "robot_motor", norm_mode_body),
                "joint_9": Motor(9, "robot_motor", norm_mode_body),
                "joint_10": Motor(10, "robot_motor", norm_mode_body),
            },
            "follower_left_pose": {
                "pose_position_x": Motor(1, "robot_motor", norm_mode_body),
                "pose_position_y": Motor(2, "robot_motor", norm_mode_body),
                "pose_position_z": Motor(3, "robot_motor", norm_mode_body),
                "pose_oriention_x": Motor(4, "robot_motor", norm_mode_body),
                "pose_oriention_y": Motor(5, "robot_motor", norm_mode_body),
                "pose_oriention_z": Motor(6, "robot_motor", norm_mode_body),
                "pose_oriention_w": Motor(7, "robot_motor", norm_mode_body),
            },
            "follower_right_pose": {
                "pose_position_x": Motor(1, "robot_motor", norm_mode_body),
                "pose_position_y": Motor(2, "robot_motor", norm_mode_body),
                "pose_position_z": Motor(3, "robot_motor", norm_mode_body),
                "pose_oriention_x": Motor(4, "robot_motor", norm_mode_body),
                "pose_oriention_y": Motor(5, "robot_motor", norm_mode_body),
                "pose_oriention_z": Motor(6, "robot_motor", norm_mode_body),
                "pose_oriention_w": Motor(7, "robot_motor", norm_mode_body),
            },
        }
    )
    leader_motors: Dict[str, Dict[str, Motor]] = field(
        default_factory=lambda norm_mode_body=norm_mode_body: {
            "leader_left_joints": {
                "joint_1": Motor(1, "robot_motor", norm_mode_body),
                "joint_2": Motor(2, "robot_motor", norm_mode_body),
                "joint_3": Motor(3, "robot_motor", norm_mode_body),
                "joint_4": Motor(4, "robot_motor", norm_mode_body),
                "joint_5": Motor(5, "robot_motor", norm_mode_body),
                "joint_6": Motor(6, "robot_motor", norm_mode_body),
                "joint_7": Motor(7, "robot_motor", norm_mode_body),
            },
            "leader_right_joints": {
                "joint_1": Motor(1, "robot_motor", norm_mode_body),
                "joint_2": Motor(2, "robot_motor", norm_mode_body),
                "joint_3": Motor(3, "robot_motor", norm_mode_body),
                "joint_4": Motor(4, "robot_motor", norm_mode_body),
                "joint_5": Motor(5, "robot_motor", norm_mode_body),
                "joint_6": Motor(6, "robot_motor", norm_mode_body),
                "joint_7": Motor(7, "robot_motor", norm_mode_body),
            },
            "leader_left_hand": {
                "thumb_spread": Motor(1, "robot_motor", norm_mode_body),
                "thumb_bend": Motor(2, "robot_motor", norm_mode_body),
                "thumb_tip": Motor(3, "robot_motor", norm_mode_body),
                "index_spread": Motor(4, "robot_motor", norm_mode_body),
                "index_bend": Motor(5, "robot_motor", norm_mode_body),
                "index_tip": Motor(6, "robot_motor", norm_mode_body),
                "middle_spread": Motor(7, "robot_motor", norm_mode_body),
                "middle_bend": Motor(8, "robot_motor", norm_mode_body),
                "middle_tip": Motor(9, "robot_motor", norm_mode_body),
                "ring_spread": Motor(10, "robot_motor", norm_mode_body),
                "ring_bend": Motor(11, "robot_motor", norm_mode_body),
                "ring_tip": Motor(12, "robot_motor", norm_mode_body),
                "pinky_spread": Motor(13, "robot_motor", norm_mode_body),
                "pinky_bend": Motor(14, "robot_motor", norm_mode_body),
                "pinky_tip": Motor(15, "robot_motor", norm_mode_body),
            },
            "leader_right_hand": {
                "thumb_spread": Motor(1, "robot_motor", norm_mode_body),
                "thumb_bend": Motor(2, "robot_motor", norm_mode_body),
                "thumb_tip": Motor(3, "robot_motor", norm_mode_body),
                "index_spread": Motor(4, "robot_motor", norm_mode_body),
                "index_bend": Motor(5, "robot_motor", norm_mode_body),
                "index_tip": Motor(6, "robot_motor", norm_mode_body),
                "middle_spread": Motor(7, "robot_motor", norm_mode_body),
                "middle_bend": Motor(8, "robot_motor", norm_mode_body),
                "middle_tip": Motor(9, "robot_motor", norm_mode_body),
                "ring_spread": Motor(10, "robot_motor", norm_mode_body),
                "ring_bend": Motor(11, "robot_motor", norm_mode_body),
                "ring_tip": Motor(12, "robot_motor", norm_mode_body),
                "pinky_spread": Motor(13, "robot_motor", norm_mode_body),
                "pinky_bend": Motor(14, "robot_motor", norm_mode_body),
                "pinky_tip": Motor(15, "robot_motor", norm_mode_body),
            },
        }
    )
        
    cameras: Dict[str, CameraConfig] = field(
        default_factory=lambda:  {
            "image_top": OpenCVCameraConfig(
                index_or_path=1,
                fps=10,
                width=640,
                height=480,
            ),
        }
    )

    use_videos: bool = False

    microphones: Dict[str, int] = field(default_factory=lambda: {}
    )