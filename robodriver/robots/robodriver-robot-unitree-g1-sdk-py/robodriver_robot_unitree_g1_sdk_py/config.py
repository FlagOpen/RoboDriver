from typing import Dict
from dataclasses import dataclass, field

from lerobot.robots.config import RobotConfig
from lerobot.cameras import CameraConfig
from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.motors import Motor, MotorNormMode


@RobotConfig.register_subclass("unitree_g1_sdk_py")
@dataclass
class UnitreeG1RobotConfig(RobotConfig):
    use_degrees = True
    norm_mode_body = (
        MotorNormMode.DEGREES if use_degrees else MotorNormMode.RANGE_M100_100
    )

    # 按组件分组：{ comp_id: { joint_name: Motor, ... }, ... }
    leader_motors: Dict[str, Dict[str, Motor]] = field(
        default_factory=lambda norm_mode_body=norm_mode_body: {
            "leader_joints": {
                "LeftHipPitch": Motor(0, "robot_motor", norm_mode_body),
                "LeftHipRoll": Motor(1, "robot_motor", norm_mode_body),
                "LeftHipYaw": Motor(2, "robot_motor", norm_mode_body),
                "LeftKnee": Motor(3, "robot_motor", norm_mode_body),
                "LeftAnklePitch": Motor(4, "robot_motor", norm_mode_body),
                "LeftAnkleRoll": Motor(5, "robot_motor", norm_mode_body),
                #右腿:
                "RightHipPitch": Motor(6, "robot_motor", norm_mode_body),
                "RightHipRoll": Motor(7, "robot_motor", norm_mode_body),
                "RightHipYaw": Motor(8, "robot_motor", norm_mode_body),
                "RightKnee": Motor(9, "robot_motor", norm_mode_body),
                "RightAnklePitch": Motor(10, "robot_motor", norm_mode_body),
                "RightAnkleRoll": Motor(11, "robot_motor", norm_mode_body),
                #腰部:
                "WaistYaw": Motor(12, "robot_motor", norm_mode_body),
                "WaistRoll": Motor(13, "robot_motor", norm_mode_body),
                "WaistPitch": Motor(14, "robot_motor", norm_mode_body),       
                #左臂:
                "LeftShoulderPitch": Motor(15, "robot_motor", norm_mode_body),
                "LeftShoulderRoll": Motor(16, "robot_motor", norm_mode_body),
                "LeftShoulderYaw": Motor(17, "robot_motor", norm_mode_body),
                "LeftElbow": Motor(18, "robot_motor", norm_mode_body),
                "LeftWristRoll": Motor(19, "robot_motor", norm_mode_body),
                "LeftWristPitch": Motor(20, "robot_motor", norm_mode_body),
                "LeftWristYaw": Motor(21, "robot_motor", norm_mode_body),     
                #右臂:
                "RightShoulderPitch": Motor(22, "robot_motor", norm_mode_body),
                "RightShoulderRoll": Motor(23, "robot_motor", norm_mode_body),
                "RightShoulderYaw": Motor(24, "robot_motor", norm_mode_body),
                "RightElbow": Motor(25, "robot_motor", norm_mode_body),
                "RightWristRoll": Motor(26, "robot_motor", norm_mode_body),
                "RightWristPitch": Motor(27, "robot_motor", norm_mode_body),  
                "RightWristYaw": Motor(28, "robot_motor", norm_mode_body), 
            },
        }
    )

    follower_motors: Dict[str, Dict[str, Motor]] = field(
        default_factory=lambda norm_mode_body=norm_mode_body: {
            "follower_joint": {
                "LeftHipPitch": Motor(0, "robot_motor", norm_mode_body),
                "LeftHipRoll": Motor(1, "robot_motor", norm_mode_body),
                "LeftHipYaw": Motor(2, "robot_motor", norm_mode_body),
                "LeftKnee": Motor(3, "robot_motor", norm_mode_body),
                "LeftAnklePitch": Motor(4, "robot_motor", norm_mode_body),
                "LeftAnkleRoll": Motor(5, "robot_motor", norm_mode_body),
                #右腿:
                "RightHipPitch": Motor(6, "robot_motor", norm_mode_body),
                "RightHipRoll": Motor(7, "robot_motor", norm_mode_body),
                "RightHipYaw": Motor(8, "robot_motor", norm_mode_body),
                "RightKnee": Motor(9, "robot_motor", norm_mode_body),
                "RightAnklePitch": Motor(10, "robot_motor", norm_mode_body),
                "RightAnkleRoll": Motor(11, "robot_motor", norm_mode_body),
                #腰部:
                "WaistYaw": Motor(12, "robot_motor", norm_mode_body),
                "WaistRoll": Motor(13, "robot_motor", norm_mode_body),
                "WaistPitch": Motor(14, "robot_motor", norm_mode_body),       
                #左臂:
                "LeftShoulderPitch": Motor(15, "robot_motor", norm_mode_body),
                "LeftShoulderRoll": Motor(16, "robot_motor", norm_mode_body),
                "LeftShoulderYaw": Motor(17, "robot_motor", norm_mode_body),
                "LeftElbow": Motor(18, "robot_motor", norm_mode_body),
                "LeftWristRoll": Motor(19, "robot_motor", norm_mode_body),
                "LeftWristPitch": Motor(20, "robot_motor", norm_mode_body),
                "LeftWristYaw": Motor(21, "robot_motor", norm_mode_body),     
                #右臂:
                "RightShoulderPitch": Motor(22, "robot_motor", norm_mode_body),
                "RightShoulderRoll": Motor(23, "robot_motor", norm_mode_body),
                "RightShoulderYaw": Motor(24, "robot_motor", norm_mode_body),
                "RightElbow": Motor(25, "robot_motor", norm_mode_body),
                "RightWristRoll": Motor(26, "robot_motor", norm_mode_body),
                "RightWristPitch": Motor(27, "robot_motor", norm_mode_body),  
                "RightWristYaw": Motor(28, "robot_motor", norm_mode_body),
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
