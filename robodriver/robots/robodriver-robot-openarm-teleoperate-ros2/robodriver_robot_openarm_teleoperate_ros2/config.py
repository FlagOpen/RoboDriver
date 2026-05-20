from typing import Dict
from dataclasses import dataclass, field

from lerobot.robots.config import RobotConfig
from lerobot.cameras import CameraConfig
from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.motors import Motor, MotorNormMode


@RobotConfig.register_subclass("openarm-teleop-ros2")
@dataclass
class OPENARMRos2RobotConfig(RobotConfig):
    use_degrees = True
    norm_mode_body = (
        MotorNormMode.DEGREES if use_degrees else MotorNormMode.RANGE_M100_100
    )


    """
      leader_motors: Dict[str, Motor] = field(
        default_factory=lambda norm_mode_body=norm_mode_body: {
            "leader_arms":{
            "left_arm_joint_1_rad": Motor(1, "sts3215", norm_mode_body),
            "left_arm_joint_2_rad": Motor(2, "sts3215", norm_mode_body),
            "left_arm_joint_3_rad": Motor(3, "sts3215", norm_mode_body),
            "left_arm_joint_4_rad": Motor(4, "sts3215", norm_mode_body),
            "left_arm_joint_5_rad": Motor(5, "sts3215", norm_mode_body),
            "left_arm_joint_6_rad": Motor(6, "sts3215", norm_mode_body),
            "left_gripper_degree_mm": Motor(7, "sts3215", MotorNormMode.RANGE_0_100),
            "right_arm_joint_1_rad": Motor(8, "sts3215", norm_mode_body),
            "right_arm_joint_2_rad": Motor(9, "sts3215", norm_mode_body),
            "right_arm_joint_3_rad": Motor(10, "sts3215", norm_mode_body),
            "right_arm_joint_4_rad": Motor(11, "sts3215", norm_mode_body),
            "right_arm_joint_5_rad": Motor(12, "sts3215", norm_mode_body),
            "right_arm_joint_6_rad": Motor(13, "sts3215", norm_mode_body),
            "right_gripper_degree_mm": Motor(14, "sts3215", MotorNormMode.RANGE_0_100),
           

            # 左臂pose数据 (位置x,y,z + 四元数x,y,z,w)
            "left_arm_pos_x": Motor(0, "sts3215", norm_mode_body),  
            "left_arm_pos_y": Motor(1, "sts3215", norm_mode_body),
            "left_arm_pos_z": Motor(2, "sts3215", norm_mode_body),
            "left_arm_quat_x": Motor(3, "sts3215", norm_mode_body),
            "left_arm_quat_y": Motor(4, "sts3215", norm_mode_body),
            "left_arm_quat_z": Motor(5, "sts3215", norm_mode_body),
            "left_arm_quat_w": Motor(6, "sts3215", norm_mode_body),
            # 右臂pose数据
            "right_arm_pos_x": Motor(7, "sts3215", norm_mode_body),
            "right_arm_pos_y": Motor(8, "sts3215", norm_mode_body),
            "right_arm_pos_z": Motor(9, "sts3215", norm_mode_body),
            "right_arm_quat_x": Motor(10, "sts3215", norm_mode_body),
            "right_arm_quat_y": Motor(11, "sts3215", norm_mode_body),
            "right_arm_quat_z": Motor(12, "sts3215", norm_mode_body),
            "right_arm_quat_w": Motor(13, "sts3215", norm_mode_body),
            
           
        }
        }
    )
    """

  

    follower_motors: Dict[str, Motor] = field(
        default_factory=lambda norm_mode_body=norm_mode_body: {
            "follower_arms":{
            "left_arm_joint_1_rad": Motor(1, "sts3215", norm_mode_body),
            "left_arm_joint_2_rad": Motor(2, "sts3215", norm_mode_body),
            "left_arm_joint_3_rad": Motor(3, "sts3215", norm_mode_body),
            "left_arm_joint_4_rad": Motor(4, "sts3215", norm_mode_body),
            "left_arm_joint_5_rad": Motor(5, "sts3215", norm_mode_body),
            "left_arm_joint_6_rad": Motor(6, "sts3215", norm_mode_body),
            "left_arm_joint_7_rad": Motor(7, "sts3215", norm_mode_body),
            "left_gripper_degree_mm": Motor(7, "sts3215", MotorNormMode.RANGE_0_100),
            "right_arm_joint_1_rad": Motor(8, "sts3215", norm_mode_body),
            "right_arm_joint_2_rad": Motor(9, "sts3215", norm_mode_body),
            "right_arm_joint_3_rad": Motor(10, "sts3215", norm_mode_body),
            "right_arm_joint_4_rad": Motor(11, "sts3215", norm_mode_body),
            "right_arm_joint_5_rad": Motor(12, "sts3215", norm_mode_body),
            "right_arm_joint_6_rad": Motor(13, "sts3215", norm_mode_body),
            "right_arm_joint_7_rad": Motor(14, "sts3215", norm_mode_body),
            "right_gripper_degree_mm": Motor(14, "sts3215", MotorNormMode.RANGE_0_100),
        }
        }
    )
    
    cameras: Dict[str, CameraConfig] = field(
        default_factory=lambda: {
            "image_top": OpenCVCameraConfig(
                index_or_path=1,
                fps=30,
                width=640,
                height=480,
            ),
            "image_wrist_left": OpenCVCameraConfig(
                index_or_path=2,
                fps=30,
                width=640,
                height=480,  
            ),
            "image_wrist_right": OpenCVCameraConfig(
                index_or_path=3,
                fps=30,
                width=640,
                height=480,  
            ),

        }
    )
    
   

    use_videos: bool = True

    microphones: Dict[str, int] = field(default_factory=lambda: {})

