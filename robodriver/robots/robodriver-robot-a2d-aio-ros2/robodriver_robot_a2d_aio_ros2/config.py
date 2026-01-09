from typing import Dict
from dataclasses import dataclass, field

from lerobot.robots.config import RobotConfig
from lerobot.cameras import CameraConfig
from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.motors import Motor, MotorNormMode


@RobotConfig.register_subclass("a2d_aio_ros2")
@dataclass
class A2DAioRos2RobotConfig(RobotConfig):
    use_degrees: bool = False

    # G1 关节通常是 -3.14 到 3.14 (弧度)，使用默认 RAW 模式或根据标定调整
    norm_mode_body = MotorNormMode.raw

    # 定义 Follower (G1 机器人本体) 的电机列表
    # 对应文档的 14 个手臂关节
    follower_motors: Dict[str, Dict[str, Motor]] = field(
        default_factory=lambda: {
            # 左臂 7 轴
            "left_arm_joint1": Motor(1, "g1_arm", norm_mode_body),
            "left_arm_joint2": Motor(2, "g1_arm", norm_mode_body),
            "left_arm_joint3": Motor(3, "g1_arm", norm_mode_body),
            "left_arm_joint4": Motor(4, "g1_arm", norm_mode_body),
            "left_arm_joint5": Motor(5, "g1_arm", norm_mode_body),
            "left_arm_joint6": Motor(6, "g1_arm", norm_mode_body),
            "left_arm_joint7": Motor(7, "g1_arm", norm_mode_body),
            # 右臂 7 轴
            "right_arm_joint1": Motor(8, "g1_arm", norm_mode_body),
            "right_arm_joint2": Motor(9, "g1_arm", norm_mode_body),
            "right_arm_joint3": Motor(10, "g1_arm", norm_mode_body),
            "right_arm_joint4": Motor(11, "g1_arm", norm_mode_body),
            "right_arm_joint5": Motor(12, "g1_arm", norm_mode_body),
            "right_arm_joint6": Motor(13, "g1_arm", norm_mode_body),
            "right_arm_joint7": Motor(14, "g1_arm", norm_mode_body),

            # --- 末端执行器 (夹爪或灵巧手) ---
            # 0=松开, 1=夹紧 (夹爪模式) [cite: 231]
            "left_gripper": Motor(15, "g1_gripper", MotorNormMode.RANGE_0_100),
            "right_gripper": Motor(16, "g1_gripper", MotorNormMode.RANGE_0_100),
        }
    )

    # Leader 定义 (如果有主手/遥操作设备，在此定义；如果是数据采集回放，保持为空或与 Follower 结构一致)
    leader_motors: Dict[str, Dict[str, Motor]] = field(
        default_factory=lambda: {}
    )

    cameras: Dict[str, CameraConfig] = field(
        default_factory=lambda: {
            "head_color": OpenCVCameraConfig("/camera/head_color", fps=30, width=1280, height=720),
            "hand_left_color":  OpenCVCameraConfig("/camera/hand_left_color",  fps=30, width=848, height=480),
            "hand_right_color": OpenCVCameraConfig("/camera/hand_right_color", fps=30, width=848, height=480),
        }
    )

    use_videos: bool = False

    microphones: Dict[str, int] = field(
        default_factory=lambda: {}
    )
