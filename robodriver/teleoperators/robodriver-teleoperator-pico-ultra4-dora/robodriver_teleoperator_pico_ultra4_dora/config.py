from dataclasses import dataclass, field
from typing import Dict, Any
import numpy as np

from lerobot.teleoperators.config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("pico_ultra4_dora")
@dataclass
class PicoUltra4DoraTeleoperatorConfig(TeleoperatorConfig):
    """
    Pico Ultra4 VR 遥操器配置

    使用 VR 控制器进行末端位姿控制，通过 IK 求解转换为关节角度
    """

    # VR 控制器配置
    vr_controller: str = "right_controller"  # 使用右手控制器
    control_trigger: str = "right_grip"      # 右手握持键激活控制
    gripper_trigger: str = "right_trigger"   # 右手扳机键控制夹爪

    # 机械臂配置
    robot_urdf_path: str = ""  # Piper URDF 文件路径
    end_effector_link: str = "link6"  # 末端执行器链接名

    # IK 求解器配置
    scale_factor: float = 1.5  # VR 控制器移动缩放因子
    control_rate_hz: int = 50  # 控制频率

    # 坐标系转换
    # VR 头显坐标系到世界坐标系的旋转矩阵
    R_headset_world: np.ndarray = field(default_factory=lambda: np.eye(3))

    # 夹爪配置
    gripper_open_pos: float = 0.85   # 完全打开位置
    gripper_close_pos: float = 0.0   # 完全闭合位置

    # Piper 机械臂关节配置
    num_joints: int = 6
    joint_names: list = field(default_factory=lambda: [
        "joint1", "joint2", "joint3", "joint4", "joint5", "joint6"
    ])
