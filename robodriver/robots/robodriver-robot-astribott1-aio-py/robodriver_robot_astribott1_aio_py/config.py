from dataclasses import dataclass, field
from typing import Dict, List, Optional

from lerobot.cameras import CameraConfig
from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.motors import Motor, MotorNormMode
from lerobot.robots.config import RobotConfig

from .sharpa_hand import SharpaHandsConfig


@RobotConfig.register_subclass("astribott1_aio_py")
@dataclass
class AstribotT1AioPyRobotConfig(RobotConfig):
    # SDK 连接
    ip: Optional[str] = None
    node_name: str = "robodriver"
    freq: float = 125.0

    # ROS2，话题名在 node.py 中写死
    ros_domain_id: int = 25
    ros_wait_first_frame_s: float = 10.0

    # 要记录的部件（真机实测自由度见注释）
    # arm_left 7 / arm_right 7 / effector_left 1 / effector_right 1 / head 2 / torso 4 / chassis 3
    record_parts: List[str] = field(
        default_factory=lambda: [
            "arm_left", "arm_right",
            "effector_left", "effector_right",
            "head", "torso", "chassis",
        ]
    )

    leader_motors: Dict[str, Dict[str, Motor]] = field(
        default_factory=lambda: {
            "arm_left": {
                "arm_left_joint1": Motor(0, "astribot_motor", MotorNormMode.RANGE_M100_100),
                "arm_left_joint2": Motor(1, "astribot_motor", MotorNormMode.RANGE_M100_100),
                "arm_left_joint3": Motor(2, "astribot_motor", MotorNormMode.RANGE_M100_100),
                "arm_left_joint4": Motor(3, "astribot_motor", MotorNormMode.RANGE_M100_100),
                "arm_left_joint5": Motor(4, "astribot_motor", MotorNormMode.RANGE_M100_100),
                "arm_left_joint6": Motor(5, "astribot_motor", MotorNormMode.RANGE_M100_100),
                "arm_left_joint7": Motor(6, "astribot_motor", MotorNormMode.RANGE_M100_100),
            },
            "arm_right": {
                "arm_right_joint1": Motor(0, "astribot_motor", MotorNormMode.RANGE_M100_100),
                "arm_right_joint2": Motor(1, "astribot_motor", MotorNormMode.RANGE_M100_100),
                "arm_right_joint3": Motor(2, "astribot_motor", MotorNormMode.RANGE_M100_100),
                "arm_right_joint4": Motor(3, "astribot_motor", MotorNormMode.RANGE_M100_100),
                "arm_right_joint5": Motor(4, "astribot_motor", MotorNormMode.RANGE_M100_100),
                "arm_right_joint6": Motor(5, "astribot_motor", MotorNormMode.RANGE_M100_100),
                "arm_right_joint7": Motor(6, "astribot_motor", MotorNormMode.RANGE_M100_100),
            },
            "effector_left":  {"effector_left_joint1":  Motor(0, "astribot_motor", MotorNormMode.RANGE_M100_100)},
            "effector_right": {"effector_right_joint1": Motor(0, "astribot_motor", MotorNormMode.RANGE_M100_100)},
            "head": {
                "head_joint1": Motor(0, "astribot_motor", MotorNormMode.RANGE_M100_100),
                "head_joint2": Motor(1, "astribot_motor", MotorNormMode.RANGE_M100_100),
            },
            "torso": {
                "torso_joint1": Motor(0, "astribot_motor", MotorNormMode.RANGE_M100_100),
                "torso_joint2": Motor(1, "astribot_motor", MotorNormMode.RANGE_M100_100),
                "torso_joint3": Motor(2, "astribot_motor", MotorNormMode.RANGE_M100_100),
                "torso_joint4": Motor(3, "astribot_motor", MotorNormMode.RANGE_M100_100),
            },
            "chassis": {
                "chassis_joint1": Motor(0, "astribot_motor", MotorNormMode.RANGE_M100_100),
                "chassis_joint2": Motor(1, "astribot_motor", MotorNormMode.RANGE_M100_100),
                "chassis_joint3": Motor(2, "astribot_motor", MotorNormMode.RANGE_M100_100),
            },
        }
    )

    follower_motors: Dict[str, Dict[str, Motor]] = field(
        default_factory=lambda: {
            "arm_left": {
                "arm_left_joint1": Motor(0, "astribot_motor", MotorNormMode.RANGE_M100_100),
                "arm_left_joint2": Motor(1, "astribot_motor", MotorNormMode.RANGE_M100_100),
                "arm_left_joint3": Motor(2, "astribot_motor", MotorNormMode.RANGE_M100_100),
                "arm_left_joint4": Motor(3, "astribot_motor", MotorNormMode.RANGE_M100_100),
                "arm_left_joint5": Motor(4, "astribot_motor", MotorNormMode.RANGE_M100_100),
                "arm_left_joint6": Motor(5, "astribot_motor", MotorNormMode.RANGE_M100_100),
                "arm_left_joint7": Motor(6, "astribot_motor", MotorNormMode.RANGE_M100_100),
            },
            "arm_right": {
                "arm_right_joint1": Motor(0, "astribot_motor", MotorNormMode.RANGE_M100_100),
                "arm_right_joint2": Motor(1, "astribot_motor", MotorNormMode.RANGE_M100_100),
                "arm_right_joint3": Motor(2, "astribot_motor", MotorNormMode.RANGE_M100_100),
                "arm_right_joint4": Motor(3, "astribot_motor", MotorNormMode.RANGE_M100_100),
                "arm_right_joint5": Motor(4, "astribot_motor", MotorNormMode.RANGE_M100_100),
                "arm_right_joint6": Motor(5, "astribot_motor", MotorNormMode.RANGE_M100_100),
                "arm_right_joint7": Motor(6, "astribot_motor", MotorNormMode.RANGE_M100_100),
            },
            "effector_left":  {"effector_left_joint1":  Motor(0, "astribot_motor", MotorNormMode.RANGE_M100_100)},
            "effector_right": {"effector_right_joint1": Motor(0, "astribot_motor", MotorNormMode.RANGE_M100_100)},
            "head": {
                "head_joint1": Motor(0, "astribot_motor", MotorNormMode.RANGE_M100_100),
                "head_joint2": Motor(1, "astribot_motor", MotorNormMode.RANGE_M100_100),
            },
            "torso": {
                "torso_joint1": Motor(0, "astribot_motor", MotorNormMode.RANGE_M100_100),
                "torso_joint2": Motor(1, "astribot_motor", MotorNormMode.RANGE_M100_100),
                "torso_joint3": Motor(2, "astribot_motor", MotorNormMode.RANGE_M100_100),
                "torso_joint4": Motor(3, "astribot_motor", MotorNormMode.RANGE_M100_100),
            },
            "chassis": {
                "chassis_joint1": Motor(0, "astribot_motor", MotorNormMode.RANGE_M100_100),
                "chassis_joint2": Motor(1, "astribot_motor", MotorNormMode.RANGE_M100_100),
                "chassis_joint3": Motor(2, "astribot_motor", MotorNormMode.RANGE_M100_100),
            },
        }
    )

    # ==== 控制权 ====
    control_mode: str = "takeover"
    takeover_priority: int = 0
    takeover_sub_priority: int = 0
    control_parts: List[str] = field(
        default_factory=lambda: ["arm_left", "arm_right", "effector_left", "effector_right"]
    )
    force_takeover: bool = False

    # ==== 相机 ====
    camera_timeout: float = 10.0

    # image_key -> AstriBot SDK 相机名
    camera_name_map: Dict[str, str] = field(
        default_factory=lambda: {
            "image_head_left":  "UDP_Camera",
            "image_head_right": "UDP_Camera",
            "image_hand_left":  "left_D405",
            "image_hand_right": "right_D405",
            "image_anjia":      "Anjia",
        }
    )

    # 从原始帧裁出子图 (x, y, w, h)
    # UDP_Camera 输出并排双目：res=1136x640 → 2272x640；按半宽切左右
    camera_crops: Dict[str, tuple] = field(
        default_factory=lambda: {
            "image_head_left":  (0,    0, 1136, 640),
            "image_head_right": (1136, 0, 1136, 640),
        }
    )

    # index_or_path 仅作为 RoboXStudio 视频流 id，实际取帧走 AstribotCamera
    cameras: Dict[str, CameraConfig] = field(
        default_factory=lambda: {
            "image_head_left":  OpenCVCameraConfig(index_or_path=1, fps=30, width=1136, height=640),
            "image_head_right": OpenCVCameraConfig(index_or_path=2, fps=30, width=1136, height=640),
            "image_hand_left":  OpenCVCameraConfig(index_or_path=3, fps=30, width=1280, height=720),
            "image_hand_right": OpenCVCameraConfig(index_or_path=4, fps=30, width=1280, height=720),
            "image_anjia":      OpenCVCameraConfig(index_or_path=5, fps=30, width=640,  height=360),
        }
    )

    use_videos: bool = False
    microphones: Dict[str, int] = field(default_factory=dict)


@RobotConfig.register_subclass("astribott1_aio_sharpa_py")
@dataclass
class AstribotT1AioSharpaPyRobotConfig(AstribotT1AioPyRobotConfig):
    """AstriBot T1 AIO + Sharpa Wave 灵巧手（22 DoF/手）。"""

    record_parts: List[str] = field(
        default_factory=lambda: [
            "arm_left", "arm_right", "head", "torso", "chassis",
        ]
    )
    sharpa_hands: SharpaHandsConfig = field(default_factory=SharpaHandsConfig)

