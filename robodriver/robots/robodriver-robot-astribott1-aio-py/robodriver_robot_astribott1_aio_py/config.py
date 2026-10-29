from dataclasses import dataclass, field
from typing import Dict, List, Optional

from lerobot.cameras import CameraConfig
from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.robots.config import RobotConfig


@RobotConfig.register_subclass("astribott1_aio_py")
@dataclass
class AstribotT1AioPyRobotConfig(RobotConfig):
    ip: Optional[str] = None
    node_name: str = "robodriver"
    freq: float = 125.0
    remote_wbc: bool = True

    # 要记录进数据集的部件，取值为 Astribot 对象的属性前缀。
    # 真机实测自由度：arm_left 7 / arm_right 7 / effector_left 1 / effector_right 1 /
    # head 2 / torso 4 / chassis 3（chassis 无关节限位，是平面位姿 x,y,yaw，不是关节角）
    record_parts: List[str] = field(
        default_factory=lambda: [
            "arm_left",
            "arm_right",
            "effector_left",
            "effector_right",
            "head",
            "torso",
            "chassis",
        ]
    )

    move_to_home_on_connect: bool = False
    home_duration: float = -1.0

    camera_timeout: float = 10.0

    # 数据集/平台里的图像 key（必须含 image）-> AstriBot 相机名。
    # 相机名在 connect 时与 cam.camera_names() 校验，不一致会直接报错并列出可用相机。
    # 真机实测：UDP_Camera 是头部左右目并排立体图(2x1280x720)；left/right_D405 是双腕 RealSense D405；
    # Anjia 是机器人端相机服务自定义的名字，需显式订阅才有画面（connect 里已处理）。
    camera_name_map: Dict[str, str] = field(
        default_factory=lambda: {
            "image_head": "UDP_Camera",
            "image_hand_left": "left_D405",
            "image_hand_right": "right_D405",
            "image_anjia": "Anjia",
        }
    )

    # 这里的 index_or_path 只作为 RoboXStudio 的视频流 id（daemon.cameras_info 读它），
    # 并不会真的去打开 /dev/video；实际取帧走 AstribotCamera。width/height 仅用于声明特征。
    cameras: Dict[str, CameraConfig] = field(
        default_factory=lambda: {
            "image_head": OpenCVCameraConfig(index_or_path=1, fps=30, width=2560, height=720),
            "image_hand_left": OpenCVCameraConfig(index_or_path=2, fps=30, width=640, height=360),
            "image_hand_right": OpenCVCameraConfig(index_or_path=3, fps=30, width=640, height=360),
            "image_anjia": OpenCVCameraConfig(index_or_path=4, fps=30, width=640, height=360),
        }
    )

    use_videos: bool = False
    microphones: Dict[str, int] = field(default_factory=dict)
