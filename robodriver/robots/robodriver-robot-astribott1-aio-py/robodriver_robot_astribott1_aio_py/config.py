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

    # ==== 外部控制（回放 / disturbance）的控制权策略 ====
    # passive  : 不主动抢控制权。若 VR 遥操作等持有租约，指令会被机器人忽略，只打印警告
    # takeover : 下发动作前 acquire_control，结束后 release_control（默认）
    #            仅在真正 send_action 时生效；正常采集只读，不会打扰遥操作
    control_mode: str = "takeover"
    # 实测 VR 遥操作持有 priority_level=0（最高），要抢占必须同为 0
    takeover_priority: int = 0
    takeover_sub_priority: int = 0
    # 只有这些部件会被下发动作，防止回放时底盘开走；不在列表里的部件只记录不控制
    control_parts: List[str] = field(
        default_factory=lambda: ["arm_left", "arm_right", "effector_left", "effector_right"]
    )
    # 目标部件仍被「非空闲」租约占用时是否强行抢占（危险，需人在现场确认）
    force_takeover: bool = False

    camera_timeout: float = 10.0

    # 数据集/平台里的图像 key（必须含 image）-> AstriBot 相机名。
    # 相机名在 connect 时与 cam.camera_names() 校验，不一致会直接报错并列出可用相机。
    # 真机实测：UDP_Camera 是头部左右目并排立体图(2x1280x720)，Web 小窗是 16:9，
    # 整张 32:9 放进去会被裁，所以拆成 image_head_left / image_head_right 两路；
    # left/right_D405 是双腕 RealSense D405；Anjia 是机器人端相机服务自定义的名字。
    camera_name_map: Dict[str, str] = field(
        default_factory=lambda: {
            "image_head_left": "UDP_Camera",
            "image_head_right": "UDP_Camera",
            "image_hand_left": "left_D405",
            "image_hand_right": "right_D405",
            "image_anjia": "Anjia",
        }
    )

    # 从原始帧裁出子图：(x, y, w, h)。头部双目按半宽切左右两目。
    # 尺寸取决于板子 multi_camera_config.json 里 UDP_Camera 的 resolution：
    #   res=1280x720 → 输出 2560x720，裁 (0,0,1280,720)/(1280,0,1280,720)
    #   res=1136x640 → 输出 2272x640，裁 (0,0,1136,640)/(1136,0,1136,640)  ← 当前
    # 改板子配置后必须同步这里，否则 get_observation() 会抛裁剪越界。
    camera_crops: Dict[str, tuple] = field(
        default_factory=lambda: {
            "image_head_left": (0, 0, 1136, 640),
            "image_head_right": (1136, 0, 1136, 640),
        }
    )

    # 这里的 index_or_path 只作为 RoboXStudio 的视频流 id（daemon.cameras_info 读它），
    # 并不会真的去打开 /dev/video；实际取帧走 AstribotCamera。width/height 仅用于声明特征。
    # width/height 必须与 camera_crops 裁出的尺寸一致（腕部为 D405 原始分辨率）。
    cameras: Dict[str, CameraConfig] = field(
        default_factory=lambda: {
            "image_head_left": OpenCVCameraConfig(index_or_path=1, fps=30, width=1136, height=640),
            "image_head_right": OpenCVCameraConfig(index_or_path=2, fps=30, width=1136, height=640),
            "image_hand_left": OpenCVCameraConfig(index_or_path=3, fps=30, width=1280, height=720),
            "image_hand_right": OpenCVCameraConfig(index_or_path=4, fps=30, width=1280, height=720),
            "image_anjia": OpenCVCameraConfig(index_or_path=5, fps=30, width=640, height=360),
        }
    )

    # ==== 关节数据源：ROS2 话题（可选，默认关闭，仍用 SDK 读关节）====
    # 打开后 get_observation/get_action 的关节部分改为订阅 ROS 话题，相机仍走 SDK。
    # 实测需要 ROS_DOMAIN_ID=25 才能发现话题；模块会在 rclpy.init() 前设置该值。
    use_ros_joints: bool = False
    ros_domain_id: int = 25
    ros_node_name: str = "robodriver_astribot_joints"
    # 话题前缀：/astribot_<part>/<suffix>
    ros_topic_prefix: str = "astribot"
    # 观测=当前关节角(RobotJointState.position, BEST_EFFORT)
    # 动作=关节指令(RobotJointController.command, RELIABLE)。机器人静止时该话题不发布，
    # 此时 get_action 回退到 obs（AIO 镜像 desired≈current）
    ros_obs_suffix: str = "joint_space_states"
    ros_action_suffix: str = "joint_space_command_forward"
    # 连接时等待首帧 obs 的超时（秒）；action 静止时无数据，不纳入硬性等待
    ros_wait_first_frame_s: float = 10.0

    use_videos: bool = False
    microphones: Dict[str, int] = field(default_factory=dict)
