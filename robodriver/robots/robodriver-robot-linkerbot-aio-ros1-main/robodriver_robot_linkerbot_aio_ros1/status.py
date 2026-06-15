from dataclasses import dataclass, field, asdict
from typing import List, Optional
import json
import abc
import draccus
import numpy as np

@dataclass
class CameraInfo:
    name: str = ""
    chinese_name: str = ""
    type: str = ""
    width: int = 0
    height: int = 0
    is_connect: bool = False

@dataclass
class CameraStatus:
    number: int = 0
    information: List[CameraInfo] = field(default_factory=list)
    
    def __post_init__(self):
        if not self.information:  # 如果 information 为空，则 number 设为 0
            self.number = 0
        else:
            self.number = len(self.information)
  
@dataclass
class ArmInfo:
    name: str = ""
    type: str = ""
    start_pose: List[float] = field(default_factory=list)
    joint_p_limit: List[float] = field(default_factory=list)
    joint_n_limit: List[float] = field(default_factory=list)
    is_connect: bool = False
    invert_direction: bool = False  # 新增：是否反转运动方向
    is_radian: bool = False         # 新增：底层数据是否已经是弧度
 
@dataclass
class ArmStatus:
    number: int = 0
    information: List[ArmInfo] = field(default_factory=list)
    
    def __post_init__(self):
        if not self.information:  # 如果 information 为空，则 number 设为 0
            self.number = 0
        else:
            self.number = len(self.information)
 
@dataclass
class HandInfo:
    name: str = ""
    type: str = ""
    joint_p_limit: List[float] = field(default_factory=list)
    joint_n_limit: List[float] = field(default_factory=list)
    is_connect: bool = False
    invert_direction: bool = False
    
@dataclass
class HandStatus:
    number: int = 0
    information: List[HandInfo] = field(default_factory=list)
    
    def __post_init__(self):
        if not self.information:
            self.number = 0
        else:
            self.number = len(self.information)

@dataclass
class Specifications:
    end_type: str = "Default"
    fps: int = 30
    camera: Optional[CameraStatus] = None
    arm: Optional[ArmStatus] = None
    hand: Optional[HandStatus] = None

@dataclass
class RobotStatus(draccus.ChoiceRegistry, abc.ABC):
    device_name: str = "Default"
    device_body: str = "Default"
    specifications: Specifications = field(default_factory=Specifications)

    @property
    def type(self) -> str:
        return self.get_choice_name(self.__class__)
    
    def to_dict(self) -> dict:
        return asdict(self)
 
    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)


@RobotStatus.register_subclass("linkerbot_aio_ros1")
@dataclass
class LinkerBotAioRos1RobotStatus(RobotStatus):
    device_name: str = "linkerbot"
    device_body: str = "linkerbot"

    def __post_init__(self):
        self.specifications.end_type = "5指灵巧手"
        self.specifications.fps = 30
        self.specifications.camera = CameraStatus(
            information=[
                CameraInfo(
                    name="image_head",
                    chinese_name="腕部摄像头",
                    type="单目视觉相机",
                    width=640,
                    height=480,
                    is_connect=False
                ),

            ]
        )

        self.specifications.arm = ArmStatus(
            information=[

                ArmInfo(
                    name="follower_arm",
                    type="linkerbot 从臂 7DOF",
                    start_pose=[],
                    joint_p_limit = [np.pi] * 7,  # 填入从臂真实的物理限位
                    joint_n_limit = [-np.pi] * 7,
                    is_connect=False,
                    invert_direction=False, # 从臂不反转
                    is_radian=True          # 从臂是弧度
                ),
                ArmInfo(
                    name="leader_arm",
                    type="主控臂 7DOF",
                    start_pose=[],
                    # 填入主臂真实的物理限位
                    joint_p_limit = [180.0]*7, 
                    joint_n_limit = [-180.0]*7,
                    is_connect=False,
                    invert_direction=True,  # 主臂动作相反，需要反转
                    is_radian=False         # 主臂是角度
                ),

            ]
        )
        self.specifications.hand = HandStatus(
            information=[
                HandInfo(
                    name="follower_hand",
                    type="从端灵巧手 10DOF",
                    # 从手真实限位：0 到 255
                    joint_p_limit = [255.0] * 10,
                    joint_n_limit = [0.0] * 10,
                    is_connect=False,
                    invert_direction=False
                ),
                HandInfo(
                    name="leader_hand",
                    type="遥操作主手 15DOF",
                    # 主手真实限位：-65535 到 65535
                    joint_p_limit = [65535.0] * 15,
                    joint_n_limit = [-65535.0] * 15,
                    is_connect=False,
                    invert_direction=False
                ),
            ]
        )