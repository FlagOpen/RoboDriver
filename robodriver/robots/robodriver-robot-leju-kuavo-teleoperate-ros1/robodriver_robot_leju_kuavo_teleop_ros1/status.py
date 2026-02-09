from dataclasses import dataclass, field, asdict
from typing import List, Optional
import json
import abc
import draccus


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
class Specifications:
    end_type: str = "Default"
    fps: int = 30
    camera: Optional[CameraStatus] = None
    arm: Optional[ArmStatus] = None

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

RobotStatus.register_subclass("leju-kuavo-teleop-ros1")
@dataclass
class LEJUKuavoRos1RobotStatus(RobotStatus):
    device_name: str = "乐聚kuavoPro4"
    device_body: str = "乐聚"

    def __post_init__(self):
        self.specifications.end_type = "灵巧手"
        self.specifications.fps = 30
        self.specifications.camera = CameraStatus(
            information=[
                CameraInfo(
                    name="image_top",
                    chinese_name="头部摄像头",
                    type="纯双目视觉相机",
                    width=640,
                    height=360,
                    is_connect=False
                ),
                CameraInfo(
                    name="image_wrist_left",
                    chinese_name="腕部左摄像头",
                    type="单目深度相机",
                    width=640,
                    height=360,
                    is_connect=False
                ),
                CameraInfo(
                    name="image_wrist_right",
                    chinese_name="腕部右摄像头",
                    type="单目深度相机",
                    width=640,
                    height=360,
                    is_connect=False
                )
            ]
        )

        self.specifications.arm = ArmStatus(
            information=[
                ArmInfo(
                    name="kuavo_left",
                    type="7DOF",
                    start_pose=[],
                    joint_p_limit=[165.0,180.0,0.0,90.0,90.0,165.0],
                    joint_n_limit=[-165.0,0.0,-190.0,-90.0,-90.0,-165.0],
                    is_connect=False
                    ),
                ArmInfo(
                    name="kuavo_right",
                    type="7DOF",
                    start_pose=[],
                    joint_p_limit=[165.0,180.0,0.0,90.0,90.0,165.0],
                    joint_n_limit=[-165.0,0.0,-190.0,-90.0,-90.0,-165.0],
                    is_connect=False
                ),
            ]
        )
