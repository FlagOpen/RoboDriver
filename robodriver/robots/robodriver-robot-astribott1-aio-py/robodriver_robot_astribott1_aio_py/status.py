import abc
import json
from dataclasses import asdict, dataclass, field
from typing import List, Optional

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
        self.number = len(self.information)


@dataclass
class ArmInfo:
    name: str = ""
    type: str = ""
    is_connect: bool = False


@dataclass
class ArmStatus:
    number: int = 0
    information: List[ArmInfo] = field(default_factory=list)

    def __post_init__(self):
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


RobotStatus.register_subclass("astribott1_aio_py")


@dataclass
class AstribotT1AioPyRobotStatus(RobotStatus):
    device_name: str = "AstriBot T1"
    device_body: str = "AstriBot"

    def __post_init__(self):
        self.specifications.end_type = "二指夹爪"
        self.specifications.fps = 30
        self.specifications.camera = CameraStatus(
            information=[
                CameraInfo(name="image_head_left",  chinese_name="头部左目",  type="双目立体相机", width=1136, height=640),
                CameraInfo(name="image_head_right", chinese_name="头部右目",  type="双目立体相机", width=1136, height=640),
                CameraInfo(name="image_hand_left",  chinese_name="左腕D405",  type="深度相机",    width=1280, height=720),
                CameraInfo(name="image_hand_right", chinese_name="右腕D405",  type="深度相机",    width=1280, height=720),
                CameraInfo(name="image_anjia",      chinese_name="Anjia相机", type="彩色相机",    width=640,  height=360),
            ]
        )
        self.specifications.arm = ArmStatus(
            information=[
                ArmInfo(name="arm_left",       type="AstriBot T1 左臂 7-DOF"),
                ArmInfo(name="arm_right",      type="AstriBot T1 右臂 7-DOF"),
                ArmInfo(name="effector_left",  type="AstriBot T1 左夹爪 1-DOF"),
                ArmInfo(name="effector_right", type="AstriBot T1 右夹爪 1-DOF"),
            ]
        )
