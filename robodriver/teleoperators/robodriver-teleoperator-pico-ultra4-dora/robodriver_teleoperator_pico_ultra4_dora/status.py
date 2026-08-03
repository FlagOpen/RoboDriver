from dataclasses import dataclass, field, asdict
from typing import List, Optional
import json
import abc
import draccus


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
        if not self.information:
            self.number = 0
        else:
            self.number = len(self.information)


@dataclass
class Specifications:
    end_type: str = "Default"
    fps: int = 30
    arm: Optional[ArmStatus] = None


@dataclass
class TeleoperatorStatus(draccus.ChoiceRegistry, abc.ABC):
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


@TeleoperatorStatus.register_subclass("pico_ultra4_dora")
@dataclass
class PicoUltra4DoraTeleoperatorStatus(TeleoperatorStatus):
    device_name: str = "Pico_Ultra4"
    device_body: str = "Pico_VR"

    def __post_init__(self):
        self.specifications.end_type = "VR 控制器"
        self.specifications.fps = 50

        self.specifications.arm = ArmStatus(
            information=[
                ArmInfo(
                    name="pico_vr_controller",
                    type="Pico Ultra4 VR 控制器",
                    start_pose=[],
                    joint_p_limit=[],
                    joint_n_limit=[],
                    is_connect=False
                ),
            ]
        )
