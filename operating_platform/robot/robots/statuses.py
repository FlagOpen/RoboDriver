from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict
import json
import requests
import threading
import time
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


# @dataclass
# class MachineInformation:

#     specifications: Specifications = field(default_factory=Specifications)
 
#     def to_dict(self) -> dict:
#         return asdict(self)
 
#     def to_json(self, indent: int = 2) -> str:
#         return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)
 
#     @classmethod
#     def from_dict(cls, data: dict) -> 'MachineInformation':
#         # 处理 specifications 字段
#         specs_data = data.get("specifications", {})
        
#         # 处理 camera 字段（可能是 dict 或 list）
#         if "camera" in specs_data:
#             if specs_data["camera"] is None:
#                 pass  # 保持 None
#             elif isinstance(specs_data["camera"], dict):
#                 if "information" in specs_data["camera"]:
#                     info_list = specs_data["camera"]["information"]
#                     if isinstance(info_list, list):
#                         specs_data["camera"]["information"] = [CameraInfo(**cam) for cam in info_list]
#                 specs_data["camera"] = CameraConfig(**specs_data["camera"])
#             elif isinstance(specs_data["camera"], list):
#                 specs_data["camera"] = CameraConfig(information=[CameraInfo(**cam) for cam in specs_data["camera"]])
        
#         # 处理 piper 字段（可能是 dict 或 list）
#         if "piper" in specs_data:
#             if specs_data["piper"] is None:
#                 pass  # 保持 None
#             elif isinstance(specs_data["piper"], dict):
#                 if "information" in specs_data["piper"]:
#                     info_list = specs_data["piper"]["information"]
#                     if isinstance(info_list, list):
#                         specs_data["piper"]["information"] = [PiperInfo(**pipe) for pipe in info_list]
#                 specs_data["piper"] = PiperConfig(**specs_data["piper"])
#             elif isinstance(specs_data["piper"], list):
#                 specs_data["piper"] = PiperConfig(information=[PiperInfo(**pipe) for pipe in specs_data["piper"]])
        
#         # 更新 specifications
#         if specs_data:
#             data["specifications"] = Specifications(**specs_data)
        
#         return cls(**data)
    