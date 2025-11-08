import abc
from dataclasses import dataclass, field
from typing import Sequence, Dict, List, Union, Optional

import draccus

from operating_platform.robot.robots.com_configs.cameras import CameraConfig
from operating_platform.robot.robots.com_configs.motors import MotorsBusConfig


@dataclass
class RobotConfig(draccus.ChoiceRegistry, abc.ABC):
    @property
    def type(self) -> str:
        return self.get_choice_name(self.__class__)

@dataclass
class ManipulatorRobotConfig(RobotConfig):
    leader_arms: Dict[str, MotorsBusConfig] = field(default_factory=dict)
    follower_arms: Dict[str, MotorsBusConfig] = field(default_factory=dict)
    cameras: Dict[str, CameraConfig] = field(default_factory=dict)

    max_relative_target: Optional[Union[List[float], float]] = None

    gripper_open_degree: Optional[float] = None

    mock: bool = False

    def __post_init__(self):
        if self.mock:
            for arm in self.leader_arms.values():
                if not arm.mock:
                    arm.mock = True
            for arm in self.follower_arms.values():
                if not arm.mock:
                    arm.mock = True
            for cam in self.cameras.values():
                if not cam.mock:
                    cam.mock = True

        if self.max_relative_target is not None and isinstance(self.max_relative_target, Sequence):
            for name in self.follower_arms:
                if len(self.follower_arms[name].motors) != len(self.max_relative_target):
                    raise ValueError(
                        f"len(max_relative_target)={len(self.max_relative_target)} but the follower arm with name {name} has "
                        f"{len(self.follower_arms[name].motors)} motors. Please make sure that the "
                        f"`max_relative_target` list has as many parameters as there are motors per arm. "
                        "Note: This feature does not yet work with robots where different follower arms have "
                        "different numbers of motors."
                    )

@dataclass
class DDSManipulatorRobotConfig(RobotConfig):
    leader_motors: Dict[str, MotorsBusConfig] = field(default_factory=dict)
    follower_motors: Dict[str, MotorsBusConfig] = field(default_factory=dict)
    cameras: Dict[str, CameraConfig] = field(default_factory=dict)

    max_relative_target: Optional[Union[List[float], float]] = None

    gripper_open_degree: Optional[float] = None

    mock: bool = False

    def __post_init__(self):
        if self.mock:
            for arm in self.leader_motors.values():
                if not arm.mock:
                    arm.mock = True
            for arm in self.follower_motors.values():
                if not arm.mock:
                    arm.mock = True
            for cam in self.cameras.values():
                if not cam.mock:
                    cam.mock = True

        if self.max_relative_target is not None and isinstance(self.max_relative_target, Sequence):
            for name in self.follower_motors:
                if len(self.follower_motors[name].motors) != len(self.max_relative_target):
                    raise ValueError(
                        f"len(max_relative_target)={len(self.max_relative_target)} but the follower arm with name {name} has "
                        f"{len(self.follower_motors[name].motors)} motors. Please make sure that the "
                        f"`max_relative_target` list has as many parameters as there are motors per arm. "
                        "Note: This feature does not yet work with robots where different follower arms have "
                        "different numbers of motors."
                    )
    
