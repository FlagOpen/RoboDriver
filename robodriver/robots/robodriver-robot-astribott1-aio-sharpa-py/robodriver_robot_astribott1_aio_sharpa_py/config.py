from dataclasses import dataclass, field
from typing import List

from lerobot.robots.config import RobotConfig

from robodriver_robot_astribott1_aio_py.config import AstribotT1AioPyRobotConfig

from .sharpa_hand import SharpaHandsConfig


@RobotConfig.register_subclass("astribott1_aio_sharpa_py")
@dataclass
class AstribotT1AioSharpaPyRobotConfig(AstribotT1AioPyRobotConfig):
    """AstriBot T1 + Sharpa hands; AstriBot's original grippers are excluded."""

    record_parts: List[str] = field(
        default_factory=lambda: ["arm_left", "arm_right", "head", "torso", "chassis"]
    )
    control_parts: List[str] = field(default_factory=lambda: ["arm_left", "arm_right"])
    sharpa_hands: SharpaHandsConfig = field(default_factory=SharpaHandsConfig)

    def __post_init__(self) -> None:
        super().__post_init__()
        excluded = {"effector_left", "effector_right"}
        if excluded.intersection(self.record_parts) or excluded.intersection(self.control_parts):
            raise ValueError("Sharpa 模式不采集或控制 AstriBot 原夹爪 effector_left/right")