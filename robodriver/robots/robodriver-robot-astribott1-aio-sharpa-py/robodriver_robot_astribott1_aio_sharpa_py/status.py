from dataclasses import dataclass

from robodriver_robot_astribott1_aio_py.status import AstribotT1AioPyRobotStatus, RobotStatus


@RobotStatus.register_subclass("astribott1_aio_sharpa_py")
@dataclass
class AstribotT1AioSharpaPyRobotStatus(AstribotT1AioPyRobotStatus):
    device_name: str = "AstriBot T1 + Sharpa Wave"

    def __post_init__(self) -> None:
        super().__post_init__()
        self.specifications.end_type = "Sharpa Wave 灵巧手"
        self.specifications.arm.information = [
            arm for arm in self.specifications.arm.information
            if arm.name not in ("effector_left", "effector_right")
        ]
        self.specifications.arm.number = len(self.specifications.arm.information)