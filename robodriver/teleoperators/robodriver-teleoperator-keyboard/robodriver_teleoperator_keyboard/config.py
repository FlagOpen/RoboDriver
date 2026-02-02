from dataclasses import dataclass, field
from typing import Dict

from lerobot.teleoperators.config import TeleoperatorConfig
from lerobot.motors import Motor, MotorNormMode


@dataclass
class Actuator:
    id: int

@TeleoperatorConfig.register_subclass("keyboard")
@dataclass
class KeyboardTeleoperatorConfig(TeleoperatorConfig):

    actuators: Dict[str, Actuator] = field(
        default_factory=lambda: {
            "left_arm_pos_x": Actuator(1),
            "left_arm_pos_y": Actuator(2),
            "left_arm_pos_z": Actuator(3),
            "left_arm_quat_x": Actuator(4),
            "left_arm_quat_y": Actuator(5),
            "left_arm_quat_z": Actuator(6),
            "left_arm_quat_w": Actuator(7),
            "right_arm_pos_x": Actuator(8),
            "right_arm_pos_y": Actuator(9),
            "right_arm_pos_z": Actuator(10),
            "right_arm_quat_x": Actuator(11),
            "right_arm_quat_y": Actuator(12),
            "right_arm_quat_z": Actuator(13),
            "right_arm_quat_w": Actuator(14),
            "left_gripper_degree_mm": Actuator(15),
            "right_gripper_degree_mm": Actuator(16),
        }
    )