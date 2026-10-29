from dataclasses import dataclass, field
from typing import Dict

from lerobot.teleoperators.config import TeleoperatorConfig
from lerobot.motors import Motor, MotorNormMode


@dataclass
class Actuator:
    id: int

@TeleoperatorConfig.register_subclass("autotask_ros2")
@dataclass
class AutoTaskTeleoperatorConfig(TeleoperatorConfig):

    actuators: Dict[str, Actuator] = field(
        default_factory=lambda: {
            "left_arm_pos_x_m": Actuator(1),
            "left_arm_pos_y_m": Actuator(2),
            "left_arm_pos_z_m": Actuator(3),
            "left_arm_quat_w": Actuator(4),
            "left_arm_quat_x": Actuator(5),
            "left_arm_quat_y": Actuator(6),
            "left_arm_quat_z": Actuator(7),

            "right_arm_pos_x_m": Actuator(8),
            "right_arm_pos_y_m": Actuator(9),
            "right_arm_pos_z_m": Actuator(10),
            "right_arm_quat_w": Actuator(11),
            "right_arm_quat_x": Actuator(12),
            "right_arm_quat_y": Actuator(13),
            "right_arm_quat_z": Actuator(14),

            "left_gripper_percent": Actuator(15),
            "right_gripper_percent": Actuator(16),
        }
    )