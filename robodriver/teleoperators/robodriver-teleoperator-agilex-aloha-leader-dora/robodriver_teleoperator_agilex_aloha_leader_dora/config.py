from typing import Dict
from dataclasses import dataclass, field

from lerobot.teleoperators.config import TeleoperatorConfig
from lerobot.motors import Motor, MotorNormMode


@TeleoperatorConfig.register_subclass("agilex_aloha_leader_dora")
@dataclass
class AgilexAlohaLeaderDoraTeleoperatorConfig(TeleoperatorConfig):
    use_degrees = True
    norm_mode_body = (
        MotorNormMode.DEGREES if use_degrees else MotorNormMode.RANGE_M100_100
    )

    leader_motors: Dict[str, Motor] = field(
        default_factory=lambda norm_mode_body=norm_mode_body: {
            "joint1_right": Motor(1, "piper-motor", norm_mode_body),
            "joint2_right": Motor(2, "piper-motor", norm_mode_body),
            "joint3_right": Motor(3, "piper-motor", norm_mode_body),
            "joint4_right": Motor(4, "piper-motor", norm_mode_body),
            "joint5_right": Motor(5, "piper-motor", norm_mode_body),
            "joint6_right": Motor(6, "piper-motor", norm_mode_body),
            "gripper_right": Motor(7, "piper-gripper", MotorNormMode.RANGE_0_100),

            "joint1_left": Motor(8, "piper-motor", norm_mode_body),
            "joint2_left": Motor(9, "piper-motor", norm_mode_body),
            "joint3_left": Motor(10, "piper-motor", norm_mode_body),
            "joint4_left": Motor(11, "piper-motor", norm_mode_body),
            "joint5_left": Motor(12, "piper-motor", norm_mode_body),
            "joint6_left": Motor(13, "piper-motor", norm_mode_body),
            "gripper_left": Motor(14, "piper-gripper", MotorNormMode.RANGE_0_100),
        }
    )
