"""TODO: Add docstring."""

import os

import pyarrow as pa
import draccus
from dora import Node
from pathlib import Path

from motors.feetech import FeetechMotorsBus, OperatingMode
from motors import Motor, MotorCalibration, MotorNormMode


GET_DEVICE_FROM = os.getenv("GET_DEVICE_FROM", "PORT")  # SN or INDEX
PORT = os.getenv("PORT")
ARM_NAME = os.getenv("ARM_NAME", "SO101-Arm")
CALIBRATION_DIR = os.getenv("CALIBRATION_DIR", "./.calibration/")
USE_DEGRESS = os.getenv("USE_DEGRESS", "True")
ARM_ROLE = os.getenv("ARM_ROLE", "follower")


def env_to_bool(env_value: str, default: bool = True) -> bool:
    """将环境变量字符串转换为布尔值"""
    if env_value is None:
        return default

    true_values = {"True", "true", "1", "yes", "on", "t", "y"}
    false_values = {"False", "false", "0", "no", "off", "f", "n"}

    value_lower = env_value.strip().lower()

    if value_lower in true_values:
        return True
    elif value_lower in false_values:
        return False
    else:
        raise ValueError(f"无效的布尔值: {env_value}")


def configure_follower(bus: FeetechMotorsBus) -> None:
    with bus.torque_disabled():
        bus.configure_motors()
        for motor in bus.motors:
            bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)
            # Set P_Coefficient to lower value to avoid shakiness (Default is 32)
            bus.write("P_Coefficient", motor, 16)
            # Set I_Coefficient and D_Coefficient to default value 0 and 32
            bus.write("I_Coefficient", motor, 0)
            bus.write("D_Coefficient", motor, 32)


def configure_leader(bus: FeetechMotorsBus) -> None:
    bus.disable_torque()
    bus.configure_motors()
    for motor in bus.motors:
        bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)


def main():
    node = Node()

    use_degrees = env_to_bool(USE_DEGRESS)
    calibration_dir = Path(CALIBRATION_DIR).resolve()
    calibration_fpath = calibration_dir / f"{ARM_NAME}.json"
    name = ARM_NAME

    try:
        with open(calibration_fpath) as f, draccus.config_type("json"):
            arm_calibration = draccus.load(dict[str, MotorCalibration], f)
    except FileNotFoundError:
        raise FileNotFoundError(f"校准文件路径不存在: {calibration_fpath}")
    except IsADirectoryError:
        raise ValueError(f"路径是目录而不是文件: {calibration_fpath}")

    norm_mode_body = (
        MotorNormMode.DEGREES if use_degrees else MotorNormMode.RANGE_M100_100
    )

    arm_bus = FeetechMotorsBus(
        port=PORT,
        motors={
            "shoulder_pan": Motor(1, "sts3215", norm_mode_body),
            "shoulder_lift": Motor(2, "sts3215", norm_mode_body),
            "elbow_flex": Motor(3, "sts3215", norm_mode_body),
            "wrist_flex": Motor(4, "sts3215", norm_mode_body),
            "wrist_roll": Motor(5, "sts3215", norm_mode_body),
            "gripper": Motor(6, "sts3215", MotorNormMode.RANGE_0_100),
        },
        calibration=arm_calibration,
    )

    arm_bus.connect()

    if ARM_ROLE == "follower":
        configure_follower(arm_bus)
    elif ARM_ROLE == "leader":
        configure_leader(arm_bus)

    ctrl_frame = 0

    for event in node:
        if event["type"] == "INPUT":
            if "action" in event["id"]:
                pass

            if event["id"] == "action_joint":
                position = event["value"].to_numpy()

                if ctrl_frame > 0:
                    continue

                goal_pos = {
                    key: position[motor.id - 1] for key, motor in arm_bus.motors.items()
                }
                arm_bus.sync_write("Goal_Position", goal_pos)

            if event["id"] == "action_joint_ctrl":
                position = event["value"].to_numpy()

                ctrl_frame = 200

                goal_pos = {
                    key: position[motor.id - 1] for key, motor in arm_bus.motors.items()
                }
                arm_bus.sync_write("Goal_Position", goal_pos)

            elif event["id"] == "get_joint":
                joint_value = []
                present_pos = arm_bus.sync_read("Present_Position")

                for motor, val in present_pos.items():
                    node.send_output(
                        f"joint_{motor}", pa.array([val], type=pa.float32())
                    )
                    joint_value.append(val)

                # joint_value = [val for _motor, val in present_pos.items()]
                metadata = {
                    "len": len(arm_bus.motors),
                    "names": list(arm_bus.motors.keys()),
                }

                node.send_output(
                    "all_joint", pa.array(joint_value, type=pa.float32()), metadata
                )

            ctrl_frame -= 1

        elif event["type"] == "STOP":
            break


if __name__ == "__main__":
    main()
