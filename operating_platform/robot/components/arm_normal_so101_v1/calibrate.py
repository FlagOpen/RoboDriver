"""TODO: Add docstring."""

import os
import time

import numpy as np
import pyarrow as pa
import draccus
from dora import Node
from pathlib import Path
import threading
import queue


from motors.feetech import FeetechMotorsBus, OperatingMode
from motors import Motor, MotorCalibration, MotorNormMode


GET_DEVICE_FROM = os.getenv("GET_DEVICE_FROM", "PORT") # SN or INDEX
PORT = os.getenv("PORT")
ARM_NAME = os.getenv("ARM_NAME", "SO101-Arm")
CALIBRATION_DIR = os.getenv("CALIBRATION_DIR", "./.calibration/")
USE_DEGRESS = os.getenv("USE_DEGRESS", "True")
ARM_ROLE = os.getenv("ARM_ROLE", "follower")

result_queue = queue.Queue()
stop_event = threading.Event()


def env_to_bool(env_value: str, default: bool = True) -> bool:
    """将环境变量字符串转换为布尔值"""
    if env_value is None:
        return default
    
    true_values = {'True', 'true', '1', 'yes', 'on', 't', 'y'}
    false_values = {'False', 'false', '0', 'no', 'off', 'f', 'n'}
    
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

def init_calibrate(name, bus: FeetechMotorsBus) -> None:
    # if self.calibration:
    #     # self.calibration is not empty here
    #     user_input = input(
    #         f"Press ENTER to use provided calibration file associated with the id {self.id}, or type 'c' and press ENTER to run calibration: "
    #     )
    #     if user_input.strip().lower() != "c":
    #         logger.info(f"Writing calibration file associated with the id {self.id} to the motors")
    #         self.bus.write_calibration(self.calibration)
    #         return

    print(f"\nRunning calibration of {name}")
    bus.disable_torque()
    for motor in bus.motors:
        bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)


def save_calibration(calibration: dict, fpath: Path | None = None) -> None:
    """
    Helper to save calibration data to the specified file.

    Args:
        fpath (Path | None): Optional path to save the calibration file. Defaults to `self.calibration_fpath`.
    """
    with open(fpath, "w") as f, draccus.config_type("json"):
        draccus.dump(calibration, f, indent=4)

def record_ranges(bus: FeetechMotorsBus, stop_event):
    try:
        range_mins, range_maxes = bus.record_ranges_of_motion(stop_event=stop_event)
        result_queue.put((range_mins, range_maxes))
    except Exception as e:
        result_queue.put(e)

def main():
    node = Node()

    use_degrees = env_to_bool(USE_DEGRESS)
    calibration_dir = Path(CALIBRATION_DIR)
    calibration_fpath = calibration_dir / f"{ARM_NAME}.json"
    name = ARM_NAME

    calibration_dir.mkdir(parents=True, exist_ok=True)

    # if calibration_fpath.is_file():
    #     self._load_calibration()
    # try:
    #     with open(calibrate_fpath) as f, draccus.config_type("json"):
    #         arm_calibration = draccus.load(dict[str, MotorCalibration], f)
    # except FileNotFoundError:
    #     raise FileNotFoundError(f"校准文件路径不存在: {fpath}")
    # except IsADirectoryError:
    #     raise ValueError(f"路径是目录而不是文件: {fpath}")

    norm_mode_body = MotorNormMode.DEGREES if use_degrees else MotorNormMode.RANGE_M100_100

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
    )

    arm_bus.connect()

    # if ARM_ROLE == "follower":
    #     configure_follower(arm_bus)
    # elif ARM_ROLE == "leader":
    #     configure_leader(arm_bus)

    init_calibrate(name, arm_bus)
    print(f"Move {name} to the middle of its range of motion and press key 'm'....")
    stop_event = threading.Event()
    # 创建并启动线程
    recording_thread = threading.Thread(
        target=record_ranges, 
        args=(arm_bus, stop_event,)
    )
    # homing_offsets = []

    for event in node:
        if event["type"] == "INPUT":
            if event["id"] == "key":
                key_chars = event["value"].to_pylist()
                key = key_chars[0]

                print(f"Received key: {key}")

                if(key == 'm' or key == 'M'):
                    homing_offsets = arm_bus.set_half_turn_homings()
                    recording_thread.start()
                    print("开始记录运动范围...")
                    print(
                        "Move all joints sequentially through their entire ranges "
                        "of motion.\nRecording positions. Press key 'e' to stop..."
                    )

                
                if(key == 'e' or key == 'E'): 
                    # range_mins, range_maxes = arm_bus.record_ranges_of_motion(stop_event=)
                    stop_event.set()
                    recording_thread.join()

                    result = result_queue.get_nowait()
                    range_mins, range_maxes = result

                    calibration = {}
                    for motor, m in arm_bus.motors.items():
                        calibration[motor] = MotorCalibration(
                            id=m.id,
                            drive_mode=0,
                            homing_offset=homing_offsets[motor],
                            range_min=range_mins[motor],
                            range_max=range_maxes[motor],
                        )

                    arm_bus.write_calibration(calibration)
                    save_calibration(calibration, calibration_fpath)
                    print("Calibration saved to", calibration_fpath)

                    print("""Calibrate Finish, Press "CTRL + C" to stop Dora dataflow""")


        elif event["type"] == "STOP":
            break


if __name__ == "__main__":
    main()
