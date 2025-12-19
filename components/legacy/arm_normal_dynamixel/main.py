import os
from functools import cache

import pyarrow as pa
from dora import Node

from .dynamixel import *

HORIZONTAL_POSITION = np.array([0, 0, 0, 0, 0, 0, 0, 0])
TARGET_90_DEGREE_POSITION = np.array([0, 0, 0, 0, 0, 0, 0, 0])
DEFAULT_DRIVE_MODE = np.array([False, False, False, False, False, False, False, False])
GRIPPER_OPEN = np.array([600])
SCALE_FACTOR = 90 / 1024  # 电机值到关节角度的比例系数
GRIPPER_SCALE = 100 / 600  # 夹爪转换比例系数


id = os.getenv(
    "ARM_ID", "arm_right"
)  # ARM_ID: Arm identifier; defaults to "arm_right" if not set
port = os.getenv(
    "ARM_PORT", "/dev/ttyUSB0"
)  # ARM_PORT: Connection USB port; defaults to /dev/ttyUSB0 if not set
ctrl_key = os.getenv(
    "CTRL_KEY", "d"
)  # CTRL_KEY: The key that controls the change of the robotic arm's functions.

start_pose_str = os.getenv("START_POSE", "0.0, 0.0, 0.0, 0.0, 0.0, 0.0")
joint_p_limit_str = os.getenv("JOINT_P_LIMIT", "0.0, 0.0, 0.0, 0.0, 0.0, 0.0")
joint_n_limit_str = os.getenv("JOINT_N_LIMIT", "0.0, 0.0, 0.0, 0.0, 0.0, 0.0")
start_pose = [float(x) for x in start_pose_str.split(",")]
joint_p_limit = [float(x) for x in joint_p_limit_str.split(",")]
joint_n_limit = [float(x) for x in joint_n_limit_str.split(",")]

node = Node()


@cache
def is_headless():
    """Detects if python is running without a monitor."""
    try:
        import pynput  # noqa

        return False
    except Exception:
        print(
            "Error trying to import pynput. Switching to headless mode. "
            "As a result, the video stream from the cameras won't be shown, "
            "and you won't be able to change the control flow with keyboards. "
            "For more info, see traceback below.\n"
        )
        traceback.print_exc()
        print()
        return True


def init_keyboard_listener():
    # Allow to exit early while recording an episode or resetting the environment,
    # by tapping the right arrow key '->'. This might require a sudo permission
    # to allow your terminal to monitor keyboard events.
    events = {}
    events["complete"] = False

    if is_headless():
        logging.warning(
            "Headless environment detected. On-screen cameras display and keyboard inputs will not be available."
        )
        listener = None
        return listener, events

    # Only import pynput if not in a headless environment
    from pynput import keyboard

    def on_press(key):
        try:
            hasattr(key, "char")
            if key.char.lower() == ctrl_key:
                print("key pressed")
                events["complete"] = True
        except Exception as e:
            print(f"Error handling key press: {e}")

    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    return listener, events


def apply_homing_offset(values: np.array, homing_offset: np.array) -> np.array:
    for i in range(len(values)):
        if values[i] is not None:
            values[i] += homing_offset[i]
    return values


def apply_drive_mode(values: np.array, drive_mode: np.array) -> np.array:
    for i in range(len(values)):
        if values[i] is not None and drive_mode[i]:
            values[i] = -values[i]
    return values


def apply_offset_and_drivemode(
    values: np.array, homing_offset: np.array, drive_mode: np.array
) -> np.array:
    values = apply_drive_mode(values, drive_mode)
    values = apply_homing_offset(values, homing_offset)
    return values


def revert_appropriate_positions(
    positions: np.array, drive_mode: list[bool]
) -> np.array:
    for i, revert in enumerate(drive_mode):
        if not revert and positions[i] is not None:
            positions[i] = -positions[i]
    return positions


def compute_corrections(
    positions: np.array, drive_mode: list[bool], target_position: np.array
) -> np.array:
    correction = revert_appropriate_positions(positions, drive_mode)

    for i in range(len(positions)):
        if correction[i] is not None:
            if drive_mode[i]:
                correction[i] -= target_position[i]
            else:
                correction[i] += target_position[i]

    return correction


def compute_nearest_rounded_positions(positions: np.array) -> np.array:
    return np.array(
        [
            round(positions[i] / 1024) * 1024 if positions[i] is not None else None
            for i in range(len(positions))
        ]
    )


class DynamixelArm:
    def __init__(self):
        arm_config = DynamixelMotorsBusConfig(
            port=port,
            motors={
                "joint_1": [1, "xl330-m288"],
                "joint_2": [2, "xl330-m288"],
                "joint_3": [3, "xl330-m288"],
                "joint_4": [4, "xl330-m288"],
                "joint_5": [5, "xl330-m288"],
                "joint_6": [6, "xl330-m288"],
                "joint_7": [7, "xl330-m288"],
                "gripper": [8, "xl330-m288"],
            },
        )
        self.arm = DynamixelMotorsBus(arm_config)
        self.arm.connect()
        self.is_connected = True

        self.init_arm()
        self.run_calibration()

        self.logs = {}

    def init_arm(self):
        self.arm.write("Torque_Enable", TorqueMode.ENABLED.value)
        self.arm.write(
            "Operating_Mode", OperatingMode.CURRENT_CONTROLLED_POSITION.value
        )
        self.arm.write("Goal_Current", 300)

        self.arm.write("Homing_Offset", 0)
        self.arm.write("Drive_Mode", DriveMode.NON_INVERTED.value)

        print("init_arm: OK!")

    def reset_arm(self):
        self.arm.write("Torque_Enable", TorqueMode.DISABLED.value)

        all_motors_except_gripper = [
            name for name in self.arm.motor_names if name != "gripper"
        ]
        self.arm.write(
            "Operating_Mode",
            OperatingMode.EXTENDED_POSITION.value,
            all_motors_except_gripper,
        )

        self.arm.write(
            "Operating_Mode", OperatingMode.CURRENT_CONTROLLED_POSITION.value, "gripper"
        )
        self.arm.write("Goal_Current", 50, "gripper")

        self.arm.write("Torque_Enable", 1, "gripper")
        self.arm.write("Goal_Position", GRIPPER_OPEN, "gripper")

        print("reset_arm: OK!")

    def run_calibration(self):
        print(f"Calibrate {id}.")
        listener, events = init_keyboard_listener()

        # Move to horizontal
        print(f"Please move the {id} to the horizontal position.)")

        while True:
            current = self.arm.read("Present_Current")
            pos = self.arm.read("Present_Position")
            print(f"current = {current}")
            print(f"pos = {pos}")
            for index, value in enumerate(current, start=0):
                if value >> 15 == 1:
                    buma = value  # 补码（这里实际上只是获取 value[0] 的值）
                    fanma = buma - 1  # 反码（这里实际上是 value[0] 减 1）
                    yuanma = int(
                        format(~fanma & 0xFFFF, "016b"), 2
                    )  # 原码（这里是对 fanma 进行按位取反操作）
                    p_current = yuanma * -1
                else:
                    p_current = float(value)
                print(f"index = {index}, value = {value}, p_current = {p_current}")
                print(f"index = {index}, current[{index}] = {current[index]}")
                print(f"index = {index}, pos[{index}] = {pos[index]}")
                print(
                    f"index = {index}, arm.motor_names[{index}] = {self.arm.motor_names[index]}"
                )
                if abs(p_current) >= 150:
                    self.arm.write(
                        "Goal_Position", pos[index], self.arm.motor_names[index]
                    )
            if events["complete"]:
                old_gripper = self.arm.read("Present_Position", "gripper")
                print(f"Will complete {id} calibration")
                while True:
                    gripper = self.arm.read("Present_Position", "gripper")
                    print(f"gripper = {gripper}")
                    if abs(gripper - old_gripper) >= 150:
                        break
                break

        # Get true horizontal_homing_offset from horizontal
        present_positions = apply_offset_and_drivemode(
            self.arm.read("Present_Position"), HORIZONTAL_POSITION, DEFAULT_DRIVE_MODE
        )
        nearest_positions = compute_nearest_rounded_positions(present_positions)
        horizontal_homing_offset = compute_corrections(
            nearest_positions, DEFAULT_DRIVE_MODE, HORIZONTAL_POSITION
        )

        # Move to 90°
        print(f"Please move the {id} to the horizontal position.)")

        # while True:
        #     current = self.arm.read("Present_Current")
        #     pos = self.arm.read("Present_Position")
        #     print(f"current = {current}")
        #     print(f"pos = {pos}")
        #     for index, value in enumerate(current, start=0):
        #         if value >> 15 == 1:
        #             buma = value  # 补码（这里实际上只是获取 value[0] 的值）
        #             fanma = buma - 1  # 反码（这里实际上是 value[0] 减 1）
        #             yuanma = int(format(~fanma & 0xFFFF, '016b'), 2)  # 原码（这里是对 fanma 进行按位取反操作）
        #             p_current = yuanma * -1
        #         else:
        #             p_current = float(value)
        #         print(f"index = {index}, value = {value}, p_current = {p_current}")
        #         print(f"index = {index}, current[{index}] = {current[index]}")
        #         print(f"index = {index}, pos[{index}] = {pos[index]}")
        #         print(f"index = {index}, arm.motor_names[{index}] = {self.arm.motor_names[index]}")
        #         if abs(p_current) >= 150:
        #             self.arm.write("Goal_Position", pos[index], self.arm.motor_names[index])
        #     if events["complete"]:
        #         old_gripper = self.arm.read("Present_Position", "gripper")
        #         print(f"Will complete {id} calibration")
        #         while True:
        #             gripper = self.arm.read("Present_Position", "gripper")
        #             print(f"gripper = {gripper}")
        #             if abs(gripper - old_gripper) >= 150:
        #                 break
        #         break

        if not is_headless():
            if listener is not None:
                listener.stop()

        # Get true drive mode from 90°
        present_positions = apply_offset_and_drivemode(
            self.arm.read("Present_Position"),
            horizontal_homing_offset,
            DEFAULT_DRIVE_MODE,
        )
        nearest_positions = compute_nearest_rounded_positions(present_positions)
        drive_mode = []
        for i in range(len(nearest_positions)):
            drive_mode.append(nearest_positions[i] != TARGET_90_DEGREE_POSITION[i])

        # Get offset from 90°
        present_positions = apply_offset_and_drivemode(
            self.arm.read("Present_Position"), HORIZONTAL_POSITION, drive_mode
        )
        nearest_positions = compute_nearest_rounded_positions(present_positions)
        homing_offset = compute_corrections(
            nearest_positions, drive_mode, TARGET_90_DEGREE_POSITION
        )

        # Invert offset for all drive_mode servos
        for i in range(len(drive_mode)):
            if drive_mode[i]:
                homing_offset[i] = -homing_offset[i]

        print("Calibration is done!")
        print("=====================================")
        print("      HOMING_OFFSET: ", " ".join([str(i) for i in homing_offset]))
        print("      DRIVE_MODE: ", " ".join([str(i) for i in drive_mode]))
        print("=====================================")
        print("run_arm_calibration OK")

        # gen72关节初始化，移动到 零位
        # ret=self.follower_arms[name].movej_cmd([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        print("机械臂回到 零位 ")

        # Set calibration
        calibration = {}
        for idx, motor_name in enumerate(self.arm.motor_names):
            calibration[motor_name] = (homing_offset[idx], drive_mode[idx])
        self.arm.set_calibration(calibration)

    def movej(self, joint):
        # clipped_joint = max(joint_n_limit[:7], min(joint_p_limit[:7], joint[:7]))
        # self.arm.rm_movej(clipped_joint, 30, 0, 0, 0)
        {}  # TODO

    def ctrl_gripper(self, gripper):
        clipped_gripper = max(0, min(100, gripper))
        # self.arm.write
        # self.arm.rm_write_single_register(self.peripheral, clipped_gripper)
        {}  # TODO

    def read_joint_degree(self):
        joint_read = []
        servo_pos = self.arm.read("Present_Position")

        for i in range(7):
            value = round(servo_pos * SCALE_FACTOR, 2)  # 数值转换

            # # 特定关节取反（3号和5号）
            # if i in {3, 5}:
            #     value = -value

            # 限幅
            clamped_value = max(joint_n_limit[i], min(joint_p_limit[i], value))

            joint_read.append(clamped_value)

        return joint_read

    def stop(self):
        # self.arm.rm_set_arm_stop()
        {}  # TODO

    def disconnect(self):
        # self.arm.rm_close_modbus_mode(1)
        self.is_connected = False
        {}  # TODO


def main():
    main_arm = DynamixelArm()

    for event in node:
        event_type = event["type"]

        if event_type == "INPUT":
            if event["id"] == "movej":
                joint = event["value"].to_pylist()
                main_arm.movej(joint)

            if event["id"] == "gripper":
                gripper = event["value"]
                main_arm.ctrl_gripper(gripper)

            if event["id"] == "read-joint":
                read_joint = main_arm.read_joint_degree()
                node.send_output("read-joint", pa.array(read_joint))


if __name__ == "__main__":
    main()
