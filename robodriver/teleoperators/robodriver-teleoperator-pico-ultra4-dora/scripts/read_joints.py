"""
读取双臂关节位置
用法：
  python read_joints.py            # 持续刷新
  python read_joints.py --once     # 只读一次
  python read_joints.py --set-home # 将当前位置保存为 home
  python read_joints.py --go-home  # 发送已保存的 home 位置
"""
import sys
import json
import time
from pathlib import Path
import numpy as np
from piper_sdk import C_PiperInterface_V2

RIGHT_CAN = "can0"
LEFT_CAN = "can1"
ANGLE_FACTOR = 57295.7795  # SDK 单位 → 弧度
HOME_FILE = Path(__file__).parent / "home_positions.json"


def read_joints(piper) -> np.ndarray:
    j = piper.GetArmJointMsgs().joint_state
    return np.array([
        j.joint_1, j.joint_2, j.joint_3,
        j.joint_4, j.joint_5, j.joint_6,
    ]) / ANGLE_FACTOR


def read_gripper(piper) -> float:
    angle_deg = piper.GetArmGripperMsgs().gripper_state.grippers_angle / 1000.0
    return np.clip(abs(angle_deg) / 80.0, 0.0, 1.0)


def connect(can_port: str):
    p = C_PiperInterface_V2(can_port)
    p.ConnectPort()
    return p


def print_state(label: str, joints: np.ndarray, gripper: float):
    j_str = "  ".join(f"{v:+.4f}" for v in joints)
    print(f"[{label}]  joints: {j_str}  gripper: {gripper:.3f}")


def save_home(right_q: np.ndarray, right_g: float, left_q: np.ndarray, left_g: float):
    data = {
        "right": {"joints": right_q.tolist(), "gripper": right_g},
        "left":  {"joints": left_q.tolist(),  "gripper": left_g},
    }
    HOME_FILE.write_text(json.dumps(data, indent=2))
    print(f"Home positions saved to {HOME_FILE}")


def load_home():
    if not HOME_FILE.exists():
        print(f"[ERROR] No home file found at {HOME_FILE}. Run --set-home first.")
        sys.exit(1)
    data = json.loads(HOME_FILE.read_text())
    right_q = np.array(data["right"]["joints"])
    right_g = data["right"]["gripper"]
    left_q  = np.array(data["left"]["joints"])
    left_g  = data["left"]["gripper"]
    return right_q, right_g, left_q, left_g


def send_joints(piper, joints: np.ndarray, speed_factor: int = 30):
    piper.MotionCtrl_2(0x01, 0x01, speed_factor, 0x00)
    vals = [round(v * ANGLE_FACTOR) for v in joints]
    piper.JointCtrl(*vals)


def send_gripper(piper, position: float, speed: int = 1000):
    pos_um = round(np.clip(position, 0.0, 1.0) * 80000)
    piper.GripperCtrl(abs(pos_um), speed, 0x01, 0)


def main():
    once     = "--once"     in sys.argv
    set_home = "--set-home" in sys.argv
    go_home  = "--go-home"  in sys.argv

    print(f"Connecting to right arm ({RIGHT_CAN}) ...")
    right = connect(RIGHT_CAN)
    print(f"Connecting to left arm  ({LEFT_CAN}) ...")
    left = connect(LEFT_CAN)
    print("Connected.\n")

    if set_home:
        right_q = read_joints(right)
        right_g = read_gripper(right)
        left_q  = read_joints(left)
        left_g  = read_gripper(left)
        print_state("RIGHT", right_q, right_g)
        print_state("LEFT ", left_q, left_g)
        save_home(right_q, right_g, left_q, left_g)
        return

    if go_home:
        right_q, right_g, left_q, left_g = load_home()
        print("Sending home positions:")
        print_state("RIGHT", right_q, right_g)
        print_state("LEFT ", left_q, left_g)
        send_joints(right, right_q)
        send_gripper(right, right_g)
        send_joints(left, left_q)
        send_gripper(left, left_g)
        print("Done.")
        return

    print("Reading joint positions...\n")
    try:
        while True:
            right_q = read_joints(right)
            right_g = read_gripper(right)
            left_q  = read_joints(left)
            left_g  = read_gripper(left)

            print_state("RIGHT", right_q, right_g)
            print_state("LEFT ", left_q, left_g)
            print()

            if once:
                break
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("Stopped.")


if __name__ == "__main__":
    main()
