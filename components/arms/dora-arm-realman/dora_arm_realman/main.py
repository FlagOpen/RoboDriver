"""TODO: Add docstring."""

import os
import time
import pyarrow as pa
import draccus
from dora import Node
from pathlib import Path
import numpy as np

from Robotic_Arm.rm_robot_interface import *


ARM_ID = os.getenv("ARM_NAME", "arm_right")
IP = os.getenv("ARM_IP", "192.168.1.18")
PORT = int(os.getenv("ARM_PORT", "8080"))

JOINT_N_LIMIT_STR = os.getenv("JOINT_N_LIMIT", "-170,-120,-170,-120,-170,-120,-170")
JOINT_P_LIMIT_STR = os.getenv("JOINT_P_LIMIT", "170,120,170,120,170,120,170")

ARM_ROLE = os.getenv("ARM_ROLE", "follower")


def env_to_bool(env_value: str, default: bool = True) -> bool:
    if env_value is None:
        return default
    true_values = {'True', 'true', '1', 'yes', 'on'}
    false_values = {'False', 'false', '0', 'no', 'off'}
    value_lower = env_value.strip().lower()
    if value_lower in true_values:
        return True
    elif value_lower in false_values:
        return False
    else:
        raise ValueError(f"无效的布尔值环境变量: {env_value}")

def parse_env_list(env_str: str, default: list, dtype=float) -> list:
    if not env_str:
        return default
    try:
        return [dtype(x.strip()) for x in env_str.split(',')]
    except Exception as e:
        raise ValueError(f"解析环境变量列表失败 '{env_str}': {e}")


class RealmanArm:
    def __init__(self, ip, port, joint_limits):
        self.arm = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
        self.ip = ip
        self.port = port
        self.joint_n_limit, self.joint_p_limit = joint_limits

        handle = self.arm.rm_create_robot_arm(self.ip, self.port)
        if handle.id == 0:
             raise ConnectionError(f"无法连接到机械臂 {self.ip}:{self.port}")

        print(f"机械臂 '{ARM_ID}' 已连接, Handle ID: {handle.id}")
        
        # 打印软件信息
        self._print_software_info()

        self.is_connected = True

    def _print_software_info(self):
        """打印机械臂软件信息"""
        software_info = self.arm.rm_get_arm_software_info()
        if software_info[0] == 0:
            info = software_info[1]
            print("\n================== Arm Software Information ==================")
            print(f"Arm Model: {info.get('product_version', 'N/A')}")
            print(f"Control Layer Version: {info.get('ctrl_info', {}).get('version', 'N/A')}")
            print("==============================================================\n")
        else:
            print(f"\n警告: 无法获取机械臂软件信息, 错误码: {software_info[0]}\n")

    def movej_canfd(self, joint):
        result = self.arm.rm_movej_canfd(joint, False, 0) # 设置为低跟随模式
        if result != 0:
            print(f"ip为{self.ip}的臂CANFD运动命令失败，错误码: {result}")
            return False
        return True

    def read_joint_state(self) -> tuple[list[float], list[float]]:
        _num, robot_info = self.arm.rm_get_current_arm_state()
        joint_degree = robot_info.get('joint', [0.0]*7)
        
        position = robot_info.get('pose', [0.0]*6)
                
        return joint_degree, position

    def read_gripper(self):
        flag, gripper_dict = self.arm.rm_get_gripper_state()
        gripper_actpos = np.array([gripper_dict['actpos']])
        return gripper_actpos

    def set_gripper(self, value):
        value = int(value)
        try:
            result = self.arm.rm_set_gripper_position(value, False, 1)
            if result != 0:
                print(f"ip为{self.ip}的臂夹爪控制失败，错误码: {result}")
                return False
            return True
        except Exception as e:
            print(f"ip为{self.ip}的臂夹爪控制异常: {e}")
            return False

    def stop(self):

        self.arm.rm_set_arm_stop()
    
    def disconnect(self):

        self.arm.rm_close_modbus_mode(1)
        self.is_connected = False


def main():
    node = Node()

    # 解析关节限制
    try:
        joint_n_limit = parse_env_list(JOINT_N_LIMIT_STR, [-170]*7)
        joint_p_limit = parse_env_list(JOINT_P_LIMIT_STR, [170]*7)
        joint_limits = (joint_n_limit, joint_p_limit)
    except ValueError as e:
        print(f"配置错误: {e}")
        return

    # 初始化机械臂
    try:
        realman_arm = RealmanArm(IP, PORT, joint_limits)
    except Exception as e:
        print(f"初始化失败: {e}")
        return

    ctrl_frame = 0

    for event in node:
        event_type = event["type"]

        if event_type == "INPUT":
            if "action" in event["id"]:
                pass
            if event["id"] == "action_joint":
                if ctrl_frame > 0:
                    continue
                
                try:
                    joint = event["value"].to_numpy()
                    joint_target = joint[:len(joint)//2] if "left" in ARM_ID else joint[len(joint)//2:]
                    realman_arm.movej_canfd(joint_target[:7])
                    realman_arm.set_gripper(joint_target[7])
                except Exception as e:
                    print(f"执行 'action_joint' 失败: {e}")

            elif event["id"] == "action_joint_ctrl":
                try:
                    joint = event["value"].to_numpy()
                    joint_target = joint[:len(joint)//2] if "left" in ARM_ID else joint[len(joint)//2:]
                    realman_arm.movej_canfd(joint_target[:7])
                    realman_arm.set_gripper(joint_target[7])
                    ctrl_frame = 200 # 持续200帧
                except Exception as e:
                    print(f"执行 'action_joint_ctrl' 失败: {e}")

            elif event["id"] == "get_joint":
                # 响应关节状态查询
                try:
                    jointstate, positon = realman_arm.read_joint_state()
                    gripper_pos = realman_arm.read_gripper()
                    combined_joint_state = np.concatenate([jointstate, gripper_pos, positon])
                    node.send_output("joint", pa.array(combined_joint_state, type=pa.float32()))
                except Exception as e:
                    print(f"执行 'get_joint' 失败: {e}")

            # 更新控制帧计数器
            if ctrl_frame > 0:
                ctrl_frame -= 1

        elif event_type == "STOP":
            print("收到停止指令，断开机械臂连接...")
            realman_arm.stop()
            break
        
        elif event_type == "ERROR":
            print(f"DORA 事件错误: {event['error']}")

    realman_arm.disconnect()
    print("程序已退出。")

if __name__ == "__main__":
    main()
