"""TODO: Add docstring."""

import os
import time

import numpy as np
import pyarrow as pa
from dora import Node
from piper_sdk import C_PiperInterface
from scipy.spatial.transform import Rotation as R


TEACH_MODE = os.getenv("TEACH_MODE", "False") in ["True", "true"]
CAN_BUS = os.getenv("CAN_BUS", "can0")
LOG_STATUS = os.getenv("LOG_STATUS", "False") in ["True", "true"]


# 坐标系变换矩阵，A为原本末端位姿坐标系，B为调整为X向前后的坐标系
R_A_TO_B = np.array([
    [0, 0, -1],
    [0, 1, 0],
    [1, 0, 0]
])
R_B_TO_A = R_A_TO_B.T


def convert_pose(pose, direction='A_to_B', input_order='ZYX', output_order='ZYX', degrees=False):
    """
    在坐标系A和B之间转换姿态
    
    参数:
    pose: 输入的姿态（旋转对象或欧拉角）, [x, y, z]
    direction: 转换方向，'A_to_B' 或 'B_to_A'
    input_order: 输入姿态的欧拉角顺序
    output_order: 输出姿态的欧拉角顺序
    degrees: 角度是否为度数
    
    返回:
    转换后的姿态
    """
    # 选择变换矩阵
    transform = R_A_TO_B if direction == 'A_to_B' else R_B_TO_A
    
    # 转换为旋转对象
    if isinstance(pose, (list, tuple, np.ndarray)):
        if len(pose) != 3:
            raise ValueError("pose应为3个欧拉角")
        rot = R.from_euler(input_order, pose, degrees=degrees)
    else:
        rot = pose
    
    # 应用变换并转换结果
    result_rot = R.from_matrix(transform @ rot.as_matrix())
    
    # 返回所需格式
    return result_rot if not output_order else result_rot.as_euler(output_order, degrees=degrees)


def enable_fun(piper: C_PiperInterface):
    """使能机械臂并检测使能状态,尝试0.05s,如果使能超时则退出程序."""
    enable_flag = all(piper.GetArmEnableStatus())
    
    timeout = 0.05  # 超时时间（秒）
    
    start_time = time.time()
    while not enable_flag:
        enable_flag = piper.EnablePiper()
        
        print(f"--------------------\n使能状态: {enable_flag} \n--------------------")

        time.sleep(0.01)
        elapsed_time = time.time() - start_time
        if elapsed_time > timeout:
            print("Piper机械臂自动使能超时....")
            break


def print_arm_status(piper: C_PiperInterface):
    """
    Print the current arm status information including ctrl_mode, arm_status, 
    mode_feed, motion_status, and err_status.
    """
    arm_status = piper.GetArmStatus()
    
    # Define lookup dictionaries for better readability
    ctrl_mode_map = {
        0x00: "待机模式",
        0x01: "CAN指令控制模式", 
        0x02: "示教模式"
    }
    
    arm_status_map = {
        0x00: "正常",
        0x01: "急停",
        0x02: "无解",
        0x03: "奇异点",
        0x04: "目标角度超过限",
        0x05: "关节通信异常",
        0x06: "关节抱闸未打开",
        0x07: "机械臂发生碰撞",
        0x08: "拖动示教时超速",
        0x09: "关节状态异常",
        0x0A: "其它异常",
        0x0B: "示教记录",
        0x0C: "示教执行",
        0x0D: "示教暂停",
        0x0E: "主控NTC过温",
        0x0F: "释放电阻NTC过温"
    }
    
    mode_feed_map = {
        0x00: "MOVE P",
        0x01: "MOVE J",
        0x02: "MOVE L",
        0x03: "MOVE C",
        0x04: "MOVE M",
        0x05: "MOVE_CPV"
    }
    
    motion_status_map = {
        0x00: "到达指定点位",
        0x01: "未到达指定点位"
    }
    
    print("=" * 50)
    print("机械臂状态信息:")
    print("=" * 50)
    
    # Print basic status information
    print(f"控制模式 (ctrl_mode): {ctrl_mode_map.get(arm_status.ctrl_mode, f'未知({arm_status.ctrl_mode})')}")
    print(f"机械臂状态 (arm_status): {arm_status_map.get(arm_status.arm_status, f'未知({arm_status.arm_status})')}")
    print(f"模式反馈 (mode_feed): {mode_feed_map.get(arm_status.mode_feed, f'未知({arm_status.mode_feed})')}")
    print(f"运动状态 (motion_status): {motion_status_map.get(arm_status.motion_status, f'未知({arm_status.motion_status})')}")
    
    # Print error status details
    print("\n故障状态 (err_status):")
    if hasattr(arm_status, 'err_status') and arm_status.err_status:
        err = arm_status.err_status
        print(f"  关节角度超限位:")
        print(f"    关节1: {'是' if err.get('joint_1_angle_limit', False) else '否'}")
        print(f"    关节2: {'是' if err.get('joint_2_angle_limit', False) else '否'}")
        print(f"    关节3: {'是' if err.get('joint_3_angle_limit', False) else '否'}")
        print(f"    关节4: {'是' if err.get('joint_4_angle_limit', False) else '否'}")
        print(f"    关节5: {'是' if err.get('joint_5_angle_limit', False) else '否'}")
        print(f"    关节6: {'是' if err.get('joint_6_angle_limit', False) else '否'}")
        
        print(f"\n  关节通信异常:")
        print(f"    关节1: {'是' if err.get('communication_status_joint_1', False) else '否'}")
        print(f"    关节2: {'是' if err.get('communication_status_joint_2', False) else '否'}")
        print(f"    关节3: {'是' if err.get('communication_status_joint_3', False) else '否'}")
        print(f"    关节4: {'是' if err.get('communication_status_joint_4', False) else '否'}")
        print(f"    关节5: {'是' if err.get('communication_status_joint_5', False) else '否'}")
        print(f"    关节6: {'是' if err.get('communication_status_joint_6', False) else '否'}")
    else:
        print("  无故障状态信息")
    
    # Print additional useful information
    print(f"\n时间戳: {arm_status.time_stamp if hasattr(arm_status, 'time_stamp') else 'N/A'}")
    print(f"频率: {arm_status.Hz if hasattr(arm_status, 'Hz') else 'N/A'} Hz")
    print("=" * 50)


def main():
    """TODO: Add docstring."""
    elapsed_time = time.time()
    piper = C_PiperInterface(CAN_BUS)
    piper.ConnectPort()

    if not TEACH_MODE:
        enable_fun(piper)

    factor = 57295.779578552  # 1000*180/3.14159265
    node = Node()

    for event in node:
        if event["type"] == "INPUT":
            # if "action" in event["id"]:
            #     enable_fun(piper)
            if event["id"] == "action_joint":
                # Do not push to many commands to fast. Limiting it to 50Hz
                if time.time() - elapsed_time > 0.02:
                    elapsed_time = time.time()
                else:
                    continue
                enable_fun(piper)

                position = event["value"].to_numpy()

                # print(f"action_joint: {position}")
                joint_0 = round(position[0] * factor)
                joint_1 = round(position[1] * factor)
                joint_2 = round(position[2] * factor)
                joint_3 = round(position[3] * factor)
                joint_4 = round(position[4] * factor)
                joint_5 = round(position[5] * factor)

                piper.MotionCtrl_2(0x01, 0x01, 100, 0x00)
                piper.JointCtrl(joint_0, joint_1, joint_2, joint_3, joint_4, joint_5)
                # piper.MotionCtrl_2(0x01, 0x01, 50, 0x00)
                if len(position) > 6 and not np.isnan(position[6]):
                    piper.GripperCtrl(int(abs(position[6] * 1000 * 100)), 1000, 0x01, 0)

            elif event["id"] == "action_endpose":
                # Do not push to many commands to fast. Limiting it to 50Hz
                if time.time() - elapsed_time > 0.02:
                    elapsed_time = time.time()
                else:
                    continue
                enable_fun(piper)

                position = event["value"].to_numpy()

                ori_rot = [
                    position[3] / np.pi * 180,
                    position[4] / np.pi * 180,
                    position[5] / np.pi * 180
                ]
                cvt_rot = convert_pose(ori_rot, 'B_to_A', 'ZYX', 'ZYX', degrees=True)
                piper.MotionCtrl_2(0x01, 0x00, 100, 0x00)
                piper.EndPoseCtrl(
                    round(position[0] * 1000 * 1000),
                    round(position[1] * 1000 * 1000),
                    round(position[2] * 1000 * 1000),
                    round(cvt_rot[0] * 1000),
                    round(cvt_rot[1] * 1000),
                    round(cvt_rot[2] * 1000),
                )
                # piper.MotionCtrl_2(0x01, 0x01, 50, 0x00)
                if len(position) > 6 and not np.isnan(position[6]):
                    piper.GripperCtrl(int(abs(position[6] * 1000 * 100)), 3000, 0x01, 0)
            
            
            elif event["id"] == "action_gripper":
                # Do not push to many commands to fast. Limiting it to 50Hz
                # if time.time() - elapsed_time > 0.02:
                #     elapsed_time = time.time()
                # else:
                #     continue

                position = event["value"].to_numpy()
                # piper.MotionCtrl_2(0x01, 0x01, 100, 0x00)

                # print(f"piper {CAN_BUS} recv position[0]: {position[0]}")
                piper.GripperCtrl(int(abs(position[0] * 1000 * 100)), 3000, 0x01, 0)
                # piper.MotionCtrl_2(0x01, 0x01, 50, 0x00)

            elif event["id"] == "tick":
                # Slave Arm
                joint = piper.GetArmJointMsgs()

                joint_value = []
                joint_value += [joint.joint_state.joint_1.real / factor]
                joint_value += [joint.joint_state.joint_2.real / factor]
                joint_value += [joint.joint_state.joint_3.real / factor]
                joint_value += [joint.joint_state.joint_4.real / factor]
                joint_value += [joint.joint_state.joint_5.real / factor]
                joint_value += [joint.joint_state.joint_6.real / factor]

                gripper = piper.GetArmGripperMsgs()
                joint_value += [gripper.gripper_state.grippers_angle / 1000 / 100]

                node.send_output("follower_jointstate", pa.array(joint_value, type=pa.float32()))

                position = piper.GetArmEndPoseMsgs()
                ori_rot = [
                    position.end_pose.RX_axis * 0.001,
                    position.end_pose.RY_axis * 0.001,
                    position.end_pose.RZ_axis * 0.001
                ]
                cvt_rot = convert_pose(ori_rot, 'A_to_B', 'ZYX', 'ZYX', degrees=True)

                position_value = []
                position_value += [position.end_pose.X_axis * 0.001 * 0.001]
                position_value += [position.end_pose.Y_axis * 0.001 * 0.001]
                position_value += [position.end_pose.Z_axis * 0.001 * 0.001]
                position_value += [cvt_rot[0] / 180 * np.pi]
                position_value += [cvt_rot[1] / 180 * np.pi]
                position_value += [cvt_rot[2] / 180 * np.pi]

                node.send_output("follower_endpose", pa.array(position_value, type=pa.float32()))

                # Master Arm
                joint = piper.GetArmJointCtrl()

                joint_value = []
                joint_value += [joint.joint_ctrl.joint_1.real / factor]
                joint_value += [joint.joint_ctrl.joint_2.real / factor]
                joint_value += [joint.joint_ctrl.joint_3.real / factor]
                joint_value += [joint.joint_ctrl.joint_4.real / factor]
                joint_value += [joint.joint_ctrl.joint_5.real / factor]
                joint_value += [joint.joint_ctrl.joint_6.real / factor]

                gripper = piper.GetArmGripperCtrl()
                joint_value += [gripper.gripper_ctrl.grippers_angle / 1000 / 100]

                node.send_output("leader_jointstate", pa.array(joint_value, type=pa.float32()))

                if LOG_STATUS:
                    print_arm_status(piper)

        elif event["type"] == "STOP":
            break


if __name__ == "__main__":
    main()
