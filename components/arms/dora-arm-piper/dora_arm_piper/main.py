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

        elif event["type"] == "STOP":
            break


if __name__ == "__main__":
    main()
