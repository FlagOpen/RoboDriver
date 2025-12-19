import os
import time

import numpy as np
import pyarrow as pa
from dora import Node
from Robotic_Arm.rm_robot_interface import *

id = os.getenv(
    "ARM_ID", "arm_right"
)  # ARM_ID: Arm identifier; defaults to "arm_right" if not set
ip = os.getenv(
    "ARM_IP", "192.168.1.18"
)  # ARM_IP: Connection IP address; defaults to "192.168.1.18" if not set
port = int(
    os.getenv("ARM_PORT", "8080")
)  # ARM_PORT: Connection port number; defaults to 8080 if not set

start_pose_str = os.getenv("START_POSE", "0.0, 0.0, 0.0, 0.0, 0.0, 0.0")
joint_p_limit_str = os.getenv("JOINT_P_LIMIT", "0.0, 0.0, 0.0, 0.0, 0.0, 0.0")
joint_n_limit_str = os.getenv("JOINT_N_LIMIT", "0.0, 0.0, 0.0, 0.0, 0.0, 0.0")
start_pose = [float(x) for x in start_pose_str.split(",")]
joint_p_limit = [float(x) for x in joint_p_limit_str.split(",")]
joint_n_limit = [float(x) for x in joint_n_limit_str.split(",")]

node = Node()


class RealmanArm:
    def __init__(self, ip, port, start_pose, joint_p_limit, joint_n_limit):
        self.arm = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
        self.ip = ip
        self.port = port
        self.start_pose = start_pose
        self.joint_p_limit = joint_p_limit
        self.joint_n_limit = joint_n_limit

        handle = self.arm.rm_create_robot_arm(self.ip, self.port)
        print("Arm ID: ", id)
        print("Arm handle ID: ", handle.id)
        print(f"Arm Connected On: {self.ip}:{self.port}")
        software_info = self.arm.rm_get_arm_software_info()
        if software_info[0] == 0:
            print("\n================== Arm Software Information ==================")
            print("Arm Model: ", software_info[1]["product_version"])
            print(
                "Algorithm Library Version: ",
                software_info[1]["algorithm_info"]["version"],
            )
            print(
                "Control Layer Software Version: ",
                software_info[1]["ctrl_info"]["version"],
            )
            print(
                "Dynamics Version: ", software_info[1]["dynamic_info"]["model_version"]
            )
            print(
                "Planning Layer Software Version: ",
                software_info[1]["plan_info"]["version"],
            )
            print("==============================================================\n")
        else:
            print(
                "\nFailed to get arm software information, Error code: ",
                software_info[0],
                "\n",
            )

        # self.arm.rm_set_tool_voltage(3)#设置末端工具接口电压为24v
        # self.arm.rm_set_modbus_mode(1, 115200, 5) #打开modbus模式
        # self.peripheral = rm_peripheral_read_write_params_t(1, 40000, 1, 1)#配置串口参数
        # print(self.peripheral)
        # self.arm.rm_write_single_register(self.peripheral, 100)#初始化夹爪为打开状态

        self.is_connected = True
        self.logs = {}

    def movej_cmd(self, joint):
        clipped_joint = max(joint_n_limit[:7], min(joint_p_limit[:7], joint[:7]))
        self.arm.rm_movej(clipped_joint, 30, 0, 0, 0)

    def movej_canfd(self, joint):
        clipped_joint = max(joint_n_limit[:7], min(joint_p_limit[:7], joint[:7]))
        self.arm.rm_movej_canfd(clipped_joint, True, 0)

    def write_single_register(self, gripper):
        clipped_gripper = max(0, min(100, gripper))
        self.arm.rm_write_single_register(self.peripheral, clipped_gripper)

    def read_joint(self):
        _num, robot_info = self.arm.rm_get_current_arm_state()
        position = robot_info["pose"][:3]
        euler = robot_info["pose"][3:]
        quaternion = self.arm.rm_algo_euler2quaternion(euler)
        pose_7d = np.concatenate([position, quaternion])
        joint_degree = robot_info["joint"]
        return joint_degree, pose_7d

    def read_lift_height(self):
        _num, lift_read = self.arm.rm_get_lift_state()
        height = lift_read["pos"]
        return height

    def stop(self):
        self.arm.rm_set_arm_stop()

    def disconnect(self):
        self.arm.rm_close_modbus_mode(1)
        self.is_connected = False

    def rm_read_gripper_actpos(self):
        flag, gripper_dict = self.arm.rm_get_gripper_state()
        gripper_actpos = np.array([gripper_dict["actpos"]]).astype(np.float64)
        return gripper_actpos


def main():
    main_arm = RealmanArm(ip, port, start_pose, joint_p_limit, joint_n_limit)
    # ---------- 新增 ----------
    last_frame_ts = time.time()  # 最近一次拿到帧的时间
    STATUS_OK = b"True"
    STATUS_OFFLINE = b"False"
    STATUS_NODATA = b"False"
    # --------------------------
    for event in node:
        event_type = event["type"]
        if event_type == "INPUT":
            if event["id"] == "tick":
                jointstate, pose = main_arm.read_joint()
                node.send_output("jointstate", pa.array(jointstate))
                node.send_output("pose", pa.array(pose))
                gripper_actpos = main_arm.rm_read_gripper_actpos()
                node.send_output("gripper", pa.array(gripper_actpos))
                lift_height = main_arm.read_lift_height()
                node.send_output("lift_height", pa.array([lift_height]))
            elif event["id"] == "hw_tick":
                # ---------- 心跳 ----------
                if time.time() - last_frame_ts > 2.0:
                    status = STATUS_NODATA
                else:
                    status = STATUS_OK
                node.send_output("hw_single", status)
                # --------------------------
            elif event["id"] == "stop":
                main_arm.stop()
        elif event_type == "ERROR":
            print("Event Error:" + event["error"])
    main_arm.disconnect()


if __name__ == "__main__":
    main()
