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
EEPOSE_MODE = os.getenv("EEPOSE_MODE", "False") in ["True", "true"]
URDF_PATH = os.getenv("URDF_PATH", "urdf/piper.urdf")

if EEPOSE_MODE:
    import casadi
    import pinocchio as pin
    from pinocchio import casadi as cpin

# 坐标系变换矩阵，A为原本末端位姿坐标系，B为调整为X向前后的坐标系
R_A_TO_B = np.array([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1]
])
R_B_TO_A = R_A_TO_B.T

R_C_TO_D = np.array([
    [0, 0, -1],
    [0, 1, 0],
    [1, 0, 0]
])
R_D_TO_C = R_C_TO_D.T


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

def convert_pose_2(pose, direction='C_to_D', input_order='ZYX', output_order='ZYX', degrees=False):
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
    transform = R_C_TO_D if direction == 'C_to_D' else R_D_TO_C
    
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
    arm_status = piper.GetArmStatus().arm_status
    
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
        print(f"    关节1: {'是' if err.joint_1_angle_limit else '否'}")
        print(f"    关节2: {'是' if err.joint_2_angle_limit else '否'}")
        print(f"    关节3: {'是' if err.joint_3_angle_limit else '否'}")
        print(f"    关节4: {'是' if err.joint_4_angle_limit else '否'}")
        print(f"    关节5: {'是' if err.joint_5_angle_limit else '否'}")
        print(f"    关节6: {'是' if err.joint_6_angle_limit else '否'}")
        
        print(f"\n  关节通信异常:")
        print(f"    关节1: {'是' if err.communication_status_joint_1 else '否'}")
        print(f"    关节2: {'是' if err.communication_status_joint_2 else '否'}")
        print(f"    关节3: {'是' if err.communication_status_joint_3 else '否'}")
        print(f"    关节4: {'是' if err.communication_status_joint_4 else '否'}")
        print(f"    关节5: {'是' if err.communication_status_joint_5 else '否'}")
        print(f"    关节6: {'是' if err.communication_status_joint_6 else '否'}")
    else:
        print("  无故障状态信息")
    
    # Print additional useful information
    print(f"\n时间戳: {arm_status.time_stamp if hasattr(arm_status, 'time_stamp') else 'N/A'}")
    print(f"频率: {arm_status.Hz if hasattr(arm_status, 'Hz') else 'N/A'} Hz")
    print("=" * 50)



class Arm_IK:
    """Inverse kinematics solver for robotic arm using Pinocchio and CasADi."""
    
    def __init__(self, urdf_path: str = None):
        np.set_printoptions(precision=5, suppress=True, linewidth=200)
    
        # 【关键修改 1】将相对路径转换为绝对路径
        if not os.path.isabs(urdf_path):
            urdf_path = os.path.abspath(urdf_path)
        
        # 【关键修改 2】获取 URDF 所在的目录，作为 mesh 搜索根目录
        urdf_dir = os.path.dirname(urdf_path)
        
        print(f"[INFO] Loading URDF: {urdf_path}")
        print(f"[INFO] Mesh Root Dir: {urdf_dir}")

        # 加载主机器人模型 (传入 package_dirs 确保主模型也能找到 mesh)
        self.robot = pin.RobotWrapper.BuildFromURDF(
            filename=urdf_path,
            package_dirs=[urdf_dir],
            verbose=False
        )

        self.mixed_jointsToLockIDs = ["joint7", "joint8"]

        self.reduced_robot = self.robot.buildReducedRobot(
            list_of_joints_to_lock=self.mixed_jointsToLockIDs,
            reference_configuration=np.array([0] * self.robot.model.nq),
        )

        self.model = self.reduced_robot.model
        
        q = pin.Quaternion(1, 0, 0, 0)
        # 注意：addFrame 是在 reduced_model 上操作的，确保 joint6 在 reduced_model 中存在
        try:
            self.model.addFrame(
                pin.Frame('ee',
                          self.model.getJointId('joint6'),
                          pin.SE3(q, np.array([0.0, 0.0, 0.0])),
                          pin.FrameType.OP_FRAME)
            )
        except Exception as e:
            print(f"Warning: Could not add frame 'ee'. Check if 'joint6' exists in reduced model. Error: {e}")

        self.data = self.model.createData()
        self.ee_frame_id = self.model.getFrameId("ee")

        # 【关键修改 3】修复 buildGeomFromUrdf 调用
        # 必须再次传入 package_dirs，因为它不会自动从 self.robot 继承
        self.geom_model = pin.buildGeomFromUrdf(
            self.robot.model, 
            urdf_path, 
            pin.GeometryType.COLLISION,
            package_dirs=[urdf_dir]  # <--- 这里必须加上！
        )
        
        # 优化碰撞对添加逻辑 (避免硬编码索引导致的错误)
        # 建议先打印所有 geometry names 确认索引，或者通过名称获取
        # 这里保留你原有的逻辑，但请确保索引范围在当前 geom_model 中有效
        # 如果报错 IndexError，说明几何体数量少于 9 个
        num_geom = len(self.geom_model.geometryObjects)
        print(f"[INFO] Number of collision geometries: {num_geom}")
        
        # 安全地添加碰撞对 (防止索引越界)
        for i in range(min(4, num_geom), min(9, num_geom)):
            for j in range(0, min(3, num_geom)):
                if i != j:
                    self.geom_model.addCollisionPair(pin.CollisionPair(i, j))
                    
        self.geometry_data = pin.GeometryData(self.geom_model)

        self.init_data = np.zeros(self.reduced_robot.model.nq)
        self.history_data = np.zeros(self.reduced_robot.model.nq)

        # Creating Casadi models and data for symbolic computing
        self.cmodel = cpin.Model(self.reduced_robot.model)
        self.cdata = self.cmodel.createData()

        # Creating symbolic variables
        self.cq = casadi.SX.sym("q", self.reduced_robot.model.nq, 1)
        self.cTf = casadi.SX.sym("tf", 4, 4)
        cpin.framesForwardKinematics(self.cmodel, self.cdata, self.cq)

        # Get the hand joint ID and define the error function
        # 注意：如果上面 addFrame 失败，这里会报错，需确保 ee_frame_id 有效
        if self.ee_frame_id >= len(self.cdata.oMf):
             raise RuntimeError("End-effector frame ID invalid. Check joint names in URDF.")

        self.error = casadi.Function(
            "error",
            [self.cq, self.cTf],
            [
                casadi.vertcat(
                    cpin.log6(
                        self.cdata.oMf[self.ee_frame_id].inverse() * cpin.SE3(self.cTf)
                    ).vector,
                )
            ],
        )

        # Defining the optimization problem
        self.opti = casadi.Opti()
        self.var_q = self.opti.variable(self.reduced_robot.model.nq)
        self.param_tf = self.opti.parameter(4, 4)
        self.totalcost = casadi.sumsqr(self.error(self.var_q, self.param_tf))
        self.regularization = casadi.sumsqr(self.var_q)

        # Setting optimization constraints and goals
        self.opti.subject_to(self.opti.bounded(
            self.reduced_robot.model.lowerPositionLimit,
            self.var_q,
            self.reduced_robot.model.upperPositionLimit)
        )
        self.opti.minimize(20 * self.totalcost + 0.01 * self.regularization)

        opts = {
            'ipopt': {
                'print_level': 0,
                'max_iter': 50,
                'tol': 1e-4
            },
            'print_time': False
        }
        self.opti.solver("ipopt", opts)

    def ik_fun(self, target_pose, gripper=0, motorstate=None, motorV=None):
        """Calculate inverse kinematics for target pose."""
        gripper = np.array([gripper/2.0, -gripper/2.0])
        if motorstate is not None:
            self.init_data = motorstate
        self.opti.set_initial(self.var_q, self.init_data)

        self.opti.set_value(self.param_tf, target_pose)
        # self.opti.set_value(self.var_q_last, self.init_data) # for smooth

        try:
            sol = self.opti.solve_limited()
            sol_q = self.opti.value(self.var_q)

            if self.init_data is not None:
                max_diff = max(abs(self.history_data - sol_q))
                self.init_data = sol_q
                if max_diff > 30.0/180.0*3.1415:
                    self.init_data = np.zeros(self.reduced_robot.model.nq)
            else:
                self.init_data = sol_q
            self.history_data = sol_q

            if motorV is not None:
                v = motorV * 0.0
            else:
                v = (sol_q - self.init_data) * 0.0

            tau_ff = pin.rnea(self.reduced_robot.model, self.reduced_robot.data, sol_q, v,
                              np.zeros(self.reduced_robot.model.nv))

            is_collision = self.check_self_collision(sol_q, gripper)

            return sol_q, tau_ff, not is_collision

        except Exception as e:
            print(f"ERROR in convergence, plotting debug info.{e}")
            return self.opti.debug.value(self.var_q), '', False

    def check_self_collision(self, q, gripper=np.array([0, 0])):
        """Check for self-collision in the robot configuration."""
        pin.forwardKinematics(self.robot.model, self.robot.data, np.concatenate([q, gripper], axis=0))
        pin.updateGeometryPlacements(self.robot.model, self.robot.data, self.geom_model, self.geometry_data)
        collision = pin.computeCollisions(self.geom_model, self.geometry_data, False)
        return collision

    def get_fk(self, q: np.ndarray) -> np.ndarray:
        """
        Calculate forward kinematics for given joint angles.

        Args:
            q: joint angles(rad) of the arm

        Returns:
            xyz_rpy: xyz(m), rpy(rad) in standard coordinate system:
                - X: forward direction
                - Y: left direction  
                - Z: upward direction
                - roll: rotation around X axis
                - pitch: rotation around Y axis
                - yaw: rotation around Z axis
        """
        pin.framesForwardKinematics(self.model, self.data, q)
        frame = self.data.oMf[self.ee_frame_id]
        
        # Convert from Pinocchio coordinate system to standard coordinate system
        rpy = pin.rpy.matrixToRpy(frame.rotation)
        rpy = convert_pose(rpy, direction='A_to_B')
        rpy_new = [-rpy[1], rpy[0], rpy[2]]
        rpy_new[1] -= np.pi/2
        rpy_new[2] -= np.pi/2

        return np.concatenate([frame.translation, rpy_new])

    def get_ik_solution(self, x, y, z, roll, pitch, yaw) -> np.ndarray:
        """
        Get inverse kinematics solution for target position and orientation.
        
        Args:
            x, y, z: target position (meters) in world coordinate system
            roll, pitch, yaw: target orientation (radians) in standard coordinate system:
                - roll: rotation around X axis (forward direction)
                - pitch: rotation around Y axis (left direction)
                - yaw: rotation around Z axis (upward direction)
        
        Returns:
            Joint angles solution or None if no solution found
        """
        # Convert from standard coordinate system to Pinocchio coordinate system
        pitch += np.pi/2
        yaw += np.pi/2

        rpy = convert_pose([pitch, -roll, yaw], direction='B_to_A')
        rot_orig = R.from_euler('xyz', rpy)

        q = rot_orig.as_quat()

        target = pin.SE3(
            pin.Quaternion(q[3], q[0], q[1], q[2]),
            np.array([x, y, z]),
        )
        sol_q, tau_ff, get_result = self.ik_fun(target.homogeneous,0)
        
        if get_result:
            return sol_q
        else:
            return None


def send_smooth_joint_move(piper, current_joints, target_joints, steps=10):
    """
    平滑地将关节从当前位置移动到目标位置
    
    Args:
        piper: 机器人控制对象
        current_joints: 当前关节角度列表 [j0, j1, j2, j3, j4, j5]
        target_joints: 目标关节角度列表 [j0, j1, j2, j3, j4, j5]
        steps: 插值步数，默认10步
    """
    
    # 生成插值轨迹
    for step in range(1, steps + 1):
        # 计算插值系数 (0.1, 0.2, 0.3, ..., 1.0)
        alpha = step / steps
        
        # 对每个关节进行线性插值
        interpolated_joints = []
        for i in range(6):
            # 线性插值公式: current + (target - current) * alpha
            joint_value = current_joints[i] + (target_joints[i] - current_joints[i]) * alpha
            joint_value = round(joint_value)  # 四舍五入取整
            interpolated_joints.append(joint_value)
        
        # 发送插值后的关节角度
        piper.MotionCtrl_2(0x01, 0x01, 100, 0x00)
        piper.JointCtrl(interpolated_joints[0], interpolated_joints[1],
                       interpolated_joints[2], interpolated_joints[3],
                       interpolated_joints[4], interpolated_joints[5])
        
        # 适当延时，控制运动速度
        time.sleep(0.005)  # 10ms延时，可以根据需要调整

def main():
    """TODO: Add docstring."""
    elapsed_time = time.time()
    piper = C_PiperInterface(CAN_BUS)
    piper.ConnectPort()
    if EEPOSE_MODE:
        arm_ik = Arm_IK(URDF_PATH)

    if not TEACH_MODE:
        enable_fun(piper)

    factor = 57295.779578552  # 1000*180/3.14159265
    node = Node()

    last_joints = [0, 0, 0, 0, 0, 0]

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
                if time.time() - elapsed_time > 0.005 and EEPOSE_MODE:
                    elapsed_time = time.time()
                else:
                    continue
                enable_fun(piper)

                position = event["value"].to_numpy()

                joints = arm_ik.get_ik_solution(
                    position[0],
                    position[1],
                    position[2],
                    position[3],
                    position[4],
                    position[5],
                )

                joints[0] = round(joints[0] * factor)
                joints[1] = round(joints[1] * factor)
                joints[2] = round(joints[2] * factor)
                joints[3] = round(joints[3] * factor)
                joints[4] = round(joints[4] * factor)
                joints[5] = round(joints[5] * factor)

                # send_smooth_joint_move(piper, last_joints, joints, steps=10)
                # last_joints = joints

                piper.MotionCtrl_2(0x01, 0x01, 100, 0x00)
                piper.JointCtrl(joints[0], joints[1], joints[2], joints[3], joints[4], joints[5])

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

                # position = piper.GetArmEndPoseMsgs()
                # ori_rot = [
                #     position.end_pose.RX_axis * 0.001,
                #     position.end_pose.RY_axis * 0.001,
                #     position.end_pose.RZ_axis * 0.001
                # ]
                # cvt_rot = convert_pose_2(ori_rot, 'C_to_D', 'ZYX', 'ZYX', degrees=True)

                # position_value = []
                # position_value += [position.end_pose.X_axis * 0.001 * 0.001]
                # position_value += [position.end_pose.Y_axis * 0.001 * 0.001]
                # position_value += [position.end_pose.Z_axis * 0.001 * 0.001]
                # position_value += [cvt_rot[0] / 180 * np.pi]
                # position_value += [cvt_rot[1] / 180 * np.pi]
                # position_value += [cvt_rot[2] / 180 * np.pi]
                if EEPOSE_MODE:
                    position_value = arm_ik.get_fk(joint_value[:6])

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
