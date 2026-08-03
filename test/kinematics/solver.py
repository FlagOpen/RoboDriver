import os
import time

import numpy as np
from scipy.spatial.transform import Rotation as R



URDF_PATH = os.getenv("URDF_PATH", "urdf/piper.urdf")


import casadi
import pinocchio as pin
from pinocchio import casadi as cpin


class Solver:
    """Forward/Inverse kinematics solver for robotic arm using Pinocchio and CasADi."""
    
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
        # rpy = convert_pose(rpy, direction='A_to_B')
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

        # rpy = convert_pose([pitch, -roll, yaw], direction='B_to_A')
        rpy = [pitch, -roll, yaw]
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


