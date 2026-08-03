import logging_mp
import threading
import time
import numpy as np
import pyarrow as pa
from dora import Node
from typing import Any, Dict, Optional

# 导入 XRoboToolkit SDK
try:
    import xrobotoolkit_sdk as xrt
    XRT_AVAILABLE = True
except ImportError:
    XRT_AVAILABLE = False
    logging_mp.warning("xrobotoolkit_sdk not available")

# 导入 Placo IK 求解器
try:
    import placo
    PLACO_AVAILABLE = True
except ImportError:
    PLACO_AVAILABLE = False
    logging_mp.warning("placo not available")


logger = logging_mp.getLogger(__name__)
CONNECT_TIMEOUT_FRAME = 10


class TeleoperatorNode:
    pass


class DoraTeleoperatorNode(TeleoperatorNode):
    pass


class PicoUltra4DoraTeleoperatorNode(DoraTeleoperatorNode):
    """
    Pico Ultra4 VR 遥操器 Dora 节点

    功能：
    1. 从 XRoboToolkit SDK 获取 VR 控制器位姿
    2. 使用 Placo 进行 IK 求解
    3. 通过 Dora 发送关节角度命令
    """

    def __init__(self, robot_urdf_path: str, config: Dict[str, Any]):
        self.node = Node("pico_ultra4_dora")
        self.config = config

        # VR 数据
        self.vr_pose: Optional[np.ndarray] = None  # [x, y, z, qx, qy, qz, qw]
        self.gripper_value: float = 0.0
        self.control_active: bool = False

        # IK 求解器
        self.robot_urdf_path = robot_urdf_path
        self.placo_robot = None
        self.placo_solver = None

        # 关节数据
        self.current_joint_positions: Optional[np.ndarray] = None
        self.target_joint_positions: Optional[np.ndarray] = None

        self.lock = threading.Lock()
        self.running = False

        # 初始化 XRoboToolkit
        if XRT_AVAILABLE:
            xrt.init()
            logger.info("XRoboToolkit SDK initialized")
        else:
            logger.error("XRoboToolkit SDK not available")

        # 初始化 Placo IK 求解器
        if PLACO_AVAILABLE and robot_urdf_path:
            self._setup_placo()
        else:
            logger.error("Placo not available or URDF path not provided")

        # 启动 VR 数据读取线程
        self.vr_thread = threading.Thread(target=self._vr_update_loop, daemon=True)

    def _setup_placo(self):
        """设置 Placo IK 求解器"""
        try:
            # 加载机器人模型
            self.placo_robot = placo.RobotWrapper(self.robot_urdf_path)

            # 创建求解器
            self.placo_solver = placo.KinematicsSolver(self.placo_robot)

            # 获取关节索引
            joint_names = self.config.get("joint_names", [f"joint{i}" for i in range(1, 7)])
            self.placo_arm_joint_slice = slice(
                self.placo_robot.get_joint_offset(joint_names[0]),
                self.placo_robot.get_joint_offset(joint_names[-1]) + 1,
            )

            # 初始化关节位置为零位
            self.current_joint_positions = np.zeros(len(joint_names))
            self.placo_robot.state.q[self.placo_arm_joint_slice] = self.current_joint_positions

            logger.info("Placo IK solver initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Placo: {e}")
            self.placo_robot = None
            self.placo_solver = None

    def _vr_update_loop(self):
        """VR 数据更新循环"""
        if not XRT_AVAILABLE:
            return

        while self.running:
            try:
                # 获取 VR 控制器位姿
                controller_name = self.config.get("vr_controller", "right_controller")
                pose = xrt.get_right_controller_pose() if "right" in controller_name else xrt.get_left_controller_pose()

                # 获取握持键状态（激活控制）
                grip_trigger = self.config.get("control_trigger", "right_grip")
                grip_value = xrt.get_right_grip() if "right" in grip_trigger else xrt.get_left_grip()

                # 获取扳机键状态（夹爪控制）
                gripper_trigger = self.config.get("gripper_trigger", "right_trigger")
                trigger_value = xrt.get_right_trigger() if "right" in gripper_trigger else xrt.get_left_trigger()

                with self.lock:
                    self.vr_pose = pose
                    self.control_active = grip_value > 0.5  # 握持键按下激活控制
                    self.gripper_value = trigger_value

            except Exception as e:
                logger.error(f"Error reading VR data: {e}")

            time.sleep(0.01)  # 100Hz 更新频率

    def _solve_ik(self, target_pose: np.ndarray) -> Optional[np.ndarray]:
        """
        使用 Placo 求解 IK

        Args:
            target_pose: 目标末端位姿 [x, y, z, qx, qy, qz, qw]

        Returns:
            关节角度数组，如果求解失败返回 None
        """
        if not PLACO_AVAILABLE or self.placo_solver is None:
            return None

        try:
            # 更新当前关节状态
            if self.current_joint_positions is not None:
                self.placo_robot.state.q[self.placo_arm_joint_slice] = self.current_joint_positions

            # 设置末端执行器目标
            end_effector_link = self.config.get("end_effector_link", "link6")

            # 提取位置和四元数
            position = target_pose[:3]
            quaternion = target_pose[3:]  # [qx, qy, qz, qw]

            # 转换为旋转矩阵
            from scipy.spatial.transform import Rotation as R
            rotation_matrix = R.from_quat(quaternion).as_matrix()

            # 设置任务
            task = self.placo_solver.add_frame_task(end_effector_link, position, rotation_matrix)
            task.configure(end_effector_link, "soft", 1.0, 1.0)

            # 求解
            self.placo_solver.solve(True)

            # 获取求解结果
            q_des = self.placo_robot.state.q[self.placo_arm_joint_slice].copy()

            return q_des

        except Exception as e:
            logger.error(f"IK solve failed: {e}")
            return None

    def start(self):
        """启动节点"""
        if self.running:
            logger.warning(f"{self} is already running.")
            return

        self.running = True
        self.vr_thread.start()

        logger.info(f"{self} started. Waiting for VR data...")

    def get_action(self) -> Optional[Dict[str, float]]:
        """
        获取当前动作（关节角度）

        Returns:
            关节角度字典，如果未激活或求解失败返回 None
        """
        with self.lock:
            if not self.control_active or self.vr_pose is None:
                return None

            # 应用缩放因子
            scale_factor = self.config.get("scale_factor", 1.5)
            scaled_pose = self.vr_pose.copy()
            scaled_pose[:3] *= scale_factor  # 缩放位置

            # 求解 IK
            joint_positions = self._solve_ik(scaled_pose)

            if joint_positions is None:
                return None

            # 构造动作字典
            joint_names = self.config.get("joint_names", [f"joint{i}" for i in range(1, 7)])
            action = {f"{name}.pos": pos for name, pos in zip(joint_names, joint_positions)}

            # 添加夹爪
            action["gripper.pos"] = self.gripper_value

            return action

    def update_current_joints(self, joint_positions: np.ndarray):
        """更新当前关节位置（用于 IK 求解的初始状态）"""
        with self.lock:
            self.current_joint_positions = joint_positions.copy()

    def stop(self):
        """停止节点"""
        self.running = False
        if XRT_AVAILABLE:
            xrt.close()
        logger.info(f"{self} stopped.")

