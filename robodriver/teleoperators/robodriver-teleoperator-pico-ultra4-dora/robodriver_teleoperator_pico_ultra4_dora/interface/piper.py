"""
松灵 Piper 机械臂硬件接口

提供与 Piper 机械臂的底层通信接口，包括：
- 关节位置控制
- 关节状态读取
- 夹爪控制
- 安全保护功能

基于 Piper SDK V2 API
"""

import time
import numpy as np
from typing import List, Optional, Union
from piper_sdk import C_PiperInterface_V2


class PiperInterface:
    """
    松灵 Piper 机械臂硬件接口类

    使用 CAN 总线与 Piper 机械臂通信
    """

    def __init__(
        self,
        can_port: str = "can0",
        dt: float = 0.02,
        filter_alpha: float = 0.6,
    ):
        """
        初始化 Piper 接口

        Args:
            can_port: CAN 端口名称（如 "can0", "can1"）
            dt: 控制周期（秒），默认 50Hz
            filter_alpha: 一阶低通滤波器系数 (0-1)，越小越平滑但延迟越大，默认 0.6
        """
        self.can_port = can_port
        self.dt = dt
        self.num_joints = 6  # Piper 有 6 个关节

        # 角度转换因子：SDK 使用的单位转换
        # factor = 1000 * 180 / π ≈ 57295.7795
        self.angle_factor = 57295.7795

        # 夹爪转换因子：SDK 使用 μm 单位
        self.gripper_factor = 1000 * 1000  # m to μm

        # 一阶低通滤波器
        self.filter_alpha = filter_alpha
        self.filtered_positions = None  # 滤波后的关节位置
        self.filtered_gripper = None    # 滤波后的夹爪位置

        # 初始化 Piper SDK
        print(f"Initializing Piper on CAN port: {can_port}")
        self.piper = C_PiperInterface_V2(can_port)

        # 连接端口
        self.piper.ConnectPort()
        print("Piper CAN port connected")

        # 使能机械臂
        self._enable_and_wait()

    def _enable_and_wait(self, timeout: float = 5.0):
        """
        使能机械臂并等待就绪

        Args:
            timeout: 超时时间（秒）
        """
        print("Enabling Piper robot...")
        start_time = time.time()

        while not self.piper.EnablePiper():
            if time.time() - start_time > timeout:
                raise TimeoutError("Failed to enable Piper robot within timeout")
            time.sleep(0.01)

        print("Piper robot enabled successfully")

    def go_home(self) -> bool:
        """
        移动到预定义的 Home 位置（零位）

        Returns:
            bool: 成功返回 True
        """
        print("Moving Piper to home position...")

        # Home 位置：所有关节归零
        home_position = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        # 设置运动模式：位置控制模式
        # ModeCtrl(enable, mode, speed_factor, unused)
        # mode: 0x01 = 位置控制模式
        self.piper.ModeCtrl(0x01, 0x01, 30, 0x00)

        # 发送关节位置命令
        self.set_joint_positions(home_position)

        # 夹爪归零
        self.set_gripper_position(0.0)

        # 等待到达
        time.sleep(2.0)

        return True

    def get_joint_positions(self) -> np.ndarray:
        """
        获取当前关节位置

        Returns:
            np.ndarray: 关节位置数组（弧度），shape: (6,)
        """
        # 获取关节消息
        joint_msgs = self.piper.GetArmJointMsgs()

        # 提取关节位置并转换为弧度
        # SDK 返回的单位需要除以 angle_factor 转换为弧度
        positions = np.array([
            joint_msgs.joint_state.joint_1 / self.angle_factor,
            joint_msgs.joint_state.joint_2 / self.angle_factor,
            joint_msgs.joint_state.joint_3 / self.angle_factor,
            joint_msgs.joint_state.joint_4 / self.angle_factor,
            joint_msgs.joint_state.joint_5 / self.angle_factor,
            joint_msgs.joint_state.joint_6 / self.angle_factor,
        ])

        return positions

    def get_joint_velocities(self) -> np.ndarray:
        """
        获取当前关节速度

        Returns:
            np.ndarray: 关节速度数组（弧度/秒），shape: (6,)

        Note:
            Piper SDK 不直接提供速度数据，这里返回零数组
            如需速度信息，可以通过位置数值微分计算
        """
        # Piper SDK 的 GetArmJointMsgs() 不包含速度信息
        # 返回零数组
        return np.zeros(self.num_joints)

    def set_joint_positions(
        self,
        positions: Union[List[float], np.ndarray],
        speed_factor: int = 100,
        **kwargs
    ) -> bool:
        """
        设置目标关节位置

        Args:
            positions: 目标关节位置（弧度），shape: (6,)
            speed_factor: 速度因子 (0-100)，默认 100
            **kwargs: 额外参数

        Returns:
            bool: 成功返回 True
        """
        if len(positions) != self.num_joints:
            print(f"Error: Expected {self.num_joints} joints, got {len(positions)}")
            return False

        # 关节限位检查（弧度）
        joint_limits = [
            (-3.14, 3.14),   # joint1
            (-2.0, 2.0),     # joint2
            (-2.5, 2.5),     # joint3
            (-3.14, 3.14),   # joint4
            (-2.0, 2.0),     # joint5
            (-3.14, 3.14),   # joint6
        ]

        # 限制关节位置在安全范围内
        clipped_positions = []
        for i, (pos, (min_val, max_val)) in enumerate(zip(positions, joint_limits)):
            if pos < min_val or pos > max_val:
                clipped_pos = np.clip(pos, min_val, max_val)
                print(f"Warning: Joint {i+1} position {pos:.3f} clipped to {clipped_pos:.3f}")
                clipped_positions.append(clipped_pos)
            else:
                clipped_positions.append(pos)

        # 一阶低通滤波：y = alpha * x + (1 - alpha) * y_prev
        clipped_positions = np.array(clipped_positions)
        if self.filtered_positions is None:
            self.filtered_positions = clipped_positions.copy()
        else:
            self.filtered_positions = (
                self.filter_alpha * clipped_positions
                + (1.0 - self.filter_alpha) * self.filtered_positions
            )
        clipped_positions = self.filtered_positions

        # 转换为 SDK 单位
        joint_0 = round(clipped_positions[0] * self.angle_factor)
        joint_1 = round(clipped_positions[1] * self.angle_factor)
        joint_2 = round(clipped_positions[2] * self.angle_factor)
        joint_3 = round(clipped_positions[3] * self.angle_factor)
        joint_4 = round(clipped_positions[4] * self.angle_factor)
        joint_5 = round(clipped_positions[5] * self.angle_factor)

        # 设置运动控制模式
        # MotionCtrl_2(enable, mode, speed_factor, unused)
        self.piper.MotionCtrl_2(0x01, 0x01, speed_factor, 0x00)

        # 发送关节控制命令
        self.piper.JointCtrl(joint_0, joint_1, joint_2, joint_3, joint_4, joint_5)

        return True

    def set_gripper_position(self, position: float, speed: int = 1000) -> bool:
        """
        设置夹爪位置

        Args:
            position: 夹爪开合度 (0.0=完全闭合, 1.0=完全打开)
            speed: 夹爪速度 (默认 1000)

        Returns:
            bool: 成功返回 True
        """
        # 限制在 0-1 范围内
        position = np.clip(position, 0.0, 1.0)

        # 一阶低通滤波
        if self.filtered_gripper is None:
            self.filtered_gripper = position
        else:
            self.filtered_gripper = (
                self.filter_alpha * position
                + (1.0 - self.filter_alpha) * self.filtered_gripper
            )
        position = self.filtered_gripper

        # 将 0-1 映射到 Piper 夹爪的实际范围（单位：μm）
        # Piper 夹爪最大开合约 0.08m = 80000μm
        gripper_pos_um = round(position * 80000)

        # 发送夹爪控制命令
        # GripperCtrl(position_um, speed, enable, force)
        self.piper.GripperCtrl(abs(gripper_pos_um), speed, 0x01, 0)

        return True

    def get_gripper_position(self) -> float:
        """
        获取当前夹爪位置

        Returns:
            float: 夹爪开合度 (0.0-1.0)
        """
        gripper_msgs = self.piper.GetArmGripperMsgs()

        # grippers_angle 单位是 0.001 度
        # 转换为 0-1 范围（假设最大开合角度约 80 度）
        angle_degrees = gripper_msgs.gripper_state.grippers_angle / 1000.0
        gripper_pos = abs(angle_degrees) / 80.0  # 归一化到 0-1

        return np.clip(gripper_pos, 0.0, 1.0)

    def enable_robot(self) -> bool:
        """使能机械臂"""
        print("Enabling Piper robot...")
        success = self.piper.EnablePiper()
        if success:
            print("Piper robot enabled")
        return success

    def disable_robot(self) -> bool:
        """失能机械臂"""
        print("Disabling Piper robot...")
        # Piper SDK 没有直接的 disable 方法
        # 可以通过设置模式为 0 来停止控制
        self.piper.ModeCtrl(0x00, 0x00, 0, 0x00)
        print("Piper robot disabled")
        return True

    def emergency_stop(self) -> bool:
        """紧急停止"""
        print("Piper emergency stop triggered!")
        # 立即停止所有运动
        self.piper.ModeCtrl(0x00, 0x00, 0, 0x00)
        return True

    def get_arm_status(self):
        """
        获取机械臂状态信息

        Returns:
            机械臂状态对象
        """
        return self.piper.GetArmStatus()

    def __del__(self):
        """析构函数，断开连接"""
        if hasattr(self, 'piper'):
            print("Disconnecting Piper...")
            self.disable_robot()
