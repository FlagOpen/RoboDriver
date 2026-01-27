#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Galbot 机器人回放脚本
用于从 parquet 文件回放机器人动作轨迹
"""

import time
import json
import argparse
import logging
import threading
from typing import Dict, List, Any, Tuple, Optional
import pandas as pd
import numpy as np

from pygalbot import GalbotRobot, ControlStatus, Trajectory, TrajectoryPoint, JointCommand
# from chassis_kinematics import FourOmniWheelKinematics

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def to_list(x: Any) -> List[float]:
    """
    将 parquet 中的 action 单元转换为 Python list
    
    Args:
        x: 输入数据，可能是 list、numpy array 或 JSON 字符串
        
    Returns:
        List[float]: 转换后的列表
        
    Raises:
        TypeError: 当输入类型不支持时
    """
    if isinstance(x, list):
        return x
    if hasattr(x, "tolist"):
        return x.tolist()
    if isinstance(x, str):
        try:
            return json.loads(x)
        except json.JSONDecodeError:
            raise ValueError(f"无法解析 JSON 字符串: {x[:50]}...")
    raise TypeError(f"不支持的动作类型: {type(x)}")


def split_action_31(action: List[float]) -> Dict[str, List[float]]:
    """
    将 31 维动作向量拆分为各个部分
    
    Args:
        action: 31 维动作向量
        
    Returns:
        Dict[str, List[float]]: 拆分后的动作部分
        
    Raises:
        ValueError: 当动作维度不正确时
    """
    if len(action) != 31:
        raise ValueError(f"期望 31 维动作，实际得到 {len(action)} 维")
    
    return {
        "right_arm": action[0:7],
        "right_gripper": [action[7]],
        "left_arm": action[8:15],
        "left_gripper": [action[15]],
        "leg": action[16:21],
        "head": action[21:23],
        "chassis_vel": action[27:31],
    }


def generate_trajectory_point(
    parts: Dict[str, List[float]], 
    time_from_start_second: float, 
    gripper_scale: float = 100.0
) -> TrajectoryPoint:
    """
    生成单个轨迹点
    
    Args:
        parts: 拆分后的动作部分
        time_from_start_second: 从轨迹开始的时间（秒）
        gripper_scale: 夹爪缩放比例
        
    Returns:
        TrajectoryPoint: 轨迹点对象
    """
    # 构建关节位置向量，按照机器人要求的顺序
    joint_pos_vec = []
    
    # 顺序: leg -> head -> left_arm -> right_arm -> left_gripper -> right_gripper
    joint_pos_vec.extend(parts["leg"])
    joint_pos_vec.extend(parts["head"])
    joint_pos_vec.extend(parts["left_arm"])
    joint_pos_vec.extend(parts["right_arm"])
    joint_pos_vec.append(parts["left_gripper"][0] * gripper_scale / 1000)
    joint_pos_vec.append(parts["right_gripper"][0] * gripper_scale / 1000)
    
    # 创建轨迹点
    trajectory_point = TrajectoryPoint()
    trajectory_point.time_from_start_second = time_from_start_second
    
    # 创建关节命令
    joint_command_vec = []
    for pos in joint_pos_vec:
        joint_cmd = JointCommand()
        joint_cmd.position = pos
        joint_command_vec.append(joint_cmd)
    
    trajectory_point.joint_command_vec = joint_command_vec
    return trajectory_point

class FourOmniWheelKinematics:
    """
    四轮万向轮底盘运动学
    用于从四个轮子的速度计算机器人本体的运动速度
    """
    
    def __init__(self, layout_type='x45', L=0.2, W=0.2):
        """
        初始化运动学参数
        
        Args:
            layout_type: 布局类型
                - 'x45': X型45°布局（最常见）
                - 'plus': +型布局（前后左右）
                - 'custom': 自定义布局
            L: 底盘长度的一半（前后方向，单位：米）
            W: 底盘宽度的一半（左右方向，单位：米）
        """
        self.L = L  # 前后方向半长
        self.W = W  # 左右方向半宽
        
        # 轮子位置（机器人坐标系：X向前，Y向左）
        # 轮子编号：1-右前，2-左前，3-左后，4-右后
        self.wheel_positions = np.array([
            [L, -W],   # 轮子1: 右前
            [L, W],    # 轮子2: 左前
            [-L, W],   # 轮子3: 左后
            [-L, -W]   # 轮子4: 右后
        ])
        
        # 根据布局类型设置轮子滚动方向
        if layout_type == 'x45':
            # X型45°布局：轮子滚动方向与底盘轴线成45°
            # 轮子方向单位向量（指向轮子自由滚动方向）
            self.wheel_directions = np.array([
                [np.sqrt(2)/2, -np.sqrt(2)/2],   # 右前: 45°方向
                [np.sqrt(2)/2, np.sqrt(2)/2],    # 左前: 135°方向
                [-np.sqrt(2)/2, np.sqrt(2)/2],   # 左后: 225°方向
                [-np.sqrt(2)/2, -np.sqrt(2)/2]   # 右后: 315°方向
            ])
            
        elif layout_type == 'plus':
            # +型布局：每个轮子正对前后左右方向
            self.wheel_directions = np.array([
                [0, -1],   # 右前: 向左滚动
                [1, 0],    # 左前: 向前滚动
                [0, 1],    # 左后: 向右滚动
                [-1, 0]    # 右后: 向后滚动
            ])
            
        elif layout_type == 'custom':
            # 自定义布局 - 需要后续设置
            self.wheel_directions = None
            
        else:
            raise ValueError(f"未知的布局类型: {layout_type}")
        
        # 计算运动学矩阵J和它的伪逆
        self.J = self._calculate_jacobian()
        self.J_pinv = np.linalg.pinv(self.J)  # 伪逆
        
    def _calculate_jacobian(self):
        """
        计算运动学雅可比矩阵J
        v_wheel = J * [vx, vy, omega]^T
        """
        J = np.zeros((4, 3))
        
        for i in range(4):
            # 轮子位置
            rx, ry = self.wheel_positions[i]
            # 轮子方向
            dx, dy = self.wheel_directions[i]
            
            # 前三列对应vx, vy, omega
            J[i, 0] = dx  # vx系数
            J[i, 1] = dy  # vy系数
            J[i, 2] = dy * rx - dx * ry  # omega系数
            
        return J
    
    def wheels_to_chassis(self, wheel_speeds):
        """
        从四个轮子的速度计算底盘运动速度
        
        Args:
            wheel_speeds: 四个轮子的速度数组 [v1, v2, v3, v4]
                          v1: 右前轮，v2: 左前轮，v3: 左后轮，v4: 右后轮
                          正值表示沿着轮子滚动方向运动
                          
        Returns:
            chassis_velocity: [vx, vy, omega]
                vx: 机器人前进速度 (m/s)
                vy: 机器人横向速度 (m/s)
                omega: 机器人旋转角速度 (rad/s)，逆时针为正
        """
        wheel_speeds = np.array(wheel_speeds).flatten()
        
        if len(wheel_speeds) != 4:
            raise ValueError("轮子速度数组必须是4个元素")
        
        # 使用伪逆计算底盘速度
        chassis_velocity = self.J_pinv @ wheel_speeds
        
        return chassis_velocity
    
    def chassis_to_wheels(self, vx, vy, omega):
        """
        从底盘速度计算四个轮子应有的速度（逆运动学）
        
        Args:
            vx: 机器人前进速度 (m/s)
            vy: 机器人横向速度 (m/s)
            omega: 机器人旋转角速度 (rad/s)
            
        Returns:
            wheel_speeds: 四个轮子应有的速度 [v1, v2, v3, v4]
        """
        chassis_velocity = np.array([vx, vy, omega])
        wheel_speeds = self.J @ chassis_velocity
        
        return wheel_speeds
    
    def get_condition_number(self):
        """
        获取运动学矩阵的条件数
        条件数越小，数值稳定性越好
        """
        cond = np.linalg.cond(self.J)
        return cond
    
    def print_configuration(self):
        """打印当前配置信息"""
        print("四轮万向轮运动学配置:")
        print(f"底盘尺寸: L={self.L:.3f}m, W={self.W:.3f}m")
        print("\n轮子位置 (机器人坐标系，X向前，Y向左):")
        for i in range(4):
            print(f"  轮子{i+1}: ({self.wheel_positions[i, 0]:.3f}, {self.wheel_positions[i, 1]:.3f}) m")
        
        print("\n轮子滚动方向 (单位向量):")
        for i in range(4):
            print(f"  轮子{i+1}: [{self.wheel_directions[i, 0]:.3f}, {self.wheel_directions[i, 1]:.3f}]")
        
        print("\n运动学矩阵 J (4x3):")
        print(self.J)
        
        print(f"\n条件数: {self.get_condition_number():.3f}")
        
        # 验证可逆性
        JtJ = self.J.T @ self.J
        det = np.linalg.det(JtJ)
        print(f"J^T*J的行列式: {det:.6f}")
        if abs(det) < 1e-10:
            print("警告: J^T*J接近奇异，运动学可能不稳定")


class ChassisController:
    """底盘速度控制器，用于同步控制底盘运动"""
    
    def __init__(self, robot: GalbotRobot, velocity_data: List[Tuple[float, List[float]]]):
        """
        初始化底盘控制器
        
        Args:
            robot: 机器人实例
            velocity_data: 速度数据列表，每个元素为 (时间戳, [vx, vy, omega])
        """
        self.robot = robot
        self.velocity_data = velocity_data
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.start_time: Optional[float] = None
        self.stop_event = threading.Event()
        
    def start(self, start_time: float):
        """
        开始底盘控制线程
        
        Args:
            start_time: 轨迹开始的系统时间
        """
        if self.running:
            logger.warning("底盘控制器已在运行")
            return
            
        self.start_time = start_time
        self.running = True
        self.stop_event.clear()
        
        self.thread = threading.Thread(target=self._control_loop, daemon=True)
        self.thread.start()
        logger.info("底盘控制线程已启动")
        
    def stop(self):
        """停止底盘控制"""
        if not self.running:
            return
            
        self.running = False
        self.stop_event.set()
        
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)
            
        # 发送停止指令
        try:
            linear_velocity = [0.0, 0.0, 0.0]
            angular_velocity = [0.0, 0.0, 0.0]
            self.robot.set_base_velocity(linear_velocity, angular_velocity)
            logger.info("底盘已停止")
        except Exception as e:
            logger.warning(f"停止底盘时出错: {e}")
            
    def _control_loop(self):
        """底盘控制主循环"""
        try:
            data_index = 0
            last_velocity = None
            
            while self.running and not self.stop_event.is_set():
                current_time = time.time() - self.start_time
                
                # 找到当前时间对应的速度
                while (data_index < len(self.velocity_data) and 
                       self.velocity_data[data_index][0] < current_time):
                    data_index += 1
                
                if data_index >= len(self.velocity_data):
                    # 所有数据已发送完毕，保持最后一个速度或停止
                    if last_velocity is not None:
                        # 保持最后一个速度直到轨迹结束
                        self._send_velocity(last_velocity)
                    time.sleep(0.01)  # 短暂休眠避免CPU占用过高
                    continue
                    
                # 发送当前速度
                target_time, velocity = self.velocity_data[data_index]
                self._send_velocity(velocity)
                last_velocity = velocity
                
                # 计算到下一个速度点的等待时间
                if data_index + 1 < len(self.velocity_data):
                    next_time = self.velocity_data[data_index + 1][0]
                    sleep_time = max(0, next_time - current_time - 0.001)  # 稍提前一点
                    if sleep_time > 0:
                        time.sleep(min(sleep_time, 0.1))  # 限制最大休眠时间
                else:
                    time.sleep(0.01)
                    
        except Exception as e:
            logger.error(f"底盘控制线程出错: {e}")
            self.running = False
            
    def _send_velocity(self, velocity: List[float]):
        """
        发送底盘速度命令
        
        Args:
            velocity: [vx, vy, omega] 速度向量
        """
        try:
            # 根据你的说明，linear_velocity 使用 xy，angular_velocity 使用 wz
            linear_velocity = [velocity[0], velocity[1], 0.0]  # x, y, z (z=0)
            angular_velocity = [0.0, 0.0, velocity[2]]  # wx, wy, wz (只使用wz)
            
            status = self.robot.set_base_velocity(linear_velocity, angular_velocity)
            
            if status != ControlStatus.SUCCESS:
                logger.warning(f"设置底盘速度失败，状态: {status}")
                
        except Exception as e:
            logger.warning(f"发送底盘速度命令时出错: {e}")


def replay_parquet(
    parquet_path: str,
    gripper_scale: float = 100.0,
    chassis_control: bool = False,
) -> None:
    """
    从 parquet 文件回放机器人动作
    
    Args:
        parquet_path: parquet 文件路径
        gripper_scale: 夹爪缩放比例
        chassis_control: 是否控制底盘速度
    """
    logger.info(f"开始回放文件: {parquet_path}")
    logger.info(f"gripper_scale={gripper_scale}, chassis_control={chassis_control}")
    
    # 初始化机器人
    robot = GalbotRobot.get_instance()
    robot.init()
    time.sleep(1.0)
    logger.info("机器人初始化完成")
    
    # 读取数据
    df = pd.read_parquet(parquet_path)
    logger.info(f"读取到 {len(df)} 条记录")
    
    # 创建运动学对象（用于底盘速度计算）
    chassis_controller = None
    if chassis_control:
        kin = FourOmniWheelKinematics(layout_type='x45', L=0.4, W=0.4)
        kin.print_configuration()
        
        # 收集底盘速度数据
        velocity_data = []
    
    # 构建轨迹
    traj = Trajectory()
    traj.joint_groups = ["leg", "head", "left_arm", "right_arm", "left_gripper", "right_gripper"]
    traj.joint_names = []
    
    points = []
    
    for i, row in df.iterrows():
        try:
            # 转换动作数据
            action = to_list(row["action"])
            parts = split_action_31(action)
            
            # 计算时间戳
            timestamp = float(row["timestamp"])
            
            # 生成轨迹点
            trajectory_point = generate_trajectory_point(parts, timestamp, gripper_scale)
            points.append(trajectory_point)
            
            # 收集底盘速度数据（如果启用）
            if chassis_control:
                # 计算底盘速度
                chassis_vel = kin.wheels_to_chassis(parts["chassis_vel"])
                velocity_data.append((timestamp, chassis_vel.tolist() if hasattr(chassis_vel, 'tolist') else chassis_vel))
                
                if i % 100 == 0:
                    logger.debug(f"时间 {timestamp:.3f}s: 底盘速度 vx={chassis_vel[0]:.3f}, "
                               f"vy={chassis_vel[1]:.3f}, omega={chassis_vel[2]:.3f}")
            
            if i % 100 == 0:
                logger.info(f"已处理 {i+1}/{len(df)} 条记录")
                
        except Exception as e:
            logger.error(f"处理第 {i} 条记录时出错: {e}")
            continue
    
    # 设置轨迹点
    traj.points = points
    logger.info(f"轨迹构建完成，共 {len(traj.points)} 个点")
    
    # 创建底盘控制器（如果启用）
    if chassis_control and velocity_data:
        chassis_controller = ChassisController(robot, velocity_data)
    
    # 执行轨迹
    logger.info("开始执行关节轨迹...")
    try:
        # 如果启用了底盘控制，先启动底盘控制器
        if chassis_controller:
            trajectory_start_time = time.time()
            chassis_controller.start(trajectory_start_time)
            
        # 执行关节轨迹（阻塞调用）
        robot.execute_joint_trajectory(traj, False)
        logger.info("关节轨迹执行完成")
        
    except Exception as e:
        logger.error(f"执行关节轨迹时出错: {e}")
        
    finally:
        # 无论成功与否，都停止底盘控制器
        if chassis_controller:
            chassis_controller.stop()
    
    logger.info("[replay] 回放完成")


def main() -> None:
    """主函数"""
    parser = argparse.ArgumentParser(
        description="Galbot 机器人回放脚本 - 从 parquet 文件回放动作轨迹",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  %(prog)s --file /path/to/episode.parquet
  %(prog)s --file /path/to/episode.parquet --chassis-control
        """
    )
    
    parser.add_argument(
        "--file", 
        required=True, 
        help="episode parquet 文件路径"
    )
    parser.add_argument(
        "--gripper-scale", 
        type=float, 
        default=100.0, 
        help="夹爪缩放比例（默认: 100.0）"
    )
    parser.add_argument(
        "--chassis-control", 
        action="store_true", 
        help="启用底盘速度控制（实验性功能）"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true", 
        help="启用详细日志输出"
    )
    
    args = parser.parse_args()
    
    # 设置日志级别
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # 执行回放
    replay_parquet(
        parquet_path=args.file,
        gripper_scale=args.gripper_scale,
        chassis_control=args.chassis_control,
    )


if __name__ == "__main__":
    main()