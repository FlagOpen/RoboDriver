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
    
    轮子顺序固定为：
        0: 左前 (Front-Left, FL)
        1: 右前 (Front-Right, FR)
        2: 右后 (Rear-Right, RR)
        3: 左后 (Rear-Left, RL)
    """

    def __init__(self, layout_type='x45', L=0.2, W=0.2):
        """
        初始化运动学参数
        
        Args:
            layout_type: 布局类型
                - 'x45': X型45°布局（最常见）
                - 'plus': +型布局（前后左右）
                - 'custom': 自定义布局（需后续手动设置 wheel_directions）
            L: 底盘长度的一半（前后方向，单位：米）
            W: 底盘宽度的一半（左右方向，单位：米）
        """
        self.L = L  # 前后方向半长（从中心到前/后轮轴）
        self.W = W  # 左右方向半宽（从中心到左/右轮轴）

        # 轮子位置（机器人坐标系：X向前，Y向左）
        # 顺序：[左前, 右前, 右后, 左后]
        self.wheel_positions = np.array([
            [ L,  W],   # 0: 左前 (FL)
            [ L, -W],   # 1: 右前 (FR)
            [-L, -W],   # 2: 右后 (RR)
            [-L,  W]    # 3: 左后 (RL)
        ])

        # 根据布局类型设置轮子滚动方向（单位向量，指向自由滚动方向）
        if layout_type == 'x45':
            s = np.sqrt(2) / 2
            self.wheel_directions = np.array([
                [-s,  s],   # 左前:（X负，Y正）
                [ s,  s],   # 右前:（X正，Y正）
                [ s, -s],   # 右后:（X正，Y负）
                [-s, -s]    # 左后:（X负，Y负）
            ])
        elif layout_type == 'plus':
            self.wheel_directions = np.array([
                [1, 0],     # 左前: 向前滚动（+X）
                [0, -1],    # 右前: 向左滚动（-Y）
                [-1, 0],    # 右后: 向后滚动（-X）
                [0, 1]      # 左后: 向右滚动（+Y）
            ])
        elif layout_type == 'custom':
            self.wheel_directions = None  # 需外部设置
        else:
            raise ValueError(f"未知的布局类型: {layout_type}")

        # 计算运动学雅可比矩阵 J 和其伪逆
        self.J = self._calculate_jacobian()
        self.J_pinv = np.linalg.pinv(self.J)

    def _calculate_jacobian(self):
        """计算运动学雅可比矩阵 J，满足 v_wheel = J @ [vx, vy, omega]^T"""
        if self.wheel_directions is None:
            # 对于 custom 布局，暂时用零向量（用户应自行设置后重建）
            directions = np.zeros((4, 2))
        else:
            directions = self.wheel_directions

        J = np.zeros((4, 3))
        for i in range(4):
            rx, ry = self.wheel_positions[i]
            dx, dy = directions[i]
            J[i, 0] = dx                    # vx 分量
            J[i, 1] = dy                    # vy 分量
            J[i, 2] = dy * rx - dx * ry     # omega 分量（绕Z轴）
        return J

    def wheels_to_chassis(self, wheel_speeds):
        """
        正运动学：由轮速计算底盘速度
        
        Args:
            wheel_speeds: [v_fl, v_fr, v_rr, v_rl]
                v_fl: 左前轮速度
                v_fr: 右前轮速度
                v_rr: 右后轮速度
                v_rl: 左后轮速度
                （正值表示沿 wheel_directions 定义的方向运动）
        
        Returns:
            [vx, vy, omega]: 机器人在本体坐标系下的速度
                vx: 前进速度 (m/s)
                vy: 横向速度 (m/s)，向左为正
                omega: 角速度 (rad/s)，逆时针为正
        """
        wheel_speeds = np.asarray(wheel_speeds).flatten()
        if wheel_speeds.size != 4:
            raise ValueError("wheel_speeds 必须包含4个元素：[左前, 右前, 右后, 左后]")
        return self.J_pinv @ wheel_speeds

    def chassis_to_wheels(self, vx, vy, omega):
        """
        逆运动学：由底盘速度计算所需轮速
        
        Args:
            vx: 前进速度 (m/s)
            vy: 横向速度 (m/s)
            omega: 角速度 (rad/s)
        
        Returns:
            [v_fl, v_fr, v_rr, v_rl]: 四个轮子的目标速度
        """
        chassis_vel = np.array([vx, vy, omega])
        return self.J @ chassis_vel

    def set_custom_directions(self, directions):
        """
        为 'custom' 布局设置轮子方向（必须按 [FL, FR, RR, RL] 顺序）
        
        Args:
            directions: shape (4, 2) 的数组，每行为 [dx, dy] 单位向量
        """
        directions = np.asarray(directions)
        if directions.shape != (4, 2):
            raise ValueError("directions 必须是 (4, 2) 形状的数组")
        self.wheel_directions = directions
        self.J = self._calculate_jacobian()
        self.J_pinv = np.linalg.pinv(self.J)

    def get_condition_number(self):
        """返回运动学矩阵 J 的条件数（越小越好）"""
        return np.linalg.cond(self.J)

    def print_configuration(self):
        """打印当前运动学配置"""
        print("四轮万向轮运动学配置")
        print(f"布局类型: {self.layout_type if hasattr(self, 'layout_type') else 'custom'}")
        print(f"底盘尺寸: L={self.L:.3f} m, W={self.W:.3f} m")
        print("\n轮子顺序: [0]左前(FL), [1]右前(FR), [2]右后(RR), [3]左后(RL)")
        print("\n轮子位置 (X向前, Y向左):")
        names = ["左前", "右前", "右后", "左后"]
        for i in range(4):
            x, y = self.wheel_positions[i]
            print(f"  [{i}] {names[i]}: ({x:+.3f}, {y:+.3f}) m")

        if self.wheel_directions is not None:
            print("\n轮子滚动方向 (单位向量):")
            for i in range(4):
                dx, dy = self.wheel_directions[i]
                print(f"  [{i}] {names[i]}: [{dx:+.3f}, {dy:+.3f}]")
        else:
            print("\n轮子滚动方向: 未设置 (custom 模式)")

        print("\n运动学矩阵 J (4x3):")
        print(self.J)
        print(f"\n条件数: {self.get_condition_number():.3f}")

        # 检查数值稳定性
        det_val = np.linalg.det(self.J.T @ self.J)
        print(f"JᵀJ 行列式: {det_val:.6e}")
        if abs(det_val) < 1e-10:
            print("⚠️ 警告: JᵀJ 接近奇异，运动学可能不稳定！")


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


# def replay_parquet(
#     parquet_path: str,
#     gripper_scale: float = 100.0,
#     chassis_control: bool = False,
# ) -> None:
#     """
#     从 parquet 文件回放机器人动作
    
#     Args:
#         parquet_path: parquet 文件路径
#         gripper_scale: 夹爪缩放比例
#         chassis_control: 是否控制底盘速度
#     """
#     logger.info(f"开始回放文件: {parquet_path}")
#     logger.info(f"gripper_scale={gripper_scale}, chassis_control={chassis_control}")
    
#     # 初始化机器人
#     robot = GalbotRobot.get_instance()
#     robot.init()
#     time.sleep(1.0)
#     logger.info("机器人初始化完成")
    
#     # 读取数据
#     df = pd.read_parquet(parquet_path)
#     logger.info(f"读取到 {len(df)} 条记录")
    
#     # 创建运动学对象（用于底盘速度计算）
#     chassis_controller = None
#     if chassis_control:
#         kin = FourOmniWheelKinematics(layout_type='x45', L=0.4, W=0.4)
#         kin.print_configuration()
        
#         # 收集底盘速度数据
#         velocity_data = []
    
#     # 构建轨迹
#     traj = Trajectory()
#     traj.joint_groups = ["leg", "head", "left_arm", "right_arm", "left_gripper", "right_gripper"]
#     traj.joint_names = []
    
#     points = []
    
#     for i, row in df.iterrows():
#         try:
#             # 转换动作数据
#             action = to_list(row["action"])
#             parts = split_action_31(action)
            
#             # 计算时间戳
#             timestamp = float(row["timestamp"])
            
#             # 生成轨迹点
#             trajectory_point = generate_trajectory_point(parts, timestamp, gripper_scale)
#             points.append(trajectory_point)
            
#             # 收集底盘速度数据（如果启用）
#             if chassis_control:
#                 # 计算底盘速度
#                 chassis_vel = kin.wheels_to_chassis(parts["chassis_vel"])
#                 velocity_data.append((timestamp, chassis_vel.tolist() if hasattr(chassis_vel, 'tolist') else chassis_vel))
                
#                 if i % 10 == 0:
#                     logger.debug(f"时间 {timestamp:.3f}s: 底盘速度 vx={chassis_vel[0]:.3f}, "
#                                f"vy={chassis_vel[1]:.3f}, omega={chassis_vel[2]:.3f}")
            
#             if i % 10 == 0:
#                 logger.info(f"已处理 {i+1}/{len(df)} 条记录")
                
#         except Exception as e:
#             logger.error(f"处理第 {i} 条记录时出错: {e}")
#             continue
    
#     # 设置轨迹点
#     traj.points = points
#     logger.info(f"轨迹构建完成，共 {len(traj.points)} 个点")
    
#     # 创建底盘控制器（如果启用）
#     if chassis_control and velocity_data:
#         chassis_controller = ChassisController(robot, velocity_data)
    
#     # 执行轨迹
#     logger.info("开始执行关节轨迹...")
#     try:
#         # 如果启用了底盘控制，先启动底盘控制器
#         if chassis_controller:
#             trajectory_start_time = time.time()
#             chassis_controller.start(trajectory_start_time)
            
#         # 执行关节轨迹（阻塞调用）
#         robot.execute_joint_trajectory(traj, False)
#         logger.info("关节轨迹执行完成")
        
#     except Exception as e:
#         logger.error(f"执行关节轨迹时出错: {e}")
        
#     finally:
#         # 无论成功与否，都停止底盘控制器
#         if chassis_controller:
#             chassis_controller.stop()
    
#     logger.info("[replay] 回放完成")


def replay_parquet(
    parquet_path: str,
    gripper_scale: float = 100.0,
    chassis_control: bool = False,
) -> None:
    logger.info(f"开始回放文件: {parquet_path}")
    
    # 初始化机器人
    robot = GalbotRobot.get_instance()
    robot.init()
    time.sleep(1.5)  # 增加初始化等待时间
    logger.info("机器人初始化完成")
    
    # 读取数据
    df = pd.read_parquet(parquet_path)
    logger.info(f"读取到 {len(df)} 条记录")
    
    # 初始化运动学（仅当需要底盘控制时）
    kin = None
    if chassis_control:
        kin = FourOmniWheelKinematics(layout_type='x45', L=0.2, W=0.2)
        kin.print_configuration()
    
    # 构建轨迹点 + 底盘速度数据（同步存储）
    traj = Trajectory()
    traj.joint_groups = ["leg", "head", "left_arm", "right_arm", "left_gripper", "right_gripper"]
    # traj.points = []
    point_list=[]
    
    chassis_velocities = []  # 同步存储底盘速度 [timestamp, vx, vy, omega]
    
    for i, row in df.iterrows():
        try:
            action = to_list(row["action"])
            parts = split_action_31(action)
            timestamp = float(row["timestamp"])
            
            # 生成轨迹点
            point_list.append(generate_trajectory_point(parts, timestamp, gripper_scale))
            
            # 计算并验证底盘速度
            if chassis_control and kin:
                wheel_speeds = parts["chassis_vel"]
                chassis_vel = kin.wheels_to_chassis(wheel_speeds)
                
                # 严格验证速度有效性
                if np.any(np.isnan(chassis_vel)) or np.any(np.isinf(chassis_vel)):
                    logger.warning(f"时间 {timestamp:.3f}s: 检测到无效底盘速度 {chassis_vel}，使用零速度替代")
                    chassis_vel = np.zeros(3)
                
                # 限制速度范围（防止过大值）
                max_speed = 2.0  # m/s
                max_omega = 3.0  # rad/s
                chassis_vel = np.clip(chassis_vel, [-max_speed, -max_speed, -max_omega], 
                                     [max_speed, max_speed, max_omega])
                
                chassis_velocities.append((timestamp, chassis_vel.tolist()))
            
            if i % 20 == 0:
                logger.info(f"已处理 {i+1}/{len(df)} 条记录")
                
        except Exception as e:
            logger.error(f"处理第 {i} 条记录时出错: {e}", exc_info=True)
            continue

    traj.points = point_list
    
    logger.info(f"轨迹构建完成，共 {len(traj.points)} 个点")
    
    # ===== 单线程同步执行方案（关键改进）=====
    logger.info("开始同步执行关节轨迹和底盘控制...")
    
    try:
        # 先发送零速度确保安全
        if chassis_control and chassis_velocities:
            _safe_set_base_velocity(robot, [0.0, 0.0, 0.0])
            time.sleep(0.1)
        
        # 使用 monotonic 时间确保时间单调递增
        start_time = time.monotonic()
        chassis_idx = 0
        # total_points = len(traj.points)
        
        # 执行轨迹（非阻塞模式）+ 同步底盘控制
        robot.execute_joint_trajectory(traj, is_blocking=False)
        
        # 主循环：同步发送底盘速度
        while chassis_idx < len(chassis_velocities):
            current_time = time.monotonic() - start_time
            target_time, velocity = chassis_velocities[chassis_idx]
            
            # 当到达目标时间点时发送速度
            if current_time >= target_time:
                _safe_set_base_velocity(robot, velocity)
                chassis_idx += 1
            
            time.sleep(0.005)  # 5ms 控制周期
        
        logger.info("关节轨迹执行完成")
        
    except Exception as e:
        logger.error(f"执行过程中出错: {e}", exc_info=True)
        # 尝试安全停止
        try:
            if chassis_control:
                _safe_set_base_velocity(robot, [0.0, 0.0, 0.0])
        except:
            pass
    
    finally:
        # 确保底盘停止
        if chassis_control:
            _safe_set_base_velocity(robot, [0.0, 0.0, 0.0])
            time.sleep(0.2)
    
    logger.info("[replay] 回放完成")


def _safe_set_base_velocity(robot: GalbotRobot, velocity: List[float]):
    """安全发送底盘速度命令（带参数验证）"""
    # 验证输入
    if not isinstance(velocity, (list, tuple)) or len(velocity) != 3:
        logger.warning(f"无效速度向量（长度错误）: {velocity}")
        return False
    
    # 检查 NaN/Inf
    if any(np.isnan(v) or np.isinf(v) for v in velocity):
        logger.warning(f"速度包含无效值: {velocity}")
        return False
    
    # 限制合理范围
    velocity = [
        np.clip(velocity[0], -2.0, 2.0),
        np.clip(velocity[1], -2.0, 2.0),
        np.clip(velocity[2], -3.0, 3.0)
    ]
    
    try:
        linear = [velocity[0], velocity[1], 0.0]
        angular = [0.0, 0.0, velocity[2]]
        
        # 再次验证计算结果
        if any(np.isnan(v) or np.isinf(v) for v in linear + angular):
            logger.warning(f"计算出的速度命令包含无效值: {linear}, {angular}")
            return False
        
        status = robot.set_base_velocity(linear, angular)
        if status != ControlStatus.SUCCESS:
            logger.debug(f"设置底盘速度警告，状态: {status}")
        return status == ControlStatus.SUCCESS
        
    except Exception as e:
        logger.warning(f"发送底盘速度时异常: {e}")
        return False

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