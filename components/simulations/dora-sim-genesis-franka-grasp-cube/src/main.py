"""
主程序 - 使用配置文件外部化参数，增强错误处理和性能监控
"""
import os
import numpy as np
import genesis as gs
import math
from keyboard_controller import KeyboardController
from config_loader import get_config_loader
from dora import Node
import cv2
import pyarrow as pa
import time
from collections import deque
import threading
from queue import Queue
import traceback
from pynput import keyboard


class FrankaSimulation:
    """Franka机械臂仿真主类"""
    
    def __init__(self, config_path: str = "config/simulation_config.yaml"):
        """
        初始化仿真环境
        
        Args:
            config_path: 配置文件路径
        """
        self.config_loader = get_config_loader(config_path)
        self.config = self.config_loader.config
        
        # 初始化变量
        self.node = None
        self.scene = None
        self.franka = None
        self.cam = None
        self.wrist_cam = None  # 腕部相机
        self.controller = None
        self.end_effector = None
        self.cube_entity = None  # 方块实体引用
        
        # 线程控制
        self.should_exit = threading.Event()
        self.step_data_queue = Queue()
        self.control_thread = None
        
        # 性能监控
        self.step_times = deque(maxlen=100)
        self.step_intervals = deque(maxlen=100)
        self.render_times = deque(maxlen=100)
        self.wrist_render_times = deque(maxlen=100)  # 腕部相机渲染时间
        self.tick_intervals = deque(maxlen=100)
        
        self.last_tick_time = time.perf_counter()
        self.print_counter = 0
        self.print_interval = self.config['performance']['print_interval']
        self.controlling = False
        self.control_joint = None
        self.control_frame = 0

        self.jnt_names = [
            'arm_joint1',
            'arm_joint2',
            'arm_joint3',
            'arm_joint4',
            'arm_joint5',
            'arm_joint6',
            'arm_joint7',
            'gripper_joint1',
            'gripper_joint2',
        ]
        
        # 错误统计
        self.ik_failures = 0
        self.total_ik_calls = 0
        
    @staticmethod
    def quat_rotate(quat, vec):
        """使用四元数旋转向量
        
        Args:
            quat: 四元数 [w, x, y, z]
            vec: 三维向量 [x, y, z]
            
        Returns:
            旋转后的向量
        """
        # 四元数旋转公式: v' = q * v * q^-1
        q = np.array(quat)
        v = np.array([0.0, vec[0], vec[1], vec[2]])  # 转换为四元数形式
        
        # 计算 q * v
        qv = np.array([
            -q[1]*v[1] - q[2]*v[2] - q[3]*v[3],
            q[0]*v[1] + q[2]*v[3] - q[3]*v[2],
            q[0]*v[2] - q[1]*v[3] + q[3]*v[1],
            q[0]*v[3] + q[1]*v[2] - q[2]*v[1]
        ])
        
        # 计算 qv * q^-1 (q^-1 = [q[0], -q[1], -q[2], -q[3]])
        q_inv = np.array([q[0], -q[1], -q[2], -q[3]])
        result = np.array([
            -qv[1]*q_inv[1] - qv[2]*q_inv[2] - qv[3]*q_inv[3],
            qv[0]*q_inv[1] + qv[2]*q_inv[3] - qv[3]*q_inv[2],
            qv[0]*q_inv[2] - qv[1]*q_inv[3] + qv[3]*q_inv[1],
            qv[0]*q_inv[3] + qv[1]*q_inv[2] - qv[2]*q_inv[1]
        ])
        
        return result[1:]  # 返回三维向量
        
    def initialize_simulation(self):
        """初始化仿真环境"""
        print("初始化仿真环境...")
        
        # 初始化Genesis引擎
        gs.init(backend=gs.gpu, logging_level="warn")
        
        # 创建Dora节点
        self.node = Node()
        
        # 获取仿真配置
        sim_config = self.config['simulation']
        robot_config = self.config['robot']
        
        # 创建场景
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(),
            viewer_options=gs.options.ViewerOptions(),
            show_viewer=True,
        )
        
        # 创建实体
        if sim_config['plane_enabled']:
            plane = self.scene.add_entity(gs.morphs.Plane())
            print("  平面实体已创建")
        
        cube_config = sim_config['cube']
        cube = self.scene.add_entity(
            gs.morphs.Box(
                size=tuple(cube_config['size']),
                pos=tuple(cube_config['position']),
            )
        )
        self.cube_entity = cube  # 保存方块引用
        print(f"  立方体实体已创建: 大小={cube_config['size']}, 位置={cube_config['position']}")
        
        # 创建Franka机械臂
        self.franka = self.scene.add_entity(gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"))
        print("  Franka机械臂已加载")
        
        # 创建相机
        cam_config = sim_config['camera']
        self.cam = self.scene.add_camera(
            res=tuple(cam_config['resolution']),
            pos=tuple(cam_config['position']),
            lookat=tuple(cam_config['lookat']),
            fov=cam_config['fov'],
            GUI=cam_config['gui_enabled']
        )
        print(f"  相机已创建: 分辨率={cam_config['resolution']}, 位置={cam_config['position']}")

        wrist_cam_config = sim_config['wrist_camera']
        self.wrist_cam = self.scene.add_camera(
            res=tuple(wrist_cam_config.get('resolution', [320, 240])),
            fov=wrist_cam_config.get('fov', 70),
            GUI=wrist_cam_config.get('gui_enabled', False)
        )
        print(f"  腕部相机已创建: 分辨率={wrist_cam_config.get('resolution', [320, 240])}")

        # 构建场景
        self.scene.build()
        print("  场景构建完成")
        
        # 配置机械臂控制参数
        self.franka.set_dofs_kp(np.array(robot_config['kp_gains']))
        self.franka.set_dofs_kv(np.array(robot_config['kv_gains']))
        self.franka.set_dofs_force_range(
            np.array(robot_config['force_min']),
            np.array(robot_config['force_max'])
        )
        print("  机械臂控制参数已配置")
        
        # 获取末端执行器
        self.end_effector = self.franka.get_link("hand")
        
        # 创建相机相对于末端执行器的变换矩阵
        # 从配置中获取相对位置和观察点
        relative_pos = np.array(wrist_cam_config.get('relative_position', [0.1, 0.0, 0.0]))
        relative_lookat = np.array(wrist_cam_config.get('relative_lookat', [0.1, 0.0, -0.1]))
        
        # 右手系，z轴向下为负
        # 计算观察方向（从相机位置指向观察点）
        view_dir = relative_lookat - relative_pos
        view_dir = view_dir / np.linalg.norm(view_dir)  # 归一化
        
        # 默认up向量（z向下为负）
        up = np.array([0.0, 0.0, -1.0])
        
        # 计算右向量
        right = np.cross(up, view_dir)
        right_norm = np.linalg.norm(right)
        if right_norm < 1e-6:
            # 如果view_dir与up平行，使用备用右向量
            right = np.array([1.0, 0.0, 0.0])
        else:
            right = right / right_norm
        
        # 重新计算实际的up向量以确保正交
        actual_up = np.cross(view_dir, right)
        actual_up = actual_up / np.linalg.norm(actual_up)
        
        # 构建旋转矩阵（相机坐标系：右向量，上向量，观察方向）
        R = np.column_stack((right, actual_up, view_dir))
        
        # 绕z轴朝下方向逆时针旋转90度
        # 从z轴负方向看（朝下），逆时针旋转90度相当于绕z轴旋转-90度
        # 旋转矩阵：R_z(-90°) = [[cos(-90°), -sin(-90°), 0],
        #                        [sin(-90°), cos(-90°), 0],
        #                        [0, 0, 1]]
        # cos(-90°) = 0, sin(-90°) = -1
        R_z = np.array([[0, 1, 0],
                        [-1, 0, 0],
                        [0, 0, 1]])
        
        # 应用额外的旋转
        R = R @ R_z
        
        # 构建变换矩阵
        offset_T = np.eye(4)
        offset_T[:3, :3] = R
        offset_T[:3, 3] = relative_pos
        
        # 将相机附加到末端执行器
        self.wrist_cam.attach(self.end_effector, offset_T)
        print(f"  腕部相机已附加到末端执行器，相对位置: {relative_pos}, 观察点: {relative_lookat}")

        
        # 初始化键盘控制器
        self.controller = KeyboardController()
        
        # 设置关节索引
        self.motors_dof = np.array(robot_config['motor_dofs'])
        self.fingers_dof = np.array(robot_config['finger_dofs'])
        
        print("仿真环境初始化完成")
        
    def move_to_initial_position(self):
        """移动到初始位置"""
        print("\n移动到初始位置...")
        
        keyboard_config = self.config['keyboard']
        motion_config = self.config['motion']
        
        initial_pos = np.array(keyboard_config['initial_position'])
        initial_euler = np.array(keyboard_config['initial_euler'])
        initial_quat = self.controller.euler_to_quat(*initial_euler)
        
        self.controller.target_pos = initial_pos.copy()
        self.controller.target_euler = initial_euler.copy()
        
        try:
            # 逆运动学计算
            qpos = self.franka.inverse_kinematics(
                link=self.end_effector,
                pos=self.controller.target_pos,
                quat=initial_quat,
            )
            
            # 路径规划
            path = self.franka.plan_path(
                qpos_goal=qpos,
                num_waypoints=motion_config['initial_path_waypoints'],
            )
            
            print(f"  路径规划完成: {len(path)}个路径点")
            
            # 执行规划路径
            for i, waypoint in enumerate(path):
                self.franka.control_dofs_position(waypoint[:-2], self.motors_dof)
                self.scene.step()
                
                if i % 50 == 0:
                    print(f"    执行路径点 {i}/{len(path)}")
            
            # 等待机械臂到达最后路径点
            for i in range(motion_config['initial_wait_steps']):
                self.scene.step()
                
            print("  初始位置移动完成")
            
        except Exception as e:
            print(f"  移动到初始位置失败: {e}")
            print("  继续执行程序...")
    
    def reset_cube(self):
        """重置方块到初始位置"""
        if self.cube_entity is not None:
            try:
                cube_config = self.config['simulation']['cube']
                initial_position = np.array(cube_config['position'], dtype=np.float64)
                
                # 重置方块位置
                self.cube_entity.set_pos(initial_position)
                
                # 重置方块物理状态（速度和角速度）
                # 注意：Genesis API可能需要不同的方法名，这里使用通用方法
                try:
                    self.cube_entity.set_linear_velocity([0.0, 0.0, 0.0])
                except AttributeError:
                    pass  # 如果方法不存在，忽略
                    
                try:
                    self.cube_entity.set_angular_velocity([0.0, 0.0, 0.0])
                except AttributeError:
                    pass  # 如果方法不存在，忽略
                
                print(f"方块已重置到位置: {initial_position}")
                
            except Exception as e:
                print(f"重置方块时发生错误: {e}")
        else:
            print("警告: 方块实体未初始化，无法重置")
    
    def step_thread_func(self):
        """控制线程函数"""
        last_step_time = time.perf_counter()
        data_counter = 0
        
        while not self.should_exit.is_set():
            try:
                # 更新目标位置和姿态
                self.controller.update_target_from_keyboard()

                # 检查是否重置方块（退格键同时重置机械臂和方块）
                if keyboard.Key.backspace in self.controller.keys_pressed:
                    # 重置方块到初始位置
                    self.reset_cube()
                    # 移除按键避免连续打印
                    self.controller.keys_pressed.discard(keyboard.Key.backspace)
                
                # 检查是否打印信息
                if "p" in self.controller.keys_pressed:
                    quat = self.controller.get_target_quat()
                    print(f"\n位置: {self.controller.target_pos}")
                    print(f"欧拉角 (度): {np.degrees(self.controller.target_euler)}")
                    print(f"四元数: {quat}")
                    # 移除按键避免连续打印
                    self.controller.keys_pressed.discard("p")
                
                # 获取当前目标四元数
                target_quat = self.controller.get_target_quat()
                
                # 逆运动学计算
                self.total_ik_calls += 1
                if self.control_frame == 0:
                    try:
                        qpos = self.franka.inverse_kinematics(
                            link=self.end_effector,
                            pos=self.controller.target_pos,
                            quat=target_quat,
                        )
                        
                        # 控制机械臂
                        self.franka.control_dofs_position(qpos[:-2], self.motors_dof)
                        
                        # 控制夹爪（位置控制）
                        self.franka.control_dofs_position(self.controller.gripper_position, self.fingers_dof)
                        
                    except Exception as e:
                        self.ik_failures += 1
                        if self.ik_failures % 10 == 0:  # 每10次失败打印一次
                            print(f"逆运动学求解失败 ({self.ik_failures}/{self.total_ik_calls}): {e}")
                            print(f"当前位置: {self.controller.target_pos}, 姿态: {self.controller.target_euler}")
                
                # 测量 scene.step 执行时间和间隔
                step_start = time.perf_counter()
                self.scene.step()
                step_end = time.perf_counter()
                
                # 计算 step 间隔时间
                step_interval = step_start - last_step_time
                last_step_time = step_end
                
                # 更新腕部相机位置（如果存在）
                if self.wrist_cam is not None:
                    try:
                        self.wrist_cam.move_to_attach()
                    except Exception as e:
                        # 如果更新失败，记录错误但不中断程序
                        if self.total_ik_calls % 500 == 0:  # 每500次打印一次错误
                            print(f"腕部相机位置更新失败: {e}")
                
                # 渲染相机图像（每4次循环渲染一次以平衡性能）
                render_data = None
                wrist_render_data = None
                position_data = None
                data_counter += 1
                if data_counter >= 4:  # 控制数据频率
                    data_counter = 0
                    
                    # 渲染主相机
                    render_start = time.perf_counter()
                    rgb, _, _, _ = self.cam.render()
                    render_end = time.perf_counter()

                    # 渲染腕部相机（如果存在）
                    wrist_rgb = None
                    wrist_render_time = 0
                    if self.wrist_cam is not None:
                        wrist_render_start = time.perf_counter()
                        wrist_rgb, _, _, _ = self.wrist_cam.render()
                        wrist_render_end = time.perf_counter()
                        wrist_render_time = wrist_render_end - wrist_render_start
                    
                    # 获取机械臂位置数据
                    position_data = self.franka.get_dofs_position()
                    
                    # 准备渲染数据
                    render_data = {
                        'rgb_image': rgb,
                        'render_time': render_end - render_start
                    }
                    
                    # 准备腕部相机渲染数据
                    if wrist_rgb is not None:
                        wrist_render_data = {
                            'rgb_image': wrist_rgb,
                            'render_time': wrist_render_time
                        }
                
                    # 将数据发送回主线程
                    self.step_data_queue.put({
                        'step_time': step_end - step_start,
                        'step_interval': step_interval,
                        'render_data': render_data,
                        'wrist_render_data': wrist_render_data,
                        'position_data': position_data
                    })
                    
            except Exception as e:
                print(f"控制线程错误: {e}")
                traceback.print_exc()
    
    def start_control_thread(self):
        """启动控制线程"""
        print("启动控制线程...")
        self.control_thread = threading.Thread(target=self.step_thread_func, daemon=True)
        self.control_thread.start()
        print("控制线程已启动")
    
    def print_performance_stats(self):
        """打印性能统计信息"""
        perf_config = self.config['performance']
        
        if perf_config['enable_tick_stats'] and self.tick_intervals:
            print(f"\n=== tick 调用间隔统计 ===")
            print(f"调用间隔 - 平均: {np.mean(self.tick_intervals)*1000:.2f}ms, "
                  f"最大: {np.max(self.tick_intervals)*1000:.2f}ms, "
                  f"最小: {np.min(self.tick_intervals)*1000:.2f}ms")
            print(f"估计频率: {1/np.mean(self.tick_intervals):.1f}Hz")

        if perf_config['enable_step_stats'] and self.step_intervals:
            print(f"\n=== step 调用间隔统计 ===")
            print(f"调用间隔 - 平均: {np.mean(self.step_intervals)*1000:.2f}ms, "
                  f"最大: {np.max(self.step_intervals)*1000:.2f}ms, "
                  f"最小: {np.min(self.step_intervals)*1000:.2f}ms")
            print(f"估计频率: {1/np.mean(self.step_intervals):.1f}Hz")
        
        if perf_config['enable_step_stats'] and self.step_times:
            print(f"\n=== scene.step 时间统计 ===")
            print(f"执行时间 - 平均: {np.mean(self.step_times)*1000:.2f}ms, "
                  f"最大: {np.max(self.step_times)*1000:.2f}ms, "
                  f"最小: {np.min(self.step_times)*1000:.2f}ms")
        
        if perf_config['enable_render_stats'] and self.render_times:
            print(f"\n=== cam.render 时间统计 ===")
            print(f"执行时间 - 平均: {np.mean(self.render_times)*1000:.2f}ms, "
                  f"最大: {np.max(self.render_times)*1000:.2f}ms, "
                  f"最小: {np.min(self.render_times)*1000:.2f}ms")
        
        if self.total_ik_calls > 0:
            success_rate = (self.total_ik_calls - self.ik_failures) / self.total_ik_calls * 100
            print(f"\n=== 逆运动学统计 ===")
            print(f"调用次数: {self.total_ik_calls}")
            print(f"失败次数: {self.ik_failures}")
            print(f"成功率: {success_rate:.1f}%")
        
        print("=" * 50)
    
    def process_tick_event(self):
        """处理tick事件 - 使用非阻塞获取和最新帧缓冲"""
        # 计算 tick 间隔时间
        current_time = time.perf_counter()
        tick_interval = current_time - self.last_tick_time
        self.tick_intervals.append(tick_interval)
        self.last_tick_time = current_time
        
        # 非阻塞获取最新数据
        latest_data = None
        try:
            # 尝试获取数据，但不阻塞
            step_data = self.step_data_queue.get_nowait()
            
            # 只保留包含渲染数据的最新帧
            if step_data.get('render_data') is not None:
                latest_data = step_data
            else:
                # 对于不包含渲染数据的帧，仍然更新性能统计
                self.step_times.append(step_data['step_time'])
                self.step_intervals.append(step_data['step_interval'])
                
        except Exception:
            # 队列为空，跳出此次tick事件
            return
        
        # 处理最新的渲染数据（如果存在）
        if latest_data is not None:
            self.step_times.append(latest_data['step_time'])
            self.step_intervals.append(latest_data['step_interval'])
            
            # 处理主相机渲染数据
            render_data = latest_data['render_data']
            self.render_times.append(render_data['render_time'])
            
            # 发送主相机彩色图像
            rgb_image = render_data['rgb_image']
            if rgb_image is not None:
                self.node.send_output(
                    "image",
                    pa.array(rgb_image.ravel()),
                    {"encoding": "rgb8", "width": rgb_image.shape[1], "height": rgb_image.shape[0]},
                )
            
            # 处理腕部相机渲染数据（如果存在）
            wrist_render_data = latest_data['wrist_render_data']
            if wrist_render_data is not None:
                self.wrist_render_times.append(wrist_render_data['render_time'])
                
                # 发送腕部相机彩色图像
                wrist_rgb_image = wrist_render_data['rgb_image']
                if wrist_rgb_image is not None:
                    self.node.send_output(
                        "wrist_image",
                        pa.array(wrist_rgb_image.ravel()),
                        {"encoding": "rgb8", "width": wrist_rgb_image.shape[1], "height": wrist_rgb_image.shape[0]},
                    )
            
            # 发送位置数据（如果存在）
            position_data = latest_data.get('position_data')
            if position_data is not None:
                # 将位置数据转换为numpy数组（如果还不是）
                if hasattr(position_data, 'cpu'):
                    # 如果是PyTorch tensor，转换为numpy
                    position_np = position_data.cpu().numpy()
                elif hasattr(position_data, 'numpy'):
                    # 如果是TensorFlow tensor，转换为numpy
                    position_np = position_data.numpy()
                else:
                    # 假设已经是numpy数组或类似结构
                    position_np = np.array(position_data)
                
                # 发送位置数据
                self.node.send_output(
                    "joint_position",
                    pa.array(position_np),
                    {"names": self.jnt_names,"shape": list(position_np.shape), "dtype": str(position_np.dtype)}
                )
    
    def run(self):
        """运行主循环"""
        try:
            # 初始化
            self.initialize_simulation()
            self.move_to_initial_position()
            
            # 打印控制说明
            self.controller.print_controls()
            print("开始键盘控制...")
            print("按 'p' 键打印当前位置和姿态")
            print(f"每{self.print_interval}次循环打印一次时间统计信息")
            
            # 启动控制线程
            self.start_control_thread()
            
            # 主事件循环
            print("\n进入主事件循环...")
            for event in self.node:
                if event["type"] == "INPUT":
                    if event["id"] == "tick":
                        self.process_tick_event()
                        
                        # 定期打印统计信息
                        self.print_counter += 1
                        if self.print_counter >= self.print_interval:
                            self.print_counter = 0
                            self.print_performance_stats()
                        
                        self.control_frame -= 1
                        if self.control_frame < 0:
                            self.control_frame = 0
                    if event["id"] == "action_joint":
                        self.control_frame = 100
                        data = event["value"].to_numpy()
                        self.handle_action_joint(data)
                
                elif event["type"] == "STOP":
                    print("程序被中断")
                    break
            
        except KeyboardInterrupt:
            print("\n程序被用户中断")
        except Exception as e:
            print(f"程序运行错误: {e}")
            traceback.print_exc()
        finally:
            self.cleanup()
    
    def cleanup(self):
        """清理资源"""
        print("\n清理资源...")
        
        # 停止控制线程
        if self.should_exit:
            self.should_exit.set()
        
        # 等待控制线程结束
        if self.control_thread:
            threading_config = self.config['threading']
            self.control_thread.join(timeout=threading_config['step_thread_timeout'])
            if self.control_thread.is_alive():
                print("警告: 控制线程未正常结束")
            else:
                print("控制线程已结束")
        
        # 清理键盘控制器
        if self.controller:
            self.controller.cleanup()
            print("键盘控制器已清理")
        
    def handle_action_joint(self, joint_positions: np.ndarray):
        """处理关节位置控制指令 - 使用keyboard_controller中的功能
        
        Args:
            joint_positions: 关节位置数组，应为9个元素（7个电机关节 + 2个夹爪关节）
        """
        print("handle_action_joint")
        try:
            # 验证输入数据
            if joint_positions is None or len(joint_positions) != 9:
                print(f"错误: 无效的关节位置数据，期望9个元素，实际收到: {len(joint_positions) if joint_positions is not None else 'None'}")
                return
            
            # 打印接收到的关节位置（调试用）
            if self.print_counter % 50 == 0:  # 每50次打印一次，避免过于频繁
                print(f"接收到关节位置指令: {joint_positions}")
            
            # 分离电机关节和夹爪关节
            motor_positions = joint_positions[:7]
            gripper_positions = joint_positions[7:]
            
            # 使用keyboard_controller中的夹爪位置设置
            # 更新controller的夹爪位置，这样后续的控制会使用这个位置
            self.controller.gripper_position = gripper_positions.copy()
            
            # 注意：这里我们不直接调用franka.control_dofs_position
            # 而是更新controller的目标，让控制线程来处理
            # 但是action_joint是直接控制指令，所以我们需要直接设置关节位置
            
            # 直接控制机械臂到指定关节位置
            self.franka.control_dofs_position(motor_positions, self.motors_dof)
            self.franka.control_dofs_position(gripper_positions, self.fingers_dof)
            
            # 使用正向运动学计算末端执行器位置和姿态
            # 更新controller的目标位置和姿态，以保持一致性
            # try:
            #     # 获取末端执行器状态
            #     # 注意：这里需要等待一步让机械臂移动到新位置
            #     self.scene.step()
                
            #     # 获取末端执行器位置和姿态
            #     end_effector_state = self.franka.get_link_state(self.end_effector)
            #     if end_effector_state is not None:
            #         # 更新controller的目标位置
            #         self.controller.target_pos = end_effector_state['position'].copy()
                    
            #         # 获取四元数姿态
            #         quat = end_effector_state['quaternion']  # [w, x, y, z]
                    
            #         # 将四元数转换为欧拉角（简化版本，实际可能需要更复杂的转换）
            #         # 这里使用简单的近似转换
            #         w, x, y, z = quat
                    
            #         # 将四元数转换为欧拉角（roll, pitch, yaw）
            #         # 使用标准转换公式
            #         sinr_cosp = 2 * (w * x + y * z)
            #         cosr_cosp = 1 - 2 * (x * x + y * y)
            #         roll = math.atan2(sinr_cosp, cosr_cosp)
                    
            #         sinp = 2 * (w * y - z * x)
            #         if abs(sinp) >= 1:
            #             pitch = math.copysign(math.pi / 2, sinp)
            #         else:
            #             pitch = math.asin(sinp)
                    
            #         siny_cosp = 2 * (w * z + x * y)
            #         cosy_cosp = 1 - 2 * (y * y + z * z)
            #         yaw = math.atan2(siny_cosp, cosy_cosp)
                    
            #         # 更新controller的目标欧拉角
            #         self.controller.target_euler = np.array([roll, pitch, yaw])
                    
            #         if self.print_counter % 100 == 0:
            #             print(f"更新controller目标: 位置={self.controller.target_pos}, 欧拉角={np.degrees(self.controller.target_euler)}")
                        
            # except Exception as e:
            #     # 正向运动学计算失败不影响关节控制
            #     if self.print_counter % 100 == 0:
            #         print(f"正向运动学计算失败，无法更新controller目标: {e}")
            
        except Exception as e:
            print(f"处理关节位置指令时发生错误: {e}")
            traceback.print_exc()


def main():
    """主函数"""
    print("=" * 60)
    print("Franka机械臂抓取立方体仿真系统 - 改进版")
    print("=" * 60)
    
    # 从环境变量获取配置文件路径，如果不存在则使用默认值
    config_path = os.environ.get("CONFIG_PATH", "config/simulation_config.yaml")
    print(f"使用配置文件: {config_path}")
    
    # 创建并运行仿真
    simulation = FrankaSimulation(config_path)
    simulation.run()


if __name__ == "__main__":
    main()
