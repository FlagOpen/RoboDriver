import numpy as np
import genesis as gs
from pynput import keyboard
import math
from keyboard_controller import KeyboardController
from dora import Node
import cv2
import pyarrow as pa
import time
from collections import deque
import threading
from queue import Queue

# init
gs.init(backend=gs.gpu, logging_level="warn")
node = Node()

# 创建场景
scene = gs.Scene(
    sim_options=gs.options.SimOptions(),
    viewer_options=gs.options.ViewerOptions(),
    show_viewer=True,
)

# 创建实体
plane = scene.add_entity(gs.morphs.Plane())
cube = scene.add_entity(
    gs.morphs.Box(
        size=(0.05, 0.05, 0.05),
        pos=(0.4, 0.0, 0.00),
    )
)
franka = scene.add_entity(gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"))
cam = scene.add_camera(
    res=(640, 480), pos=(1.0, 0.0, 1.0), lookat=(0, 0, 0), fov=60, GUI=False
)

# 构建场景
scene.build()

# 关节索引定义
motors_dof = np.arange(7)
fingers_dof = np.arange(7, 9)

# 设置控制增益
franka.set_dofs_kp(
    np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100]),
)
franka.set_dofs_kv(
    np.array([450, 450, 350, 350, 200, 200, 200, 10, 10]),
)
franka.set_dofs_force_range(
    np.array([-87, -87, -87, -87, -12, -12, -12, -100, -100]),
    np.array([87, 87, 87, 87, 12, 12, 12, 100, 100]),
)

# 获取末端执行器
end_effector = franka.get_link("hand")

# 初始化键盘控制器
controller = KeyboardController()

# 初始位置
initial_pos = np.array([0.4, 0.0, 0.4])
initial_euler = np.array([math.pi, 0.0, 0.0])
initial_quat = controller.euler_to_quat(*initial_euler)

controller.target_pos = initial_pos.copy()
controller.target_euler = initial_euler.copy()

# 先移动到初始位置,这个IK感觉有问题
qpos = franka.inverse_kinematics(
    link=end_effector,
    pos=controller.target_pos,
    quat=initial_quat,
)
path = franka.plan_path(
    qpos_goal=qpos,
    num_waypoints=200,  # 2s duration
)
# execute the planned path
for waypoint in path:
    franka.control_dofs_position(waypoint[:-2], motors_dof)
    scene.step()

# allow robot to reach the last waypoint
for i in range(100):
    scene.step()

# 初始化时间统计变量
step_times = deque(maxlen=100)  # 保存最近100次的执行时间
step_intervals = deque(maxlen=100)  # 保存最近100次的间隔时间
render_times = deque(maxlen=100)  # 保存最近100次的执行时间
tick_intervals = deque(maxlen=100)  # 保存最近100次的间隔时间

last_tick_time = time.perf_counter()  # tick 上一次调用时间
print_counter = 0  # 打印计数器
print_interval = 100  # 每100次打印一次统计信息

controller.print_controls()
print("开始键盘控制...")
print("按 'p' 键打印当前位置和姿态")
print("每100次循环打印一次时间统计信息")

# 创建线程通信队列和事件
should_exit = threading.Event()
step_data_queue = Queue()

def control_thread_func():
    """在线程中执行所有 tick 逻辑"""
    last_step_time = time.perf_counter()  # 线程内记录上一次 step 的时间
    
    while not should_exit.is_set():
        try:
            # 更新目标位置和姿态
            controller.update_target_from_keyboard()

            # 检查是否打印信息
            if "p" in controller.keys_pressed:
                quat = controller.get_target_quat()
                print(f"\n位置: {controller.target_pos}")
                print(f"欧拉角 (度): {np.degrees(controller.target_euler)}")
                print(f"四元数: {quat}")
                # 移除按键避免连续打印
                controller.keys_pressed.discard("p")

            # 获取当前目标四元数
            target_quat = controller.get_target_quat()

            try:
                # 逆运动学计算
                qpos = franka.inverse_kinematics(
                    link=end_effector,
                    pos=controller.target_pos,
                    quat=target_quat,
                )
                
                # 控制机械臂
                franka.control_dofs_position(qpos[:-2], motors_dof)
                
                # 控制夹爪（位置控制）
                franka.control_dofs_position(controller.gripper_position, fingers_dof)
                
            except Exception as e:
                print(f"逆运动学求解失败: {e}")
                print(f"当前位置: {controller.target_pos}, 姿态: {controller.target_euler}")

            # 测量 scene.step 执行时间和间隔
            step_start = time.perf_counter()
            scene.step()
            step_end = time.perf_counter()
            
            # 计算 step 间隔时间
            step_interval = step_start - last_step_time
            last_step_time = step_end
            
            # 将统计数据发送回主线程
            step_data_queue.put({
                'step_time': step_end - step_start,
                'step_interval': step_interval
            })
                
        except Exception:
            # 队列超时，继续等待
            pass

# 启动控制线程
control_thread = threading.Thread(target=control_thread_func, daemon=True)
control_thread.start()

########################## 主控制循环 ##########################
for event in node:
    if event["type"] == "INPUT":
        if event["id"] == "tick":
            # 计算 tick 间隔时间
            current_time = time.perf_counter()
            tick_interval = current_time - last_tick_time
            tick_intervals.append(tick_interval)
            last_tick_time = current_time

            # 从线程获取统计数据
            try:
                step_data = step_data_queue.get(timeout=0.01)
                step_times.append(step_data['step_time'])
                step_intervals.append(step_data['step_interval'])
            except Exception:
                pass

            # 测量 cam.render 执行时间
            render_start = time.perf_counter()
            rgb, _, _, _ = cam.render()
            render_end = time.perf_counter()
            render_times.append(render_end - render_start)

            # Send Color Image
            rgb_image = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
            ret, frame = cv2.imencode("." + "jpeg", rgb_image)
            if ret:
                node.send_output(
                    "image",
                    pa.array(frame),
                    {"encoding": "jpeg", "width": int(640), "height": int(480)},
                )
        
        # 每 print_interval 次循环打印一次统计信息
        print_counter += 1
        if print_counter >= print_interval:
            print_counter = 0
            if step_intervals:
                print(f"\n=== scene.step 调用间隔统计 ===")
                print(f"调用间隔 - 平均: {np.mean(step_intervals)*1000:.2f}ms, "
                      f"最大: {np.max(step_intervals)*1000:.2f}ms, "
                      f"最小: {np.min(step_intervals)*1000:.2f}ms")
                print(f"估计频率: {1/np.mean(step_intervals):.1f}Hz")

            if step_times:
                print(f"\n=== scene.step 时间统计 ===")
                print(f"执行时间 - 平均: {np.mean(step_times)*1000:.2f}ms, "
                      f"最大: {np.max(step_times)*1000:.2f}ms, "
                      f"最小: {np.min(step_times)*1000:.2f}ms")
            
            print("-" * 50)

            if tick_intervals:
                print(f"\n=== tick 调用间隔统计 ===")
                print(f"调用间隔 - 平均: {np.mean(tick_intervals)*1000:.2f}ms, "
                        f"最大: {np.max(tick_intervals)*1000:.2f}ms, "
                        f"最小: {np.min(tick_intervals)*1000:.2f}ms")
                print(f"估计频率: {1/np.mean(tick_intervals):.1f}Hz")

            if render_times:
                print(f"\n=== cam.render 时间统计 ===")
                print(f"执行时间 - 平均: {np.mean(render_times)*1000:.2f}ms, "
                      f"最大: {np.max(render_times)*1000:.2f}ms, "
                      f"最小: {np.min(render_times)*1000:.2f}ms")
            
            print("=" * 50)

    elif event["type"] == "STOP":
        print("程序被中断")
        should_exit.set()
        
        break


should_exit.set()
control_thread.join(timeout=2)
controller.listener.stop()
print("程序结束")
