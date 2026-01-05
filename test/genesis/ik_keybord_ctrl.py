import numpy as np
import genesis as gs
from pynput import keyboard
import threading
import time
import math

########################## 键盘控制类 ##########################
class KeyboardController:
    def __init__(self):
        self.keys_pressed = set()
        self.target_pos = np.array([0.4, 0.0, 0.4])  # 初始目标位置
        self.target_euler = np.array([0.0, 0.0, 0.0])  # 欧拉角 (roll, pitch, yaw)，单位：弧度
        self.position_step = 0.002  # 位置移动步长
        self.orientation_step = 0.005  # 姿态旋转步长，单位：弧度
        self.gripper_force = np.array([3.0, 3.0])  # 夹爪力控制
        
        # 启动键盘监听
        self.listener = keyboard.Listener(
            on_press=self.on_press,
            on_release=self.on_release
        )
        self.listener.start()
    
    def on_press(self, key):
        try:
            self.keys_pressed.add(key.char)
        except AttributeError:
            self.keys_pressed.add(key)
    
    def on_release(self, key):
        try:
            self.keys_pressed.discard(key.char)
        except AttributeError:
            self.keys_pressed.discard(key)
    
    def euler_to_quat(self, roll, pitch, yaw):
        """
        将欧拉角转换为四元数
        使用ZYX顺序（先绕Z轴旋转，然后Y轴，最后X轴）
        参数：roll(x), pitch(y), yaw(z) 单位：弧度
        """
        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)
        cp = math.cos(pitch * 0.5)
        sp = math.sin(pitch * 0.5)
        cr = math.cos(roll * 0.5)
        sr = math.sin(roll * 0.5)
        
        w = cr * cp * cy + sr * sp * sy
        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy
        
        return np.array([w, x, y, z])  # 四元数格式：[w, x, y, z]
        # return np.array([0, 1, 0, 0])
    
    def update_target_from_keyboard(self):
        """根据按键更新目标位置和姿态"""
        # 位置控制 (使用方向键和功能键)
        if keyboard.Key.up in self.keys_pressed:    # 向前 (X轴+)
            self.target_pos[0] += self.position_step
        if keyboard.Key.down in self.keys_pressed:  # 向后 (X轴-)
            self.target_pos[0] -= self.position_step
        if keyboard.Key.left in self.keys_pressed:  # 向左 (Y轴+)
            self.target_pos[1] += self.position_step
        if keyboard.Key.right in self.keys_pressed: # 向右 (Y轴-)
            self.target_pos[1] -= self.position_step
        if '=' in self.keys_pressed or '+' in self.keys_pressed:  # 向上 (Z轴+)
            self.target_pos[2] += self.position_step
        if '-' in self.keys_pressed:                # 向下 (Z轴-)
            self.target_pos[2] -= self.position_step
        
        # 姿态控制 (欧拉角) - 使用数字键
        if '6' in self.keys_pressed:  # 增加绕X轴旋转 (roll+)
            self.target_euler[0] += self.orientation_step
        if '4' in self.keys_pressed:  # 减少绕X轴旋转 (roll-)
            self.target_euler[0] -= self.orientation_step
        if '8' in self.keys_pressed:  # 增加绕Y轴旋转 (pitch+)
            self.target_euler[1] += self.orientation_step
        if '2' in self.keys_pressed:  # 减少绕Y轴旋转 (pitch-)
            self.target_euler[1] -= self.orientation_step
        if '7' in self.keys_pressed:  # 增加绕Z轴旋转 (yaw+)
            self.target_euler[2] += self.orientation_step
        if '9' in self.keys_pressed:  # 减少绕Z轴旋转 (yaw-)
            self.target_euler[2] -= self.orientation_step
        
        # 重置姿态 (使用空格键)
        if keyboard.Key.space in self.keys_pressed:
            self.target_euler = np.array([math.pi, 0.0, 0.0])
        
        # 夹爪控制 (使用b/n键)
        if 'b' in self.keys_pressed:  # 夹紧
            self.gripper_force = np.array([-3.0, -3.0])
        elif 'n' in self.keys_pressed:  # 松开
            self.gripper_force = np.array([3.0, 3.0])
        
        # 位置重置 (使用退格键)
        if keyboard.Key.backspace in self.keys_pressed:
            self.target_pos = np.array([0.4, 0.0, 0.4])
            self.target_euler = np.array([math.pi, 0.0, 0.0])
    
    def get_target_quat(self):
        """获取当前目标姿态的四元数表示"""
        return self.euler_to_quat(
            self.target_euler[0],  # roll
            self.target_euler[1],  # pitch
            self.target_euler[2]   # yaw
        )
    
    def print_controls(self):
        """打印控制说明"""
        print("\n" + "="*50)
        print("机械臂键盘控制说明")
        print("="*50)
        print("位置控制:")
        print("  方向键↑/↓ - 前/后移动 (X轴)")
        print("  方向键←/→ - 左/右移动 (Y轴)")
        print("  +/= 键   - 向上移动 (Z轴)")
        print("  - 键     - 向下移动 (Z轴)")
        print("\n姿态控制 (欧拉角):")
        print("  6/4 键 - 增加/减少绕X轴旋转 (Roll)")
        print("  8/2 键 - 增加/减少绕Y轴旋转 (Pitch)")
        print("  7/9 键 - 增加/减少绕Z轴旋转 (Yaw)")
        print("  空格键  - 重置姿态为初始值")
        print("\n夹爪控制:")
        print("  B 键 - 夹紧")
        print("  N 键 - 松开")
        print("\n其他:")
        print("  退格键 - 重置位置和姿态")
        print("  P 键  - 打印当前位置和姿态")
        print("  ESC键 - 退出程序")
        print("="*50)
        print(f"当前位置: {self.target_pos}")
        print(f"当前欧拉角 (roll,pitch,yaw): {self.target_euler}")
        print(f"四元数: {self.get_target_quat()}")
        print("="*50 + "\n")

########################## init ##########################
gs.init(backend=gs.gpu, logging_level="info")

########################## 创建场景 ##########################
scene = gs.Scene(
    sim_options=gs.options.SimOptions(),
    viewer_options=gs.options.ViewerOptions(),
    vis_options = gs.options.VisOptions(
        show_world_frame = False, # visualize the coordinate frame of `world` at its origin
        world_frame_size = 1.0, # length of the world frame in meter
        show_link_frame  = False, # do not visualize coordinate frames of entity links
        show_cameras     = False, # do not visualize mesh and frustum of the cameras added
        plane_reflection = True, # turn on plane reflection
        ambient_light    = (0.1, 0.1, 0.1), # ambient light setting
    ),
    show_viewer=True,
)

########################## 创建实体 ##########################
plane = scene.add_entity(
    gs.morphs.Plane(),
)
cube = scene.add_entity(
    gs.morphs.Box(
        size=(0.05, 0.05, 0.05),
        pos=(0.4, 0.0, 0.00),
    )
)
franka = scene.add_entity(
    gs.morphs.MJCF(file='xml/franka_emika_panda/panda.xml'),
)
########################## 构建场景 ##########################
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
    np.array([87,  87,  87,  87,  12,  12,  12,  100,  100]),
)

# 获取末端执行器
end_effector = franka.get_link('hand')

########################## 初始化键盘控制器 ##########################
controller = KeyboardController()
# controller.print_controls()

# 初始位置（lift后的位置）
initial_pos = np.array([0.4, 0.0, 0.4])
initial_euler = np.array([math.pi, 0.0, 0.0])

# initial_pos = np.array([0.0, 0.0, 0])
# initial_euler = np.array([0.0, 0.0, 0.0])

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
    qpos_goal     = qpos,
    num_waypoints = 200, # 2s duration
)
# execute the planned path
for waypoint in path:
    franka.control_dofs_position(waypoint[:-2], motors_dof)
    scene.step()

# allow robot to reach the last waypoint
for i in range(100):
    scene.step()

controller.print_controls()
print("开始键盘控制...")
print("按 'p' 键打印当前位置和姿态")
print("按 ESC 键退出程序")

########################## 主控制循环 ##########################
try:
    while True:
        # 更新目标位置和姿态
        controller.update_target_from_keyboard()
        
        # 检查是否退出
        if keyboard.Key.esc in controller.keys_pressed:
            print("退出程序")
            break
        
        # 检查是否打印信息
        if 'p' in controller.keys_pressed:
            quat = controller.get_target_quat()
            print(f"\n位置: {controller.target_pos}")
            print(f"欧拉角 (度): {np.degrees(controller.target_euler)}")
            print(f"四元数: {quat}")
            # 移除按键避免连续打印
            controller.keys_pressed.discard('p')
        
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
            
            # 控制夹爪
            franka.control_dofs_force(controller.gripper_force, fingers_dof)
            
        except Exception as e:
            print(f"逆运动学求解失败: {e}")
            print(f"当前位置: {controller.target_pos}, 姿态: {controller.target_euler}")
        
        scene.step()

except KeyboardInterrupt:
    print("程序被中断")

finally:
    controller.listener.stop()
    print("程序结束")