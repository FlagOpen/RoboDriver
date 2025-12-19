import numpy as np
from pynput import keyboard
import math
from typing import Set, Any
from config_loader import get_config_loader


class KeyboardController:
    def __init__(self, config_path: str = "config/simulation_config.yaml"):
        """
        初始化键盘控制器
        
        Args:
            config_path: 配置文件路径
        """
        # 加载配置
        self.config_loader = get_config_loader(config_path)
        keyboard_config = self.config_loader.get_keyboard_config()
        
        # 从配置初始化参数
        self.keys_pressed: Set[Any] = set()
        self.target_pos = np.array(keyboard_config['initial_position'], dtype=np.float64)
        self.target_euler = np.array(keyboard_config['initial_euler'], dtype=np.float64)
        self.position_step = keyboard_config['position_step']
        self.orientation_step = keyboard_config['orientation_step']
        self.gripper_position = np.array(keyboard_config['gripper_open_position'], dtype=np.float64)
        
        # 配置中的夹爪位置设置
        self.gripper_close_position = np.array(keyboard_config['gripper_close_position'], dtype=np.float64)
        self.gripper_open_position = np.array(keyboard_config['gripper_open_position'], dtype=np.float64)
        
        # 状态标志
        self.is_gripper_closed = False
        self.last_print_time = 0
        self.print_interval = 1.0  # 打印间隔（秒）
        
        # 启动键盘监听
        self.listener = keyboard.Listener(
            on_press=self.on_press, 
            on_release=self.on_release
        )
        self.listener.start()
        
        print("键盘控制器初始化完成")

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
        import time
        
        current_time = time.time()
        
        # 位置控制 (使用方向键和功能键)
        if keyboard.Key.up in self.keys_pressed:  # 向前 (X轴+)
            self.target_pos[0] += self.position_step
        if keyboard.Key.down in self.keys_pressed:  # 向后 (X轴-)
            self.target_pos[0] -= self.position_step
        if keyboard.Key.left in self.keys_pressed:  # 向左 (Y轴+)
            self.target_pos[1] += self.position_step
        if keyboard.Key.right in self.keys_pressed:  # 向右 (Y轴-)
            self.target_pos[1] -= self.position_step
        if "=" in self.keys_pressed or "+" in self.keys_pressed:  # 向上 (Z轴+)
            self.target_pos[2] += self.position_step
        if "-" in self.keys_pressed:  # 向下 (Z轴-)
            self.target_pos[2] -= self.position_step

        # 姿态控制 (欧拉角) - 使用数字键
        if "6" in self.keys_pressed:  # 增加绕X轴旋转 (roll+)
            self.target_euler[0] += self.orientation_step
        if "4" in self.keys_pressed:  # 减少绕X轴旋转 (roll-)
            self.target_euler[0] -= self.orientation_step
        if "8" in self.keys_pressed:  # 增加绕Y轴旋转 (pitch+)
            self.target_euler[1] += self.orientation_step
        if "2" in self.keys_pressed:  # 减少绕Y轴旋转 (pitch-)
            self.target_euler[1] -= self.orientation_step
        if "7" in self.keys_pressed:  # 增加绕Z轴旋转 (yaw+)
            self.target_euler[2] += self.orientation_step
        if "9" in self.keys_pressed:  # 减少绕Z轴旋转 (yaw-)
            self.target_euler[2] -= self.orientation_step

        # 重置姿态 (使用空格键)
        if keyboard.Key.space in self.keys_pressed:
            keyboard_config = self.config_loader.get_keyboard_config()
            self.target_euler = np.array(keyboard_config['initial_euler'], dtype=np.float64)
            # 移除按键避免连续重置
            self.keys_pressed.discard(keyboard.Key.space)

        # 夹爪控制 (使用b/n键) - 切换模式
        if "b" in self.keys_pressed and not self.is_gripper_closed:
            self.gripper_position = self.gripper_close_position.copy()
            self.is_gripper_closed = True
            if current_time - self.last_print_time > self.print_interval:
                print("夹爪: 夹紧")
                self.last_print_time = current_time
        elif "n" in self.keys_pressed and self.is_gripper_closed:
            self.gripper_position = self.gripper_open_position.copy()
            self.is_gripper_closed = False
            if current_time - self.last_print_time > self.print_interval:
                print("夹爪: 松开")
                self.last_print_time = current_time

        # 位置重置 (使用退格键)
        if keyboard.Key.backspace in self.keys_pressed:
            keyboard_config = self.config_loader.get_keyboard_config()
            self.target_pos = np.array(keyboard_config['initial_position'], dtype=np.float64)
            self.target_euler = np.array(keyboard_config['initial_euler'], dtype=np.float64)
            # 移除按键避免连续重置
            self.keys_pressed.discard(keyboard.Key.backspace)
            if current_time - self.last_print_time > self.print_interval:
                print("位置和姿态已重置")
                self.last_print_time = current_time
                
        # 速度控制模式切换 (使用v键)
        if "v" in self.keys_pressed:
            # 切换步长大小
            if self.position_step == 0.002:
                self.position_step = 0.005
                self.orientation_step = 0.01
                print("控制模式: 高速模式")
            else:
                self.position_step = 0.002
                self.orientation_step = 0.005
                print("控制模式: 标准模式")
            self.keys_pressed.discard("v")

    def get_target_quat(self):
        """获取当前目标姿态的四元数表示"""
        return self.euler_to_quat(
            self.target_euler[0],  # roll
            self.target_euler[1],  # pitch
            self.target_euler[2],  # yaw
        )

    def print_controls(self):
        """打印控制说明"""
        print("\n" + "=" * 60)
        print("机械臂键盘控制说明")
        print("=" * 60)
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
        print("  B 键 - 夹紧 (切换)")
        print("  N 键 - 松开 (切换)")
        print(f"  当前状态: {'夹紧' if self.is_gripper_closed else '松开'}")
        print("\n其他控制:")
        print("  退格键 - 重置机械臂位置和姿态，同时重置方块位置")
        print("  P 键  - 打印当前位置和姿态")
        print("  V 键  - 切换速度模式 (标准/高速)")
        print(f"  当前速度模式: {'高速' if self.position_step > 0.002 else '标准'}")
        print("  ESC键 - 退出程序")
        print("=" * 60)
        print(f"当前位置: {self.target_pos}")
        print(f"当前欧拉角 (度): {np.degrees(self.target_euler)}")
        print(f"四元数: {self.get_target_quat()}")
        print(f"位置步长: {self.position_step}")
        print(f"姿态步长: {self.orientation_step} rad ({np.degrees(self.orientation_step):.2f}°)")
        print("=" * 60 + "\n")
        
    def get_status(self) -> dict:
        """获取当前控制器状态"""
        return {
            'position': self.target_pos.tolist(),
            'euler_angles': self.target_euler.tolist(),
            'quaternion': self.get_target_quat().tolist(),
            'gripper_closed': self.is_gripper_closed,
            'position_step': self.position_step,
            'orientation_step': self.orientation_step,
            'keys_pressed': list(self.keys_pressed)
        }
        
    def cleanup(self):
        """清理资源"""
        if self.listener:
            self.listener.stop()
            print("键盘监听器已停止")
