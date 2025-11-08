import asyncio
import json
import base64
import time
import os
import sys
import numpy as np
import cv2
import websockets
import threading
import keyboard


os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, 'galbot'))

from galbot.core_proto import time_pb2, header_pb2
from galbot.spatial_proto import twist_pb2, pose_pb2
from galbot.singorix_proto import singorix_command_pb2, singorix_sensor_pb2, singorix_error_pb2, singorix_target_pb2
from galbot.sensor_proto import imu_pb2, image_pb2, camera_pb2, joy_pb2
from galbot.tf2_proto import tf2_message_pb2

from operating_platform.utils.colored_logging import setup_colored_logger

logger = setup_colored_logger(__name__)

class RobotSocket:
    def __init__(self, robot_ip, bridge_port=10800):
        self.robot_ip = robot_ip
        self.bridge_port = bridge_port
        self.ws = None
        self.uri = f"ws://{self.robot_ip}:{self.bridge_port}"

        self.latest_states = {}
        self.state_lock = threading.Lock()

        self.arm_joint_data = {
            "right_arm": {},
            "left_arm": {}
        }
        self.gripper_data = {
            "right_gripper": {},
            "left_gripper": {}
        }

        # Protobuf 类型映射
        self.protobuf_type_map = {
            "galbot.sensor_proto.CompressedImage": image_pb2.CompressedImage,
            "galbot.sensor_proto.CameraInfo": camera_pb2.CameraInfo,
            "galbot.singorix_proto.SingoriXSensor": singorix_sensor_pb2.SingoriXSensor,
            "galbot.singorix_proto.SingoriXError": singorix_error_pb2.SingoriXError,
            "galbot.singorix_proto.SingoriXTarget": singorix_target_pb2.SingoriXTarget,
            "galbot.tf2_proto.TF2Message": tf2_message_pb2.TF2Message,
            "galbot.sensor_proto.Joy": joy_pb2.Joy
        }

        # 异步任务控制
        self.running = False
        self.task = None

    async def connect(self):
        """建立 WebSocket 连接"""
        try:
            self.ws = await websockets.connect(self.uri)
            print(f"✅ WebSocket 已连接: {self.uri}")
            self.running = True
            await self.listen()
        except Exception as e:
            print(f"❌ 连接失败: {e}")

    async def listen(self):
        """监听 WebSocket 消息"""
        try:
            async for message in self.ws:
                try:
                    msg_json = json.loads(message)
                    op = msg_json.get("op")

                    if op == "message":
                        await self._process_protobuf_message(msg_json)
                    elif op == "heartbeat":
                        await self._process_heartbeat(msg_json)
                    elif op == "error":
                        await self._process_error(msg_json)
                    else:
                        print(f"⚠️ 未知操作类型: {op}")

                except json.JSONDecodeError:
                    print(f"❌ JSON 解析失败: {message[:100]}...")
                except Exception as e:
                    print(f"❌ 处理消息时出错: {e}")

        except websockets.exceptions.ConnectionClosed as e:
            print(f"🔌 连接关闭: {e}")
        finally:
            self.running = False

    async def _process_protobuf_message(self, message):
        """处理 protobuf 消息"""
        topic = message.get("topic")
        type_str = message.get("type")
        data_b64 = message.get("data")

        if not all([topic, type_str, data_b64]):
            print("❌ 缺少必要字段")
            return

        pb_class = self.protobuf_type_map.get(type_str)
        if not pb_class:
            print(f"❌ 未知 protobuf 类型: {type_str}")
            return

        try:
            data_bytes = base64.b64decode(data_b64)

            if not data_bytes:
                raise ValueError(f"解码后得到空字节数据 (topic: {topic})")
            
            pb_message = pb_class()
            pb_message.ParseFromString(data_bytes)

            if pb_message is None:
                raise ValueError(f"创建protobuf消息对象失败 (topic: {topic})")
            
            if "singorix/wbcs/sensor" in topic:
                self._parse_and_store_joint_data(pb_message)

            with self.state_lock:
                self.latest_states[topic] = {
                    "message": pb_message,
                    "timestamp": message.get("pub_ts", 0),
                    "received": time.time_ns()
                }

            # print(f"📥 接收到 {topic} 消息: {type_str}")

        except Exception as e:
            print(f"❌ 解析 protobuf 失败: {e}")

    async def _process_heartbeat(self, message):
        ts = message.get("ts", 0)
        print(f"💓 心跳时间戳: {ts}")

    async def _process_error(self, message):
        error_msg = message.get("msg", "未知错误")
        print(f"❗ 错误消息: {error_msg}")

    def _parse_and_store_joint_data(self, sensor_msg):
        """解析 SingoriXSensor 消息，提取并存储 arm 和 gripper 数据"""
        if not sensor_msg.joint_sensor_map:
            return

        with self.state_lock:
            for group_name, joint_sensor in sensor_msg.joint_sensor_map.items():
                n = len(joint_sensor.name)
                if n == 0:
                    continue

                # 构建当前组的关节数据字典
                joint_data = {}
                for i in range(n):
                    name = joint_sensor.name[i] if i < len(joint_sensor.name) else f"joint{i}"
                    joint_data[name] = {
                        "position": joint_sensor.position[i] if i < len(joint_sensor.position) else 0.0,
                        "velocity": joint_sensor.velocity[i] if i < len(joint_sensor.velocity) else 0.0,
                        "effort": joint_sensor.effort[i] if i < len(joint_sensor.effort) else 0.0,
                        "current": joint_sensor.current[i] if i < len(joint_sensor.current) else 0.0,
                    }

                # 存储到对应结构
                if group_name == "right_arm":
                    self.arm_joint_data["right_arm"] = joint_data
                elif group_name == "left_arm":
                    self.arm_joint_data["left_arm"] = joint_data
                elif group_name == "right_gripper":
                    self.gripper_data["right_gripper"] = joint_data
                elif group_name == "left_gripper":
                    self.gripper_data["left_gripper"] = joint_data

    def get_arm_state(self, side):
        """获取指定臂的关节状态 ('left' 或 'right')"""
        key = f"{side}_arm"
        with self.state_lock:
            return self.arm_joint_data.get(key, {}).copy()

    def get_gripper_state(self, side):
        """获取指定夹爪状态 ('left' 或 'right')"""
        key = f"{side}_gripper"
        with self.state_lock:
            return self.gripper_data.get(key, {}).copy()

    def get_all_arm_states(self):
        """获取所有臂状态"""
        with self.state_lock:
            return {k: v.copy() for k, v in self.arm_joint_data.items()}

    def get_all_gripper_states(self):
        """获取所有夹爪状态"""
        with self.state_lock:
            return {k: v.copy() for k, v in self.gripper_data.items()}

    def get_latest_state(self, topic):
        """同步方法，供外部调用"""
        with self.state_lock:
            return self.latest_states.get(topic)

    def get_all_topics(self):
        """同步方法，获取所有主题"""
        with self.state_lock:
            return list(self.latest_states.keys())

    # ========== 新增的关节控制方法 ==========
    
    def create_joint_command(self, position=None, velocity=None, acceleration=None, effort=None):
        """创建单个关节命令"""
        joint_cmd = singorix_command_pb2.JointCommand()
        
        # 设置头部信息
        joint_cmd.header.timestamp.sec = int(time.time())
        joint_cmd.header.timestamp.nanosec = int((time.time() - int(time.time())) * 1e9)
        
        # 设置关节参数
        if position is not None:
            joint_cmd.position = position
        if velocity is not None:
            joint_cmd.velocity = velocity
        if acceleration is not None:
            joint_cmd.acceleration = acceleration
        if effort is not None:
            joint_cmd.effort = effort
            
        return joint_cmd

    def create_group_command(self, joint_commands, time_from_start_sec=0):
        """创建组命令，包含多个关节命令"""
        group_cmd = singorix_command_pb2.GroupCommand()
        
        # 设置时间偏移
        group_cmd.time_from_start.sec = int(time_from_start_sec)
        group_cmd.time_from_start.nanosec = int((time_from_start_sec - int(time_from_start_sec)) * 1e9)
        
        # 添加关节命令
        for joint_cmd in joint_commands:
            group_cmd.joint_commands.append(joint_cmd)
        
        return group_cmd

    def create_target_config(self, target_data, target_type, target_sampling, target_priority=0):
        """创建目标配置"""
        config = singorix_target_pb2.TargetConfig()
        config.target_data = target_data
        config.target_type = target_type
        config.target_sampling = target_sampling
        config.target_priority = target_priority
        return config

    def create_target_group_trajectory(self, target_config, joint_names, group_commands):
        """创建目标组轨迹"""
        trajectory = singorix_target_pb2.TargetGroupTrajectory()
        trajectory.target_config.CopyFrom(target_config)
        
        # 设置关节名称
        trajectory.joint_names.extend(joint_names)
        
        # 添加组命令
        for group_cmd in group_commands:
            trajectory.group_commands.append(group_cmd)
        
        return trajectory

    def create_singorix_target(self, group_name, trajectory):
        """创建SingoriX目标消息"""
        target = singorix_target_pb2.SingoriXTarget()
        
        # 设置头部信息
        target.header.timestamp.sec = int(time.time())
        target.header.timestamp.nanosec = int((time.time() - int(time.time())) * 1e9)
        target.header.frame_id = "world"
        
        # 添加目标组轨迹
        target.target_group_trajectory_map[group_name].CopyFrom(trajectory)
        
        return target

    async def send_joint_positions(self, group_name, joint_names, positions, 
                                 target_type=singorix_target_pb2.TARGET_TYPE_APPEND,
                                 target_sampling=singorix_target_pb2.TARGET_SAMPLING_LINEAR_INTERPOLATE):
        """发送关节位置控制命令"""
        try:
            # 创建关节命令
            joint_commands = []
            for pos in positions:
                joint_cmd = self.create_joint_command(position=pos)
                joint_commands.append(joint_cmd)
            
            # 创建组命令
            group_cmd = self.create_group_command(joint_commands)
            
            # 创建目标配置
            target_config = self.create_target_config(
                target_data=singorix_target_pb2.TARGET_DATA_JOINT_POS,
                target_type=target_type,
                target_sampling=target_sampling
            )
            
            # 创建目标轨迹
            trajectory = self.create_target_group_trajectory(
                target_config, joint_names, [group_cmd]
            )
            
            # 创建目标消息
            target_msg = self.create_singorix_target(group_name, trajectory)
            
            # 序列化消息
            serialized_msg = target_msg.SerializeToString()
            data_b64 = base64.b64encode(serialized_msg).decode('utf-8')
            
            # 构建发送消息
            message = {
                "op": "message",
                "topic": "omnilink_comm/robot_simple_target",
                "type": "galbot.singorix_proto.SingoriXTarget",
                "pub_ts": int((time.time() - int(time.time())) * 1e9),  
                "data": data_b64
            }
            
            # 发送消息
            if self.ws:
                await self.ws.send(json.dumps(message))
                print(f"✅ 已发送关节位置命令到 {group_name}: {positions}")
            else:
                print("❌ WebSocket 连接未建立")
                
        except Exception as e:
            print(f"❌ 发送关节位置命令失败: {e}")

    async def shutdown(self):
        """关闭连接"""
        self.running = False
        if self.ws:
            await self.ws.close()
        print("🔌 WebSocket 已关闭")


class MenuController:
    def __init__(self, robot_socket):
        self.robot_socket = robot_socket
        self.current_menu = "main"  # 当前菜单级别
        self.selected_part = None   # 选择的部位
        self.selected_joint = None  # 选择的关节
        self.joint_step = 0.1       # 关节调整步长
        
        # 关节名称定义
        self.right_arm_joints = ["right_arm_joint1", "right_arm_joint2", "right_arm_joint3", 
                                "right_arm_joint4", "right_arm_joint5", "right_arm_joint6", "right_arm_joint7"]
        self.left_arm_joints = ["left_arm_joint1", "left_arm_joint2", "left_arm_joint3", 
                               "left_arm_joint4", "left_arm_joint5", "left_arm_joint6", "left_arm_joint7"]
        self.right_gripper_joints = ["right_gripper_joint1"]
        self.left_gripper_joints = ["left_gripper_joint1"]

    def display_menu(self):
        """显示当前菜单"""
        os.system('cls' if os.name == 'nt' else 'clear')  # 清屏
        
        if self.current_menu == "main":
            self.display_main_menu()
        elif self.current_menu == "part_selected":
            self.display_part_menu()
        elif self.current_menu == "joint_selected":
            self.display_joint_control_menu()

    def display_main_menu(self):
        """显示主菜单"""
        print("=== 🤖 机器人控制菜单 ===")
        print("1. 右臂控制")
        print("2. 左臂控制")
        print("3. 右夹爪控制")
        print("4. 左夹爪控制")
        print("5. 退出")
        print("\n使用数字键选择选项")

    def display_part_menu(self):
        """显示部位控制菜单"""
        print(f"=== 🎮 {self.selected_part} 控制 ===")
        
        # 获取当前状态
        if "arm" in self.selected_part:
            state = self.robot_socket.get_arm_state("right" if "right" in self.selected_part else "left")
            joints = self.right_arm_joints if "right" in self.selected_part else self.left_arm_joints
        else:
            state = self.robot_socket.get_gripper_state("right" if "right" in self.selected_part else "left")
            joints = self.right_gripper_joints if "right" in self.selected_part else self.left_gripper_joints
        
        # 显示关节状态
        for i, joint in enumerate(joints):
            pos = state.get(joint, {}).get('position', 0.0) if state else 0.0
            print(f"{i+1}. {joint}: {pos:.4f}")
        
        print(f"{len(joints)+1}. 返回主菜单")
        print("\n使用数字键选择关节，或按 'r' 返回")

    def display_joint_control_menu(self):
        """显示关节控制菜单"""
        part_type = "arm" if "arm" in self.selected_part else "gripper"
        side = "right" if "right" in self.selected_part else "left"
        
        # 获取当前状态
        if part_type == "arm":
            state = self.robot_socket.get_arm_state(side)
        else:
            state = self.robot_socket.get_gripper_state(side)
        
        joints = self.right_arm_joints if side == "right" and part_type == "arm" else \
                 self.left_arm_joints if side == "left" and part_type == "arm" else \
                 self.right_gripper_joints if side == "right" else self.left_gripper_joints
        
        current_pos = state.get(self.selected_joint, {}).get('position', 0.0) if state else 0.0
        
        print(f"=== 🎮 {self.selected_part} - {self.selected_joint} 控制 ===")
        print(f"当前位置: {current_pos:.4f}")
        print("1. 增加位置 (+)")
        print("2. 减少位置 (-)")
        print("3. 设置自定义位置")
        print("4. 返回上一级")
        print("5. 返回主菜单")
        print("\n使用数字键选择操作")

    async def handle_input(self, key):
        """处理键盘输入"""
        try:
            if self.current_menu == "main":
                await self.handle_main_menu_input(key)
            elif self.current_menu == "part_selected":
                await self.handle_part_menu_input(key)
            elif self.current_menu == "joint_selected":
                await self.handle_joint_control_input(key)
        except Exception as e:
            print(f"处理输入时出错: {e}")

    async def handle_main_menu_input(self, key):
        """处理主菜单输入"""
        if key == b'1':
            self.selected_part = "right_arm"
            self.current_menu = "part_selected"
        elif key == b'2':
            self.selected_part = "left_arm"
            self.current_menu = "part_selected"
        elif key == b'3':
            self.selected_part = "right_gripper"
            self.current_menu = "part_selected"
        elif key == b'4':
            self.selected_part = "left_gripper"
            self.current_menu = "part_selected"
        elif key == b'5':
            return "exit"
        return None

    async def handle_part_menu_input(self, key):
        """处理部位菜单输入"""
        part_type = "arm" if "arm" in self.selected_part else "gripper"
        side = "right" if "right" in self.selected_part else "left"
        
        joints = self.right_arm_joints if side == "right" and part_type == "arm" else \
                 self.left_arm_joints if side == "left" and part_type == "arm" else \
                 self.right_gripper_joints if side == "right" else self.left_gripper_joints
        
        if key == b'r':
            self.current_menu = "main"
            self.selected_part = None
        else:
            try:
                choice = int(key) - 1
                if 0 <= choice < len(joints):
                    self.selected_joint = joints[choice]
                    self.current_menu = "joint_selected"
                elif choice == len(joints):
                    self.current_menu = "main"
                    self.selected_part = None
            except ValueError:
                pass

    async def handle_joint_control_input(self, key):
        """处理关节控制输入"""
        part_type = "arm" if "arm" in self.selected_part else "gripper"
        side = "right" if "right" in self.selected_part else "left"
        
        # 获取当前状态
        if part_type == "arm":
            state = self.robot_socket.get_arm_state(side)
        else:
            state = self.robot_socket.get_gripper_state(side)
        
        current_pos = state.get(self.selected_joint, {}).get('position', 0.0) if state else 0.0
        
        if key == b'1':  # 增加位置
            new_pos = current_pos + self.joint_step
            await self.send_joint_command(side, part_type, self.selected_joint, new_pos)
        elif key == b'2':  # 减少位置
            new_pos = current_pos - self.joint_step
            await self.send_joint_command(side, part_type, self.selected_joint, new_pos)
        elif key == b'3':  # 设置自定义位置
            try:
                custom_pos = float(input("请输入目标位置: "))
                await self.send_joint_command(side, part_type, self.selected_joint, custom_pos)
            except ValueError:
                print("无效的位置值")
        elif key == b'4':  # 返回上一级
            self.current_menu = "part_selected"
            self.selected_joint = None
        elif key == b'5':  # 返回主菜单
            self.current_menu = "main"
            self.selected_part = None
            self.selected_joint = None

    async def send_joint_command(self, side, part_type, joint_name, position):
        """发送关节控制命令"""
        group_name = f"{side}_{part_type}"
        
        if part_type == "arm":
            joints = self.right_arm_joints if side == "right" else self.left_arm_joints
        else:
            joints = self.right_gripper_joints if side == "right" else self.left_gripper_joints
        
        # 获取当前所有关节位置
        if part_type == "arm":
            state = self.robot_socket.get_arm_state(side)
        else:
            state = self.robot_socket.get_gripper_state(side)
        
        positions = []
        for joint in joints:
            if joint == joint_name:
                positions.append(position)
            else:
                positions.append(state.get(joint, {}).get('position', 0.0) if state else 0.0)
        
        # 发送命令
        await self.robot_socket.send_joint_positions(group_name, joints, positions)
        print(f"已发送命令: {joint_name} -> {position:.4f}")


async def main():
    robot_ip = "127.0.0.1"
    robot_socket = RobotSocket(robot_ip)
    menu_controller = MenuController(robot_socket)
    
    connect_task = asyncio.create_task(robot_socket.connect())
    
    # 等待连接建立
    await asyncio.sleep(1)

    # # 保存原始终端设置（用于恢复）
    # old_settings = None
    # if os.name != 'nt':
    #     fd = sys.stdin.fileno()
    #     old_settings = termios.tcgetattr(fd)
    #     # 设置为 raw 模式，支持单字符输入
    #     tty.setraw(fd, termios.TCSANOW)
    
    try:
        exit_program = False
        while not exit_program:
            # 显示菜单
            menu_controller.display_menu()
            
            # 非阻塞地检查按键输入
            # if os.name == 'nt':  # Windows
            #     if msvcrt.kbhit():
            #         key = msvcrt.getch()
            #         result = await menu_controller.handle_input(key)
            #         if result == "exit":
            #             exit_program = True
            # else:  # Linux/macOS
            #     # 使用 select 检测是否有输入
            #     if select.select([sys.stdin], [], [], 0.1)[0]:  # 0.1秒超时
            #         key = sys.stdin.read(1).encode('utf-8')  # 读取一个字符并编码为 bytes，保持接口一致
            #         result = await menu_controller.handle_input(key)
            #         if result == "exit":
            #             exit_program = True
            if keyboard.is_pressed('1'):
                await menu_controller.handle_input(b'1')
            elif keyboard.is_pressed('2'):
                await menu_controller.handle_input(b'2')
            elif keyboard.is_pressed('3'):
                await menu_controller.handle_input(b'3')
            elif keyboard.is_pressed('4'):
                await menu_controller.handle_input(b'4')
            elif keyboard.is_pressed('5'):
                await menu_controller.handle_input(b'5')
            elif keyboard.is_pressed('r'):
                await menu_controller.handle_input(b'r')
            
            # 短暂休眠以减少CPU使用
            await asyncio.sleep(0.1)
            
    except KeyboardInterrupt:
        logger.info("👋 用户中断，正在关闭程序...")
    except Exception as e:
        logger.error(f"💥 主循环异常: {e}")
    finally:
        # # 恢复终端设置
        # if old_settings is not None:
        #     fd = sys.stdin.fileno()
        #     termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

        logger.info("🛑 正在关闭机器人连接...")
        await robot_socket.shutdown()
        if not connect_task.done():
            connect_task.cancel()
            try:
                await connect_task
            except asyncio.CancelledError:
                pass

        await asyncio.sleep(0.1)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("👋 用户中断，正在关闭程序...")
    except Exception as e:
        logger.error(f"💥 程序异常: {e}")
    finally:
        cv2.destroyAllWindows()
        logger.info("✅ 程序已安全退出")