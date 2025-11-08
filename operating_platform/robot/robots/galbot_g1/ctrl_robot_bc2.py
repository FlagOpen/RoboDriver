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


async def main():
    robot_ip = "127.0.0.1"
    robot_socket = RobotSocket(robot_ip)
    connect_task = asyncio.create_task(robot_socket.connect())
    
    # 等待连接建立
    await asyncio.sleep(1)
    
    # 定义关节名称 (根据实际机器人调整)
    right_arm_joints = ["right_arm_joint1", "right_arm_joint2", "right_arm_joint3", "right_arm_joint4", 
                        "right_arm_joint5", "right_arm_joint6", "right_arm_joint7"]
    
    left_arm_joints = ["left_arm_joint1", "left_arm_joint2", "left_arm_joint3", "left_arm_joint4", 
                       "left_arm_joint5", "left_arm_joint6", "left_arm_joint7"]
    
    right_gripper_joints = ["right_gripper_joint1",]
    left_gripper_joints = ["left_gripper_joint1",]
    
    try:
        while True:
            # 非阻塞地轮询状态
            topics = robot_socket.get_all_topics()
            
            if "singorix/wbcs/sensor" in topics:
                state = robot_socket.get_latest_state("singorix/wbcs/sensor")
                if state:
                    print(f"⏱️ 传感器数据接收时间: {state['received']}")

            # 打印存储的 arm 和 gripper 数据
            right_arm = robot_socket.get_arm_state("right")
            left_arm = robot_socket.get_arm_state("left")
            right_gripper = robot_socket.get_gripper_state("right")
            left_gripper = robot_socket.get_gripper_state("left")

            print("\n=== 🤖 实时关节状态 ===")
            if right_arm:
                pos_str = ", ".join([f"{k}: {v['position']:.4f}" for k, v in right_arm.items()])
                print(f"👉 右臂: {pos_str}")
            if left_arm:
                pos_str = ", ".join([f"{k}: {v['position']:.4f}" for k, v in left_arm.items()])
                print(f"👈 左臂: {pos_str}")
            if right_gripper:
                pos_str = ", ".join([f"{k}: {v['position']:.4f}" for k, v in right_gripper.items()])
                print(f"✋ 右夹爪: {pos_str}")
            if left_gripper:
                pos_str = ", ".join([f"{k}: {v['position']:.4f}" for k, v in left_gripper.items()])
                print(f"✋ 左夹爪: {pos_str}")
            
            # 示例: 发送控制命令
            print("\n=== 🎮 控制命令示例 ===")
            print("1. 发送右臂关节位置命令")
            print("2. 发送左臂关节位置命令")
            print("3. 发送右夹爪命令")
            print("4. 发送左夹爪命令")
            print("5. 退出")
            
            try:
                choice = input("请选择操作 (1-5): ")
                
                if choice == "1":
                    # 示例: 发送右臂关节位置
                    positions = [0.1, -0.2, 0.3, -0.4, 0.5, -0.6]  # 换成从获取的位置开始变化
                    # await robot_socket.send_joint_positions("right_arm", right_arm_joints, positions)
                    
                elif choice == "2":
                    # 示例: 发送左臂关节位置
                    positions = [-0.1, 0.2, -0.3, 0.4, -0.5, 0.6]  # 换成从获取的位置开始变化
                    # await robot_socket.send_joint_positions("left_arm", left_arm_joints, positions)
                    
                elif choice == "3":
                    # 示例: 发送右夹爪命令
                    # position = 0.5  # 夹爪位置 (0-1)
                    # await robot_socket.send_gripper_command("right_gripper", position)
                    position = [0.0,]
                    await robot_socket.send_joint_positions("right_gripper", right_gripper_joints, position)
                    
                elif choice == "4":
                    # 示例: 发送左夹爪命令
                    # position = 0.5  # 夹爪位置 (0-1)
                    # await robot_socket.send_gripper_command("left_gripper", position)

                    position = [0.0,]
                    await robot_socket.send_joint_positions("left_gripper", left_gripper_joints, position)
                    
                elif choice == "5":
                    break
                    
            except EOFError:
                # 处理无输入的情况
                await asyncio.sleep(1)
                continue
                
            await asyncio.sleep(1.0)
            
    except KeyboardInterrupt:
        logger.info("👋 用户中断，正在关闭程序...")
    except Exception as e:
        logger.error(f"💥 主循环异常: {e}")
    finally:
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