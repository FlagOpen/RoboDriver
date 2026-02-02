import asyncio
import logging_mp
import threading
import websockets
import cv2
import os
import queue
import base64
import json
import numpy as np
import time

from typing import Dict, Any
from pathlib import Path
import sys

os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
current_dir = os.path.dirname(os.path.abspath(__file__))
# sys.path.append(os.path.join(current_dir, 'galbot'))
sys.path.insert(0, current_dir)

from galbot.core_proto import time_pb2, header_pb2
from galbot.spatial_proto import twist_pb2, pose_pb2
from galbot.singorix_proto import singorix_command_pb2, singorix_sensor_pb2, singorix_error_pb2, singorix_target_pb2
from galbot.sensor_proto import imu_pb2, image_pb2, camera_pb2, joy_pb2
from galbot.tf2_proto import tf2_message_pb2


logger = logging_mp.get_logger(__name__)
CONNECT_TIMEOUT_FRAME = 10


class RobotNode:
    pass

class SocketRobotNode(RobotNode):
    pass

class AsyncLoopManager:
    """管理异步事件循环的线程"""
    def __init__(self):
        self.loop = None
        self.thread = None
        self.running = False
        
    def start_loop(self):
        """在新线程中启动事件循环"""
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.running = True
        self.loop.run_forever()
    
    def start(self):
        """启动线程"""
        self.thread = threading.Thread(target=self.start_loop, daemon=True)
        self.thread.start()
        
        # 等待循环初始化
        for _ in range(10):  # 最多等待1秒
            if self.loop and self.loop.is_running():
                break
            time.sleep(0.1)
        else:
            raise RuntimeError("无法启动事件循环")
    
    def stop(self):
        """停止事件循环和线程"""
        if self.loop and self.loop.is_running():
            self.loop.call_soon_threadsafe(self.loop.stop)
            self.running = False
            if self.thread:
                self.thread.join(timeout=1.0)
    
    def run_coroutine(self, coro):
        """在事件循环线程中运行协程"""
        if self.loop and self.loop.is_running():
            return asyncio.run_coroutine_threadsafe(coro, self.loop)
        else:
            raise RuntimeError("事件循环未运行")


class GalbotG1AIOSDKRCRobotNode(SocketRobotNode):
    def __init__(self, robot_ip, bridge_port=10800):
        self.robot_ip = robot_ip
        self.bridge_port = bridge_port
        self.ws = None
        self.uri = f"ws://{self.robot_ip}:{self.bridge_port}"
        self.task = None  # listen 任务

        # 异步循环管理器
        self.loop_manager = AsyncLoopManager()

        # 状态存储（线程安全）
        self.latest_states = {}
        self.state_lock = threading.Lock()

        # 图像缓存（线程安全）
        # self.latest_images = {}  # topic -> numpy array
        self.image_lock = threading.Lock()

        self.recv_images: Dict[str, Any] = {}
        self.recv_follower_arm_right: list[float] = []
        self.recv_follower_arm_left: list[float] = []
        self.recv_follower_gripper_right: list[float] = []
        self.recv_follower_gripper_left: list[float] = []
        self.recv_follower_leg: list[float] = []
        self.recv_follower_head: list[float] = []
        self.recv_follower_chassis: list[float] = []
        self.recv_follower_chassis_velocity: list[float] = []

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
        self.connection_event = threading.Event()

    def start(self):
        """启动连接（非阻塞）"""
        self.loop_manager.start()
        self.loop_manager.run_coroutine(self.connect())
        
        # 等待连接建立或超时
        if not self.connection_event.wait(timeout=10.0):
            raise ConnectionError("连接机器人超时")

    async def connect(self):
        """建立 WebSocket 连接"""
        try:
            self.ws = await websockets.connect(self.uri)
            logger.info(f"✅ WebSocket 已连接: {self.uri}")
            self.running = True
            self.connection_event.set()
            self.task = asyncio.create_task(self.listen())
        except Exception as e:
            logger.error(f"❌ 连接失败: {e}")
            self.connection_event.set()  # 即使失败也设置事件，避免阻塞
            raise

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
                        logger.debug(f"⚠️ 未知操作类型: {op}")

                except json.JSONDecodeError:
                    logger.error(f"❌ JSON 解析失败: {message[:100]}...")
                except Exception as e:
                    logger.error(f"❌ 处理消息时出错: {e}")

        except websockets.exceptions.ConnectionClosed as e:
            logger.info(f"🔌 连接关闭: {e}")
        except Exception as e:
            logger.error(f"❌ 监听消息时发生错误: {e}")
        finally:
            self.running = False

    async def _process_protobuf_message(self, message):
        """处理 protobuf 消息"""
        topic = message.get("topic")
        type_str = message.get("type")
        data_b64 = message.get("data")

        if not all([topic, type_str, data_b64]):
            logger.error("❌ 缺少必要字段")
            return

        pb_class = self.protobuf_type_map.get(type_str)
        if not pb_class:
            logger.error(f"❌ 未知 protobuf 类型: {type_str}")
            return

        try:
            data_bytes = base64.b64decode(data_b64)
            if not data_bytes:
                raise ValueError(f"解码后得到空字节数据 (topic: {topic})")

            pb_message = pb_class()
            pb_message.ParseFromString(data_bytes)
            if pb_message is None:
                raise ValueError(f"创建protobuf消息对象失败 (topic: {topic})")

            # 处理图像：缓存到 latest_images，不在这里显示
            if any(cam in topic for cam in [
                "/right_arm_camera/color/image_raw",
                "/left_arm_camera/color/image_raw",
                "/front_head_camera/right_color/image_raw",
                "/front_head_camera/left_color/image_raw"
            ]) and isinstance(pb_message, image_pb2.CompressedImage):
                np_arr = np.frombuffer(pb_message.data, np.uint8)
                image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                if image is not None:
                    image = cv2.resize(image, (640, 480))
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    with self.image_lock:
                        if "right" in topic and "head" in topic:
                            self.recv_images["image_head_right"] = image
                        elif "left" in topic and "head" in topic:
                            self.recv_images["image_head_left"] = image
                        elif "right" in topic and "arm" in topic:
                            self.recv_images["image_arm_right"] = image
                        elif "left" in topic and "arm" in topic:
                            self.recv_images["image_arm_left"] = image

            # 处理传感器数据
            elif "singorix/wbcs/sensor" in topic:
                self._parse_and_store_joint_data(pb_message)

            # 通用状态存储
            with self.state_lock:
                self.latest_states[topic] = {
                    "message": pb_message,
                    "timestamp": message.get("pub_ts", 0),
                    "received": time.time_ns()
                }

        except Exception as e:
            logger.error(f"❌ 解析 protobuf 失败: {e}")

    async def _process_heartbeat(self, message):
        ts = message.get("ts", 0)
        logger.debug(f"💓 心跳时间戳: {ts}")

    async def _process_error(self, message):
        error_msg = message.get("msg", "未知错误")
        logger.error(f"❗ 错误消息: {error_msg}")

    def _parse_and_store_joint_data(self, sensor_msg):
        """解析 SingoriXSensor 消息，提取并存储 arm 和 gripper 数据"""
        if not sensor_msg.joint_sensor_map:
            return
        
        # logger.info(f"{sensor_msg.joint_sensor_map}")

        with self.state_lock:
            for group_name, joint_sensor in sensor_msg.joint_sensor_map.items():
                if not joint_sensor.name:
                    continue

                joint_data = list(joint_sensor.position)

                if group_name == "right_arm":
                    self.recv_follower_arm_right = joint_data
                elif group_name == "left_arm":
                    self.recv_follower_arm_left = joint_data
                elif group_name == "right_gripper":
                    self.recv_follower_gripper_right = joint_data
                elif group_name == "left_gripper":
                    self.recv_follower_gripper_left = joint_data
                elif group_name == "leg":
                    self.recv_follower_leg = joint_data
                elif group_name == "head":
                    self.recv_follower_head = joint_data
                elif group_name == "chassis":
                    self.recv_follower_chassis = joint_data
                    self.recv_follower_chassis_velocity = joint_sensor.velocity

                # logger.info(f"Group: {group_name}, Data: {joint_data}")

    def shutdown(self):
        """关闭连接"""
        self.running = False
        if self.loop_manager:
            self.loop_manager.run_coroutine(self._async_shutdown())
            self.loop_manager.stop()

    async def _async_shutdown(self):
        """异步关闭连接"""
        if self.ws:
            await self.ws.close()
        if self.task and not self.task.done():
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass
        logger.info("🔌 WebSocket 已关闭")