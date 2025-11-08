import asyncio
import json
import base64
import time
import os
import sys

os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, 'galbot'))

# Protobuf imports (保持不变)
from galbot.core_proto import time_pb2, header_pb2
from galbot.spatial_proto import twist_pb2, pose_pb2
from galbot.singorix_proto import singorix_command_pb2, singorix_sensor_pb2, singorix_error_pb2, singorix_target_pb2
from galbot.sensor_proto import imu_pb2, image_pb2, camera_pb2, joy_pb2
from galbot.tf2_proto import tf2_message_pb2

import threading  # 仅用于非异步部分（如外部调用 get_latest_state）

class RobotSocket:
    def __init__(self, robot_ip, bridge_port=10800):
        self.robot_ip = robot_ip
        self.bridge_port = bridge_port
        self.ws = None
        self.uri = f"ws://{self.robot_ip}:{self.bridge_port}"

        # 状态存储（线程安全）
        self.latest_states = {}
        self.state_lock = threading.Lock()

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
            pb_message = pb_class()
            pb_message.ParseFromString(data_bytes)

            with self.state_lock:
                self.latest_states[topic] = {
                    "message": pb_message,
                    "timestamp": message.get("pub_ts", 0),
                    "received": time.time_ns()
                }

            print(f"📥 接收到 {topic} 消息: {type_str}")

        except Exception as e:
            print(f"❌ 解析 protobuf 失败: {e}")

    async def _process_heartbeat(self, message):
        ts = message.get("ts", 0)
        print(f"💓 心跳时间戳: {ts}")

    async def _process_error(self, message):
        error_msg = message.get("msg", "未知错误")
        print(f"❗ 错误消息: {error_msg}")

    def get_latest_state(self, topic):
        """同步方法，供外部调用"""
        with self.state_lock:
            return self.latest_states.get(topic)

    def get_all_topics(self):
        """同步方法，获取所有主题"""
        with self.state_lock:
            return list(self.latest_states.keys())

    async def shutdown(self):
        """关闭连接"""
        self.running = False
        if self.ws:
            await self.ws.close()
        print("🔌 WebSocket 已关闭")


# ========================
# 异步主程序示例
# ========================
async def main():
    robot_ip = "127.0.0.1"
    robot_socket = RobotSocket(robot_ip)

    # 启动连接（不会阻塞，listen 在后台运行）
    connect_task = asyncio.create_task(robot_socket.connect())

    try:
        while True:
            # 非阻塞地轮询状态（不影响 WebSocket 接收）
            topics = robot_socket.get_all_topics()
            print(f"📊 当前活跃主题: {topics}")

            if "/singorix/wbcs/sensor" in topics:
                state = robot_socket.get_latest_state("/singorix/wbcs/sensor")
                if state:
                    print(f"⏱️ 传感器数据接收时间: {state['received']}")

            await asyncio.sleep(1.0)  # 异步等待，不阻塞事件循环

    except KeyboardInterrupt:
        print("🛑 正在关闭...")
    finally:
        await robot_socket.shutdown()
        await connect_task  # 等待连接任务结束


# ========================
# 启动入口
# ========================
if __name__ == "__main__":
    import websockets  # 必须导入

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("👋 程序被用户中断")