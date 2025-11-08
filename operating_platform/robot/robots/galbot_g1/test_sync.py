import sys
import os
import json
import base64
import struct

os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, 'galbot'))

from galbot.core_proto import time_pb2, header_pb2
from galbot.spatial_proto import twist_pb2, pose_pb2
from galbot.singorix_proto import singorix_command_pb2, singorix_sensor_pb2, singorix_error_pb2, singorix_target_pb2
from galbot.sensor_proto import imu_pb2, image_pb2, camera_pb2, joy_pb2
from galbot.tf2_proto import tf2_message_pb2

import socket
import threading
import time
import websocket

class RobotSocket:
    def __init__(self, robot_ip, bridge_port=10800):
        self.robot_ip = robot_ip
        self.bridge_port = bridge_port
        self.ws = None

        self.latest_states = {}
        self.state_lock = threading.Lock()

        self.protobuf_type_map = {
            "galbot.sensor_proto.CompressedImage": image_pb2.CompressedImage,
            "galbot.sensor_proto.CameraInfo": camera_pb2.CameraInfo,
            "galbot.singorix_proto.SingoriXSensor": singorix_sensor_pb2.SingoriXSensor,
            "galbot.singorix_proto.SingoriXError": singorix_error_pb2.SingoriXError,
            "galbot.singorix_proto.SingoriXTarget": singorix_target_pb2.SingoriXTarget,
            "galbot.tf2_proto.TF2Message": tf2_message_pb2.TF2Message,
            "galbot.sensor_proto.Joy": joy_pb2.Joy
        }

    def start_state_listener(self):
        """启动 WebSocket 连接并监听消息"""
        ws_url = f"ws://{self.robot_ip}:{self.bridge_port}"

        def on_message(ws, message):
            # message 是字符串（WebSocket 自动处理帧解码）
            try:
                msg_json = json.loads(message)
                op = msg_json.get("op")

                if op == "message":
                    self._process_protobuf_message(msg_json)
                elif op == "heartbeat":
                    self._process_heartbeat(msg_json)
                elif op == "error":
                    self._process_error(msg_json)

            except json.JSONDecodeError as e:
                print(f"JSON解析错误: {e}")
            except Exception as e:
                print(f"处理消息时出错: {e}")

        def on_error(ws, error):
            print(f"WebSocket错误: {error}")

        def on_close(ws, close_status_code, close_msg):
            print("WebSocket连接关闭")

        def on_open(ws):
            print("WebSocket连接已建立")

        # 创建 WebSocket App
        self.ws = websocket.WebSocketApp(
            ws_url,
            on_open=on_open,
            on_message=on_message,
            on_error=on_error,
            on_close=on_close
        )

        # 在新线程中运行 WebSocket
        def run_ws():
            self.ws.run_forever()

        thread = threading.Thread(target=run_ws, daemon=True)
        thread.start()
        print(f"已连接到 WebSocket: {ws_url}")

    def _process_protobuf_message(self, message):
        """处理protobuf消息（保持不变）"""
        topic = message.get("topic")
        type_str = message.get("type")
        data_b64 = message.get("data")

        if not all([topic, type_str, data_b64]):
            print("缺少必要的消息字段")
            return

        pb_class = self.protobuf_type_map.get(type_str)
        if not pb_class:
            print(f"未知的protobuf类型: {type_str}")
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

            print(f"接收到 {topic} 消息: {type_str}")

        except Exception as e:
            print(f"解析protobuf消息错误: {e}")

    def _process_heartbeat(self, message):
        ts = message.get("ts", 0)
        print(f"接收到心跳, 时间戳: {ts}")

    def _process_error(self, message):
        error_msg = message.get("msg", "未知错误")
        print(f"接收到错误: {error_msg}")

    def get_latest_state(self, topic):
        with self.state_lock:
            return self.latest_states.get(topic)

    def get_all_topics(self):
        with self.state_lock:
            return list(self.latest_states.keys())

    def shutdown(self):
        if self.ws:
            self.ws.close()


# 使用示例
if __name__ == "__main__":
    robot_ip = "127.0.0.1"  # 替换为你的机器人IP

    robot_socket = RobotSocket(robot_ip)
    robot_socket.start_state_listener()

    try:
        while True:
            # 获取所有主题
            topics = robot_socket.get_all_topics()
            print(f"当前活跃主题: {topics}")
            
            # 示例：获取特定主题的状态
            if "/singorix/wbcs/sensor" in topics:
                state = robot_socket.get_latest_state("/singorix/wbcs/sensor")
                if state:
                    print(f"传感器数据接收时间: {state['received']}")
                    # 这里可以根据需要访问具体的protobuf消息字段
            
            time.sleep(1.0)  # 等待1秒

    except KeyboardInterrupt:
        print("正在关闭...")
    finally:
        robot_socket.shutdown()
