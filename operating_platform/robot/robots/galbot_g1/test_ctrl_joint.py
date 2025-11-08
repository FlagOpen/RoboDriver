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

            if not data_bytes:
                raise ValueError(f"解码后得到空字节数据 (topic: {topic})")
            
            pb_message = pb_class()
            pb_message.ParseFromString(data_bytes)

            if pb_message is None:
                raise ValueError(f"创建protobuf消息对象失败 (topic: {topic})")
            
            # if "/right_arm_camera/color/image_raw" in topic:
            #     show_compressed_image_from_proto(pb_message)

            if "/right_arm_camera/color/image_raw" in topic:
                show_compressed_image_from_proto(pb_message, "Right Arm Camera")
            elif "/left_arm_camera/color/image_raw" in topic:
                show_compressed_image_from_proto(pb_message, "Left Arm Camera")
            elif "/front_head_camera/right_color/image_raw" in topic:
                show_compressed_image_from_proto(pb_message, "Front Head Right Camera")
            elif "/front_head_camera/left_color/image_raw" in topic:
                show_compressed_image_from_proto(pb_message, "Front Head Left Camera")
            elif "singorix/wbcs/sensor" in topic:
            # elif "singorix_omnilink/scaled_device_robot_data" in topic:
                print(f"📊 Sensor message size: {len(data_bytes)} bytes")
                # print(f"pb_message: {pb_message}")
                show_sensor_from_proto(pb_message)

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

def show_compressed_image_from_proto(compressed_image_msg, window_name="Image"):
    """
    显示 CompressedImage protobuf 消息中的图像
    """
    data = compressed_image_msg.data
    # print(f"Image Header frame: {compressed_image_msg.header.frame_id}")
    # format = compressed_image_msg.format  # 可选，OpenCV 自动识别

    np_arr = np.frombuffer(data, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError("无法解码图像，请检查数据是否完整或格式是否支持。")

    cv2.imshow(window_name, image)
    cv2.waitKey(1)

def show_sensor_from_proto(sensor_msg):
    # print("✅ Get Joint successfully")
    # print(f"Header frame: {sensor_msg.header.frame_id}")
    # print(f"sensor_msg: {sensor_msg}")
    # # print(MessageToJson(sensor_msg))

    # # 调试：打印所有字段
    # print("Fields present:", [field.name for field, value in sensor_msg.ListFields()])

    if not sensor_msg.joint_sensor_map:
        print("⚠️  No joint data in sensor message.")
        return
    
    field = sensor_msg.joint_sensor_map
    
    # if hasattr(field, 'items'):  # 是 map
    #     for joint_name, joint_data in field.items():
    #         print(f"Joint: {joint_name}, Pos: {joint_data.position}")
    # else:  # 是 repeated（旧格式）
    #     for entry in field:  # 假设每个 entry 有 .key 和 .value
    #         joint_name = entry.key
    #         joint_data = entry.value
    #         print(f"Joint: {joint_name}, Pos: {joint_data.position}")

    for group_name, joint_sensor in sensor_msg.joint_sensor_map.items():
        # print(f"=== Joint Group: {group_name} ===")
        # if joint_sensor.header:
        #     print(f"  Header: {joint_sensor.header.timestamp.sec}.{joint_sensor.header.timestamp.nanosec}")

        n = len(joint_sensor.name)
        if n == 0:
            print("  ⚠️  No joints in this group.")
            continue

        for i in range(n):
            name = joint_sensor.name[i] if i < len(joint_sensor.name) else "N/A"
            pos = joint_sensor.position[i] if i < len(joint_sensor.position) else 0.0
            vel = joint_sensor.velocity[i] if i < len(joint_sensor.velocity) else 0.0
            eff = joint_sensor.effort[i] if i < len(joint_sensor.effort) else 0.0
            curr = joint_sensor.current[i] if i < len(joint_sensor.current) else 0.0

            # print(f"  Joint[{i}]: {name} | pos={pos:.4f} rad | vel={vel:.4f} rad/s | eff={eff:.4f} Nm | curr={curr:.4f} A")


async def main():
    robot_ip = "127.0.0.1"
    robot_socket = RobotSocket(robot_ip)

    # 启动连接任务（后台运行）
    connect_task = asyncio.create_task(robot_socket.connect())

    try:
        # 等待连接建立（可选：加超时或重试机制）
        await asyncio.wait_for(connect_task, timeout=10.0)
        logger.info("🔌 机器人连接已建立")

    except asyncio.TimeoutError:
        logger.error("⏳ 连接超时，程序退出")
        await robot_socket.shutdown()
        return
    except Exception as e:
        logger.error(f"❌ 连接失败: {e}")
        await robot_socket.shutdown()
        return
    
    try:
         while True:
            # 非阻塞地轮询状态（不影响 WebSocket 接收）
            topics = robot_socket.get_all_topics()
            print(f"📊 当前活跃主题: {topics}")

            if "singorix/wbcs/sensor" in topics:
                state = robot_socket.get_latest_state("singorix/wbcs/sensor")
                if state:
                    print(f"⏱️ 传感器数据接收时间: {state['received']}")

            await asyncio.sleep(1.0)

    # except KeyboardInterrupt:
    #     logger.info("👋 key 用户中断，正在关闭程序...")
    except Exception as e:
        logger.error(f"💥 主循环异常: {e}")
    finally:
        logger.info("🛑 正在关闭机器人连接...")
        await robot_socket.shutdown()
        # 确保 connect_task 完成或取消
        if not connect_task.done():
            connect_task.cancel()
            try:
                await connect_task
            except asyncio.CancelledError:
                pass  # 正常取消，忽略

        await asyncio.sleep(0.1)


# ========================
# 启动入口
# ========================
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
