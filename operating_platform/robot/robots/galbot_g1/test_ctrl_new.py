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
import logging  # 临时备用

os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, 'galbot'))

from galbot.core_proto import time_pb2, header_pb2
from galbot.spatial_proto import twist_pb2, pose_pb2
from galbot.singorix_proto import singorix_command_pb2, singorix_sensor_pb2, singorix_error_pb2, singorix_target_pb2
from galbot.sensor_proto import imu_pb2, image_pb2, camera_pb2, joy_pb2
from galbot.tf2_proto import tf2_message_pb2

# 如果 colored_logging 有问题，临时用标准 logger
try:
    from operating_platform.utils.colored_logging import setup_colored_logger
    logger = setup_colored_logger(__name__)
except Exception:
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class RobotSocket:
    def __init__(self, robot_ip, bridge_port=10800):
        self.robot_ip = robot_ip
        self.bridge_port = bridge_port
        self.ws = None
        self.uri = f"ws://{self.robot_ip}:{self.bridge_port}"
        self.task = None  # listen 任务

        # 状态存储（线程安全）
        self.latest_states = {}
        self.state_lock = threading.Lock()

        # 图像缓存（线程安全）
        self.latest_images = {}  # topic -> numpy array
        self.image_lock = threading.Lock()

        # 新增：结构化存储特定关节组数据
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

    async def connect(self):
        """建立 WebSocket 连接"""
        try:
            self.ws = await websockets.connect(self.uri)
            logger.info(f"✅ WebSocket 已连接: {self.uri}")
            self.running = True
            self.task = asyncio.create_task(self.listen())
        except Exception as e:
            logger.error(f"❌ 连接失败: {e}")

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
                        logger.warning(f"⚠️ 未知操作类型: {op}")

                except json.JSONDecodeError:
                    logger.error(f"❌ JSON 解析失败: {message[:100]}...")
                except Exception as e:
                    logger.error(f"❌ 处理消息时出错: {e}")

        except websockets.exceptions.ConnectionClosed as e:
            logger.info(f"🔌 连接关闭: {e}")
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

            # ��️ 处理图像：缓存到 latest_images，不在这里显示
            if any(cam in topic for cam in [
                "/right_arm_camera/color/image_raw",
                "/left_arm_camera/color/image_raw",
                "/front_head_camera/right_color/image_raw",
                "/front_head_camera/left_color/image_raw"
            ]) and isinstance(pb_message, image_pb2.CompressedImage):
                np_arr = np.frombuffer(pb_message.data, np.uint8)
                image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                if image is not None:
                    with self.image_lock:
                        self.latest_images[topic] = image

            # �� 处理传感器数据
            elif "singorix/wbcs/sensor" in topic:
                self._parse_and_store_joint_data(pb_message)

            # 通用状态存储
            with self.state_lock:
                self.latest_states[topic] = {
                    "message": pb_message,
                    "timestamp": message.get("pub_ts", 0),
                    "received": time.time_ns()
                }

            # logger.info(f"📥 接收到 {topic} 消息: {type_str}")

        except Exception as e:
            logger.error(f"❌ 解析 protobuf 失败: {e}")

    async def _process_heartbeat(self, message):
        ts = message.get("ts", 0)
        logger.info(f"💓 心跳时间戳: {ts}")

    async def _process_error(self, message):
        error_msg = message.get("msg", "未知错误")
        logger.error(f"❗ 错误消息: {error_msg}")

    def _parse_and_store_joint_data(self, sensor_msg):
        """解析 SingoriXSensor 消息，提取并存储 arm 和 gripper 数据"""
        if not sensor_msg.joint_sensor_map:
            return

        with self.state_lock:
            for group_name, joint_sensor in sensor_msg.joint_sensor_map.items():
                n = len(joint_sensor.name)
                if n == 0:
                    continue

                joint_data = {}
                for i in range(n):
                    name = joint_sensor.name[i] if i < len(joint_sensor.name) else f"joint{i}"
                    joint_data[name] = {
                        "position": joint_sensor.position[i] if i < len(joint_sensor.position) else 0.0,
                        "velocity": joint_sensor.velocity[i] if i < len(joint_sensor.velocity) else 0.0,
                        "effort": joint_sensor.effort[i] if i < len(joint_sensor.effort) else 0.0,
                        "current": joint_sensor.current[i] if i < len(joint_sensor.current) else 0.0,
                    }

                if group_name == "right_arm":
                    self.arm_joint_data["right_arm"] = joint_data
                elif group_name == "left_arm":
                    self.arm_joint_data["left_arm"] = joint_data
                elif group_name == "right_gripper":
                    self.gripper_data["right_gripper"] = joint_data
                elif group_name == "left_gripper":
                    self.gripper_data["left_gripper"] = joint_data

    def get_arm_state(self, side):
        key = f"{side}_arm"
        with self.state_lock:
            return self.arm_joint_data.get(key, {}).copy()

    def get_gripper_state(self, side):
        key = f"{side}_gripper"
        with self.state_lock:
            return self.gripper_data.get(key, {}).copy()

    def get_all_arm_states(self):
        with self.state_lock:
            return {k: v.copy() for k, v in self.arm_joint_data.items()}

    def get_all_gripper_states(self):
        with self.state_lock:
            return {k: v.copy() for k, v in self.gripper_data.items()}

    def get_latest_state(self, topic):
        with self.state_lock:
            return self.latest_states.get(topic)

    def get_all_topics(self):
        with self.state_lock:
            return list(self.latest_states.keys())

    def get_latest_image(self, topic):
        with self.image_lock:
            return self.latest_images.get(topic)

    def get_all_image_topics(self):
        with self.image_lock:
            return list(self.latest_images.keys())

    async def shutdown(self):
        """关闭连接"""
        self.running = False
        if self.ws:
            await self.ws.close()
        if self.task and not self.task.done():
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass
        logger.info("🔌 WebSocket 已关闭")


async def main():
    robot_ip = "127.0.0.1"
    robot_socket = RobotSocket(robot_ip)
    await robot_socket.connect()

    # # 等待连接建立
    # try:
    #     await asyncio.wait_for(asyncio.sleep(0.1), timeout=10.0)
    #     if not robot_socket.running:
    #         raise Exception("WebSocket 未成功连接")
    #     logger.info("🔌 机器人连接已建立")
    # except (asyncio.TimeoutError, Exception) as e:
    #     logger.error(f"⏳ 连接失败: {e}")
    #     await robot_socket.shutdown()
    #     return

    # ��️ 图像窗口映射
    topic_to_window = {
        "/right_arm_camera/color/image_raw": "Right Arm Camera",
        "/left_arm_camera/color/image_raw": "Left Arm Camera",
        "/front_head_camera/right_color/image_raw": "Front Head Right Camera",
        "/front_head_camera/left_color/image_raw": "Front Head Left Camera",
    }

    try:
        while robot_socket.running:
            # 显示所有缓存图像（在主线程安全调用）
            for topic, window_name in topic_to_window.items():
                img = robot_socket.get_latest_image(topic)
                if img is not None:
                    cv2.imshow(window_name, img)
            cv2.waitKey(1)  # 非阻塞刷新

            # 打印状态
            topics = robot_socket.get_all_topics()
            logger.info(f"📊 当前活跃主题: {topics}")

            # 打印关节数据
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

            await asyncio.sleep(1.0)

    # except KeyboardInterrupt:
    #     logger.info("👋 用户中断，正在关闭程序...")
    except Exception as e:
        logger.error(f"💥 主循环异常: {e}")
    finally:
        logger.info("🛑 正在关闭机器人连接...")
        await robot_socket.shutdown()
        cv2.destroyAllWindows()
        logger.info("✅ 主程序已安全退出")


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