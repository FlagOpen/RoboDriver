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


from operating_platform.robot.robots.utils import RobotDeviceNotConnectedError
from operating_platform.robot.robots.configs import GalbotG1RobotConfig
from operating_platform.robot.robots.com_configs.cameras import CameraConfig, OpenCVCameraConfig
from operating_platform.robot.robots.camera import Camera
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


async def Robot_main(robot_socket: RobotSocket):

    await robot_socket.connect()

    # 等待连接建立
    try:
        await asyncio.wait_for(asyncio.sleep(0.1), timeout=10.0)
        if not robot_socket.running:
            raise Exception("WebSocket 未成功连接")
        logger.info("🔌 机器人连接已建立")
    except (asyncio.TimeoutError, Exception) as e:
        logger.error(f"⏳ 连接失败: {e}")
        await robot_socket.shutdown()
        return

    # 图像窗口映射
    topic_to_window = {
        "/right_arm_camera/color/image_raw": "Right Arm Camera",
        "/left_arm_camera/color/image_raw": "Left Arm Camera",
        "/front_head_camera/right_color/image_raw": "Front Head Right Camera",
        "/front_head_camera/left_color/image_raw": "Front Head Left Camera",
    }

    try:
        while robot_socket.running:


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

    except Exception as e:
        logger.error(f"💥 主循环异常: {e}")
    finally:
        logger.info("🛑 正在关闭机器人连接...")

        cv2.destroyAllWindows()
        logger.info("✅ 主程序已安全退出")


class GalbotG1Manipulator:
    def __init__(self, config: GalbotG1RobotConfig):
        self.config = config
        self.robot_type = self.config.type

        self.use_videos = self.config.use_videos

        self.microphones = self.config.microphones

        robot_ip = "127.0.0.1"
        self.robot_socket = RobotSocket(robot_ip)
        asyncio.run(Robot_main(self.robot_socket))


        self.leader_arms = {}
        self.leader_arms['main_leader'] = self.config.leader_arms["main"]

        self.follower_arms = {}
        self.follower_arms['main_follower'] = self.config.follower_arms["main"]

        self.cameras = make_cameras_from_configs(self.config.cameras)
        
        self.connect_excluded_cameras = ["image_pika_pose"]

        self.recv_image_thread = threading.Thread(target=recv_image_server, daemon=True)
        self.recv_image_thread.start()

        self.recv_joint_thread = threading.Thread(target=recv_joint_server, daemon=True)
        self.recv_joint_thread.start()

        
        self.is_connected = False
        self.logs = {}



    def get_motor_names(self, arms: dict[str, dict]) -> list:
        return [f"{arm}_{motor}" for arm, bus in arms.items() for motor in bus.motors]

    @property
    def camera_features(self) -> dict:
        cam_ft = {}
        for cam_key, cam in self.cameras.items():
            key = f"observation.images.{cam_key}"
            cam_ft[key] = {
                "shape": (cam.height, cam.width, cam.channels),
                "names": ["height", "width", "channels"],
                "info": None,
            }
        return cam_ft
    
    @property
    def microphone_features(self) -> dict:
        mic_ft = {}
        for mic_key, mic in self.microphones.items():
            key = f"observation.audio.{mic_key}"
            mic_ft[key] = {
                "shape": (1,),
                "names": ["channels"],
                "info": None,
            }
        return mic_ft
    
    @property
    def motor_features(self) -> dict:
        action_names = self.get_motor_names(self.leader_arms)
        state_names = self.get_motor_names(self.follower_arms)
        return {
            "action": {
                "dtype": "float32",
                "shape": (len(action_names),),
                "names": action_names,
            },
            "observation.state": {
                "dtype": "float32",
                "shape": (len(state_names),),
                "names": state_names,
            },
        }
    
    def connect(self):
        timeout = 50  # 统一的超时时间（秒）
        start_time = time.perf_counter()

        # 定义所有需要等待的条件及其错误信息
        conditions = [
            (
                lambda: all(name in recv_images for name in self.cameras if name not in self.connect_excluded_cameras),
                lambda: [name for name in self.cameras if name not in recv_images],
                "等待摄像头图像超时"
            ),
            (
                lambda: all(
                    any(name in key for key in recv_joint)
                    for name in self.leader_arms
                ),
                lambda: [name for name in self.leader_arms if not any(name in key for key in recv_joint)],
                "等待主臂关节角度超时"
            ),
            (
                lambda: all(
                    any(name in key for key in recv_joint)
                    for name in self.follower_arms
                ),
                lambda: [name for name in self.follower_arms if not any(name in key for key in recv_joint)],
                "等待从臂关节角度超时"
            ),
        ]

        # 跟踪每个条件是否已完成
        completed = [False] * len(conditions)

        while True:
            # 检查每个未完成的条件
            for i in range(len(conditions)):
                if not completed[i]:
                    condition_func = conditions[i][0]
                    if condition_func():
                        completed[i] = True

            # 如果所有条件都已完成，退出循环
            if all(completed):
                break

            # 检查是否超时
            if time.perf_counter() - start_time > timeout:
                failed_messages = []
                for i in range(len(completed)):
                    if not completed[i]:
                        condition_func, get_missing, base_msg = conditions[i]
                        missing = get_missing()

                        # 重新检查条件是否满足（可能刚好在最后一次检查后满足）
                        if condition_func():
                            completed[i] = True
                            continue

                        # 如果没有 missing，也视为满足
                        if not missing:
                            completed[i] = True
                            continue

                        # 计算已接收的项
                        if i == 0:
                            received = [name for name in self.cameras if name not in missing]
                        else:
                            received = [name for name in self.follower_arms if name not in missing]

                        # 构造错误信息
                        msg = f"{base_msg}: 未收到 [{', '.join(missing)}]; 已收到 [{', '.join(received)}]"
                        failed_messages.append(msg)

                # 如果所有条件都已完成，break
                if not failed_messages:
                    break

                # 抛出超时异常
                raise TimeoutError(f"连接超时，未满足的条件: {'; '.join(failed_messages)}")

            # 减少 CPU 占用
            time.sleep(0.01)

        # ===== 新增成功打印逻辑 =====
        success_messages = []
        # 摄像头连接状态
        if conditions[0][0]():
            cam_received = [name for name in self.cameras 
                        if name in recv_images and name not in self.connect_excluded_cameras]
            success_messages.append(f"摄像头: {', '.join(cam_received)}")

        # 主臂数据状态
        arm_data_types = ["主臂关节角度",]
        for i, data_type in enumerate(arm_data_types, 1):
            if conditions[i][0]():
                arm_received = [name for name in self.leader_arms 
                            if any(name in key for key in (recv_joint,)[i-1])]
                success_messages.append(f"{data_type}: {', '.join(arm_received)}")
        
        # 从臂数据状态
        arm_data_types = ["从臂关节角度",]
        for i, data_type in enumerate(arm_data_types, 1):
            if conditions[i][0]():
                arm_received = [name for name in self.follower_arms 
                            if any(name in key for key in (recv_joint,)[i-1])]
                success_messages.append(f"{data_type}: {', '.join(arm_received)}")
        
        # 打印成功连接信息
        print("\n[连接成功] 所有设备已就绪:")
        for msg in success_messages:
            print(f"  - {msg}")
        print(f"  总耗时: {time.perf_counter() - start_time:.2f}秒\n")
        # ===========================

        self.is_connected = True
    
    @property
    def features(self):
        return {**self.motor_features, **self.camera_features}

    @property
    def has_camera(self):
        return len(self.cameras) > 0

    @property
    def num_cameras(self):
        return len(self.cameras)

    def teleop_step(
        self, record_data=False, 
    ) -> None | tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:

        if not self.is_connected:
            raise RobotDeviceNotConnectedError(
                "Aloha is not connected. You need to run `robot.connect()`."
            )

        if not record_data:
            return

        follower_joint = {}
        for name in self.follower_arms:
            for match_name in recv_joint:
                if name in match_name:
                    now = time.perf_counter()

                    byte_array = np.zeros(6, dtype=np.float32)
                    pose_read = recv_joint[match_name]

                    byte_array[:6] = pose_read[:]
                    byte_array = np.round(byte_array, 3)
                    
                    follower_joint[name] = torch.from_numpy(byte_array)

                    self.logs[f"read_follower_{name}_joint_dt_s"] = time.perf_counter() - now
                    
        leader_joint = {}
        for name in self.leader_arms:
            for match_name in recv_joint:
                if name in match_name:
                    now = time.perf_counter()

                    byte_array = np.zeros(6, dtype=np.float32)
                    pose_read = recv_joint[match_name]

                    byte_array[:6] = pose_read[:]
                    byte_array = np.round(byte_array, 3)
                    
                    leader_joint[name] = torch.from_numpy(byte_array)

                    self.logs[f"read_leader_{name}_joint_dt_s"] = time.perf_counter() - now

        #记录当前关节角度
        state = []
        for name in self.follower_arms:
            if name in follower_joint:
                state.append(follower_joint[name])
        state = torch.cat(state)

        #将关节目标位置添加到 action 列表中
        action = []
        for name in self.leader_arms:
            if name in leader_joint:
                action.append(leader_joint[name])
        action = torch.cat(action)

        # Capture images from cameras
        images = {}
        for name in self.cameras:
            now = time.perf_counter()
            images[name] = recv_images[name]
            images[name] = torch.from_numpy(images[name])
            self.logs[f"read_camera_{name}_dt_s"] = time.perf_counter() - now

        # Populate output dictionnaries and format to pytorch
        obs_dict, action_dict = {}, {}
        obs_dict["observation.state"] = state
        action_dict["action"] = action
        for name in self.cameras:
            obs_dict[f"observation.images.{name}"] = images[name]

        # print("end teleoperate record")
        return obs_dict, action_dict


    # def capture_observation(self):
    #     if not self.is_connected:
    #         raise RobotDeviceNotConnectedError(
    #             "KochRobot is not connected. You need to run `robot.connect()`."
    #         )

    #     #调用从臂api获取当前关节角度 
    #     for name in self.leader_arms:
    #         now = time.perf_counter()
    #         self.pDll.Get_Joint_Degree(self.nSocket,self.joint_obs_read)  
    #         #夹爪通信获取当前夹爪开合度
    #         #   giper_read=ctypes.c_int()
    #         #   self.pDll.Get_Read_Holding_Registers(self.nSocket,1,40005,1,ctypes.byref(giper_read))
    #         #   #八位数组存储关节和夹爪数据
    #         self.joint_obs_present[:7]=self.joint_obs_read[:]
    #         #   self.joint_obs_present[7]=giper_read.value
    #         if self.gipflag_send==1:
    #             self.joint_obs_present[7]=100
    #         elif self.gipflag_send==0:
    #             self.joint_obs_present[7]=10
    #         # self.joint_obs_present = np.zeros(8)  # 创建一个包含八个0的 NumPy 数组
    #         self.logs[f"read_follower_{name}_pos_dt_s"] = time.perf_counter() - now

    #     # Create state by concatenating follower current position
    #     #上传当前机械臂状态
    #     state = []
    #     self.joint_obs_present = np.round(self.joint_obs_present, 2)
    #     joint_array_np = np.array( self.joint_obs_present)
    #     state = np.array([joint_array_np], dtype=np.float32)
    #     state = np.concatenate(state, dtype=np.float32)

    #     # Capture images from cameras
    #     images = {}
    #     for name in self.cameras:
    #         now = time.perf_counter()
    #         images[name] = self.cameras[name].async_read()
    #         self.logs[f"read_camera_{name}_dt_s"] = self.cameras[name].logs["delta_timestamp_s"]
    #         self.logs[f"async_read_camera_{name}_dt_s"] = time.perf_counter() - now

    #     # Populate output dictionnaries and format to pytorch
    #     obs_dict = {}
    #     obs_dict["observation.state"] = torch.from_numpy(state)
    #     for name in self.cameras:
    #         # Convert to pytorch format: channel first and float32 in [0,1]
    #         img = torch.from_numpy(images[name])
    #         img = img.type(torch.float32) / 255
    #         img = img.permute(2, 0, 1).contiguous()
    #         obs_dict[f"observation.images.{name}"] = img
    #     return obs_dict    def capture_observation(self):

    # def capture_observation(self):
    #     if not self.is_connected:
    #         raise RobotDeviceNotConnectedError(
    #             "KochRobot is not connected. You need to run `robot.connect()`."
    #         )

    #     follower_pos = {}
    #     for name in self.follower_arms:
    #         now = time.perf_counter()
    #         eight_byte_array = np.zeros(8, dtype=np.float32)
    #         joint_obs_read = self.follower_arms[name].async_read_joint_degree()

    #         #夹爪通信获取当前夹爪开合度
    #         # giper_read=ctypes.c_int()
    #         # self.pDll.Get_Read_Holding_Registers(self.nSocket,1,40000,1,ctypes.byref(giper_read))
    #         #   #八位数组存储关节和夹爪数据
    #         eight_byte_array[:7] = joint_obs_read[:]
    #         # self.joint_obs_present[7]=giper_read.value
    #         eight_byte_array[7] = self.follower_arms[name].old_grasp
    #         # self.joint_obs_present = np.zeros(8)  # 创建一个包含八个0的 NumPy 数组
    #         eight_byte_array = np.round(eight_byte_array, 2)
    #         follower_pos[name] = torch.from_numpy(eight_byte_array)
    #         self.logs[f"read_follower_{name}_pos_dt_s"] = time.perf_counter() - now

    #     # Create state by concatenating follower current position
    #     #上传当前机械臂状态
    #     state = []
    #     for name in self.follower_arms:
    #         if name in follower_pos:
    #             state.append(follower_pos[name])    
    #     state = torch.cat(state)

    #     # Capture images from cameras
    #     images = {}
    #     for name in self.cameras:
    #         now = time.perf_counter()
    #         images[name] = self.cameras[name].async_read()
    #         images[name] = torch.from_numpy(images[name])
    #         self.logs[f"read_camera_{name}_dt_s"] = self.cameras[name].logs["delta_timestamp_s"]
    #         self.logs[f"async_read_camera_{name}_dt_s"] = time.perf_counter() - now

    #     # Populate output dictionnaries and format to pytorch
    #     obs_dict = {}
    #     obs_dict["observation.state"] = state
    #     for name in self.cameras:
    #         obs_dict[f"observation.images.{name}"] = images[name]
    #     return obs_dict

    def send_action(self, action: dict[str, Any]):
        """The provided action is expected to be a vector."""
        if not self.is_connected:
            raise RobotDeviceNotConnectedError(
                "KochRobot is not connected. You need to run `robot.connect()`."
            )

        for name in self.leader_arms:
            goal_joint = [ val for key, val in action.items() if name in key and "joint" in key]
            # goal_gripper = [ val for key, val in action.items() if name in key and "gripper" in key]

            # goal_joint = action[(arm_index*arm_action_dim+from_idx):(arm_index*arm_action_dim+to_idx)]
            # goal_gripper = action[arm_index*arm_action_dim + 12]
            # arm_index += 1
            goal_joint_numpy = np.array([t.item() for t in goal_joint], dtype=np.float32)
            # goal_gripper_numpy = np.array([t.item() for t in goal_gripper], dtype=np.float32)
            # position = np.concatenate([goal_joint_numpy, goal_gripper_numpy], axis=0)

            so101_zmq_send(f"action_joint_{name}", goal_joint_numpy, wait_time_s=0.01)


    def disconnect(self):
        if not self.is_connected:
            raise RobotDeviceNotConnectedError(
                "Aloha is not connected. You need to run `robot.connect()` before disconnecting."
            )

        self.is_connected = False
        global running_recv_image_server
        global running_recv_joint_server
        running_recv_image_server = False
        running_recv_joint_server = False

        self.robot_socket.shutdown()

        self.recv_image_thread.join()
        self.recv_joint_thread.join()
        

    def __del__(self):
        if getattr(self, "is_connected", False):
            self.disconnect()
