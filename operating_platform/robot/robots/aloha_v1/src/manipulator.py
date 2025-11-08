import time
import zmq
import torch
import threading
import cv2
import json
import numpy as np

from typing import Any

from operating_platform.robot.robots.utils import RobotDeviceNotConnectedError
from operating_platform.robot.robots.aloha_v1 import AlohaRobotConfig
from operating_platform.robot.robots.aloha_v1 import AlohaRobotStatus
from operating_platform.robot.robots.com_configs.cameras import CameraConfig, OpenCVCameraConfig

from operating_platform.robot.robots.camera import Camera


recv_images = {}
recv_master_jointstats = {}
recv_follower_jointstats = {}
recv_follower_pose = {}
lock = threading.Lock()  # 线程锁

# IPC Address
ipc_address_image = "ipc:///tmp/dora-zeromq-image"
ipc_address_piper = "ipc:///tmp/dora-zeromq-piper"

context = zmq.Context()
running_recv_image_server = True
running_recv_piper_server = True

socket_image = context.socket(zmq.PAIR)
socket_image.connect(ipc_address_image)
socket_image.setsockopt(zmq.SNDHWM, 2000)
socket_image.setsockopt(zmq.SNDBUF, 2**25)
socket_image.setsockopt(zmq.SNDTIMEO, 2000)
socket_image.setsockopt(zmq.RCVTIMEO, 2000)  # 设置接收超时（毫秒）
socket_image.setsockopt(zmq.LINGER, 0)

socket_piper = context.socket(zmq.PAIR)
socket_piper.connect(ipc_address_piper)
socket_piper.setsockopt(zmq.SNDHWM, 2000)
socket_piper.setsockopt(zmq.SNDBUF, 2**25)
socket_piper.setsockopt(zmq.SNDTIMEO, 2000)
socket_piper.setsockopt(zmq.RCVTIMEO, 2000)  # 设置接收超时（毫秒）
socket_piper.setsockopt(zmq.LINGER, 0)

def piper_zmq_send(event_id, buffer, wait_time_s):
    buffer_bytes = buffer.tobytes()
    print(f"zmq send event_id:{event_id}, value:{buffer}")
    try:
        socket_piper.send_multipart([
            event_id.encode('utf-8'),
            buffer_bytes
        ], flags=zmq.NOBLOCK)
    except zmq.Again:
        pass
    time.sleep(wait_time_s)

def recv_img_server():
    """接收数据线程"""
    while running_recv_image_server:
        try:
            message_parts = socket_image.recv_multipart()
            if len(message_parts) < 2:
                continue  # 协议错误

            event_id = message_parts[0].decode('utf-8')
            buffer_bytes = message_parts[1]
            metadata = json.loads(message_parts[2].decode('utf-8'))

            if 'image' in event_id:
                # 解码图像
                img_array = np.frombuffer(buffer_bytes, dtype=np.uint8)
                encoding = metadata["encoding"].lower()
                width = metadata["width"]
                height = metadata["height"]

                if encoding == "bgr8":
                    channels = 3
                    frame = (
                        img_array.reshape((height, width, channels))
                        .copy()  # Copy So that we can add annotation on the image
                    )
                elif encoding == "rgb8":
                    channels = 3
                    frame = (img_array.reshape((height, width, channels)))
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                elif encoding in ["jpeg", "jpg", "jpe", "bmp", "webp", "png"]:
                    channels = 3
                    frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                if frame is not None:
                    with lock:
                        recv_images[event_id] = frame

        except zmq.Again:
            print(f"Manipulator Receive Image Timeout")
            continue
        except Exception as e:
            print("Manipulator Receive Image Recv Error:", e)
            break

def recv_piper_server():
    """接收数据线程"""
    while running_recv_piper_server:
        try:
            message_parts = socket_piper.recv_multipart()
            if len(message_parts) < 2:
                continue  # 协议错误

            event_id = message_parts[0].decode('utf-8')
            buffer_bytes = message_parts[1]

            if 'jointstat' in event_id and 'master' in event_id:
                joint_array = np.frombuffer(buffer_bytes, dtype=np.float32)
                if joint_array is not None:
                    with lock:
                        recv_master_jointstats[event_id] = joint_array

            if 'jointstat' in event_id and 'follower' in event_id:
                joint_array = np.frombuffer(buffer_bytes, dtype=np.float32)
                if joint_array is not None:
                    with lock:
                        recv_follower_jointstats[event_id] = joint_array

            if 'endpose' in event_id and 'follower' in event_id:
                pose_array = np.frombuffer(buffer_bytes, dtype=np.float32)
                if pose_array is not None:
                    with lock:
                        recv_follower_pose[event_id] = pose_array

        except zmq.Again:
            print(f"Manipulator Receive Piper Timeout")
            continue
        except Exception as e:
            print("Manipulator Receive Piper Recv Error:", e)
            break


class OpenCVCamera:
    def __init__(self, config: OpenCVCameraConfig):
        self.config = config
        self.camera_index = config.camera_index
        self.port = None

        # Store the raw (capture) resolution from the config.
        self.capture_width = config.width
        self.capture_height = config.height

        # If rotated by ±90, swap width and height.
        if config.rotation in [-90, 90]:
            self.width = config.height
            self.height = config.width
        else:
            self.width = config.width
            self.height = config.height

        self.fps = config.fps
        self.channels = config.channels
        self.color_mode = config.color_mode
        self.mock = config.mock

        self.camera = None
        self.is_connected = False
        self.thread = None
        self.stop_event = None
        self.color_image = None
        self.logs = {}



def make_cameras_from_configs(camera_configs: dict[str, CameraConfig]) -> list[Camera]:
    cameras = {}

    for key, cfg in camera_configs.items():
        if cfg.type == "opencv":
            cameras[key] = OpenCVCamera(cfg)
        else:
            raise ValueError(f"The camera type '{cfg.type}' is not valid.")

    return cameras



class AlohaManipulator:
    def __init__(self, config: AlohaRobotConfig):
        self.config = config
        self.status = AlohaRobotStatus()
        self.robot_type = self.config.type

        self.use_videos = self.config.use_videos

        self.follower_arms = {}
        self.follower_arms['right'] = self.config.right_leader_arm.motors
        self.follower_arms['left'] = self.config.left_leader_arm.motors

        self.connect_excluded_cameras = ["image_pika_pose"]

        self.cameras = make_cameras_from_configs(self.config.cameras)
        self.microphones = self.config.microphones

        self.recv_img_thread = threading.Thread(target=recv_img_server, daemon=True)
        self.recv_img_thread.start()

        self.recv_piper_thread = threading.Thread(target=recv_piper_server, daemon=True)
        self.recv_piper_thread.start()
        
        self.is_connected = False
        self.logs = {}
        self.frame_counter = 0  # 帧计数器

        

    def get_motor_names(self, arm: dict[str, dict]) -> list:
        return [f"{arm}_{motor}" for arm, motors in arm.items() for motor in motors]

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
    def motor_features(self) -> dict:
        action_names = self.get_motor_names(self.follower_arms)
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

        conditions = [
            (
                lambda: all(name in recv_images for name in self.cameras if name not in self.connect_excluded_cameras),
                lambda: [name for name in self.cameras if name not in recv_images],
                "等待摄像头图像超时"
            ),
            (
                lambda: all(
                    any(name in key for key in recv_follower_jointstats)
                    for name in self.follower_arms
                ),
                lambda: [name for name in self.follower_arms if not any(name in key for key in recv_follower_jointstats)],
                "等待机械臂从臂关节超时"
            ),
            (
                lambda: all(
                    any(name in key for key in recv_master_jointstats)
                    for name in self.follower_arms
                ),
                lambda: [name for name in self.follower_arms if not any(name in key for key in recv_master_jointstats)],
                "等待机械臂主臂关节超时"
            )
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
        
        # 机械臂数据状态
        arm_data_types = ["主臂关节角度", "从臂关节角度"]
        for i, data_type in enumerate(arm_data_types, 1):
            if conditions[i][0]():
                arm_received = [name for name in self.follower_arms 
                            if any(name in key for key in (recv_master_jointstats, recv_follower_jointstats)[i-1])]
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
    
    def follower_record(self):
        follower_joint = {}
        for name in self.follower_arms:
            for match_name in recv_follower_jointstats:
                if name in match_name:
                    now = time.perf_counter()

                    byte_array = np.zeros(6, dtype=np.float32)
                    joint_read = recv_follower_jointstats[match_name]

                    byte_array[:6] = joint_read[:6]
                    byte_array = np.round(byte_array, 3)
                    
                    follower_joint[name] = torch.from_numpy(byte_array)

                    self.logs[f"read_follower_{name}_joint_dt_s"] = time.perf_counter() - now

        follower_pos = {}
        for name in self.follower_arms:
            for match_name in recv_follower_pose:
                if name in match_name:
                    now = time.perf_counter()

                    byte_array = np.zeros(6, dtype=np.float32)
                    pose_read = recv_follower_pose[match_name]

                    byte_array[:6] = pose_read[:]
                    byte_array = np.round(byte_array, 3)
                    
                    follower_pos[name] = torch.from_numpy(byte_array)

                    self.logs[f"read_follower_{name}_pos_dt_s"] = time.perf_counter() - now

        follower_gripper = {}
        for name in self.follower_arms:
            for match_name in recv_follower_jointstats:
                if name in match_name:
                    now = time.perf_counter()

                    byte_array = np.zeros(1, dtype=np.float32)
                    gripper_read = recv_follower_jointstats[match_name][6]

                    byte_array[0] = gripper_read
                    byte_array = np.round(byte_array, 3)
                    
                    follower_gripper[name] = torch.from_numpy(byte_array)

                    self.logs[f"read_follower_{name}_gripper_dt_s"] = time.perf_counter() - now
        
        return follower_joint, follower_pos, follower_gripper
    
    def master_record(self):
        master_joint = {}
        for name in self.follower_arms:
            for match_name in recv_master_jointstats:
                if name in match_name:
                    now = time.perf_counter()

                    byte_array = np.zeros(6, dtype=np.float32)
                    joint_read = recv_master_jointstats[match_name]

                    byte_array[:6] = joint_read[:6]
                    byte_array = np.round(byte_array, 3)
                    
                    master_joint[name] = torch.from_numpy(byte_array)

                    self.logs[f"read_master_{name}_joint_dt_s"] = time.perf_counter() - now

        master_gripper = {}
        for name in self.follower_arms:
            for match_name in recv_master_jointstats:
                if name in match_name:
                    now = time.perf_counter()

                    byte_array = np.zeros(1, dtype=np.float32)
                    gripper_read = recv_master_jointstats[match_name][6]

                    byte_array[0] = gripper_read
                    byte_array = np.round(byte_array, 3)
                    
                    master_gripper[name] = torch.from_numpy(byte_array)

                    self.logs[f"read_master_{name}_gripper_dt_s"] = time.perf_counter() - now
        
        return master_joint, master_gripper

    def teleop_step(
        self, record_data=False, 
    ) -> None | tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        self.frame_counter += 1

        if not self.is_connected:
            raise RobotDeviceNotConnectedError(
                "Aloha is not connected. You need to run `robot.connect()`."
            )

        if not record_data:
            return

        follower_joint, follower_pos, follower_gripper = self.follower_record()

        # master_joint, master_pos, master_gripper = self.master_record()
        master_joint, master_gripper = self.master_record()

        #记录当前关节角度
        state = []
        for name in self.follower_arms:
            if name in follower_joint:
                state.append(follower_joint[name])
            if name in follower_pos:
                state.append(follower_pos[name])
            if name in follower_gripper:
                state.append(follower_gripper[name])
        state = torch.cat(state)

        #将关节目标位置添加到 action 列表中
        action = []
        for name in self.follower_arms:
            if name in master_joint:
                action.append(master_joint[name])
            if name in follower_pos:
                action.append(follower_pos[name])
            if name in master_gripper:
                action.append(master_gripper[name])
        action = torch.cat(action)

        # Capture images from cameras
        
        images = {}
        for name in self.cameras:
            now = time.perf_counter()
            
            images[name] = recv_images[name]

            # images[name] = self.cameras[name].async_read()
            images[name] = torch.from_numpy(images[name])
            # self.logs[f"read_camera_{name}_dt_s"] = self.cameras[name].logs["delta_timestamp_s"]
            self.logs[f"read_camera_{name}_dt_s"] = time.perf_counter() - now

        # Populate output dictionnaries and format to pytorch
        obs_dict, action_dict = {}, {}
        obs_dict["observation.state"] = state
        action_dict["action"] = action
        for name in self.cameras:
            obs_dict[f"observation.images.{name}"] = images[name]
        
        # print("end teleoperate record")

        return obs_dict, action_dict


    def send_action(self, action: dict[str, Any]):
        """The provided action is expected to be a vector."""
        if not self.is_connected:
            raise RobotDeviceNotConnectedError(
                "KochRobot is not connected. You need to run `robot.connect()`."
            )

        for name in self.follower_arms:
            goal_joint = [ val for key, val in action.items() if name in key and "joint" in key]
            goal_gripper = [ val for key, val in action.items() if name in key and "gripper" in key]

            # goal_joint = action[(arm_index*arm_action_dim+from_idx):(arm_index*arm_action_dim+to_idx)]
            # goal_gripper = action[arm_index*arm_action_dim + 12]
            # arm_index += 1
            goal_joint_numpy = np.array([t.item() for t in goal_joint], dtype=np.float32)
            goal_gripper_numpy = np.array([t.item() for t in goal_gripper], dtype=np.float32)
            position = np.concatenate([goal_joint_numpy, goal_gripper_numpy], axis=0)

            piper_zmq_send(f"action_joint_{name}", position, wait_time_s=0.01)
            # piper_zmq_send(f"action_gripper_{name}", goal_gripper_numpy, wait_time_s=0.01)

            # action_sent.append(goal_joint)

    # def update_status(self) -> str:

    #     for i in range(self.status.specifications.camera.number):
    #         match_name = self.status.specifications.camera.information[i].name
    #         for name in recv_images_status:
    #             if match_name in name:
    #                 self.status.specifications.camera.information[i].is_connect = True if recv_images_status[name]>0 else False

    #     for i in range(self.status.specifications.arm.number):
    #         match_name = self.status.specifications.arm.information[i].name
    #         for name in recv_joint_status:
    #             if match_name in name:
    #                 self.status.specifications.arm.information[i].is_connect = True if recv_joint_status[name]>0 else False

    #     return self.status.to_json()

    def disconnect(self):
        if not self.is_connected:
            raise RobotDeviceNotConnectedError(
                "Aloha is not connected. You need to run `robot.connect()` before disconnecting."
            )
        

        self.is_connected = False
        
        global running_recv_image_server
        global running_recv_piper_server

        running_recv_image_server = False
        running_recv_piper_server = False

        self.recv_img_thread.join()
        self.recv_piper_thread.join()

        socket_image.close()
        socket_piper.close()

        context.term()
        

    def __del__(self):
        if getattr(self, "is_connected", False):
            self.disconnect()
