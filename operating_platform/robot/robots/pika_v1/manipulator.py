import pickle
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
import os
import ctypes
import platform
import sys
import json
import numpy as np
import torch

from concurrent.futures import ThreadPoolExecutor
from collections import deque
from functools import cache

import threading
import cv2

import zmq


from operating_platform.robot.robots.utils import RobotDeviceNotConnectedError
from operating_platform.robot.robots.pika_v1 import PikaV1RobotConfig
from operating_platform.robot.robots.com_configs.cameras import CameraConfig, OpenCVCameraConfig

from operating_platform.robot.robots.camera import Camera
from operating_platform.robot.robots.pika_v1.pika_trans_visual_dual import Transformer


# IPC Address
pika_ipc_address = "ipc:///tmp/dr-robot-pika-v1"
vive_ipc_address = "ipc:///tmp/dr-component-vive"
gripper_ipc_address = "ipc:///tmp/dr-component-pika-gripper"


recv_images = {}
recv_pose = {}
recv_rotation = {}
recv_gripper = {}
lock = threading.Lock()  # 线程锁
zmq_context = zmq.Context()

pika_socket = zmq_context.socket(zmq.PAIR)
pika_socket.bind(pika_ipc_address)
pika_socket.setsockopt(zmq.RCVTIMEO, 300)
pika_running_server = True

def pika_recv_server():
    """接收数据线程"""
    while pika_running_server:
        try:

            message_parts = pika_socket.recv_multipart()
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
                        # print(f"Received event_id = {event_id}")
                        recv_images[event_id] = frame

            # if 'gripper' in event_id:
            #     gripper_array = np.frombuffer(buffer_bytes, dtype=np.float32)
            #     if gripper_array is not None:
            #         with lock:
            #             recv_gripper[event_id] = gripper_array


        except zmq.Again:
            print(f"Pika Received Timeout")
            continue
        except Exception as e:
            print("recv error:", e)
            break

vive_socket = zmq_context.socket(zmq.PAIR)
vive_socket.bind(vive_ipc_address)
vive_socket.setsockopt(zmq.RCVTIMEO, 300)
vive_running_server = True

def vive_recv_server():
    """接收数据线程"""
    while vive_running_server:
        try:

            message_parts = vive_socket.recv_multipart()
            if len(message_parts) < 2:
                continue  # 协议错误

            event_id = message_parts[0].decode('utf-8')
            buffer_bytes = message_parts[1]

            if 'pose' in event_id:
                pose_array = np.frombuffer(buffer_bytes, dtype=np.float32)
                if pose_array is not None:
                    # print(f"Received pose data for event_id: {event_id}")
                    # print(f"Pose array shape: {pose_array.shape}")
                    # print(f"Pose array values: {pose_array}")
                    with lock:
                        recv_pose[event_id] = pose_array
                # else:
                #     print(f"Failed to parse pose data for event_id: {event_id}")
            
            if 'rotation' in event_id:
                rotation_array = np.frombuffer(buffer_bytes, dtype=np.float32)
                if rotation_array is not None:
                    with lock:
                        recv_rotation[event_id] = rotation_array

        except zmq.Again:
            print(f"VIVE Received Timeout")
            continue
        except Exception as e:
            print("recv error:", e)
            break


gripper_socket = zmq_context.socket(zmq.PAIR)
gripper_socket.bind(gripper_ipc_address)
gripper_socket.setsockopt(zmq.RCVTIMEO, 300)
gripper_running_server = True

def gripper_recv_server():
    """接收数据线程"""
    while gripper_running_server:
        try:

            message_parts = gripper_socket.recv_multipart()
            if len(message_parts) < 2:
                continue  # 协议错误

            event_id = message_parts[0].decode('utf-8')
            buffer_bytes = message_parts[1]
            metadata = json.loads(message_parts[2].decode('utf-8'))
            
            if 'gripper' in event_id:
                gripper_array = np.frombuffer(buffer_bytes, dtype=np.float32)
                if gripper_array is not None:
                    with lock:
                        recv_gripper[event_id] = gripper_array

        except zmq.Again:
            print(f"Pika Gripper Received Timeout")
            continue
        except Exception as e:
            print("recv error:", e)
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



class PikaV1Manipulator:
    def __init__(self, config: PikaV1RobotConfig):
        self.config = config
        self.robot_type = self.config.type

        self.follower_arms = {}
        self.follower_arms['right'] = self.config.right_leader_arm.motors
        self.follower_arms['left'] = self.config.left_leader_arm.motors
        
        self.use_videos = self.config.use_videos
        self.cameras = make_cameras_from_configs(self.config.cameras)
        self.microphones = self.config.microphones
        
        self.connect_excluded_cameras = ["image_pika_pose"]

        pika_recv_thread = threading.Thread(target=pika_recv_server, daemon=True)
        pika_recv_thread.start()

        vive_recv_thread = threading.Thread(target=vive_recv_server, daemon=True)
        vive_recv_thread.start()

        gripper_recv_thread = threading.Thread(target=gripper_recv_server, daemon=True)
        gripper_recv_thread.start()

        self.pika_transferorm = Transformer()
        
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

        # 定义所有需要等待的条件及其错误信息
        conditions = [
            (
                lambda: all(name in recv_images for name in self.cameras if name not in self.connect_excluded_cameras),
                lambda: [name for name in self.cameras if name not in recv_images],
                "等待摄像头图像超时"
            ),
            (
                lambda: all(
                    any(name in key for key in recv_rotation)
                    for name in self.follower_arms
                ),
                lambda: [name for name in self.follower_arms if not any(name in key for key in recv_rotation)],
                "等待旋转角度数据超时"
            ),
            (
                lambda: all(
                    any(name in key for key in recv_pose)
                    for name in self.follower_arms
                ),
                lambda: [name for name in self.follower_arms if not any(name in key for key in recv_pose)],
                "等待机械臂末端位姿超时"
            ),
            (
                lambda: all(
                    any(name in key for key in recv_gripper)
                    for name in self.follower_arms
                ),
                lambda: [name for name in self.follower_arms if not any(name in key for key in recv_gripper)],
                "等待机械臂夹爪超时"
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
        arm_data_types = ["旋转角度", "末端位姿", "夹爪状态"]
        for i, data_type in enumerate(arm_data_types, 1):
            if conditions[i][0]():
                arm_received = [name for name in self.follower_arms 
                            if any(name in key for key in (recv_rotation, recv_pose, recv_gripper)[i-1])]
                success_messages.append(f"{data_type}: {', '.join(arm_received)}")
        
        # 打印成功连接信息
        print("\n[连接成功] 所有设备已就绪:")
        for msg in success_messages:
            print(f"  - {msg}")
        print(f"  总耗时: {time.perf_counter() - start_time:.2f}秒\n")
        # ===========================

        print("\nPika获取初始位姿和旋转角度中...请将Pika双爪都平行正放, 夹爪朝向前方。")
        while True:
            follower_pos = {}
            for name in self.follower_arms:
                for match_name in recv_pose:
                    if name in match_name:
                        byte_array = np.zeros(3, dtype=np.float32)
                        pose_read = recv_pose[match_name]

                        byte_array[:3] = pose_read[:]
                        byte_array = np.round(byte_array, 3)
                        
                        follower_pos[name] = byte_array

            follower_rotation = {}
            for name in self.follower_arms:
                for match_name in recv_rotation:
                    if name in match_name:
                        byte_array = np.zeros(4, dtype=np.float32)
                        rotation_read = recv_rotation[match_name]

                        byte_array[:4] = rotation_read[:]
                        byte_array = np.round(byte_array, 3)
                        
                        follower_rotation[name] = byte_array

            all_trans_success = True
            for name in self.follower_arms:
                # print(f"Calling trans with - name: {name}, position: {follower_pos[name]}, rotation: {follower_rotation[name]}")

                result = self.pika_transferorm.trans(position=follower_pos[name], rotation=follower_rotation[name], name=name)

                if result is None:
                    all_trans_success = False

            if all_trans_success == True:
                break

            time.sleep(0.033)  # 等待33毫秒，约30FPS

            if time.perf_counter() - start_time > timeout:
                raise TimeoutError("获取初始位姿和旋转角度超时，请检查设备连接状态。")
        
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
        self.frame_counter += 1

        if not self.is_connected:
            raise RobotDeviceNotConnectedError(
                "Aloha is not connected. You need to run `robot.connect()`."
            )

        # for name in self.leader_arms:
        #     # if name == "left":
        #     #     continue
        #     # 读取领导臂电机数据
        #     read_start = time.perf_counter()
        #     leader_pos = self.leader_arms[name].async_read("Present_Position")
        #     # self.follower_arms[name].leader_pos = leader_pos
        #     self.logs[f"read_leader_{name}_pos_dt_s"] = time.perf_counter() - read_start

        #     # 电机数据到关节角度的转换，关节角度处理（向量化操作）
        #     for i in range(7):
        #         # 数值转换
        #         value = round(leader_pos[i] * SCALE_FACTOR, 2)

        #         # 特定关节取反（3号和5号）
        #         if i in {3, 5}:
        #             value = -value

        #         # 限幅
        #         clamped_value = max(self.follower_arms[name].joint_n_limit[i], min(self.follower_arms[name].joint_p_limit[i], value))

        #         # 移动平均滤波
        #         # filter_value = self.follower_arms[name].filters[i].update(clamped_value)

        #         # if abs(filter_value - self.filters[i].get_last()) / WINDOW_SIZE > 180 ##超180度/s位移限制，暂时不弄

        #         # 直接使用内存视图操作

        #         # self.follower_arms[name].joint_teleop_write[i] = filter_value
        #         self.follower_arms[name].joint_teleop_write[i] = clamped_value

        #     # 电机角度到夹爪开合度的换算
        #     giper_value = leader_pos[7] * GRIPPER_SCALE
        #     self.follower_arms[name].clipped_gripper = max(0, min(100, int(giper_value)))

        #     # 机械臂执行动作（调用透传API，控制gen72移动到目标位置）
        #     write_start = time.perf_counter()
        #     self.follower_arms[name].movej_canfd(self.follower_arms[name].joint_teleop_write)
        #     if self.frame_counter % 5 == 0:
        #         self.frame_counter = 0
        #         self.follower_arms[name].write_single_register(self.follower_arms[name].clipped_gripper)
        #     self.logs[f"write_follower_{name}_goal_pos_dt_s"] = time.perf_counter() - write_start

        # print("end teleoperate")

        if not record_data:
            return

        follower_pos = {}
        for name in self.follower_arms:
            for match_name in recv_pose:
                if name in match_name:
                    now = time.perf_counter()

                    byte_array = np.zeros(3, dtype=np.float32)
                    pose_read = recv_pose[match_name]

                    byte_array[:3] = pose_read[:]
                    byte_array = np.round(byte_array, 3)
                    
                    follower_pos[name] = byte_array

                    self.logs[f"read_follower_{name}_pos_dt_s"] = time.perf_counter() - now

        follower_rotation = {}
        for name in self.follower_arms:
            for match_name in recv_rotation:
                if name in match_name:
                    now = time.perf_counter()

                    byte_array = np.zeros(4, dtype=np.float32)
                    rotation_read = recv_rotation[match_name]

                    byte_array[:4] = rotation_read[:]
                    byte_array = np.round(byte_array, 3)
                    
                    follower_rotation[name] = byte_array

                    self.logs[f"read_follower_{name}_rotation_dt_s"] = time.perf_counter() - now
        
        # 从VIVE坐标系变换到PIKA夹爪坐标系（相对第一帧，两夹爪中值）
        for name in self.follower_arms:
            # print(f"Calling trans with - name: {name}, position: {follower_pos[name]}, rotation: {follower_rotation[name]}")

            result = self.pika_transferorm.trans(position=follower_pos[name], rotation=follower_rotation[name], name=name)
            if result is None:
                follower_pos[name] = np.zeros(3, dtype=np.float32)
                follower_rotation[name] = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)  # 单位四元数
            else:
                try:
                    new_pos, new_rot, *optional = result
                    if optional and optional[0] is not None and "left" in name:
                        recv_images["image_pika_pose"] = optional[0]
                    follower_pos[name] = np.asarray(new_pos, dtype=np.float32).copy()
                    follower_rotation[name] = np.asarray(new_rot, dtype=np.float32).copy()
                except (TypeError, ValueError) as e:
                    follower_pos[name] = np.zeros(3, dtype=np.float32)
                    follower_rotation[name] = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)

            follower_pos[name] = torch.from_numpy(follower_pos[name])
            follower_rotation[name] = torch.from_numpy(follower_rotation[name])

        
        follower_gripper = {}
        for name in self.follower_arms:
            for match_name in recv_gripper:
                if name in match_name:
                    now = time.perf_counter()

                    byte_array = np.zeros(1, dtype=np.float32)
                    gripper_read = recv_gripper[match_name]

                    byte_array[:1] = gripper_read[:]
                    byte_array = np.round(byte_array, 3)
                    
                    follower_gripper[name] = torch.from_numpy(byte_array)

                    self.logs[f"read_follower_{name}_gripper_dt_s"] = time.perf_counter() - now

        #记录当前关节角度
        state = []
        for name in self.follower_arms:
            if name in follower_pos:
                state.append(follower_pos[name])
            if name in follower_rotation:
                state.append(follower_rotation[name])
            if name in follower_gripper:
                state.append(follower_gripper[name])
        state = torch.cat(state)

        #将关节目标位置添加到 action 列表中
        action = []
        for name in self.follower_arms:
            if name in follower_pos:
                action.append(follower_pos[name])
            if name in follower_rotation:
                action.append(follower_rotation[name])
            if name in follower_gripper:
                action.append(follower_gripper[name])
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

        # obs_dict["observation.images.image_pika_pose"] = torch.from_numpy(recv_images["image_pika_pose"])
        
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




    # def send_action(self, action: torch.Tensor):
    #     """The provided action is expected to be a vector."""
    #     if not self.is_connected:
    #         raise RobotDeviceNotConnectedError(
    #             "KochRobot is not connected. You need to run `robot.connect()`."
    #         )
    #     from_idx = 0
    #     to_idx = 8
    #     index = 0
    #     action_sent = []
    #     for name in self.follower_arms:

    #         goal_pos = action[index*8+from_idx:index*8+to_idx]
    #         index+=1

    #         for i in range(7):
    #             self.follower_arms[name].joint_send[i] = max(self.follower_arms[name].joint_n_limit[i], min(self.follower_arms[name].joint_p_limit[i], goal_pos[i]))
            
            
    #         self.follower_arms[name].movej_canfd(self.follower_arms[name].joint_send)
    #         # if (goal_pos[7]<50):
    #         #     # ret_giper = self.pDll.Write_Single_Register(self.nSocket, 1, 40000, int(follower_goal_pos_array[7]), 1, 1)
    #         #     ret_giper = self.pDll.Write_Single_Register(self.nSocket, 1, 40000, 0 , 1, 1)
    #         #     self.gipflag_send=0
    #         # #状态为闭合，且需要张开夹爪
    #         # if (goal_pos[7]>=50):
    #         #     # ret_giper = self.pDll.Write_Single_Register(self.nSocket, 1, 40000, int(follower_goal_pos_array[7]), 1, 1)
    #         #     ret_giper = self.pDll.Write_Single_Register(self.nSocket, 1, 40000, 100, 1, 1)
    #         #     self.gipflag_send=1
    #         gripper_value = max(0, min(100, int(goal_pos[7])))

    #         self.frame_counter += 1

    #         self.follower_arms[name].old_grasp = gripper_value
    #         if self.frame_counter % 5 == 0:
    #             self.frame_counter = 0
    #             self.follower_arms[name].write_single_register(gripper_value)
    #         action_sent.append(goal_pos)

    #     return torch.cat(action_sent)

    def disconnect(self):
        if not self.is_connected:
            raise RobotDeviceNotConnectedError(
                "Aloha is not connected. You need to run `robot.connect()` before disconnecting."
            )
        
        # for name in self.follower_arms:
        #     self.follower_arms[name].disconnect()

        # for name in self.leader_arms:
        #     self.leader_arms[name].disconnect()

        # for name in self.cameras:
        #     self.cameras[name].disconnect()

        self.is_connected = False
        global pika_running_server
        global vive_running_server
        global gripper_running_server
        pika_running_server = False
        vive_running_server = False
        gripper_running_server = False

        self.pika_transferorm.close()
        

    def __del__(self):
        if getattr(self, "is_connected", False):
            self.disconnect()
