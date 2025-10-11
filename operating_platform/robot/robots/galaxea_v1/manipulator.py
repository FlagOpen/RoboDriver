# import pickle
import time
# from dataclasses import dataclass, field, replace
# from pathlib import Path
# import os
# import ctypes
# import platform
# import sys
import json
import numpy as np
import torch

# from concurrent.futures import ThreadPoolExecutor
# from collections import deque
# from functools import cache
from typing import Any

import threading
import cv2

import zmq

from operating_platform.robot.robots.utils import RobotDeviceNotConnectedError
from operating_platform.robot.robots.configs import GALAXEARobotConfig
from operating_platform.robot.robots.com_configs.cameras import CameraConfig, OpenCVCameraConfig

from operating_platform.robot.robots.camera import Camera
# from operating_platform.robot.robots.pika_v1.pika_trans_visual_dual import Transformer


ipc_address_image = "ipc:///tmp/ros2-zeromq-galaxea-image"
ipc_address_joint = "ipc:///tmp/ros2-zeromq-galaxea-joint"

recv_images = {}
recv_joint = {}
lock = threading.Lock()  # 线程锁

running_recv_image_server = True
running_recv_joint_server = True

zmq_context = zmq.Context()

socket_image = zmq_context.socket(zmq.PAIR)
socket_image.connect(ipc_address_image)
socket_image.setsockopt(zmq.RCVTIMEO, 2000)

socket_joint = zmq_context.socket(zmq.PAIR)
socket_joint.connect(ipc_address_joint)
socket_joint.setsockopt(zmq.RCVTIMEO, 2000)

def galaxea_zmq_send(event_id, buffer, wait_time_s):
    buffer_bytes = buffer.tobytes()
    print(f"zmq send event_id:{event_id}, value:{buffer}")
    try:
        socket_joint.send_multipart([
            event_id.encode('utf-8'),
            buffer_bytes
        ], flags=zmq.NOBLOCK)
    except zmq.Again:
        pass
    time.sleep(wait_time_s)

def recv_image_server():
    """接收数据线程"""
    while running_recv_image_server:
        try:
            message_parts = socket_image.recv_multipart()
            if len(message_parts) < 2:
                continue  # 协议错误

            event_id = message_parts[0].decode('utf-8')
            buffer_bytes = message_parts[1]
            metadata = json.loads(message_parts[2].decode('utf-8'))

            # print(f"Received event_id = {event_id}")
            # print(f"len(message_parts) = {len(message_parts)}")
            
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
                
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # BGR → RGB 转换
                
                if frame is not None:
                    with lock:
                        # print(f"Received event_id = {event_id}")
                        recv_images[event_id] = frame

        except zmq.Again:
            print(f"GALAXEA Image Received Timeout")
            continue
        except Exception as e:
            print("recv image error:", e)
            break


def recv_joint_server():
    """接收数据线程"""
    while running_recv_joint_server:
        try:
            message_parts = socket_joint.recv_multipart()
            if len(message_parts) < 2:
                continue  # 协议错误

            event_id = message_parts[0].decode('utf-8')
            buffer_bytes = message_parts[1]   
            joint_array = np.frombuffer(buffer_bytes, dtype=np.float32)
            if joint_array is not None:
                with lock:
                    recv_joint[event_id] = joint_array

        except zmq.Again:
            print(f"GALAXEA Joint Received Timeout")
            continue
        except Exception as e:
            print("recv joint error:", e)
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



class GALAXEAManipulator:
    def __init__(self, config: GALAXEARobotConfig):
        self.config = config
        self.robot_type = self.config.type
        print(self.robot_type)
        self.use_videos = self.config.use_videos
        self.microphones = self.config.microphones # galaxea没有audio

        self.leader_arms = {}
        self.leader_arms['main_leader'] = self.config.lead_arms["main"]

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
        return [f"{motor}" for arm, bus in arms.items() for motor in bus.motors]

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
        timeout = 5  # 统一的超时时间（秒）
        start_time = time.perf_counter()

        # 定义所有需要等待的条件及其错误信息
        conditions = [
            (
                lambda: all(name in recv_images for name in self.cameras if name not in self.connect_excluded_cameras),
                lambda: [name for name in self.cameras if name not in recv_images],
                "等待摄像头图像超时"
            ),
            # (
            #     lambda: all(
            #         any(name in key for key in recv_joint)
            #         for name in self.leader_arms
            #     ),
            #     lambda: [name for name in self.leader_arms if not any(name in key for key in recv_joint)],
            #     "等待主臂关节角度超时"
            # ),
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
                            received = []

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

                    byte_array = np.zeros(30, dtype=np.float32)
                    pose_read = recv_joint[match_name]

                    byte_array[:30] = pose_read[:]
                    byte_array = np.round(byte_array, 3)
                    
                    follower_joint[name] = torch.from_numpy(byte_array)

                    self.logs[f"read_follower_{name}_joint_dt_s"] = time.perf_counter() - now
                    
        leader_joint = {}
        for name in self.leader_arms:
            for match_name in recv_joint:
                if name in match_name:
                    now = time.perf_counter()

                    byte_array = np.zeros(14, dtype=np.float32)
                    pose_read = recv_joint[match_name]

                    byte_array[:14] = pose_read[:]
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
        if action:
            action = torch.cat(action)
        else:
            action = state

        
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

            galaxea_zmq_send(f"action_joint_{name}", goal_joint_numpy, wait_time_s=0.01)


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

        self.is_connected = False
        global running_recv_image_server
        global running_recv_joint_server
        running_recv_image_server = False
        running_recv_joint_server = False

        self.recv_image_thread.join()
        self.recv_joint_thread.join()
        

    def __del__(self):
        if getattr(self, "is_connected", False):
            self.disconnect()

if __name__ == '__main__':
    config = GALAXEARobotConfig()  # 实例化
    galaxea = GALAXEAManipulator(config)
    galaxea.connect()