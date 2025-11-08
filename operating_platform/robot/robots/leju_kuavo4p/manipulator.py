import asyncio
import json
import base64
import time
import os
import sys
import numpy as np
import cv2

import threading
import logging
import torch
from typing import Any, Dict, Optional, List, Tuple
from pathlib import Path
import sys
import rospy
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image, JointState
from kuavo_msgs.msg import sensorsData 

from operating_platform.robot.robots.utils import RobotDeviceNotConnectedError
from operating_platform.robot.robots.leju_kuavo4p.config import LejuKuavo4pRobotConfig
from operating_platform.robot.robots.com_configs.cameras import CameraConfig, OpenCVCameraConfig, DDSCameraConfig
from operating_platform.robot.robots.camera import Camera


# 日志设置
try:
    from operating_platform.utils.colored_logging import setup_colored_logger
    logger = setup_colored_logger(__name__)
except Exception:
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class RobotROSNode:
    def __init__(self):
        rospy.init_node('wanx_data_listener', anonymous=True)

        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/camera/color/image_raw", Image, self.image_callback)
        self.sensor_sub = rospy.Subscriber("/sensors_data_raw", sensorsData, self.sensor_callback)
        self.dexhand_sub = rospy.Subscriber("/dexhand/state", JointState, self.dexhand_callback)
        
        # 图像缓存（线程安全）
        self.latest_images = {}
        self.image_lock = threading.Lock()

        # 状态数据（线程安全）
        self.arm_joint_data = {}
        self.dexhand_data = {}
        self.head_data = {}
        self.state_lock = threading.Lock()  # ← 补充：用于保护状态数据

        # 控制标志
        self.running = False
        self.connection_event = threading.Event()
        self.spin_thread = None

    def loop(self):
        """运行 ROS spin，阻塞直到 shutdown"""
        try:
            rospy.spin()
        except rospy.ROSInterruptException:
            pass
        finally:
            self.running = False

    def start(self):
        """启动 ROS 节点在一个独立线程中"""
        if self.running:
            logger.warning("Node is already running.")
            return

        self.running = True
        self.spin_thread = threading.Thread(target=self.loop, daemon=True)
        self.spin_thread.start()
        logger.info("Robot ROS node started. Waiting for images and sensor data...")

    def stop(self):
        """停止节点（可选，rospy 通常靠 Ctrl+C）"""
        self.running = False
        if self.spin_thread and self.spin_thread.is_alive():
            self.spin_thread.join(timeout=1.0)
        rospy.signal_shutdown("Node stopped manually.")

    def image_callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "rgb8")
            with self.image_lock:
                self.latest_images["/camera/color/image_raw"] = cv_image.copy()  # copy for safety
        except CvBridgeError as e:
            logger.error("CvBridge Error: %s", e)

    def sensor_callback(self, msg):
        joint_q = msg.joint_data.joint_q

        if len(joint_q) < 28:
            rospy.logwarn("Received joint_q has fewer than 28 elements (%d). Skipping.", len(joint_q))
            return

        # 下半身关节索引和名称
        lower_body_indices = list(range(0, 12))
        lower_body_names = [
            "l_leg_roll", "l_leg_yaw", "l_leg_pitch", "l_knee", "l_foot_pitch", "l_foot_roll",
            "r_leg_roll", "r_leg_yaw", "r_leg_pitch", "r_knee", "r_foot_pitch", "r_foot_roll"
        ]

        # 上半身关节索引和名称（注意：12~25 是14个，+26,27 共16个）
        upper_body_indices = list(range(12, 26)) + [26, 27]
        upper_body_names = [
            "l_arm_pitch", "l_arm_roll", "l_arm_yaw", "l_forearm_pitch",
            "l_hand_yaw", "l_hand_pitch", "l_hand_roll",
            "r_arm_pitch", "r_arm_roll", "r_arm_yaw", "r_forearm_pitch",
            "r_hand_yaw", "r_hand_pitch", "r_hand_roll",
            "head_yaw", "head_pitch"
        ]

        # 构建日志（可选：生产环境可关闭）
        lines_upper = [f"  {name}: {joint_q[idx]:.4f}" for name, idx in zip(upper_body_names, upper_body_indices)]
        lines_lower = [f"  {name}: {joint_q[idx]:.4f}" for name, idx in zip(lower_body_names, lower_body_indices)]
        log_message = (
            "Upper body joint angles (rad):\n" + "\n".join(lines_upper) +
            "\n\nLower body joint angles (rad):\n" + "\n".join(lines_lower)
        )
        logger.debug(log_message)  # 改为 debug，避免刷屏；如需 info 可改回

        # 更新状态（线程安全）
        with self.state_lock:
            self.arm_joint_data["left_arm"] = list(joint_q[12:19])   # 7 DoF
            self.arm_joint_data["right_arm"] = list(joint_q[19:26])  # 7 DoF
            self.head_data["head"] = [joint_q[26], joint_q[27]]
        
    def dexhand_callback(self, msg):
        dexhand_position = msg.position
        with self.state_lock:
            self.dexhand_data["left_dexhand"] = list(dexhand_position[0:6])
            self.dexhand_data["right_dexhand"] = list(dexhand_position[6:12])

    def get_arm_state(self, side):
        """获取手臂状态，side: 'left' or 'right'"""
        if side not in ["left", "right"]:
            raise ValueError("side must be 'left' or 'right'")
        key = f"{side}_arm"
        with self.state_lock:
            return self.arm_joint_data.get(key, []).copy()

    def get_dexhand_state(self, side):
        """预留接口：获取灵巧手状态"""
        key = f"{side}_dexhand"
        with self.state_lock:
            return self.dexhand_data.get(key, {}).copy()
    
    def get_head_state(self):
        """获取头部状态"""
        with self.state_lock:
            return self.head_data.copy()

    def get_latest_image(self, topic="/camera/color/image_raw"):
        """获取最新图像"""
        with self.image_lock:
            img = self.latest_images.get(topic)
            return img.copy() if img is not None else None

    def get_all_image_topics(self):
        """获取所有已订阅的图像话题（当前只有一个）"""
        with self.image_lock:
            return list(self.latest_images.keys())


class DDSOpenCVCamera:
    def __init__(self, config: DDSCameraConfig):
        self.config = config
        self.camera_index = config.camera_index
        self.port = None
        self.topic = config.topic

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


def make_cameras_from_configs(camera_configs: Dict[str, CameraConfig]) -> List[Camera]:
    cameras = {}

    for key, cfg in camera_configs.items():
        if cfg.type == "ddscamera":
            cameras[key] = DDSOpenCVCamera(cfg)
        else:
            raise ValueError(f"The camera type '{cfg.type}' is not valid.")

    return cameras


class LejuKuavo4pManipulator:
    def __init__(self, config: LejuKuavo4pRobotConfig):
        self.config = config
        self.robot_type = self.config.type
        self.use_videos = self.config.use_videos
        self.microphones = self.config.microphones

        self.robot_ros_node = RobotROSNode()
        self.robot_ros_node.start()

        self.follower_motors = {}
        self.follower_motors['right_arm'] = self.config.follower_motors["right_arm"]
        self.follower_motors['left_arm'] = self.config.follower_motors["left_arm"]
        self.follower_motors['head'] = self.config.follower_motors["head"]
        self.follower_motors['right_dexhand'] = self.config.follower_motors["right_dexhand"]
        self.follower_motors['left_dexhand'] = self.config.follower_motors["left_dexhand"]

        self.cameras = make_cameras_from_configs(self.config.cameras)
        
        self.connect_excluded_cameras = ["image_robot_pose"]

        self.is_connected = False
        self.logs = {}

    def get_motor_names(self, arms: Dict[str, dict]) -> List:
        return [f"{arm}_{motor}" for arm, bus in arms.items() for motor in bus.motors]

    @property
    def camera_features(self) -> Dict:
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
    def microphone_features(self) -> Dict:
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
    def motor_features(self) -> Dict:
        action_names = self.get_motor_names(self.follower_motors)
        state_names = self.get_motor_names(self.follower_motors)
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
        timeout = 20  # 统一的超时时间（秒）
        start_time = time.perf_counter()

        # 定义所有需要等待的条件及其错误信息
        conditions = [
            (
                lambda: all(camera.topic in self.robot_ros_node.get_all_image_topics() for name, camera in self.cameras.items() if name not in self.connect_excluded_cameras),
                lambda: [name for name, camera in self.cameras.items() if camera.topic not in self.robot_ros_node.get_all_image_topics()],
                "等待摄像头图像超时"
            ),
            (
                lambda: all(
                    any(name in key for key in self.robot_ros_node.arm_joint_data)
                    or any(name in key for key in self.robot_ros_node.dexhand_data)
                    or any(name in key for key in self.robot_ros_node.head_data)
                    for name in self.follower_motors
                ),
                lambda: [
                    name for name in self.follower_motors
                    if not any(name in key for key in self.robot_ros_node.arm_joint_data)
                    and not any(name in key for key in self.robot_ros_node.dexhand_data)
                    and not any(name in key for key in self.robot_ros_node.head_data)
                ],
                "等待机器人关节角度超时"
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
                            received = [name for name in self.follower_motors if name not in missing]

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
            cam_received = [name for name, camera in self.cameras.items() 
                        if camera.topic in self.robot_ros_node.get_all_image_topics() and name not in self.connect_excluded_cameras]
            success_messages.append(f"摄像头: {', '.join(cam_received)}")

        # # 主臂数据状态
        # arm_data_types = ["主臂关节角度",]
        # for i, data_type in enumerate(arm_data_types, 1):
        #     if conditions[i][0]():
        #         arm_received = [name for name in self.leader_arms 
        #                     if any(name in key for key in (recv_joint,)[i-1])]
        #         success_messages.append(f"{data_type}: {', '.join(arm_received)}")
        
        # 从臂数据状态
        arm_data_types = ["从臂关节角度",]
        for i, data_type in enumerate(arm_data_types, 1):
            if conditions[i][0]():
                arm_received = [
                    name for name in self.follower_motors 
                    if any(name in key for key in (self.robot_ros_node.arm_joint_data,)[i-1])
                    or any(name in key for key in (self.robot_ros_node.dexhand_data,)[i-1])
                    or any(name in key for key in (self.robot_ros_node.head_data,)[i-1])
                ]
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
    ) -> Optional[Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]]:

        if not self.is_connected:
            raise RobotDeviceNotConnectedError(
                "Aloha is not connected. You need to run `robot.connect()`."
            )

        if not record_data:
            return

        follower_joint = {}
        for name in self.follower_motors:
            now = time.perf_counter()
            
            # 查找匹配的关节数据
            match_name = None
            for potential_match in self.robot_ros_node.arm_joint_data:
                if name in potential_match:
                    match_name = potential_match
                    break
            
            if match_name is None:
                continue  # 如果没有匹配项则跳过
            
            try:
                # 使用列表推导式一次性构建数组
                joint_data = self.robot_ros_node.arm_joint_data[match_name]
                # positions = [value for value in joint_data.values()]
                
                # 直接使用torch从列表创建张量
                byte_array = torch.tensor(joint_data, dtype=torch.float32)
                
                follower_joint[name] = byte_array
                
            except Exception as e:
                print(f"Error processing joint data for {name}: {e}")
                continue
            
            self.logs[f"read_follower_{name}_joint_dt_s"] = time.perf_counter() - now

        follower_dexhand = {}
        for name in self.follower_motors:
            now = time.perf_counter()
            
            # 查找匹配的关节数据
            match_name = None
            for potential_match in self.robot_ros_node.dexhand_data:
                if name in potential_match:
                    match_name = potential_match
                    break
            
            if match_name is None:
                continue  # 如果没有匹配项则跳过
            
            try:
                # 使用列表推导式一次性构建数组
                joint_data = self.robot_ros_node.dexhand_data[match_name]
                # positions = [value for value in joint_data.values()]
                
                # 直接使用torch从列表创建张量
                byte_array = torch.tensor(joint_data, dtype=torch.float32)
                byte_array = byte_array / 100.0
                
                follower_dexhand[name] = byte_array
                
            except Exception as e:
                print(f"Error processing joint data for {name}: {e}")
                continue
            
            self.logs[f"read_follower_{name}_joint_dt_s"] = time.perf_counter() - now

        follower_head = {}
        for name in self.follower_motors:
            now = time.perf_counter()
            
            # 查找匹配的关节数据
            match_name = None
            for potential_match in self.robot_ros_node.head_data:
                if name in potential_match:
                    match_name = potential_match
                    break
            
            if match_name is None:
                continue  # 如果没有匹配项则跳过
            
            try:
                # 使用列表推导式一次性构建数组
                joint_data = self.robot_ros_node.head_data[match_name]
                # positions = [value for value in joint_data.values()]
                
                # 直接使用torch从列表创建张量
                byte_array = torch.tensor(joint_data, dtype=torch.float32)
                
                follower_head[name] = byte_array
                
            except Exception as e:
                print(f"Error processing joint data for {name}: {e}")
                continue
            
            self.logs[f"read_follower_{name}_joint_dt_s"] = time.perf_counter() - now


        #记录当前关节角度
        state = []
        for name in self.follower_motors:
            if name in follower_joint:
                state.append(follower_joint[name])
            if name in follower_head:
                state.append(follower_head[name])
            if name in follower_dexhand:
                state.append(follower_dexhand[name])

        state = torch.cat(state)

        #将关节目标位置添加到 action 列表中
        action = []
        for name in self.follower_motors:
            if name in follower_joint:
                action.append(follower_joint[name])
            if name in follower_head:
                action.append(follower_head[name])
            if name in follower_dexhand:
                action.append(follower_dexhand[name])

        action = torch.cat(action)

        # Capture images from cameras
        images = {}
        for name, camera in self.cameras.items():
            now = time.perf_counter()
            
            images[name] = self.robot_ros_node.get_latest_image(camera.topic)
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

    # def send_action(self, action: dict[str, Any]):
    #     """The provided action is expected to be a vector."""
    #     if not self.is_connected:
    #         raise RobotDeviceNotConnectedError(
    #             "KochRobot is not connected. You need to run `robot.connect()`."
    #         )

    #     for name in self.leader_arms:
    #         goal_joint = [ val for key, val in action.items() if name in key and "joint" in key]
    #         # goal_gripper = [ val for key, val in action.items() if name in key and "gripper" in key]

    #         # goal_joint = action[(arm_index*arm_action_dim+from_idx):(arm_index*arm_action_dim+to_idx)]
    #         # goal_gripper = action[arm_index*arm_action_dim + 12]
    #         # arm_index += 1
    #         goal_joint_numpy = np.array([t.item() for t in goal_joint], dtype=np.float32)
    #         # goal_gripper_numpy = np.array([t.item() for t in goal_gripper], dtype=np.float32)
    #         # position = np.concatenate([goal_joint_numpy, goal_gripper_numpy], axis=0)

    #         so101_zmq_send(f"action_joint_{name}", goal_joint_numpy, wait_time_s=0.01)

    def disconnect(self):
        if not self.is_connected:
            raise RobotDeviceNotConnectedError(
                "Aloha is not connected. You need to run `robot.connect()` before disconnecting."
            )

        self.is_connected = False
        self.robot_ros_node.stop()
