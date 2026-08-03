import logging_mp
import threading
import cv2
import queue
import json
import numpy as np
import pyarrow as pa
import time
from typing import Any, Dict

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, CompressedImage
from message_filters import Subscriber, ApproximateTimeSynchronizer
from geometry_msgs.msg import PoseStamped
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy, HistoryPolicy


logger = logging_mp.get_logger(__name__)
CONNECT_TIMEOUT_FRAME = 10


class AgilexAlohaEEposeROS2RobotNode(Node):
    def __init__(self):
        super().__init__('robodriver_agilex_aloha_eepose_ros2_node')
        
        self.qos = QoSProfile(
            durability=DurabilityPolicy.VOLATILE,
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        self.publisher_left_arm_eepose = self.create_publisher(
            PoseStamped, '/motion_target/target_pose_arm_left', self.qos
        )
        self.publisher_right_arm_eepose = self.create_publisher(
            PoseStamped, '/motion_target/target_pose_arm_right', self.qos
        )
        self.publisher_left_gripper = self.create_publisher(
            JointState, "/motion_target/target_position_gripper_left", self.qos  
        )
        self.publisher_right_gripper = self.create_publisher(
            JointState, "/motion_target/target_position_gripper_right", self.qos  
        )
     
        self.last_follow_send_time_ns = 0
        self.min_interval_ns = 1e9 / 30

        self._init_message_follow_filters()
        self._init_image_message_filters()

        self.recv_images: Dict[str, float] = {}
        self.recv_follower: list[float] = []
        self.recv_images_status: Dict[str, int] = {}
        self.recv_follower_status: list[int] = []

        self.lock = threading.Lock()
        self.running = False
        

    def _init_message_follow_filters(self):
        sub_arm_left = Subscriber(self, JointState, '/hdas/feedback_arm_left')
        sub_arm_right = Subscriber(self, JointState, '/hdas/feedback_arm_right')
        sub_gripper_left = Subscriber(self, JointState, '/hdas/feedback_gripper_left')
        sub_gripper_right = Subscriber(self, JointState, '/hdas/feedback_gripper_right')
        sub_torso = Subscriber(self, JointState, '/hdas/feedback_torso')
        
        self.sync = ApproximateTimeSynchronizer(
            [sub_arm_left, sub_arm_right, sub_gripper_left, sub_gripper_right, sub_torso],
            queue_size=10,
            slop=0.1
        )
        self.sync.registerCallback(self.synchronized_follow_callback)
 
    def synchronized_follow_callback(self, arm_left, arm_right, gripper_left, gripper_right, torso):
        try:
            current_time_ns = time.time_ns()
            if (current_time_ns - self.last_follow_send_time_ns) < self.min_interval_ns:
                return
            self.last_follow_send_time_ns = current_time_ns
 
            left_pos = np.array(arm_left.position, dtype=np.float32)
            right_pos = np.array(arm_right.position, dtype=np.float32)
            left_arm_data = left_pos
            right_arm_data = right_pos
 
            gripper_left_pos = np.array(gripper_left.position, dtype=np.float32)
            gripper_right_pos = np.array(gripper_right.position, dtype=np.float32)
            torso_pos = np.array(torso.position, dtype=np.float32)
            torso_pos = torso_pos[:-1]
           
            merged_data = np.concatenate([left_arm_data, gripper_left_pos, right_arm_data, gripper_right_pos, torso_pos])
            with self.lock:
                self.recv_follower = merged_data
                self.recv_follower_status = [CONNECT_TIMEOUT_FRAME, CONNECT_TIMEOUT_FRAME, CONNECT_TIMEOUT_FRAME, CONNECT_TIMEOUT_FRAME, CONNECT_TIMEOUT_FRAME]
        except Exception as e:
            self.get_logger().error(f"Synchronized follow callback error: {e}")

    def _init_image_message_filters(self):
        sub_camera_top_left = Subscriber(self, CompressedImage, '/hdas/camera_head/left_raw/image_raw_color/compressed')
        sub_camera_top_right = Subscriber(self, CompressedImage, '/hdas/camera_head/right_raw/image_raw_color/compressed')
        sub_camera_wrist_left = Subscriber(self, CompressedImage, '/hdas/camera_wrist_left/color/image_raw/compressed')
        sub_camera_wrist_right = Subscriber(self, CompressedImage, '/hdas/camera_wrist_right/color/image_raw/compressed')
 
        self.image_sync = ApproximateTimeSynchronizer(
            [sub_camera_top_left, sub_camera_top_right, sub_camera_wrist_left, sub_camera_wrist_right],
            queue_size=5,
            slop=0.1
        )
        self.image_sync.registerCallback(self.image_synchronized_callback)

    def image_synchronized_callback(self, top_left, top_right, wrist_left, wrist_right):
        try:
            self.images_recv(top_left, "image_top_left", 1280, 720)
            self.images_recv(top_right, "image_top_right", 1280, 720)
            self.images_recv(wrist_left, "image_wrist_left", 640, 360)
            self.images_recv(wrist_right, "image_wrist_right", 640, 360)
        except Exception as e:
            self.get_logger().error(f"Image synchronized callback error: {e}")
    
    def images_recv(self, msg, event_id, width, height, encoding="jpeg"):
        try:
            if 'image' in event_id:
                img_array = np.frombuffer(msg.data, dtype=np.uint8)
                if encoding == "bgr8":
                    channels = 3
                    frame = img_array.reshape((height, width, channels)).copy()
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                elif encoding == "rgb8":
                    channels = 3
                    frame = img_array.reshape((height, width, channels))
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                elif encoding in ["jpeg", "jpg", "jpe", "bmp", "webp", "png"]:
                    channels = 3
                    frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                elif encoding == "depth16":
                    frame = np.frombuffer(msg.data, dtype=np.uint16).reshape(height, width,1)
                
                if frame is not None:
                    with self.lock:
                        self.recv_images[event_id] = frame
                        self.recv_images_status[event_id] = CONNECT_TIMEOUT_FRAME
        except Exception as e:
            logger.error(f"recv image error: {e}")

    def ros_replay(self, array):
        try:
            def normalize_precision(val, decimals=3):
                val = float(val)
                if np.isnan(val) or np.isinf(val):
                    self.get_logger().warning(f"检测到非法值 {val}，替换为0.0")
                    return 0.0
                return round(val, decimals)
            
            left_arm_eepose = [normalize_precision(v) for v in array[0:7]]
            right_arm_eepose = [normalize_precision(v) for v in array[7:14]]
            left_gripper = [normalize_precision(v) for v in array[14:15]]
            right_gripper = [normalize_precision(v) for v in array[15:16]]

            msg = PoseStamped()
            msg.pose.position.x = left_arm_eepose[0]
            msg.pose.position.y = left_arm_eepose[1]
            msg.pose.position.z = left_arm_eepose[2]
            msg.pose.orientation.x = left_arm_eepose[3]
            msg.pose.orientation.y = left_arm_eepose[4]
            msg.pose.orientation.z = left_arm_eepose[5]
            msg.pose.orientation.w = left_arm_eepose[6]
            self.publisher_left_arm_eepose.publish(msg)

            msg = PoseStamped()
            msg.pose.position.x = right_arm_eepose[0]
            msg.pose.position.y = right_arm_eepose[1]
            msg.pose.position.z = right_arm_eepose[2]
            msg.pose.orientation.x = right_arm_eepose[3]
            msg.pose.orientation.y = right_arm_eepose[4]
            msg.pose.orientation.z = right_arm_eepose[5]
            msg.pose.orientation.w = right_arm_eepose[6]
            self.publisher_right_arm_eepose.publish(msg)

            msg = JointState()
            msg.position = left_gripper
            self.publisher_left_gripper.publish(msg)

            msg = JointState()
            msg.position = right_gripper
            self.publisher_right_gripper.publish(msg)

        except Exception as e:
            self.get_logger().error(f"Error during replay at frame: {e}")
            raise

    def destroy(self):
        super().destroy_node()