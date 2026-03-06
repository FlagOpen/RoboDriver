import threading
import time
from typing import Dict

import numpy as np
import cv2
import rclpy
from rclpy.node import Node as ROS2Node
from sensor_msgs.msg import JointState, CompressedImage
from message_filters import Subscriber, ApproximateTimeSynchronizer
from geometry_msgs.msg import PoseStamped
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy, HistoryPolicy

import logging_mp

CONNECT_TIMEOUT_FRAME = 10
logger = logging_mp.get_logger(__name__)

class DobotNova2Ros2RobotNode(ROS2Node):
    def __init__(self):
        super().__init__('ros2_recv_pub_driver')
        self.stop_spin = False  # 初始化停止标志
        self.qos = QoSProfile(
            durability=DurabilityPolicy.VOLATILE,
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        self.qos_best_effort = QoSProfile(
            durability=DurabilityPolicy.VOLATILE,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        
     
        self.last_main_send_time_ns = 0
        self.last_follow_send_time_ns = 0
        self.min_interval_ns = 1e9 / 30  # 30Hz

        self._init_message_filters()
        self._init_image_message_filters()

        self.recv_images: Dict[str, float] = {}
        self.recv_leader: Dict[str, float] = {}
        self.recv_follower: Dict[str, float] = {}
        self.recv_images_status: Dict[str, int] = {}
        self.recv_leader_status: Dict[str, int] = {}
        self.recv_follower_status: Dict[str, int] = {}

        self.lock = threading.Lock()

    def _init_message_filters(self):
        arm_left = Subscriber(self, JointState, '/joint_states_robot_left')
        arm_right = Subscriber(self, JointState, '/joint_states_robot_right')
 
        self.sync = ApproximateTimeSynchronizer(
            [arm_left, arm_right],
            queue_size=10,
            slop=0.1
        )
        self.sync.registerCallback(self.synchronized_callback)
 
    def synchronized_callback(self, arm_left, arm_right):
        try:
            current_time_ns = time.time_ns()
            if (current_time_ns - self.last_follow_send_time_ns) < self.min_interval_ns:
                return
            self.last_follow_send_time_ns = current_time_ns
 
            left_pos = np.array(arm_left.position, dtype=np.float32)
            right_pos = np.array(arm_right.position, dtype=np.float32)

            left_arm_data = left_pos
            right_arm_data = right_pos
           
            left_merged_data = np.concatenate([left_arm_data])
            right_merged_data = np.concatenate([right_arm_data])
            with self.lock:
                self.recv_follower['follower_arms'] = left_merged_data
                self.recv_follower_status['follower_arms'] = CONNECT_TIMEOUT_FRAME
                self.recv_leader['leader_arms'] = right_merged_data
                self.recv_leader_status['leader_arms'] = CONNECT_TIMEOUT_FRAME
        except Exception as e:
            self.get_logger().error(f"Synchronized follow callback error: {e}")

    def _init_image_message_filters(self):
        sub_camera_top = Subscriber(self, CompressedImage, '/camera/camera_top/color/image_rect_raw/compressed')
        sub_camera_wrist_left = Subscriber(self, CompressedImage, '/camera/camera_left/color/image_rect_raw/compressed')
        sub_camera_wrist_right = Subscriber(self, CompressedImage, '/camera/camera_right/color/image_rect_raw/compressed')
 
        self.image_sync = ApproximateTimeSynchronizer(
            [sub_camera_top, sub_camera_wrist_left, sub_camera_wrist_right],
            queue_size=5,
            slop=0.1
        )
        self.image_sync.registerCallback(self.image_synchronized_callback)

    def image_synchronized_callback(self, top, wrist_left, wrist_right):
        try:
            self.images_recv(top, "image_top", 1280, 720)
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

    def destroy(self):
        self.stop_spin = True
        super().destroy_node()

# 保留ros_spin_thread函数（供外部调用）
def ros_spin_thread(node):
    while rclpy.ok() and not getattr(node, "stop_spin", False):
        rclpy.spin_once(node, timeout_sec=0.01)