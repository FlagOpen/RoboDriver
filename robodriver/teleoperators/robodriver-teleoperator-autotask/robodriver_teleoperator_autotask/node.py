import logging_mp
import threading
import numpy as np
import pyarrow as pa
from dora import Node
from typing import Any, Dict
import time

import rclpy
from rclpy.node import Node as ROS2Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy, HistoryPolicy
from message_filters import Subscriber, ApproximateTimeSynchronizer
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped


logger = logging_mp.get_logger(__name__)


class AutoTaskTeleoperatorNode(ROS2Node):
    def __init__(self):
        super().__init__('autotask_ros2_recv')

        self.last_main_send_time_ns = 0
        self.min_interval_ns = 1e9 / 30

        self._init_message_main_filters()
        
        self.recv_data: list[float] = []
        self.lock = threading.Lock()
        self.running = False


    def _init_message_main_filters(self):
        sub_pose_left = Subscriber(self, PoseStamped, '/autotask_motion_target/target_pose_arm_left')
        sub_pose_right = Subscriber(self, PoseStamped, '/autotask_motion_target/target_pose_arm_right')
        sub_gripper_left = Subscriber(self, JointState, '/autotask_motion_target/target_position_gripper_left')
        sub_gripper_right = Subscriber(self, JointState, '/autotask_motion_target/target_position_gripper_right')

        self.pose_sync = ApproximateTimeSynchronizer(
            [sub_pose_left, sub_pose_right, sub_gripper_left, sub_gripper_right],
            queue_size=50,
            slop=0.1
        )
        self.pose_sync.registerCallback(self.synchronized_main_callback)
 
    def synchronized_main_callback(self, pose_left, pose_right, gripper_left, gripper_right):
        try:
            current_time_ns = time.time_ns()
            if (current_time_ns - self.last_main_send_time_ns) < self.min_interval_ns:
                return
            self.last_main_send_time_ns = current_time_ns
 
            gripper_left_pose = np.array(gripper_left.position, dtype=np.float32)
            gripper_right_pose = np.array(gripper_right.position, dtype=np.float32)

            left_pos = np.array([
                pose_left.pose.position.x, pose_left.pose.position.y, pose_left.pose.position.z,
                pose_left.pose.orientation.w, pose_left.pose.orientation.x, pose_left.pose.orientation.y, pose_left.pose.orientation.z
            ], dtype=np.float32)

            right_pos = np.array([
                pose_right.pose.position.x, pose_right.pose.position.y, pose_right.pose.position.z,
                pose_right.pose.orientation.w, pose_right.pose.orientation.x, pose_right.pose.orientation.y, pose_right.pose.orientation.z
            ], dtype=np.float32)

            merged_data = np.concatenate([left_pos, right_pos, gripper_left_pose, gripper_right_pose])
            with self.lock:
                self.recv_data = merged_data
        except Exception as e:
            self.get_logger().error(f"Pose callback error: {e}")
        
    def destroy(self):
        super().destroy_node()
