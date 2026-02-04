# robodriver_robot_galbot_aio_ros2/node.py

import threading

import cv2
import numpy as np
import logging_mp
import rclpy

from rclpy.node import Node
from std_msgs.msg import Header
from sensor_msgs.msg import Image, JointState
from geometry_msgs.msg import Pose
from nav_msgs.msg import Odometry
from typing import Dict, Any


logger = logging_mp.get_logger(__name__)

CONNECT_TIMEOUT_FRAME = 10


# 由生成脚本根据 JSON 自动生成
NODE_CONFIG = {
    "leader_joint_topics": {
        "right_arm": {"topic": "/right_arm", "msg": "JointState"},
        "left_arm": {"topic": "/left_arm", "msg": "JointState"},
        "right_gripper": {"topic": "/right_gripper", "msg": "JointState"},
        "left_gripper": {"topic": "/left_gripper", "msg": "JointState"},
        "leg": {"topic": "/leg", "msg": "JointState"},
        "head": {"topic": "/head", "msg": "JointState"},
    },
    "follower_joint_topics": {
        "right_arm": {"topic": "/right_arm", "msg": "JointState"},
        "left_arm": {"topic": "/left_arm", "msg": "JointState"},
        "right_gripper": {"topic": "/right_gripper", "msg": "JointState"},
        "left_gripper": {"topic": "/left_gripper", "msg": "JointState"},
        "leg": {"topic": "/leg", "msg": "JointState"},
        "head": {"topic": "/head", "msg": "JointState"},
    },
    "camera_topics": {
        "image_top_right": {"topic": "/front_head_camera/right_color/image_raw", "msg": "Image"},
        "image_top_left": {"topic": "/front_head_camera/left_color/image_raw", "msg": "Image"},
        "image_wrist_right": {"topic": "/right_arm_camera/color/image_raw", "msg": "Image"},
        "image_wrist_left": {"topic": "/left_arm_camera/color/image_raw", "msg": "Image"},
    }
}




class GALBOTAioRos2Node(Node):
    """
    ROS2 → 本地数据存储（无 ZMQ，无 Dora）
    leader / follower / camera 多 topic，按 JSON 配置自动订阅。
    """

    def __init__(
        self,
        leader_joint_topics: Dict[str, Dict[str, str]] = NODE_CONFIG["leader_joint_topics"],
        follower_joint_topics: Dict[str, Dict[str, str]] = NODE_CONFIG["follower_joint_topics"],
        camera_topics: Dict[str, Dict[str, str]] = NODE_CONFIG["camera_topics"],
    ):
        super().__init__("galbot_aio_ros2_direct")

        # ---- 从参数 / NODE_CONFIG 中拿配置 ----
        self.leader_joint_cfgs = leader_joint_topics or {}
        self.follower_joint_cfgs = follower_joint_topics or {}
        self.camera_cfgs = camera_topics or {}

        if not self.leader_joint_cfgs:
            raise RuntimeError("leader_joint_topics is empty")
        if not self.follower_joint_cfgs:
            raise RuntimeError("follower_joint_topics is empty")

        # 相机 topic 简化一个 name -> topic 的 dict
        self.camera_topics: Dict[str, str] = {
            name: info["topic"] for name, info in self.camera_cfgs.items()
        }

        # ---- 各种缓存 ----
        self.recv_images: Dict[str, np.ndarray] = {}
        self.recv_images_status: Dict[str, int] = {}

        self.recv_follower: Dict[str, Any] = {}
        self.recv_follower_status: Dict[str, int] = {}

        self.recv_leader: Dict[str, Any] = {}
        self.recv_leader_status: Dict[str, int] = {}

        self.lock = threading.Lock()
        self.running = False


        # ---- follower side: 订阅所有 follower_joint_topics ----
        for comp_name, cfg in self.follower_joint_cfgs.items():
            topic = cfg["topic"]
            msg_name = cfg.get("msg", "JointState")

            if msg_name == "JointState":
                msg_cls = JointState
                callback = lambda msg, cname=comp_name: self._joint_callback_follower(
                    cname, msg
                )
            elif msg_name == "Pose":
                msg_cls = Pose
                callback = lambda msg, cname=comp_name: self._pose_callback_follower(
                    cname, msg
                )
            elif msg_name == "Odometry":
                msg_cls = Odometry
                callback = lambda msg, cname=comp_name: self._odom_callback_follower(
                    cname, msg
                )
            else:
                raise RuntimeError(f"Unsupported follower msg type: {msg_name}")

            self.create_subscription(
                msg_cls,
                topic,
                callback,
                10,
            )
            logger.info(
                f"[Direct] Follower subscriber '{comp_name}' at {topic} ({msg_name})"
            )

        # ---- leader side: 订阅所有 leader_joint_topics ----
        for comp_name, cfg in self.leader_joint_cfgs.items():
            topic = cfg["topic"]
            msg_name = cfg.get("msg", "JointState")

            if msg_name == "JointState":
                msg_cls = JointState
                callback = lambda msg, cname=comp_name: self._joint_callback_leader(
                    cname, msg
                )
            elif msg_name == "Pose":
                msg_cls = Pose
                callback = lambda msg, cname=comp_name: self._pose_callback_leader(
                    cname, msg
                )
            elif msg_name == "Odometry":
                msg_cls = Odometry
                callback = lambda msg, cname=comp_name: self._odom_callback_leader(
                    cname, msg
                )
            else:
                raise RuntimeError(f"Unsupported leader msg type: {msg_name}")

            self.create_subscription(
                msg_cls,
                topic,
                callback,
                10,
            )
            logger.info(
                f"[Direct] Leader subscriber '{comp_name}' at {topic} ({msg_name})"
            )

        self.pub_action_joint_states = self.create_publisher(
            JointState,
            topic="/joint_states",
            qos_profile=10,
        )

        # ---- cameras: 订阅所有 camera_topics（目前只支持 Image）----
        self.camera_subs = []
        for cam_name, cfg in self.camera_cfgs.items():
            topic = cfg["topic"]
            msg_name = cfg.get("msg", "Image")

            if msg_name != "Image":
                raise RuntimeError(f"Unsupported camera msg type: {msg_name}")

            sub = self.create_subscription(
                Image,
                topic,
                lambda msg, cname=cam_name: self._image_callback(cname, msg),
                10,
            )
            self.camera_subs.append(sub)
            logger.info(f"[Direct] Camera '{cam_name}' subscribed: {topic} ({msg_name})")

        logger.info("[Direct] READY (ROS2 callbacks active).")

    # ======================
    # callbacks
    # ======================

    def _image_callback(self, cam_name: str, msg: Image):
        try:
            with self.lock:
                event_id = f"{cam_name}"

                data = np.frombuffer(msg.data, dtype=np.uint8)
                h, w = msg.height, msg.width
                encoding = msg.encoding.lower()

                frame = None
                try:
                    if encoding == "bgr8":
                        frame = data.reshape((h, w, 3))
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    elif encoding == "rgb8":
                        frame = data.reshape((h, w, 3))
                    elif encoding in ["jpeg", "jpg", "png", "bmp", "webp"]:
                        frame = cv2.imdecode(data, cv2.IMREAD_COLOR)
                    
                except Exception as e:
                    logger.error(f"Image decode error ({encoding}): {e}")

                if frame is not None:
                    self.recv_images[event_id] = frame
                    self.recv_images_status[event_id] = CONNECT_TIMEOUT_FRAME

        except Exception as e:
            logger.error(f"Image callback error ({cam_name}): {e}")

    # ---------- JointState ----------

    def _joint_callback_follower(self, comp_name: str, msg: JointState):
        try:
            with self.lock:
                event_id = comp_name
                self.recv_follower[event_id] = {
                    name: position
                    for name, position in zip(msg.name, msg.position)
                }
                self.recv_follower_status[event_id] = CONNECT_TIMEOUT_FRAME
        except Exception as e:
            logger.error(f"Joint callback error (follower:{comp_name}): {e}")

    def _joint_callback_leader(self, comp_name: str, msg: JointState):
        try:
            with self.lock:
                event_id = comp_name
                self.recv_leader[event_id] = {
                    name: position
                    for name, position in zip(msg.name, msg.position)
                }
                self.recv_leader_status[event_id] = CONNECT_TIMEOUT_FRAME
        except Exception as e:
            logger.error(f"Joint callback error (leader:{comp_name}): {e}")

        # ---------- Pose ----------

    def _pose_callback_follower(self, comp_name: str, msg: Pose):
        """
        follower 侧 Pose 回调
        合并 position + orientation -> [px, py, pz, qx, qy, qz, qw]
        """
        try:
            with self.lock:
                vec = np.array(
                    [
                        msg.position.x,
                        msg.position.y,
                        msg.position.z,
                        msg.orientation.x,
                        msg.orientation.y,
                        msg.orientation.z,
                        msg.orientation.w,
                    ],
                    dtype=float,
                )
                event_id = f"{comp_name}"
                self.recv_follower[event_id] = vec
                self.recv_follower_status[event_id] = CONNECT_TIMEOUT_FRAME
        except Exception as e:
            logger.error(f"Pose callback error (follower:{comp_name}): {e}")

    def _pose_callback_leader(self, comp_name: str, msg: Pose):
        """
        leader 侧 Pose 回调
        合并 position + orientation -> [px, py, pz, qx, qy, qz, qw]
        """
        try:
            with self.lock:
                vec = np.array(
                    [
                        msg.position.x,
                        msg.position.y,
                        msg.position.z,
                        msg.orientation.x,
                        msg.orientation.y,
                        msg.orientation.z,
                        msg.orientation.w,
                    ],
                    dtype=float,
                )
                event_id = f"{comp_name}"
                self.recv_leader[event_id] = vec
                self.recv_leader_status[event_id] = CONNECT_TIMEOUT_FRAME
        except Exception as e:
            logger.error(f"Pose callback error (leader:{comp_name}): {e}")

    # ---------- Odometry ----------

    def _odom_callback_follower(self, comp_name: str, msg: Odometry):
        """
        follower 侧 Odometry 回调
        合并:
          - pose.position        (3)
          - pose.orientation     (4)
          - twist.linear         (3)
          - twist.angular        (3)
        -> 13 维向量
        """
        try:
            with self.lock:
                vec = np.array(
                    [
                        # position
                        msg.pose.pose.position.x,
                        msg.pose.pose.position.y,
                        msg.pose.pose.position.z,
                        # orientation
                        msg.pose.pose.orientation.x,
                        msg.pose.pose.orientation.y,
                        msg.pose.pose.orientation.z,
                        msg.pose.pose.orientation.w,
                        # linear velocity
                        msg.twist.twist.linear.x,
                        msg.twist.twist.linear.y,
                        msg.twist.twist.linear.z,
                        # angular velocity
                        msg.twist.twist.angular.x,
                        msg.twist.twist.angular.y,
                        msg.twist.twist.angular.z,
                    ],
                    dtype=float,
                )
                event_id = f"{comp_name}"
                self.recv_follower[event_id] = vec
                self.recv_follower_status[event_id] = CONNECT_TIMEOUT_FRAME
        except Exception as e:
            logger.error(f"Odometry callback error (follower:{comp_name}): {e}")

    def _odom_callback_leader(self, comp_name: str, msg: Odometry):
        """
        leader 侧 Odometry 回调
        同上，合成 13 维向量
        """
        try:
            with self.lock:
                vec = np.array(
                    [
                        # position
                        msg.pose.pose.position.x,
                        msg.pose.pose.position.y,
                        msg.pose.pose.position.z,
                        # orientation
                        msg.pose.pose.orientation.x,
                        msg.pose.pose.orientation.y,
                        msg.pose.pose.orientation.z,
                        msg.pose.pose.orientation.w,
                        # linear velocity
                        msg.twist.twist.linear.x,
                        msg.twist.twist.linear.y,
                        msg.twist.twist.linear.z,
                        # angular velocity
                        msg.twist.twist.angular.x,
                        msg.twist.twist.angular.y,
                        msg.twist.twist.angular.z,
                    ],
                    dtype=float,
                )
                event_id = f"{comp_name}"
                self.recv_leader[event_id] = vec
                self.recv_leader_status[event_id] = CONNECT_TIMEOUT_FRAME
        except Exception as e:
            logger.error(f"Odometry callback error (leader:{comp_name}): {e}")

    def ros2_send(self, action: dict[str, Any]):

        msg = JointState()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = list(action.keys())
        msg.position = [float(value) for value in action.values()]
        msg.velocity = []
        msg.effort = []
        self.pub_action_joint_states.publish(msg)

    # ======================
    # spin 线程控制
    # ======================

    def start(self):
        """启动 ROS2 spin 线程"""
        if self.running:
            return

        self.running = True
        self.spin_thread = threading.Thread(target=self._spin_loop, daemon=True)
        self.spin_thread.start()

        logger.info("[ROS2] Node started (spin thread running)")

    def _spin_loop(self):
        """独立线程执行 ROS2 spin"""
        try:
            rclpy.spin(self)
        except Exception as e:
            logger.error(f"[ROS2] Spin error: {e}")

    def stop(self):
        """停止 ROS2"""
        if not self.running:
            return

        self.running = False
        rclpy.shutdown()

        if getattr(self, "spin_thread", None):
            self.spin_thread.join(timeout=1.0)

        logger.info("[ROS2] Node stopped.")
