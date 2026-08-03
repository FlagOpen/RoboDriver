#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
第 1 部分：ROS1 → ZMQ
- 订阅 Realman AIO 的各个 ROS1 topic
- 把解码后的 numpy 向量 / 图像通过 ZMQ PUB 发出去
- 不依赖 robot.py，也不依赖新的 ZMQ Node 类
"""

import time
from typing import Dict
import threading

import cv2
import numpy as np
import zmq
import pickle
import rospy

from sensor_msgs.msg import Image, JointState
from geometry_msgs.msg import Pose, PoseStamped
from nav_msgs.msg import Odometry

import logging_mp

logger = logging_mp.getLogger(__name__)

CONNECT_TIMEOUT_FRAME = 10

# 默认的 ROS1 话题配置，如需修改话题名或类型，改这里即可
NODE_CONFIG = {
    "leader_joint_topics": {
        "leader_left_joints": {
            "topic": "/arm_left/joint_states",
            "msg": "JointState"
        },
        "leader_right_joints": {
            "topic": "/arm_right/joint_states",
            "msg": "JointState"
        }
    },
    "follower_joint_topics": {
        "left_arm": {
            "topic": "/arm_left/end_effector_pose",
            "msg": "PoseStamped"
        },
        "right_arm": {
            "topic": "/arm_right/end_effector_pose",
            "msg": "PoseStamped"
        },
        "follower_left_joints": {
            "topic": "/arm_left/joint_states",
            "msg": "JointState"
        },
        "follower_right_joints": {
            "topic": "/arm_right/joint_states",
            "msg": "JointState"
        }
    },
    "camera_topics": {
        "image_top": {
            "topic": "/camera_top/color/image_raw",
            "msg": "Image"
        },
        "image_left": {
            "topic": "/camera_left/color/image_raw",
            "msg": "Image"
        },
        "image_right": {
            "topic": "/camera_right/color/image_raw",
            "msg": "Image"
        }
    }
}


class RealmanRos1ZmqBridge:
    def __init__(
        self,
        zmq_endpoint: str = "tcp://*:6000",
        zmq_bind: bool = True,
        leader_joint_topics: Dict[str, Dict[str, str]] = NODE_CONFIG["leader_joint_topics"],
        follower_joint_topics: Dict[str, Dict[str, str]] = NODE_CONFIG["follower_joint_topics"],
        camera_topics: Dict[str, Dict[str, str]] = NODE_CONFIG["camera_topics"],
    ):
        # ---------- ZMQ ----------
        self.ctx = zmq.Context.instance()
        self.socket = self.ctx.socket(zmq.PUB)

        if zmq_bind:
            self.socket.bind(zmq_endpoint)
            logger.info(f"[ZMQ Bridge] PUB bind at {zmq_endpoint}")
        else:
            self.socket.connect(zmq_endpoint)
            logger.info(f"[ZMQ Bridge] PUB connect to {zmq_endpoint}")

        # ---------- ROS1 ----------
        rospy.init_node("realman_ros1_zmq_bridge", anonymous=False)

        # 配置
        self.leader_joint_cfgs = leader_joint_topics or {}
        self.follower_joint_cfgs = follower_joint_topics or {}
        self.camera_cfgs = camera_topics or {}

        if not self.leader_joint_cfgs:
            raise RuntimeError("leader_joint_topics is empty")
        if not self.follower_joint_cfgs:
            raise RuntimeError("follower_joint_topics is empty")

        # ---------- 本地缓存 ----------
        # camera: name -> (frame, timestamp)
        self.recv_images = {}
        # follower: name -> (vec, kind, timestamp)
        self.recv_follower = {}
        # leader: name -> (vec, kind, timestamp)
        self.recv_leader = {}

        # 线程同步
        self.lock = threading.Lock()
        self.running = True

        # ---------- follower 订阅 ----------
        for comp_name, cfg in self.follower_joint_cfgs.items():
            topic = cfg["topic"]
            msg_name = cfg.get("msg", "JointState")

            if msg_name == "JointState":
                msg_cls = JointState
                cb = lambda msg, cname=comp_name: self._joint_callback_follower(cname, msg)
            elif msg_name == "Pose":
                msg_cls = Pose
                cb = lambda msg, cname=comp_name: self._pose_callback_follower(cname, msg)
            elif msg_name == "PoseStamped":
                msg_cls = PoseStamped
                cb = lambda msg, cname=comp_name: self._pose_stamped_callback_follower(cname, msg)
            elif msg_name == "Odometry":
                msg_cls = Odometry
                cb = lambda msg, cname=comp_name: self._odom_callback_follower(cname, msg)
            else:
                raise RuntimeError(f"Unsupported follower msg type: {msg_name}")

            rospy.Subscriber(topic, msg_cls, cb, queue_size=10)
            logger.info(f"[Bridge] Follower sub '{comp_name}' @ {topic} ({msg_name})")

        # ---------- leader 订阅 ----------
        for comp_name, cfg in self.leader_joint_cfgs.items():
            topic = cfg["topic"]
            msg_name = cfg.get("msg", "JointState")

            if msg_name == "JointState":
                msg_cls = JointState
                cb = lambda msg, cname=comp_name: self._joint_callback_leader(cname, msg)
            elif msg_name == "Pose":
                msg_cls = Pose
                cb = lambda msg, cname=comp_name: self._pose_callback_leader(cname, msg)
            elif msg_name == "PoseStamped":
                msg_cls = PoseStamped
                cb = lambda msg, cname=comp_name: self._pose_stamped_callback_leader(cname, msg)
            elif msg_name == "Odometry":
                msg_cls = Odometry
                cb = lambda msg, cname=comp_name: self._odom_callback_leader(cname, msg)
            else:
                raise RuntimeError(f"Unsupported leader msg type: {msg_name}")

            rospy.Subscriber(topic, msg_cls, cb, queue_size=10)
            logger.info(f"[Bridge] Leader sub '{comp_name}' @ {topic} ({msg_name})")

        # ---------- camera 订阅 ----------
        for cam_name, cfg in self.camera_cfgs.items():
            topic = cfg["topic"]
            msg_name = cfg.get("msg", "Image")

            if msg_name != "Image":
                raise RuntimeError(f"Unsupported camera msg type: {msg_name}")

            rospy.Subscriber(
                topic,
                Image,
                lambda msg, cname=cam_name: self._image_callback(cname, msg),
                queue_size=10,
            )
            logger.info(f"[Bridge] Camera '{cam_name}' @ {topic} ({msg_name})")

        logger.info("[Bridge] ROS1 → ZMQ READY")

        # ---------- 启动轮询发送线程 ----------
        self.sender_thread = threading.Thread(
            target=self._send_loop, kwargs={"hz": 30.0}, daemon=True
        )
        self.sender_thread.start()

    # ======================
    # ZMQ helper
    # ======================

    def _send_zmq(self, topic: str, payload: dict):
        try:
            wire = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
            self.socket.send_multipart([topic.encode("utf-8"), wire])
        except Exception as e:
            logger.error(f"[Bridge] ZMQ send error on topic={topic}: {e}")

    @staticmethod
    def _stamp_to_sec(msg):
        header = getattr(msg, "header", None)
        stamp = getattr(header, "stamp", None)
        if stamp is None:
            return time.time()
        return stamp.secs + stamp.nsecs * 1e-9

    # ======================
    # ROS 回调：只更新缓存，不直接发 ZMQ
    # ======================

    def _image_callback(self, cam_name: str, msg: Image):
        try:
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
                logger.error(f"[Bridge] Image decode error ({encoding}): {e}")

            if frame is None:
                return

            ts = self._stamp_to_sec(msg)

            with self.lock:
                self.recv_images[cam_name] = (frame, ts)

        except Exception as e:
            logger.error(f"[Bridge] Image callback error ({cam_name}): {e}")

    # ---------- JointState ----------

    def _joint_callback_follower(self, comp_name: str, msg: JointState):
        try:
            vec = np.array(msg.position, dtype=float)
            ts = self._stamp_to_sec(msg)
            with self.lock:
                # kind 标记为 follower_joint
                self.recv_follower[comp_name] = (vec, "follower_joint", ts)
        except Exception as e:
            logger.error(f"[Bridge] Joint callback error (follower:{comp_name}): {e}")

    def _joint_callback_leader(self, comp_name: str, msg: JointState):
        try:
            vec = np.array(msg.position, dtype=float)
            ts = self._stamp_to_sec(msg)
            with self.lock:
                self.recv_leader[comp_name] = (vec, "leader_joint", ts)
        except Exception as e:
            logger.error(f"[Bridge] Joint callback error (leader:{comp_name}): {e}")

    # ---------- Pose / PoseStamped ----------

    def _pose_to_vec(self, pose: Pose):
        return np.array(
            [
                pose.position.x,
                pose.position.y,
                pose.position.z,
                pose.orientation.x,
                pose.orientation.y,
                pose.orientation.z,
                pose.orientation.w,
            ],
            dtype=float,
        )

    def _pose_callback_follower(self, comp_name: str, msg: Pose):
        try:
            vec = self._pose_to_vec(msg)
            ts = self._stamp_to_sec(msg)
            with self.lock:
                self.recv_follower[comp_name] = (vec, "follower_pose", ts)
        except Exception as e:
            logger.error(f"[Bridge] Pose callback error (follower:{comp_name}): {e}")

    def _pose_callback_leader(self, comp_name: str, msg: Pose):
        try:
            vec = self._pose_to_vec(msg)
            ts = self._stamp_to_sec(msg)
            with self.lock:
                self.recv_leader[comp_name] = (vec, "leader_pose", ts)
        except Exception as e:
            logger.error(f"[Bridge] Pose callback error (leader:{comp_name}): {e}")

    def _pose_stamped_callback_follower(self, comp_name: str, msg: PoseStamped):
        try:
            vec = self._pose_to_vec(msg.pose)
            ts = self._stamp_to_sec(msg)
            with self.lock:
                self.recv_follower[comp_name] = (vec, "follower_pose", ts)
        except Exception as e:
            logger.error(f"[Bridge] PoseStamped callback error (follower:{comp_name}): {e}")

    def _pose_stamped_callback_leader(self, comp_name: str, msg: PoseStamped):
        try:
            vec = self._pose_to_vec(msg.pose)
            ts = self._stamp_to_sec(msg)
            with self.lock:
                self.recv_leader[comp_name] = (vec, "leader_pose", ts)
        except Exception as e:
            logger.error(f"[Bridge] PoseStamped callback error (leader:{comp_name}): {e}")

    # ---------- Odometry ----------

    def _odom_to_vec(self, msg: Odometry):
        return np.array(
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

    def _odom_callback_follower(self, comp_name: str, msg: Odometry):
        try:
            vec = self._odom_to_vec(msg)
            ts = self._stamp_to_sec(msg)
            with self.lock:
                self.recv_follower[comp_name] = (vec, "follower_odom", ts)
        except Exception as e:
            logger.error(f"[Bridge] Odometry callback error (follower:{comp_name}): {e}")

    def _odom_callback_leader(self, comp_name: str, msg: Odometry):
        try:
            vec = self._odom_to_vec(msg)
            ts = self._stamp_to_sec(msg)
            with self.lock:
                self.recv_leader[comp_name] = (vec, "leader_odom", ts)
        except Exception as e:
            logger.error(f"[Bridge] Odometry callback error (leader:{comp_name}): {e}")

    # ======================
    # 轮询发送线程：每一轮所有话题各发一次
    # ======================

    def _send_loop(self, hz: float = 30.0):
        period = 1.0 / float(hz)

        while not rospy.is_shutdown() and self.running:
            # 拍快照，减小锁持有时间
            with self.lock:
                images = dict(self.recv_images)
                follower = dict(self.recv_follower)
                leader = dict(self.recv_leader)

            # 1) cameras
            for cam_name, (frame, ts) in images.items():
                payload = {
                    "kind": "camera",
                    "name": cam_name,
                    "timestamp": ts,
                    "frame": frame,
                }
                topic = f"camera/{cam_name}"
                # print(topic)
                self._send_zmq(topic, payload)

            # 2) follower
            for comp_name, (vec, kind, ts) in follower.items():
                payload = {
                    "kind": kind,
                    "name": comp_name,
                    "timestamp": ts,
                    "values": vec,
                }
                topic = f"follower/{comp_name}"
                # print(topic)
                self._send_zmq(topic, payload)

            # 3) leader
            for comp_name, (vec, kind, ts) in leader.items():
                payload = {
                    "kind": kind,
                    "name": comp_name,
                    "timestamp": ts,
                    "values": vec,
                }
                topic = f"leader/{comp_name}"
                # print(topic)
                self._send_zmq(topic, payload)

            time.sleep(period)


def main():
    bridge = RealmanRos1ZmqBridge()
    logger.info("RealmanRos1ZmqBridge running ...")
    rospy.spin()
    bridge.running = False
    logger.info("RealmanRos1ZmqBridge stopped.")


if __name__ == "__main__":
    main()
