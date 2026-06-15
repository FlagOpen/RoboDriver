# robodriver_robot_realman_aio_ros1/node.py

import threading
from typing import Dict

import cv2  
import numpy as np
import zmq
import pickle

import logging_mp

logger = logging_mp.getLogger(__name__)

CONNECT_TIMEOUT_FRAME = 10

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


class RealmanAioRos1Node:
    """
    第 2 部分：ZMQ → 本地数据存储
    - 不再依赖 ROS1 / ROS2
    - 通过 ZMQ SUB 接收第一个进程发来的数据
    - 提供 recv_images / recv_follower / recv_leader 给 Robot 调用
    """

    def __init__(
        self,
        zmq_endpoint: str = "tcp://127.0.0.1:6000",
        leader_joint_topics: Dict[str, Dict[str, str]] = NODE_CONFIG["leader_joint_topics"],
        follower_joint_topics: Dict[str, Dict[str, str]] = NODE_CONFIG["follower_joint_topics"],
        camera_topics: Dict[str, Dict[str, str]] = NODE_CONFIG["camera_topics"],
    ):
        self.leader_joint_cfgs = leader_joint_topics or {}
        self.follower_joint_cfgs = follower_joint_topics or {}
        self.camera_cfgs = camera_topics or {}

        self.recv_images: Dict[str, np.ndarray] = {}
        self.recv_images_status: Dict[str, int] = {}

        self.recv_follower: Dict[str, np.ndarray] = {}
        self.recv_follower_status: Dict[str, int] = {}

        self.recv_leader: Dict[str, np.ndarray] = {}
        self.recv_leader_status: Dict[str, int] = {}

        self.lock = threading.Lock()
        self.running = False


        self._ctx = zmq.Context.instance()
        self._socket = self._ctx.socket(zmq.SUB)
        self._socket.connect(zmq_endpoint)
        self._socket.setsockopt_string(zmq.SUBSCRIBE, "")

        logger.info(f"[ZMQ Node] RealmanAioRos1Node connected to {zmq_endpoint}")

        self.spin_thread = None



    def start(self):
        """启动 ZMQ 接收线程（兼容原来的 start 接口）"""
        if self.running:
            return

        self.running = True
        self.spin_thread = threading.Thread(target=self._spin_loop, daemon=True)
        self.spin_thread.start()

        logger.info("[ZMQ Node] Node started (receiver thread running)")

    def _spin_loop(self):
        poller = zmq.Poller()
        poller.register(self._socket, zmq.POLLIN)

        while self.running:
            try:
                socks = dict(poller.poll(timeout=100))  # 100 ms
            except zmq.ZMQError as e:
                logger.error(f"[ZMQ Node] Poll error: {e}")
                break

            if self._socket in socks and socks[self._socket] == zmq.POLLIN:
                try:
                    topic_b, payload_b = self._socket.recv_multipart(flags=zmq.NOBLOCK)
                except zmq.Again:
                    continue
                except Exception as e:
                    logger.error(f"[ZMQ Node] recv_multipart error: {e}")
                    continue

                try:
                    payload = pickle.loads(payload_b)
                    self._handle_payload(payload)
                except Exception as e:
                    logger.error(f"[ZMQ Node] payload decode error: {e}")



    def stop(self):
        """停止 ZMQ 接收线程（兼容原来的 stop 接口）"""
        if not self.running:
            return

        self.running = False

        if self.spin_thread is not None:
            self.spin_thread.join(timeout=1.0)

        try:
            self._socket.close(0)
        except Exception:
            pass

        logger.info("[ZMQ Node] Node stopped.")


    def _handle_payload(self, payload: dict):
        kind = payload.get("kind")
        name = payload.get("name")

        if kind is None or name is None:
            return

        with self.lock:
            if kind == "camera":
                frame = payload.get("frame")
                if frame is not None:
                    self.recv_images[name] = frame
                    self.recv_images_status[name] = CONNECT_TIMEOUT_FRAME

            elif kind.startswith("follower_"):
                values = payload.get("values")
                if values is not None:
                    vec = np.asarray(values, dtype=float)
                    self.recv_follower[name] = vec
                    self.recv_follower_status[name] = CONNECT_TIMEOUT_FRAME

            elif kind.startswith("leader_"):
                values = payload.get("values")
                if values is not None:
                    vec = np.asarray(values, dtype=float)
                    self.recv_leader[name] = vec
                    self.recv_leader_status[name] = CONNECT_TIMEOUT_FRAME

            else:
                logger.debug(f"[ZMQ Node] Unhandled kind={kind}, name={name}")
