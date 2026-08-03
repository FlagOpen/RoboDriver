# robodriver_robot_linkerbot_aio_ros1/node.py

import threading
from typing import Dict

import cv2
import numpy as np
import zmq
import pickle
import json
import logging_mp

logger = logging_mp.get_logger(__name__)

# 超时计数器初始值
CONNECT_TIMEOUT_FRAME = 10

class LinkerBotAioRos1Node:
    """
    第 2 部分：ZMQ → 本地数据存储
    - [优化] 移除了静态 CONFIG，现在的节点是自适应的。
    - 只要 ZMQ 发送端发来什么名字的数据，这里就自动缓存什么。
    """

    def __init__(
        self,
        zmq_endpoint: str = "tcp://127.0.0.1:6000",
        zmq_cmd_endpoint: str = "tcp://127.0.0.1:5556"
        # 原有的 topic 配置参数已移除，实现配置解耦
    ):
        # ---------- 本地缓存 (对外接口) ----------
        # 结构: name -> numpy_array
        self.recv_images: Dict[str, np.ndarray] = {}
        self.recv_images_status: Dict[str, int] = {}

        self.recv_follower: Dict[str, np.ndarray] = {}
        self.recv_follower_status: Dict[str, int] = {}

        self.recv_leader: Dict[str, np.ndarray] = {}
        self.recv_leader_status: Dict[str, int] = {}

        # ---------- 内部状态 ----------
        self.lock = threading.Lock()
        self.running = False

        # ---------- ZMQ 连接 ----------
        self._ctx = zmq.Context.instance()
        self._socket = self._ctx.socket(zmq.SUB)
        self._socket.connect(zmq_endpoint)
        self._socket.setsockopt_string(zmq.SUBSCRIBE, "") # 订阅所有消息

        logger.info(f"[ZMQ Node] LinkerBotAioRos1Node connected to {zmq_endpoint}")

        self.spin_thread = None

        self._cmd_socket = self._ctx.socket(zmq.PUB)
        # 注意这里是 connect 而不是 bind，因为 Repo 1 的 Hub 已经 bind 了 5556
        self._cmd_socket.connect(zmq_cmd_endpoint) 
        logger.info(f"[ZMQ Node] Command publisher connected to {zmq_cmd_endpoint}")

    def start(self):
        """启动 ZMQ 接收线程"""
        if self.running:
            return

        self.running = True
        self.spin_thread = threading.Thread(target=self._spin_loop, daemon=True)
        self.spin_thread.start()

        logger.info("[ZMQ Node] Node started (receiver thread running)")

    def stop(self):
        """停止 ZMQ 接收线程"""
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

    def _spin_loop(self):
        """
        后台轮询逻辑 (保持原架构不变)
        """
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
                    # 接收多帧消息: [topic, payload]
                    topic_b, payload_b = self._socket.recv_multipart(flags=zmq.NOBLOCK)
                except zmq.Again:
                    continue
                except Exception as e:
                    logger.error(f"[ZMQ Node] recv_multipart error: {e}")
                    continue

                try:
                    # 反序列化
                    payload = pickle.loads(payload_b)
                    self._handle_payload(payload)
                except Exception as e:
                    logger.error(f"[ZMQ Node] payload decode error: {e}")

    def _handle_payload(self, payload: dict):
        """
        处理逻辑：
        这里会自动匹配发送端发来的 kind 和 name，无需查表。
        """
        kind = payload.get("kind")
        name = payload.get("name")
        # print(f"get:{name}")
        if kind is None or name is None:
            return

        with self.lock:
            # 1. 图像数据
            if kind == "camera":
                frame = payload.get("frame")
                if frame is not None:
                    self.recv_images[name] = frame
                    self.recv_images_status[name] = CONNECT_TIMEOUT_FRAME

            # 2. 机械臂/从手数据 (匹配 kind="follower_joint" / "follower_pose")
            elif kind.startswith("follower_"):
                values = payload.get("values")
                if values is not None:
                    vec = np.asarray(values, dtype=float)
                    self.recv_follower[name] = vec
                    self.recv_follower_status[name] = CONNECT_TIMEOUT_FRAME

            # 3. 主手/遥控数据 (匹配 kind="leader_joint" / "leader_pose")
            elif kind.startswith("leader_"):
                values = payload.get("values")
                if values is not None:
                    vec = np.asarray(values, dtype=float)
                    self.recv_leader[name] = vec
                    self.recv_leader_status[name] = CONNECT_TIMEOUT_FRAME

            else:
                # 仅在调试时开启，避免刷屏
                # logger.debug(f"[ZMQ Node] Unhandled kind={kind}, name={name}")
                pass

    def send_control_command(self, device_key: str, positions: list):
        """
        向底层发送控制指令
        :param device_key: 对应 settings.yaml 中的 key，比如 "leader_left_joints"
        :param positions: 目标角度列表
        """
        payload = {
            "device_key": device_key,
            "command": {
                "position": positions
            }
        }
        try:
            # Repo 1 的 _receive_loop 期望直接收到 JSON 字符串
            self._cmd_socket.send_string(json.dumps(payload))
        except Exception as e:
            logger.error(f"[ZMQ Node] Failed to send command: {e}")