
import threading
from typing import Any, Dict, List, Optional

import cv2  
import numpy as np
import zmq
import pickle
import json

import logging_mp

logger = logging_mp.get_logger(__name__)

CONNECT_TIMEOUT_FRAME = 10

NODE_CONFIG = {
    "leader_joint_topics": {
        "leader_joints": {
            "topic": "/robot/lowstate/raw",
            "msg": "JointState"
        }
    },
    "follower_joint_topics": {
        "follower_joints": {
            "topic": "/robot/lowstate/raw",
            "msg": "JointState"
        }
    },
    "camera_topics": {
        "image_top": {
            "topic": "/robot/camera",
            "msg": "Image"
        }
    }
}


class DeeproboticsX30Ros1Node:
    """
    第 2 部分：ZMQ → 本地数据存储
    - 不再依赖 ROS1 / ROS2
    - 通过 ZMQ SUB 接收第一个进程发来的数据
    - 提供 recv_images / recv_follower / recv_leader 给 Robot 调用
    """

    def __init__(
        self,
        zmq_endpoint: str = "tcp://192.168.1.106:5555",
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

        logger.info(f"[ZMQ Node] connected to {zmq_endpoint}")

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
                    message_parts = self._socket.recv_multipart()                    
                    if len(message_parts) < 2:
                        if message_parts:
                            message = message_parts[0]  
                            # 在bytes中查找空格
                            if b' ' in message:
                                prefix, json_bytes = message.split(b' ', 1)
                                
                                try:
                                    # 将bytes解码为字符串
                                    json_str = json_bytes.decode('utf-8')
                                    
                                    # 解析JSON数据
                                    raw_data = json.loads(json_str)

                                    # 处理数据
                                    # processed_data = self.extract_lowstate_data(raw_data)
                                    self.recv_leader["leader_joints"] = raw_data
                                    self.recv_leader_status["leader_joints"] = CONNECT_TIMEOUT_FRAME
                    
                                    self.recv_follower["follower_joint"] = raw_data
                                    self.recv_follower_status["follower_joint"] = CONNECT_TIMEOUT_FRAME
                                    
                                except UnicodeDecodeError as e:
                                    logger.debug(f"UTF-8解码失败: {e}")
                                except json.JSONDecodeError as e:
                                    logger.debug(f"JSON解析失败: {e}")
                                    logger.debug(f"原始JSON: {json_str}")
                                except Exception as e:
                                    logger.debug(f"处理状态数据失败: {e}")
                    else:
                        # 解析相机消息
                        topic_metadata = message_parts[0].decode('utf-8')
                        jpeg_data = message_parts[1]

                        if " " in topic_metadata:
                            _, metadata_str = topic_metadata.split(" ", 1)
                            
                            try:
                                # 解析元数据
                                metadata = json.loads(metadata_str)
                                
                                # 解码图像
                                frame = self.decode_camera_image(jpeg_data)

                                if frame is not None:
                                    self.recv_images["image_top"] = frame
                                    self.recv_images_status["image_top"] = CONNECT_TIMEOUT_FRAME

                            except Exception as e:
                                logger.debug(f"处理相机数据失败: {e}")
                except zmq.Again:
                    continue
                except Exception as e:
                    logger.error(f"相机接收错误: {e}")


    def decode_camera_image(self, jpeg_data: bytes) -> Optional[np.ndarray]:
        """解码JPEG图像数据"""
        try:
            img_array = np.frombuffer(jpeg_data, dtype=np.uint8)
            frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            
            if frame is not None:
                # 转换为RGB格式
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                return frame_rgb
        except Exception as e:
            logger.warning(f"解码图像失败: {e}")
        
        return None
    

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
