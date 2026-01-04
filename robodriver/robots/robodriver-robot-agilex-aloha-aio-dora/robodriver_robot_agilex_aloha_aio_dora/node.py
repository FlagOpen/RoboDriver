import logging_mp
import threading
import cv2
import queue
import json
import numpy as np
import pyarrow as pa
import time
from dora import Node
from typing import Any, Dict


logger = logging_mp.get_logger(__name__)
CONNECT_TIMEOUT_FRAME = 10


class RobotNode:
    pass

class DoraRobotNode(RobotNode):
    pass

class AgilexAlohaAIODoraRobotNode(DoraRobotNode):
    def __init__(self):
        self.node = Node("agilex_aloha_aio_dora")
        
        # Update dictionaries to match Agilex Aloha hardware
        self.send_queue = queue.Queue(maxsize=100)
        self.recv_images: Dict[str, Any] = {}
        self.recv_leader_joint_right: list[float] = []
        self.recv_leader_joint_left: list[float] = []
        self.recv_follower_joint_right: list[float] = []
        self.recv_follower_joint_left: list[float] = []
        self.recv_follower_endpose_right: list[float] = []
        self.recv_follower_endpose_left: list[float] = []
        self.recv_images_status: Dict[str, int] = {}
        self.recv_leader_joint_right_status: int = 0
        self.recv_leader_joint_left_status: int = 0
        self.recv_follower_joint_right_status: int = 0
        self.recv_follower_joint_left_status: int = 0
        self.recv_follower_endpose_right_status: int = 0
        self.recv_follower_endpose_left_status: int = 0

        self.lock = threading.Lock()

        self.thread = threading.Thread(target=self.dora_run, daemon=True)
        self.running = False

        self.last_send_time = time.time()

    def dora_send(self, event_id, data):
        """线程安全的发送方法 - 只负责入队"""
        # logger.debug(f"{self} queue send event_id:{event_id}, value:{data}")
        try:
            # 限制发送频率 (可选)
            if time.time() - self.last_send_time < 0.005:  # 200Hz上限
                return
            # 转换为列表确保类型安全
            data_list = [float(x) for x in data]
            self.send_queue.put_nowait((event_id, data_list))

            self.last_send_time = time.time()
        except queue.Full:
            logger.warning(f"Send queue full! Dropping {event_id} event")
        except Exception as e:
            logger.error(f"Queue error: {str(e)}")

    def dora_run(self):
        """统一处理接收和发送的线程 - 确保所有Node操作在单线程完成"""
        logger.info(f"{self} starting receive thread")
        logger.debug(f"Queue size: {self.send_queue.qsize()}, Running: {self.running}")
        while self.running:

            if self.send_queue.qsize() > 50:
                logger.warning(f"Send queue backing up! Size: {self.send_queue.qsize()}")
            # 1. 先处理所有待发送的消息
            while not self.send_queue.empty():
                try:
                    event_id, data = self.send_queue.get_nowait()
                    # 关键：在单线程环境中转换为PyArrow
                    arrow_array = pa.array(list(map(float, data)), type=pa.float32())
                    # logger.debug(f"{self} \nsend event_id: {event_id}, \nvalue: {data}")
                    self.node.send_output(event_id, arrow_array)
                    self.send_queue.task_done()
                except queue.Empty:
                    break
                except Exception as e:
                    logger.error(f"Send error: {str(e)}")
                    self.send_queue.task_done()  # 防止卡死

            # 2. 再处理接收事件 (非阻塞模式)
            try:
                event = self.node.next(timeout=0.001)  # 1ms超时
                if event is None:
                    continue
                    
                if event["type"] == "INPUT":
                    self._handle_input_event(event)
                elif event["type"] == "STOP":
                    break
                    
            except Exception as e:
                if self.running:  # 避免在停止时记录无用错误
                    logger.error(f"Receive error: {str(e)}")

        logger.warning(f"{self} receive thread stopped.")

    def _handle_input_event(self, event):
        """处理输入事件 - 保持线程安全"""
        event_id = event["id"]
        data = event["value"].to_numpy()
        meta_data = event["metadata"]

        with self.lock:
            # logger.debug(f"{self} recv event_id:{event_id}, value:{data}")
            if 'image' in event_id:
                self._process_image(event_id, data, meta_data)
            elif 'joint' in event_id:
                self._process_joint(event_id, data)
            elif 'endpose' in event_id:
                self._process_endpose(event_id, data)

    def _process_image(self, event_id, data, meta_data):
        """处理图像数据"""
        encoding = meta_data["encoding"].lower()
        width = meta_data["width"]
        height = meta_data["height"]
        frame = None

        try:
            if encoding == "bgr8":
                frame = data.reshape((height, width, 3))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            elif encoding == "rgb8":
                frame = data.reshape((height, width, 3))
            elif encoding in ["jpeg", "jpg", "jpe", "bmp", "webp", "png"]:
                frame = cv2.imdecode(data, cv2.IMREAD_COLOR)
        except Exception as e:
            logger.error(f"Image decode error ({encoding}): {str(e)}")

        if frame is not None:
            self.recv_images[event_id] = frame
            self.recv_images_status[event_id] = CONNECT_TIMEOUT_FRAME

    def _process_joint(self, event_id, data):
        """处理关节数据"""
        if data is not None:
            if "leader" in event_id and "right" in event_id:
                self.recv_leader_joint_right = data
                self.recv_leader_joint_right_status = CONNECT_TIMEOUT_FRAME
            elif "leader" in event_id and "left" in event_id:
                self.recv_leader_joint_left = data
                self.recv_leader_joint_left_status = CONNECT_TIMEOUT_FRAME
            elif "follower" in event_id and "right" in event_id:
                self.recv_follower_joint_right = data
                self.recv_follower_joint_right_status = CONNECT_TIMEOUT_FRAME
            elif "follower" in event_id and "left" in event_id:
                self.recv_follower_joint_left = data
                self.recv_follower_joint_left_status = CONNECT_TIMEOUT_FRAME

    def _process_endpose(self, event_id, data):
        """处理末端位姿数据"""
        if data is not None:
            if "right" in event_id:
                self.recv_follower_endpose_right = data
                self.recv_follower_endpose_right_status = CONNECT_TIMEOUT_FRAME
            elif "left" in event_id:
                self.recv_follower_endpose_left = data
                self.recv_follower_endpose_left_status = CONNECT_TIMEOUT_FRAME

    def start(self):
        """Start Dora node thread"""
        if self.running:
            logger.warning(f"{self} is already running.")
            return

        self.running = True
        self.thread.start()

        logger.info(f"{self} started. Waiting for images and sensor data...")

    def stop(self):
        """安全停止节点"""
        self.running = False
        self.thread.join(timeout=1.0)
        
        # 清空发送队列
        while not self.send_queue.empty():
            try:
                self.send_queue.get_nowait()
                self.send_queue.task_done()
            except:
                break
                
        logger.info(f"{self} stopped")
