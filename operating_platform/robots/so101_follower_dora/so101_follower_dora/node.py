import json
import threading
from typing import Any, Dict

import cv2
import logging_mp
import pyarrow as pa
from dora import Node

logger = logging_mp.get_logger(__name__)
CONNECT_TIMEOUT_FRAME = 10


class RobotNode:
    pass


class DoraRobotNode(RobotNode):
    pass


class SO101FollowerDoraRobotNode(DoraRobotNode):
    def __init__(self):
        self.node = Node("so101_follower_dora")

        self.recv_images: Dict[str, Any] = {}
        self.recv_joint: Dict[str, Any] = {}
        self.recv_images_status: Dict[str, int] = {}
        self.recv_joint_status: Dict[str, int] = {}
        self.lock = threading.Lock()

        self.thread = threading.Thread(target=self.dora_recv, daemon=True, args=(1,))
        self.running = False

    def dora_recv(self, timeout: float):
        # for event in self.node:
        while self.running:
            event = self.node.next(timeout)
            if event["type"] == "INPUT":
                event_id = event["id"]
                data = event["value"].to_numpy()
                meta_data = json.dumps(event["metadata"])

                if "image" in event_id:
                    img_array = data
                    encoding = meta_data["encoding"].lower()
                    width = meta_data["width"]
                    height = meta_data["height"]

                    if encoding == "bgr8":
                        channels = 3
                        frame = img_array.reshape((height, width, channels))
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    elif encoding == "rgb8":
                        channels = 3
                        frame = img_array.reshape((height, width, channels))
                    elif encoding in ["jpeg", "jpg", "jpe", "bmp", "webp", "png"]:
                        channels = 3
                        frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

                    if frame is not None:
                        with self.lock:
                            self.recv_images[event_id] = frame
                            self.recv_images_status[event_id] = CONNECT_TIMEOUT_FRAME

                elif "joint" in event_id:
                    if data is not None:
                        with self.lock:
                            self.recv_joint[event_id] = data
                            self.recv_joint_status[event_id] = CONNECT_TIMEOUT_FRAME

            elif event["type"] == "STOP":
                break

        logger.warning("Dora Node is stopped.")

    def dora_send(self, event_id, data):
        # buffer_bytes = buffer.tobytes()
        logger.debug(f"zmq send event_id:{event_id}, value:{data}")
        self.node.send_output(event_id, pa.array(data, type=pa.float32()))

    def start(self):
        """Start Dora node thread"""
        if self.running:
            logger.warning("Node is already running.")
            return

        self.running = True
        self.thread.start()

        logger.info("Robot Dora node started. Waiting for images and sensor data...")
