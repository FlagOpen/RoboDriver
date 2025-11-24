import json
import threading
import time
from typing import Dict

import cv2
import logging_mp
import numpy as np
import zmq

logger = logging_mp.get_logger(__name__)
CONNECT_TIMEOUT_FRAME = 10


ipc_address_image = "ipc:///tmp/dora-zeromq-so101-image"
ipc_address_joint = "ipc:///tmp/dora-zeromq-so101-joint"

recv_images = {}
recv_joint = {}
recv_images_status: Dict[str, int] = {}
recv_joint_status: Dict[str, int] = {}
lock = threading.Lock()  # 线程锁

running_recv_image_server = True
running_recv_joint_server = True

zmq_context = zmq.Context()

socket_image = zmq_context.socket(zmq.PAIR)
socket_image.connect(ipc_address_image)
socket_image.setsockopt(zmq.RCVTIMEO, 2000)

socket_joint = zmq_context.socket(zmq.PAIR)
socket_joint.connect(ipc_address_joint)
socket_joint.setsockopt(zmq.RCVTIMEO, 2000)


def so101_zmq_send(event_id, buffer, wait_time_s):
    buffer_bytes = buffer.tobytes()
    logger.debug(f"zmq send event_id:{event_id}, value:{buffer}")
    try:
        socket_joint.send_multipart(
            [event_id.encode("utf-8"), buffer_bytes], flags=zmq.NOBLOCK
        )
    except zmq.Again:
        pass
    time.sleep(wait_time_s)


def recv_image_server():
    """接收数据线程"""
    while running_recv_image_server:
        try:
            message_parts = socket_image.recv_multipart()
            if len(message_parts) < 2:
                continue  # 协议错误

            event_id = message_parts[0].decode("utf-8")
            buffer_bytes = message_parts[1]
            metadata = json.loads(message_parts[2].decode("utf-8"))

            if "image" in event_id:
                # 解码图像
                img_array = np.frombuffer(buffer_bytes, dtype=np.uint8)
                encoding = metadata["encoding"].lower()
                width = metadata["width"]
                height = metadata["height"]

                if encoding == "bgr8":
                    channels = 3
                    frame = img_array.reshape(
                        (height, width, channels)
                    ).copy()  # Copy So that we can add annotation on the image
                elif encoding == "rgb8":
                    channels = 3
                    frame = img_array.reshape((height, width, channels))
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                elif encoding in ["jpeg", "jpg", "jpe", "bmp", "webp", "png"]:
                    channels = 3
                    frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                if frame is not None:
                    with lock:
                        # print(f"Received event_id = {event_id}")
                        recv_images[event_id] = frame
                        recv_images_status[event_id] = CONNECT_TIMEOUT_FRAME

        except zmq.Again:
            logger.warning("SO101 Image Received Timeout")
            continue
        except Exception as e:
            logger.error("recv image error:", e)
            break


def recv_joint_server():
    """接收数据线程"""
    while running_recv_joint_server:
        try:
            message_parts = socket_joint.recv_multipart()
            if len(message_parts) < 2:
                continue  # 协议错误

            event_id = message_parts[0].decode("utf-8")
            buffer_bytes = message_parts[1]

            if "joint" in event_id:
                joint_array = np.frombuffer(buffer_bytes, dtype=np.float32)
                if joint_array is not None:
                    with lock:
                        recv_joint[event_id] = joint_array
                        recv_joint_status[event_id] = CONNECT_TIMEOUT_FRAME

        except zmq.Again:
            logger.warning("SO101 Joint Received Timeout")
            continue
        except Exception as e:
            logger.error("recv joint error:", e)
            break
