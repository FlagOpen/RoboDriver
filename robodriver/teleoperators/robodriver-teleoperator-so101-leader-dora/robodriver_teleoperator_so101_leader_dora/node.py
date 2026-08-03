import logging_mp
import threading
import numpy as np
import pyarrow as pa
from dora import Node
from typing import Any, Dict


logger = logging_mp.getLogger(__name__)
CONNECT_TIMEOUT_FRAME = 10


class TeleoperatorNode:
    pass

class DoraTeleoperatorNode(TeleoperatorNode):
    pass

class SO101LeaderDoraTeleoperatorNode(DoraTeleoperatorNode):
    def __init__(self):
        self.node = Node("so101_leader_dora")
        
        self.recv_joint: Dict[str, float] = {}
        self.recv_joint_status: Dict[str, int] = {}
        self.lock = threading.Lock()

        self.thread = threading.Thread(target=self.dora_recv, daemon=True, args=(1,))
        self.running = False

    def dora_recv(self, timeout: float):
        while self.running:
            event = self.node.next(timeout)
            event_type = event["type"]
            # logger.debug(f"{self} recv event:{event}, type:{event_type}")

            if event_type == "INPUT":
                event_id = event["id"]
                data = event["value"].to_numpy()
                # meta_data = json.loads(event["metadata"])
                # logger.debug(f"{self} \nrecv event_id:{event_id}, value:{data}")

                if 'joint' in event_id:
                    if data is not None:
                        with self.lock:
                            scalar_value = data.item()
                            self.recv_joint[event_id] = scalar_value
                            self.recv_joint_status[event_id] = CONNECT_TIMEOUT_FRAME

            elif event["type"] == "STOP":
                break
        
        logger.warning(f"{self} is stopped.")

    def dora_send(self, event_id, data):
        logger.debug(f"{self} send event_id:{event_id}, value:{data}")
        data_array = np.array(data, dtype=np.float32, copy=True)
        self.node.send_output(event_id, pa.array(data_array, type=pa.float32()))

    def start(self):
        """Start Dora node thread"""
        if self.running:
            logger.warning(f"{self} is already running.")
            return

        self.running = True
        self.thread.start()

        logger.info(f"{self} started. Waiting for sensor data...")
