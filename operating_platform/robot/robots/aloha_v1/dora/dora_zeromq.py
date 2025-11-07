import zmq
import threading
import pyarrow as pa
import time
from dora import Node
import numpy as np
import queue
import json

node = Node()

# IPC Address
ipc_address = "ipc:///tmp/dora-zeromq"
ipc_address_piper = "ipc:///tmp/dorobot-piper"

context = zmq.Context()
running_recv_image_server = True
running_recv_piper_server = True

socket_image = context.socket(zmq.PAIR)
socket_image.bind(ipc_address)
socket_image.setsockopt(zmq.SNDHWM, 2000)
socket_image.setsockopt(zmq.SNDBUF, 2**25)
socket_image.setsockopt(zmq.SNDTIMEO, 2000)
socket_image.setsockopt(zmq.RCVTIMEO, 2000)  # 设置接收超时（毫秒）
socket_image.setsockopt(zmq.LINGER, 0)

socket_piper = context.socket(zmq.PAIR)
socket_piper.bind(ipc_address_piper)
socket_piper.setsockopt(zmq.SNDHWM, 2000)
socket_piper.setsockopt(zmq.SNDBUF, 2**25)
socket_piper.setsockopt(zmq.SNDTIMEO, 2000)
socket_piper.setsockopt(zmq.RCVTIMEO, 2000)  # 设置接收超时（毫秒）
socket_piper.setsockopt(zmq.LINGER, 0)

running_server = True

# 创建线程安全队列 (在全局作用域)
output_queue = queue.Queue()

def recv_server():
    while running_server:
        try:
            message_parts = socket_piper.recv_multipart()
            if message_parts and len(message_parts) >= 2:
                event_id = message_parts[0].decode('utf-8')
                buffer_bytes = message_parts[1]
                
                # print(f"zmq recv event_id:{event_id}")
                array = np.frombuffer(buffer_bytes, dtype=np.float32).copy()
                # print(f"zmq recv array:{array}")

                # ✅ 仅将数据放入队列，不再操作 node
                if 'action_joint_right' in event_id:
                    output_queue.put(("action_joint_right", array))
                if 'action_joint_left' in event_id:
                    output_queue.put(("action_joint_left", array))
                if 'action_gripper_right' in event_id:
                    output_queue.put(("action_gripper_right", array))
                if 'action_gripper_left' in event_id:
                    output_queue.put(("action_gripper_left", array))
                    
        except zmq.Again:
            print(f"Dora ZeroMQ Received Timeout")
            time.sleep(0.01)
            continue
            
        except Exception as e:
            print("recv error:", e)
            break



if __name__ == "__main__":


    server_thread = threading.Thread(target=recv_server)
    server_thread.start()

    try:
        for event in node:
            # 1. 先处理队列中的待发送数据
            while not output_queue.empty():
                try:
                    port, array = output_queue.get_nowait()
                    # ✅ 此时在主线程，安全操作 node
                    node.send_output(port, pa.array(array, type=pa.float32()))
                    # print(f"MAIN THREAD: send event_id : <{port}> succese")
                    # print(f"MAIN THREAD: send array : <{array}> succese")
                except queue.Empty:
                    break

            if event["type"] == "INPUT":
                event_id = event["id"]
                buffer_bytes = event["value"].to_numpy().tobytes()
                meta_bytes = json.dumps(event["metadata"]).encode('utf-8')
                            
                # 处理接收到的数据
                # print(f"Send event: {event_id}")
                # print(f"Buffer size: {len(buffer_bytes)} bytes")

                if "image" in event_id:
                    try:
                        socket_image.send_multipart([
                            event_id.encode('utf-8'),
                            buffer_bytes,
                            meta_bytes
                        ], flags=zmq.NOBLOCK)
                    except zmq.Again:
                        pass
                else:
                    try:
                        socket_piper.send_multipart([
                            event_id.encode('utf-8'),
                            buffer_bytes,
                            meta_bytes
                        ], flags=zmq.NOBLOCK)
                    except zmq.Again:
                        pass
                
            elif event["type"] == "STOP":
                break
    
    finally:
        # Close server 
        running_server = False
        server_thread.join()

        # Close zmq
        socket_image.close()
        socket_piper.close()

        context.term()
