import threading

import cv2
import numpy as np
import zmq

# IPC Address
ipc_address = "ipc:///tmp/dora-zeromq"

context = zmq.Context()
socket = context.socket(zmq.PAIR)
socket.connect(ipc_address)
socket.setsockopt(zmq.RCVTIMEO, 300)  # 设置接收超时（毫秒）

running_server = True
recv_images = {}  # 缓存每个 event_id 的最新帧
lock = threading.Lock()  # 线程锁


def recv_server():
    """接收数据线程"""
    while running_server:
        try:
            message_parts = socket.recv_multipart()
            if len(message_parts) < 2:
                continue  # 协议错误

            event_id = message_parts[0].decode("utf-8")
            buffer_bytes = message_parts[1]

            # 解码图像
            img_array = np.frombuffer(buffer_bytes, dtype=np.uint8)
            frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            if frame is not None:
                with lock:
                    # print(f"Received event_id = {event_id}")
                    recv_images[event_id] = frame
        except zmq.Again:
            # 接收超时，继续循环
            print("Received Timeout")
            continue
        except Exception as e:
            print("recv error:", e)
            break


def main():
    global running_server

    print("start")

    recv_thread = threading.Thread(target=recv_server, daemon=True)
    recv_thread.start()

    try:
        while True:
            # 复制当前图像缓存（线程安全）
            with lock:
                images_to_show = {k: v.copy() for k, v in recv_images.items()}

            # 显示图像
            for event_id, frame in images_to_show.items():
                window_name = f"Stream {event_id}"
                cv2.imshow(window_name, frame)

            # 检测退出键
            key = cv2.waitKey(1)
            if key == ord("q"):
                break

            pass
    except KeyboardInterrupt:
        print("用户中断")
    finally:
        # 清理资源
        running_server = False
        cv2.destroyAllWindows()
        print("end")


if __name__ == "__main__":
    main()
