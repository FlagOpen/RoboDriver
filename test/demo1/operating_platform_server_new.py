import json
import logging
import threading
import time

import cv2
import numpy as np
from flask import Flask, Response, jsonify, request
from flask_cors import CORS
from gevent import monkey

monkey.patch_all()

app = Flask(__name__)
CORS(app)


# 使用线程安全的数据结构
class StreamManager:
    def __init__(self):
        self.lock = threading.RLock()
        self.video_streams = {}
        self.stream_status = {}
        self.video_list = {"total": 0, "streams": []}

    def init_streams(self):
        if self.video_list["total"] == 0:
            return

        with self.lock:
            # 停止所有现有流
            for stream_id, stream in list(self.video_streams.items()):
                stream.stop()

            # 创建新流
            new_video_streams = {}
            new_stream_status = {}

            for stream in self.video_list["streams"]:
                stream_id = stream["id"]
                new_video_streams[stream_id] = VideoStream(stream_id, stream["name"])
                new_stream_status[stream_id] = {
                    "id": stream_id,
                    "name": str(stream["name"]),
                    "active": False,
                }

            self.video_streams = new_video_streams
            self.stream_status = new_stream_status
            logging.info(f"Streams initialized: {list(new_video_streams.keys())}")

    def update_video_list(self, new_list):
        with self.lock:
            self.video_list = new_list
            self.init_streams()
            print(f"self.video_streams: {self.video_streams}")
            print(f"self.stream_status: {self.stream_status}")

    def get_stream(self, stream_id):
        with self.lock:
            return self.video_streams.get(stream_id)

    def get_status(self, stream_id):
        with self.lock:
            return self.stream_status.get(stream_id)

    def get_all_status(self):
        with self.lock:
            return self.stream_status.copy()

    def active_stream_count(self):
        with self.lock:
            return sum(1 for s in self.stream_status.values() if s["active"])


# 全局流管理器
stream_manager = StreamManager()


class VideoStream:
    def __init__(self, stream_id, name):
        self.stream_id = stream_id
        self.name = name
        self.running = False
        self.frame_buffers = [None, None]  # 双缓冲
        self.buffer_index = 0
        self.lock = threading.Lock()
        self.last_update_time = time.time()

    def start(self):
        if self.running:
            return True
        self.running = True
        self.last_update_time = time.time()
        return True

    def stop(self):
        self.running = False

    def update_frame(self, frame_data):
        # if not self.running:

        #     return

        try:
            # 解码
            img = cv2.imdecode(np.frombuffer(frame_data, np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                print("Failed to decode image frame data")
                return False

            # 直接显示原始尺寸图像（避免不必要的resize）
            cv2.imshow(self.name, img)
            cv2.waitKey(1)

            compressed_frame = frame_data

            print("complete img encode")

            with self.lock:
                self.buffer_index = 1 - self.buffer_index
                self.frame_buffers[self.buffer_index] = compressed_frame
                self.last_update_time = time.time()
            return True
        except Exception as e:
            logging.error(f"Frame update failed: {e}")
            print(f"Error occurred: {e}")  # 打印异常信息
            return False

    def get_frame(self):
        if not self.running:
            return self.generate_blank_frame()

        # 检查流是否超时（5秒无更新）
        if time.time() - self.last_update_time > 5:
            self.running = False
            return self.generate_blank_frame()

        with self.lock:
            return self.frame_buffers[self.buffer_index] or self.generate_blank_frame()

    @staticmethod
    def generate_blank_frame():
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        _, jpeg = cv2.imencode(".jpg", blank)
        return jpeg.tobytes()


@app.route("/api/info")
def system_info():
    """获取系统信息"""
    return jsonify(
        {
            "status": "running",
            "streams_active": stream_manager.active_stream_count(),
            "total_streams": len(stream_manager.get_all_status()),
            "timestamp": time.time(),
            "streams": stream_manager.get_all_status(),
        }
    )


@app.route("/robot/stream_info", methods=["POST"])
def update_streams():
    """更新视频流配置"""
    try:
        new_list = request.get_json()
        new_list = json.loads(new_list)
        print(f"new_list: {new_list}")
        if "streams" not in new_list:
            raise ValueError("Invalid stream info format")

        stream_manager.update_video_list(new_list)
        return jsonify({"status": "streams updated"})
    except Exception as e:
        logging.error(f"Update streams failed: {e}")
        return jsonify({"error": str(e)}), 400


@app.route("/api/stream_info", methods=["GET"])
def get_streams():
    """获取可用视频流列表"""
    status = stream_manager.get_all_status()
    streams = [{"id": sid, "name": info["name"]} for sid, info in status.items()]
    print(f"sget_treams:{streams}")
    return jsonify({"streams": streams})


@app.route("/api/start_stream", methods=["POST"])
def start_stream():
    """启动指定视频流"""
    try:
        data = request.get_json()
        stream_id = data.get("stream_id")

        print(f"stream_id:{stream_id}")
        stream = stream_manager.get_stream(int(stream_id))

        if not stream:
            return jsonify({"error": "Invalid stream ID"}), 404

        if stream.start():
            with stream_manager.lock:
                if status := stream_manager.get_status(stream_id):
                    status["active"] = True
            return jsonify({"status": "started"})
        return jsonify({"error": "Stream start failed"}), 500
    except Exception as e:
        logging.error(f"Start stream failed: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/get_stream/<int:stream_id>")
def stream_video(stream_id):
    """视频流生成端点"""
    stream = stream_manager.get_stream(stream_id)
    if not stream:
        return jsonify({"error": "Stream not found"}), 404

    def generate():
        try:
            while True:
                frame = stream.get_frame()
                yield (
                    b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
                )
                time.sleep(0.1)  # ~30 FPS
        except GeneratorExit:
            logging.info(f"Client disconnected from stream: {stream_id}")
        except Exception as e:
            logging.error(f"Stream generation failed: {e}")

    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/robot/update_stream/<int:stream_id>", methods=["POST"])
def update_frame(stream_id):
    """更新视频流帧数据"""
    stream = stream_manager.get_stream(stream_id)
    if not stream:
        return jsonify({"error": "Stream not found"}), 404

    frame_data = request.get_data()
    if not frame_data:
        return jsonify({"error": "No frame data received"}), 400

    if stream.update_frame(frame_data):
        return jsonify({"status": "frame updated"})
    return jsonify({"error": "Frame update failed"}), 500


@app.route("/api/stop_stream/<int:stream_id>", methods=["POST"])
def stop_stream(stream_id):
    """停止指定视频流"""
    stream = stream_manager.get_stream(stream_id)
    if not stream:
        return jsonify({"error": "Stream not found"}), 404

    stream.stop()
    with stream_manager.lock:
        if status := stream_manager.get_status(stream_id):
            status["active"] = False

    return jsonify({"status": "stopped"})


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # 初始空配置
    stream_manager.update_video_list({"total": 0, "streams": []})
    app.run(host="0.0.0.0", port=8080, debug=False, threaded=True)
