#!/usr/bin/env python3
import zmq
import json
import pickle
import time
import threading
import numpy as np
import cv2 

class DataCollectionAdapter:
    def __init__(self, config: dict):
        """传入统一的 yaml 配置"""
        self.config = config
        self.ctx = zmq.Context.instance()
        
        # 1. 自动解析配置
        # ZMQ Hub 绑定的是 0.0.0.0，但我们作为订阅端连接时，需要连 127.0.0.1 
        hub_bind_addr = config.get('zmq', {}).get('pub_address', "tcp://0.0.0.0:5555")
        hub_sub_address = hub_bind_addr.replace("0.0.0.0", "127.0.0.1")
        
        legacy_pub_address = config.get('adapter', {}).get('pub_address', "tcp://0.0.0.0:6000")
        self.fps = config.get('adapter', {}).get('fps', 30.0)
        
        # 2. 初始化 Sockets
        self.sub_socket = self.ctx.socket(zmq.SUB)
        self.sub_socket.connect(hub_sub_address)
        self.sub_socket.setsockopt_string(zmq.SUBSCRIBE, "")
        
        self.pub_socket = self.ctx.socket(zmq.PUB)
        self.pub_socket.bind(legacy_pub_address)
        
        self.running = False
        self.state_cache = {}  
        self.lock = threading.Lock()
        
        print(f"🚀 Unified Adapter Ready: [SUB {hub_sub_address}] -> [PUB {legacy_pub_address}] @ {self.fps} Hz")

    def start(self):
        """由外部主程序调用启动"""
        if self.running:
            return
        self.running = True
        self.recv_thread = threading.Thread(target=self._recv_loop, daemon=True)
        self.send_thread = threading.Thread(target=self._send_loop, kwargs={"hz": self.fps}, daemon=True)
        self.recv_thread.start()
        self.send_thread.start()
        print("✅ Data Collection Adapter started.")

    def stop(self):
        """安全停止"""
        self.running = False
        # 不强制 close socket，让 context 自动回收或进程退出时回收，防止阻塞
        print("🛑 Data Collection Adapter stopped.")

    def _recv_loop(self):
        """严谨的 3 段式接收，防止 Topic 和 Meta 倒置"""
        while self.running:
            try:
                parts = self.sub_socket.recv_multipart()
                if len(parts) != 3: continue

                # 确保 Topic 是字符串
                topic = parts[0].decode('utf-8')
                # 确保 Meta 是字典
                try:
                    meta = json.loads(parts[1].decode('utf-8'))
                except: continue 
                
                raw_payload = parts[2]

                with self.lock:
                    # 关键修复：使用显式标签区分类型
                    if meta.get("is_image") is True:
                        self.state_cache[topic] = {
                            "_type": "image",
                            "meta": meta,
                            "bytes": bytes(raw_payload)
                        }
                    else:
                        try:
                            json_data = json.loads(raw_payload.decode('utf-8'))
                            # 注入时间戳
                            json_data["_timestamp"] = meta.get("timestamp", time.time())
                            json_data["_type"] = "json"
                            self.state_cache[topic] = json_data
                        except: pass
            except Exception as e:
                if self.running: print(f"❌ Recv Error: {e}")

    def _send_loop(self, hz=30.0):
        period = 1.0 / hz
        while self.running:
            loop_start = time.time()
            with self.lock:
                items = list(self.state_cache.items())

            for comp_name, data in items:
                try:
                    # 跳过无效数据
                    if not isinstance(data, dict) or "_type" not in data: continue
                    
                    payload = None
                    ros_ts = data.get("_timestamp", time.time())

                    if data["_type"] == "image":
                        meta = data.get("meta", {})
                        raw_bytes = data["bytes"]
                        frame = None

                        try:
                            # 分流处理：Raw 格式 (通常来自 ROS Camera 驱动)
                            if meta.get("format") == "raw" or meta.get("encoding") == "rgb8":
                                # 从二进制 buffer 恢复 [H, W, C] 矩阵
                                h = meta.get("height", 480)
                                w = meta.get("width", 640)
                                # 恢复为 numpy 数组
                                frame = np.frombuffer(raw_bytes, dtype=np.uint8).reshape(h, w, 3)
                                
                                # 颜色转换：ROS 默认 RGB，但 OpenCV/Node.py 通常使用 BGR
                                if meta.get("encoding") == "rgb8":
                                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                            
                            # 分流处理：压缩格式 (JPEG, PNG)
                            else:
                                img_np = np.frombuffer(raw_bytes, dtype=np.uint8)
                                frame = cv2.imdecode(img_np, cv2.IMREAD_COLOR)

                            if frame is not None:
                                payload = {
                                    "kind": "camera",
                                    "name": comp_name,
                                    "frame": frame,
                                    "timestamp": ros_ts # 优先使用透传的 ROS 时间戳
                                }
                            else:
                                print(f"⚠️ Warning: Decoded frame is None for {comp_name}")

                        except Exception as e:
                            print(f"❌ Image Reshape/Decode Error on [{comp_name}]: {e}")

                    # --- 2. 状态/位姿处理 ---
                    elif data["_type"] == "json":
                        # 自动判断：是 Pose 还是 Joint
                        if "position" in data and isinstance(data["position"], dict):
                            # --- 核心修复：处理嵌套字典的 Pose (XYZ + Q) ---
                            pos = data["position"]
                            ori = data.get("orientation", {})
                            
                            # 显式提取，确保顺序固定为 [x, y, z, qx, qy, qz, qw]
                            pos_list = [pos.get('x', 0.0), pos.get('y', 0.0), pos.get('z', 0.0)]
                            ori_list = [ori.get('x', 0.0), ori.get('y', 0.0), ori.get('z', 0.0), ori.get('w', 1.0)]
                            
                            pos_values = pos_list + ori_list
                            kind_suffix = "pose"
                        else:
                            # 处理普通列表格式的 Joint
                            pos_values = data.get("position", [])
                            kind_suffix = "joint"

                        if pos_values:
                            prefix = "leader" if any(x in comp_name.lower() for x in ["master", "leader", "hp"]) else "follower"
                            payload = {
                                "kind": f"{prefix}_{kind_suffix}",
                                "name": comp_name,
                                "values": np.array(pos_values, dtype=np.float32),
                                "timestamp": ros_ts
                            }

                    if payload:
                        wire = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
                        self.pub_socket.send_multipart([comp_name.encode(), wire])

                except Exception as e:
                    # 增加 Topic 打印，方便定位具体哪个传感器坏了
                    print(f"⚠️ Send Error on [{comp_name}]: {e}")

            elapsed = time.time() - loop_start
            if elapsed < period:
                time.sleep(period - elapsed)

if __name__ == "__main__":
    adapter = DataCollectionAdapter()
    try:
        while True:
            # time.sleep(5)
            # print(f"📊 Active Topics: {list(adapter.state_cache.keys())}")
            pass
    except KeyboardInterrupt:
        adapter.running = False