import zmq
import json
import threading
import time

class ZmqHub:
    def __init__(self, config: dict):
        self.config = config
        # --- 新增：Debug 控制参数 ---
        self.debug = config.get('zmq', {}).get('debug', True) 
        self.log_interval = 100  # 高频数据每 100 帧打印一次
        self.pub_count = {}      # 用于计数不同 topic 的发送次数
        
        self.pub_address = config.get('zmq', {}).get('pub_address', "tcp://0.0.0.0:5555")
        self.sub_address = config.get('zmq', {}).get('sub_address', "tcp://0.0.0.0:5556")
        
        self.context = zmq.Context()
        self.send_lock = threading.Lock()
        # --- 1. 初始化 Publisher (状态上报) ---
        self.publisher = self.context.socket(zmq.PUB)
        self.publisher.bind(self.pub_address)
        print(f"[ZMQ] 📡 Publisher bound to {self.pub_address}")
        
        # --- 2. 初始化 Subscriber (接收控制指令) ---
        self.subscriber = self.context.socket(zmq.SUB)
        self.subscriber.bind(self.sub_address)
        self.subscriber.setsockopt_string(zmq.SUBSCRIBE, "") 
        print(f"[ZMQ] 🎧 Subscriber bound to {self.sub_address}")
        
        self.running = False
        self.control_callback = None 
        self.sub_thread = None

    def start(self):
        self.running = True
        self.sub_thread = threading.Thread(target=self._receive_loop, daemon=True)
        self.sub_thread.start()

    def stop(self):
        self.running = False
        self.context.term()

    def publish(self, topic: str, data: dict):
        """高速发布状态数据 (统一改为三段式，透传 ROS 时间戳)"""
        try:
            # 1. 优先获取 ROS 传来的 timestamp，没有则回退到当前系统时间
            ros_ts = data.get("timestamp", time.time())
            
            # 2. 构造 Meta，标识这是一个 JSON 数据包
            meta = {
                "type": "json", 
                "timestamp": ros_ts,
                "is_image": False  # 显式告知接收端这不是图像
            }
            
            topic_b = topic.encode('utf-8')
            meta_b = json.dumps(meta).encode('utf-8')
            data_b = json.dumps(data).encode('utf-8')

            with self.send_lock:
                self.publisher.send_multipart([topic_b, meta_b, data_b])

            if self.debug:
                self._sampled_log(topic, f"Sending State [TS: {ros_ts}] -> {data}")

        except Exception as e:
            if self.debug: print(f"[ZMQ PUB Error] {e}")
    
    def publish_image(self, topic: str, image_bytes: bytes, metadata: dict):
        """发送二进制图像数据 (使用 ROS 传入的 metadata)"""
        try:
            # metadata 已经在 ROS2Connector 中包含了 ROS 的 timestamp
            ros_ts = metadata.get("timestamp", time.time())
            
            # 为了方便 Adapter 识别，确保 metadata 里有 is_image 标记
            metadata["is_image"] = True 
            
            topic_b = topic.encode('utf-8')
            meta_b = json.dumps(metadata).encode('utf-8')
            
            # 发送 3 段：[Topic, Meta, Bytes]
            with self.send_lock:
                self.publisher.send_multipart([topic_b, meta_b, image_bytes])
        
            if self.debug:
                log_msg = f"Sending Image [TS: {ros_ts}] | DataSize: {len(image_bytes)/1024:.2f} KB"
                self._sampled_log(topic, log_msg)

        except Exception as e:
            if self.debug: print(f"[ZMQ IMG Error] {e}")
    
    def register_control_callback(self, callback_func):
        self.control_callback = callback_func

    def _receive_loop(self):
        """接收 ZMQ 控制指令"""
        while self.running:
            try:
                events = self.subscriber.poll(100)
                if events:
                    msg = self.subscriber.recv_string()
                    payload = json.loads(msg)
                    
                    # --- 新增：控制指令 Log (指令通常低频，建议全量打印) ---
                    if self.debug:
                        print(f" \033[92m[ZMQ RECV]\033[0m Topic: {payload.get('device_key')} | Command: {payload.get('command')}")

                    device_key = payload.get("device_key")
                    command = payload.get("command")
                    
                    if device_key and command and self.control_callback:
                        self.control_callback(device_key, command)
            except Exception as e:
                if self.running:
                    print(f"[ZMQ] Error receiving control: {e}")
                time.sleep(0.02)

    # --- 工具方法：采样打印防止刷屏 ---
    def _sampled_log(self, topic: str, message: str):
        if topic not in self.pub_count:
            self.pub_count[topic] = 0
        self.pub_count[topic] += 1
        
        if self.pub_count[topic] % self.log_interval == 0:
            print(f"[ZMQ PUB Sampling][{topic}] {message}")