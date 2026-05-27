import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
import threading
import time
from src.core.base_connector import BaseRobotConnector

class ROS2Connector(BaseRobotConnector, Node):
    def __init__(self, config: dict, zmq_hub):
        # 1. 初始化 ROS 2 基础环境
        if not rclpy.ok():
            rclpy.init()
            
        # 2. 初始化父类
        BaseRobotConnector.__init__(self, config, zmq_hub)
        Node.__init__(self, 'linkerbot_ros2_hub')
        
        # 3. 并发配置：使用重入回调组
        self.callback_group = ReentrantCallbackGroup()
        self.ros_executor = MultiThreadedExecutor(num_threads=8)
        self.ros_executor.add_node(self)
        
        self.subscribers_dict = {}
        self.publishers_dict = {}
        
        # 4. 注册 ZMQ 回调
        if self.zmq_hub:
            self.zmq_hub.register_control_callback(self.send_control)
        
 
        
        # 5. 定义旋转线程
        self.spin_thread = threading.Thread(target=self._run_executor, daemon=True)

    def _run_executor(self):
        try:
            self.ros_executor.spin()
        except Exception as e:
            if rclpy.ok():
                self.get_logger().error(f"ROS 2 Executor Error: {e}")

    def start(self):
        self._auto_wire()   
        
        if self.zmq_hub:
            self.zmq_hub.start()
        self.get_logger().info("🚀 Starting ROS 2 Connector Spin Thread...")
        if not self.spin_thread.is_alive():
            self.spin_thread.start()

    def stop(self):
        self.get_logger().info("🛑 Stopping ROS 2 Connector...")
        if self.zmq_hub:
            self.zmq_hub.stop()

        self.ros_executor.shutdown()
        self.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

    def _auto_wire(self):
        """全新的自动挂载逻辑，区分 sensors 和 actuators"""
        devices = self.config.get('devices', {})
        
        # 1. 挂载 Sensors (Subscribers)
        sensors = devices.get('sensors', {})
        for key, meta in sensors.items():
            msg_class_str = meta['msg_class']
            msg_cls = self.load_msg_class(msg_class_str)
            sub_topic = meta['topic']
            
            # 使用通用的回调闭包
            cb = lambda msg, k=key, t=msg_class_str, m=meta: self._generic_sub_cb(msg, k, t, m)
            self.subscribers_dict[key] = self.create_subscription(msg_cls, sub_topic, cb, 10, callback_group=self.callback_group)
            self.get_logger().info(f"[AutoWire] SUB: {key} -> {sub_topic} [{msg_class_str}]")

        # 2. 挂载 Actuators (Publishers)
        actuators = devices.get('actuators', {})
        for key, meta in actuators.items():
            msg_class_str = meta['msg_class']
            msg_cls = self.load_msg_class(msg_class_str)
            pub_topic = meta['topic']
            
            self.publishers_dict[key] = {
                'pub': self.create_publisher(msg_cls, pub_topic, 10, callback_group=self.callback_group),
                'msg_class_str': msg_class_str,
                'msg_cls': msg_cls,
                'meta': meta
            }
            self.get_logger().info(f"[AutoWire] PUB: {key} -> {pub_topic} [{msg_class_str}]")

    def _generic_sub_cb(self, msg, key, msg_class_str, meta):
        """通用的数据接收回调：查表找 Decode Handler"""
        
        # 特殊处理图像，保证零拷贝性能
        if msg_class_str == "sensor_msgs.msg.Image":
            ts = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9 if hasattr(msg, 'header') else time.time()
            metadata = {
                "width": msg.width, "height": msg.height, "step": msg.step,
                "encoding": msg.encoding, "timestamp": ts, "format": "raw"
            }
            self._update_and_publish_image(key, bytes(msg.data), metadata)
            return

        handler = self._handlers.get(msg_class_str, {}).get('decode')
        if handler:
            # 将 msg 和 yaml 里的 meta (比如 format="hand") 一起传给解码器
            data = handler(msg, meta)
            # ROS2 时间戳透传
            if hasattr(msg, 'header'):
                data["timestamp"] = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            else:
                data["timestamp"] = time.time()
                
            self._update_and_publish(key, data)
        else:
            self.get_logger().warn(f"No decode handler registered for {msg_class_str}", throttle_duration_sec=5.0)

    def send_control(self, device_key: str, command_data: dict):
        """通用的控制下发逻辑：查表找 Encode Handler"""
        if device_key not in self.publishers_dict:
            # 静默忽略，因为 ZMQ 可能会收到不是给 ROS 的指令
            return False
            
        pub_info = self.publishers_dict[device_key]
        msg_class_str = pub_info['msg_class_str']
        msg_cls = pub_info['msg_cls']
        publisher = pub_info['pub']
        meta = pub_info['meta']
        
        handler = self._handlers.get(msg_class_str, {}).get('encode')
        if handler:
            # 将原始指令、消息类和配置元数据传给编码器
            msg = handler(msg_cls, command_data, meta)
            
            # 自动注入时间戳 (如果自定义包有 header 的话)
            if hasattr(msg, 'header'):
                msg.header.stamp = self.get_clock().now().to_msg()
                if not hasattr(msg.header, 'frame_id') or not msg.header.frame_id:
                    msg.header.frame_id = "base_link"
                    
            publisher.publish(msg)
            return True
        else:
            self.get_logger().error(f"No encode handler registered for {msg_class_str}")
            return False