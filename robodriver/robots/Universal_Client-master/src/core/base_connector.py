from abc import ABC, abstractmethod
import importlib

class BaseRobotConnector(ABC):
    def __init__(self, config: dict, zmq_hub):
        self.config = config
        self.zmq_hub = zmq_hub
        self.state = {}
        self.images = {}
        
        # --- 新增：动态加载缓存与处理器注册表 ---
        self._msg_classes = {}
        self._handlers = {}  # 格式: { "pkg.msg.Type": {"decode": func, "encode": func} }

    def load_msg_class(self, msg_class_str: str):
        """动态反射加载 ROS 消息类"""
        if msg_class_str not in self._msg_classes:
            module_path, class_name = msg_class_str.rsplit('.', 1)
            module = importlib.import_module(module_path)
            self._msg_classes[msg_class_str] = getattr(module, class_name)
        return self._msg_classes[msg_class_str]

    def register_handler(self, msg_class_str: str, decode_fn=None, encode_fn=None):
        """
        向外暴露的插件注册接口
        :param msg_class_str: 例如 "sensor_msgs.msg.JointState"
        :param decode_fn: 用于解析订阅话题数据的函数
        :param encode_fn: 用于封装发布话题数据的函数
        """
        if msg_class_str not in self._handlers:
            self._handlers[msg_class_str] = {}
        if decode_fn: 
            self._handlers[msg_class_str]['decode'] = decode_fn
        if encode_fn: 
            self._handlers[msg_class_str]['encode'] = encode_fn

    @abstractmethod
    def start(self):
        """启动底层的节点/线程"""
        pass

    @abstractmethod
    def stop(self):
        """优雅关闭节点"""
        pass

    @abstractmethod
    def send_control(self, device_key: str, command_data: dict):
        """通用发送控制指令接口"""
        pass

    def get_state(self, device_key: str = None):
        """获取当前缓存的状态"""
        if device_key is None:
            return self.state.copy()
        return self.state.get(device_key, None)

    def get_image(self, device_key: str):
        """获取指定相机的最新 Base64/Raw 图像"""
        return self.images.get(device_key, None)

    def _update_and_publish(self, key: str, data: dict):
        """内部方法：更新状态并触发 ZMQ 发送"""
        self.state[key] = data
        if self.zmq_hub:
            self.zmq_hub.publish(key, data)
            
    def _update_and_publish_image(self, key: str, image_bytes: bytes, metadata: dict):
        """内部方法：更新缓存并触发 ZMQ 发送"""
        self.images[key] = {
            "metadata": metadata,
            "data": image_bytes
        }
        if self.zmq_hub:
            self.zmq_hub.publish_image(key, image_bytes, metadata)