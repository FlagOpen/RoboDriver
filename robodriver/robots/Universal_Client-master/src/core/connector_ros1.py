import rospy
import time
from src.core.base_connector import BaseRobotConnector

class ROS1Connector(BaseRobotConnector):
    def __init__(self, config: dict, zmq_hub):
        super().__init__(config, zmq_hub)
        
        if not rospy.core.is_initialized():
            rospy.init_node('linkerbot_ros1_hub', disable_signals=True)
            
        self.subscribers_dict = {}
        self.publishers_dict = {}
        
        if self.zmq_hub:
            self.zmq_hub.register_control_callback(self.send_control)
            

    def start(self):
        rospy.loginfo("🚀 Starting ROS 1 Connector...")
        self._auto_wire()

        if self.zmq_hub:
            self.zmq_hub.start()

    def stop(self):
        rospy.loginfo("🛑 Stopping ROS 1 Connector...")
        if self.zmq_hub:
            self.zmq_hub.stop()
        for sub in self.subscribers_dict.values():
            sub.unregister()
        for pub in self.publishers_dict.values():
            pub['pub'].unregister()
        rospy.signal_shutdown("System closing")

    def _auto_wire(self):
        devices = self.config.get('devices', {})
        
        # 1. 挂载 Sensors (Subscribers)
        sensors = devices.get('sensors', {})
        for key, meta in sensors.items():
            msg_class_str = meta['msg_class']
            msg_cls = self.load_msg_class(msg_class_str)
            sub_topic = meta['topic']
            
            # 图像需要大 buffer
            queue_size = 1
            buff_size = 65536 * 100 if msg_class_str == "sensor_msgs.msg.Image" else 65536
            
            cb = lambda msg, k=key, t=msg_class_str, m=meta: self._generic_sub_cb(msg, k, t, m)
            self.subscribers_dict[key] = rospy.Subscriber(sub_topic, msg_cls, cb, queue_size=queue_size, buff_size=buff_size)
            rospy.loginfo(f"[AutoWire] SUB: {key} -> {sub_topic} [{msg_class_str}]")

        # 2. 挂载 Actuators (Publishers)
        actuators = devices.get('actuators', {})
        for key, meta in actuators.items():
            msg_class_str = meta['msg_class']
            msg_cls = self.load_msg_class(msg_class_str)
            pub_topic = meta['topic']
            
            self.publishers_dict[key] = {
                'pub': rospy.Publisher(pub_topic, msg_cls, queue_size=10),
                'msg_class_str': msg_class_str,
                'msg_cls': msg_cls,
                'meta': meta
            }
            rospy.loginfo(f"[AutoWire] PUB: {key} -> {pub_topic} [{msg_class_str}]")

    def _generic_sub_cb(self, msg, key, msg_class_str, meta):
        # 图像零拷贝特判
        if msg_class_str == "sensor_msgs.msg.Image":
            ts = msg.header.stamp.to_sec() if hasattr(msg, 'header') else time.time()
            metadata = {
                "width": msg.width, "height": msg.height, "step": msg.step,
                "encoding": msg.encoding, "timestamp": ts, "format": "raw"
            }
            self._update_and_publish_image(key, msg.data, metadata)
            return

        handler = self._handlers.get(msg_class_str, {}).get('decode')
        if handler:
            data = handler(msg, meta)
            data["timestamp"] = msg.header.stamp.to_sec() if hasattr(msg, 'header') else time.time()
            self._update_and_publish(key, data)
        else:
            rospy.logwarn(f"No decode handler registered for {msg_class_str}")

    def send_control(self, device_key: str, command_data: dict):
        if device_key not in self.publishers_dict:
            return False
            
        pub_info = self.publishers_dict[device_key]
        msg_class_str = pub_info['msg_class_str']
        msg_cls = pub_info['msg_cls']
        publisher = pub_info['pub']
        meta = pub_info['meta']
        
        handler = self._handlers.get(msg_class_str, {}).get('encode')
        if handler:
            msg = handler(msg_cls, command_data, meta)
            if hasattr(msg, 'header'):
                msg.header.stamp = rospy.Time.now()
                if not hasattr(msg.header, 'frame_id') or not msg.header.frame_id:
                    msg.header.frame_id = "base_link"
            publisher.publish(msg)
            return True
        else:
            rospy.logerr(f"No encode handler registered for {msg_class_str}")
            return False