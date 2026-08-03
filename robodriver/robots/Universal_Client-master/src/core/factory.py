from src.utils.config_loader import ConfigLoader
from src.core.zmq_hub import ZmqHub
from src.plugins import std_ros_handlers, lbot_handlers

def create_connector(config_path="config/settings.yaml"):
    config = ConfigLoader(config_path).load()
    middleware = config.get('app', {}).get('middleware', 'ros1').lower()
    
    # 初始化纯 Python 的 ZMQ Hub
    zmq_hub = ZmqHub(config)
    
    if middleware == 'ros1':
        print("🔧 Initializing ROS 1 Backend...")
        from src.core.connector_ros1 import ROS1Connector
        connector = ROS1Connector(config, zmq_hub)
        
    elif middleware == 'ros2':
        print("🔧 Initializing ROS 2 Backend...")
        from src.core.connector_ros2 import ROS2Connector
        connector = ROS2Connector(config, zmq_hub)
        
    elif middleware == 'dora':
        print("🔧 Initializing Dora-RS Backend...")
        raise NotImplementedError("dora-rs support coming soon")
    else:
        raise ValueError(f"Unknown middleware: {middleware}")
        

    std_ros_handlers.register(connector)
    lbot_handlers.register(connector)
        
    return connector