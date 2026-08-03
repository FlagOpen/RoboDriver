import os
import pytest
from src.utils.config_loader import ConfigLoader
from src.core.factory import create_connector

def test_config_loader():
    """测试配置加载器是否能正确读取 YAML"""
    # 确保配置文件存在
    assert os.path.exists("config/settings.yaml"), "配置文件丢失"
    
    loader = ConfigLoader("config/settings.yaml")
    data = loader.load()
    
    # 验证关键配置项
    assert "app" in data
    assert data["app"]["name"] == "LinkerBot-ZMQ-Hub"
    assert data["app"]["middleware"] in ["ros1", "ros2"]
    
    assert "zmq" in data
    assert "pub_address" in data["zmq"]
    
    assert "devices" in data
    assert "leader_left_joints" in data["devices"]

def test_config_loader_file_not_found():
    """测试读取不存在的文件是否抛出异常"""
    loader = ConfigLoader("config/fake_settings.yaml")
    with pytest.raises(FileNotFoundError):
        loader.load()

# 由于 GitHub CI 上没有安装 ROS 环境，直接 create_connector 会报错找不到 rclpy/rospy
# 我们可以在这里测试一些纯 Python 的类，比如 ZmqHub 的初始化
def test_zmq_hub_init():
    """测试 ZMQ Hub 是否能正常初始化 (纯 Python)"""
    from src.core.zmq_hub import ZmqHub
    mock_config = {
        "zmq": {
            "pub_address": "tcp://127.0.0.1:15555",
            "sub_address": "tcp://127.0.0.1:15556",
            "debug": False
        }
    }
    hub = ZmqHub(mock_config)
    assert hub.pub_address == "tcp://127.0.0.1:15555"
    assert hub.running is False
    
    # 验证方法存在
    assert hasattr(hub, "publish")
    assert hasattr(hub, "publish_image")