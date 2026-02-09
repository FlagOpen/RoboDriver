from .config import LEJUKuavoRos1Config
from .robot import LEJUKuavoRos1Robot
from .status import LEJUKuavoRos1RobotStatus

# 为 make_device_from_device_class 添加兼容性别名
LEJUKuavoRos1 = LEJUKuavoRos1Robot
