import logging_mp
from pprint import pformat

from operating_platform.robot.robots.so101_v1 import SO101RobotStatus

logging_mp.basic_config(level=logging_mp.INFO)
logger = logging_mp.get_logger(__name__)

status = SO101RobotStatus()
logger.info(pformat(status.to_dict(), width=120, depth=6))
logger.info(pformat(status.to_json()))
# print(pformat(status.to_dict()))