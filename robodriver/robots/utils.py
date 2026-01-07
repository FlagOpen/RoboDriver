import platform
import time
from typing import cast

import logging_mp
from lerobot.robots import Robot, RobotConfig

from robodriver.robots.statuses import RobotStatus
from robodriver.utils.import_utils import make_device_from_device_class

logger = logging_mp.get_logger(__name__)


def busy_wait(seconds):
    if platform.system() == "Darwin":
        # On Mac, `time.sleep` is not accurate and we need to use this while loop trick,
        # but it consumes CPU cycles.
        end_time = time.perf_counter() + seconds
        while time.perf_counter() < end_time:
            pass
    else:
        # On Linux time.sleep is accurate
        if seconds > 0:
            time.sleep(seconds)

def safe_disconnect(func):
    def wrapper(robot, *args, **kwargs):
        try:
            return func(robot, *args, **kwargs)
        except Exception as e:
            if robot.is_connected:
                robot.disconnect()
            raise e

    return wrapper

def make_robot_from_config(config: RobotConfig) -> Robot:
    logger.info("In make_robot_from_config")
    logger.info(f"make robot type: {config.type}")

    try:
        return cast(Robot, make_device_from_device_class(config))
    except Exception as e:
        logger.critical(f"Can't create robot with config {config}")
        raise ValueError(f"Error creating robot with config {config}: {e}") from e

def safe_update_status(robot: Robot) -> str:
    if hasattr(robot, "update_status"):
        return robot.update_status()
    else:
        return RobotStatus().to_json()
