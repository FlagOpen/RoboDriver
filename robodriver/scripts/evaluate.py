import asyncio
from dataclasses import asdict, dataclass
from pprint import pformat
from typing import List
import logging_mp
from lerobot.robots import RobotConfig

from robodriver.utils import parser
from robodriver.utils.import_utils import register_third_party_devices
from robodriver.utils.utils import git_branch_log

from robodriver.robots.utils import (
    make_robot_from_config,
)


logging_mp.basic_config(level=logging_mp.INFO)
logger = logging_mp.get_logger(__name__)


@dataclass
class ControlPipelineConfig:
    robot: RobotConfig
    @classmethod
    def __get_path_fields__(cls) -> List[str]:
        """This enables the parser to load config from the policy using `--policy.path=local/dir`"""
        return ["control.policy"]


@parser.wrap()
async def async_main(cfg: ControlPipelineConfig):
    logger.info(pformat(asdict(cfg)))


    try:
        robot = make_robot_from_config(cfg.robot)
    except Exception as e:
        logger.critical(f"Failed to create robot: {e}")
        raise

    logger.info("Make robot success")
    logger.info(f"robot.type: {robot.robot_type}")

    if not robot.is_connected:
        robot.connect()
    logger.info("Connect robot success")
    
    # 在下面实现推理代码
    observation = robot.get_observation() # 从臂和图像信息
    action = None
    if action:
        robot.send_action(action)

def main():
    git_branch_log()

    register_third_party_devices()
    logger.info(f"Registered robot types: {list(RobotConfig._choice_registry.keys())}")

    asyncio.run(async_main())


if __name__ == "__main__":
    main()

# 启动命令：python -m robodriver.scripts.evaluate --robot.type=galaxealite-aio-ros2