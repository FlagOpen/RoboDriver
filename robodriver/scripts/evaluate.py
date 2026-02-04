import asyncio
from dataclasses import asdict, dataclass
from pprint import pformat
from typing import List

import time
import cv2
import logging_mp

from lerobot.robots import RobotConfig

from robodriver.robots.daemon import Daemon
from robodriver.utils.utils import git_branch_log
from robodriver.utils.import_utils import register_third_party_devices
from robodriver.utils import parser
from robodriver.utils.constants import DEFAULT_FPS
from robodriver.robots.utils import (
    Robot,
    busy_wait,
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
    robot = make_robot_from_config(cfg.robot)
    robot.connect()

    while True:
        start_loop_t = time.perf_counter()
        robot.get_observation()

        robot.send_action()
        dt_s = time.perf_counter() - start_loop_t

        busy_wait(1 / 30 - dt_s)


def main():
    git_branch_log()

    register_third_party_devices()
    logger.info(f"Registered robot types: {list(RobotConfig._choice_registry.keys())}")

    asyncio.run(async_main())


if __name__ == "__main__":
    main()
