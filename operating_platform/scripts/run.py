import asyncio
from dataclasses import asdict, dataclass
from pprint import pformat
from typing import List

import cv2
import logging_mp
from lerobot.robots import RobotConfig
from lerobot.teleoperators import TeleoperatorConfig

from operating_platform.core.coordinator import Coordinator
from operating_platform.core.monitor import Monitor
from operating_platform.robots.daemon import Daemon

# from operating_platform.robot.robots.configs import RobotConfig
from operating_platform.teleoperators.utils import make_teleoperator_from_config
from operating_platform.utils import parser
from operating_platform.utils.constants import DEFAULT_FPS
from operating_platform.utils.import_utils import register_third_party_devices
from operating_platform.utils.utils import git_branch_log

# from lerobot.teleoperators import make_teleoperator_from_config

logging_mp.basic_config(level=logging_mp.INFO)
logger = logging_mp.get_logger(__name__)


@dataclass
class ControlPipelineConfig:
    robot: RobotConfig
    teleop: TeleoperatorConfig | None = None

    @classmethod
    def __get_path_fields__(cls) -> List[str]:
        """This enables the parser to load config from the policy using `--policy.path=local/dir`"""
        return ["control.policy"]


@parser.wrap()
async def async_main(cfg: ControlPipelineConfig):
    logger.info(pformat(asdict(cfg)))

    # robot = make_robot_from_config(cfg.robot)
    teleop = (
        make_teleoperator_from_config(cfg.teleop) if cfg.teleop is not None else None
    )
    if teleop is not None:
        teleop.connect()

    daemon = Daemon(fps=DEFAULT_FPS)
    daemon.start(cfg.robot)

    monitor = Monitor(daemon)
    monitor.start()

    coordinator = Coordinator(daemon, teleop)
    await coordinator.start()

    coordinator.stream_info(daemon.cameras_info)
    await coordinator.update_stream_info_to_server()

    try:
        while True:
            daemon.update()
            observation = daemon.get_observation()

            if teleop is not None:
                action = teleop.get_action()
                daemon.set_obs_action(action)
                daemon.set_pre_action(action)

            if observation is not None:
                tasks = []
                for key in observation:
                    if "image" in key and "depth" not in key:
                        img = cv2.cvtColor(observation[key], cv2.COLOR_RGB2BGR)
                        # name = key[len("observation.images."):]
                        tasks.append(coordinator.update_stream_async(key, img))
                        cv2.imshow(key, img)
                cv2.waitKey(1)
                if tasks:
                    try:
                        await asyncio.wait_for(
                            asyncio.gather(*tasks, return_exceptions=True), timeout=0.2
                        )
                    except asyncio.TimeoutError:
                        pass
            else:
                logger.warning("observation is none")
            await asyncio.sleep(0)
    except KeyboardInterrupt:
        logger.info("coordinator and daemon stop")
    finally:
        daemon.stop()
        await coordinator.stop()


def main():
    git_branch_log()

    register_third_party_devices()
    logger.info(f"Registered robot types: {list(RobotConfig._choice_registry.keys())}")
    logger.info(
        f"Registered teleoperator types: {list(TeleoperatorConfig._choice_registry.keys())}"
    )

    asyncio.run(async_main())


if __name__ == "__main__":
    main()
