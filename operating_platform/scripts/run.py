import cv2
import asyncio
import logging_mp


from pprint import pformat
from typing import Dict, List
from dataclasses import dataclass, asdict

from operating_platform.core.coordinator import Coordinator
from operating_platform.core.monitor import Monitor
from operating_platform.robot.daemon import Daemon
from operating_platform.robot.robots.configs import RobotConfig

from operating_platform.utils import parser
from operating_platform.utils.utils import git_branch_log
from operating_platform.utils.constants import DEFAULT_FPS


logging_mp.basic_config(level=logging_mp.INFO)
logger = logging_mp.get_logger(__name__)


@dataclass
class ControlPipelineConfig:
    robot: RobotConfig

    @classmethod
    def __get_path_fields__(cls) -> List[str]:
        """This enables the parser to load config from the policy using `--policy.path=local/dir`"""
        return ["control.policy"]


async def async_main(cfg: ControlPipelineConfig):
    git_branch_log()
    logger.info(f"Registered robot types: {list(RobotConfig._choice_registry.keys())}")
    logger.info(pformat(asdict(cfg)))

    daemon = Daemon(fps=DEFAULT_FPS)
    daemon.start(cfg.robot)

    monitor = Monitor(daemon)
    monitor.start()

    coordinator = Coordinator(daemon)
    await coordinator.start()

    coordinator.stream_info(daemon.cameras_info)
    await coordinator.update_stream_info_to_server()

    try:
        while True:
            daemon.update()
            observation = daemon.get_observation()
            if observation is not None:
                tasks = []
                for key in observation:
                    if "image" in key and "depth" not in key:
                        img = cv2.cvtColor(observation[key].numpy(), cv2.COLOR_RGB2BGR)
                        name = key[len("observation.images."):]
                        tasks.append(
                            coordinator.update_stream_async(name, img)
                        )
                if tasks:
                    try:
                        await asyncio.wait_for(
                            asyncio.gather(*tasks, return_exceptions=True),
                            timeout=0.2
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


@parser.wrap()
def main(cfg: ControlPipelineConfig):
    asyncio.run(async_main(cfg))


if __name__ == "__main__":
    main()
