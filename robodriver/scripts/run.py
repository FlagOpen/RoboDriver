import asyncio
from dataclasses import asdict, dataclass
from pprint import pformat
from typing import List

import cv2
import logging_mp
from lerobot.robots import RobotConfig
from lerobot.teleoperators import TeleoperatorConfig

from robodriver.core.coordinator import Coordinator
from robodriver.core.monitor import Monitor
from robodriver.core.simulator import SimulatorConfig
from robodriver.core.simulator import Simulator
from robodriver.robots.daemon import Daemon

# from operating_platform.robot.robots.configs import RobotConfig
from robodriver.teleoperators.utils import make_teleoperator_from_config
from robodriver.utils import parser
from robodriver.utils.constants import DEFAULT_FPS
from robodriver.utils.import_utils import register_third_party_devices
from robodriver.utils.utils import git_branch_log

# from lerobot.teleoperators import make_teleoperator_from_config

logging_mp.basic_config(level=logging_mp.INFO)
logger = logging_mp.get_logger(__name__)


@dataclass
class ControlPipelineConfig:
    robot: RobotConfig
    teleop: TeleoperatorConfig | None = None
    sim: SimulatorConfig | None = None

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
    sim = Simulator(backend=cfg.sim.backend, show_viewer=cfg.sim.show_viewer, arm_config=cfg.sim.arm_config, urdf_path = cfg.sim.urdf_path, mjcf_path=cfg.sim.mjcf_path) if cfg.sim is not None and (cfg.sim.arm_config is not None or cfg.sim.urdf_path is not None or cfg.sim.mjcf_path is not None) else None
    observation_sim = None
    if teleop is not None:
        teleop.connect()

    daemon = Daemon(fps=DEFAULT_FPS)
    daemon.start(cfg.robot)

    monitor = Monitor(daemon)
    monitor.start()

    coordinator = Coordinator(daemon, teleop)
    await coordinator.start()

    coordinator.stream_info(daemon.cameras_info)
    if sim is not None:
        coordinator.stream_info_add("image_sim", 21)
    await coordinator.update_stream_info_to_server()

    try:
        while True:
            daemon.update()
            observation = daemon.get_observation()

            if teleop is not None:
                action = teleop.get_action()
                daemon.set_obs_action(action)
                daemon.set_pre_action(action)
            else:
                action = daemon.robot.get_action()
                daemon.set_obs_action(action)

            if sim is not None:
                observation_sim = sim.update(action, prefix="leader_", suffix=".pos")

            tasks = []
            if observation is not None:
                for key in observation:
                    if "image" in key and "depth" not in key:
                        img = cv2.cvtColor(observation[key], cv2.COLOR_RGB2BGR)
                        # name = key[len("observation.images."):]
                        tasks.append(coordinator.update_stream_async(key, img))
                        # cv2.imshow(key, img)
            
            if observation_sim is not None:
                observation_sim = cv2.cvtColor(observation_sim, cv2.COLOR_RGB2BGR)
                tasks.append(coordinator.update_stream_async("image_sim", observation_sim))
                # cv2.imshow("image_sim", observation_sim)
            
            if tasks:
                try:
                    await asyncio.wait_for(
                        asyncio.gather(*tasks, return_exceptions=True), timeout=0.2
                    )
                except asyncio.TimeoutError:
                    pass

            
            else:
                logger.warning("observation is none")
            
            # cv2.waitKey(1)
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
