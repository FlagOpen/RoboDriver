from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import draccus
import logging_mp

from operating_platform.dataset.dorobot_dataset import DoRobotDataset
from operating_platform.robot.robots.utils import Robot, busy_wait
from operating_platform.utils.utils import log_say

logging_mp.basic_config(level=logging_mp.INFO)
logger = logging_mp.get_logger(__name__)


@dataclass
class DatasetReplayConfig:
    # Dataset identifier. By convention it should match '{hf_username}/{dataset_name}' (e.g. `lerobot/test`).
    repo_id: str
    # Episode to replay.
    episode: int
    # Root directory where the dataset will be stored (e.g. 'dataset/path').
    root: str | Path | None = None
    # Limit the frames per second. By default, uses the policy fps.
    fps: int = 30


@dataclass
class ReplayConfig:
    robot: Robot
    dataset: DatasetReplayConfig
    # Use vocal synthesis to read events.
    play_sounds: bool = False


@draccus.wrap()
def replay(cfg: ReplayConfig):
    # logging.info(pformat(asdict(cfg)))

    # robot = make_robot_from_config(cfg.robot)
    dataset = DoRobotDataset(
        cfg.dataset.repo_id, root=cfg.dataset.root, episodes=[cfg.dataset.episode]
    )
    actions = dataset.hf_dataset.select_columns("action")
    # robot.connect()
    robot = cfg.robot
    log_say("Replaying episode", cfg.play_sounds, blocking=True)
    for idx in range(dataset.num_frames):
        start_episode_t = time.perf_counter()

        action_array = actions[idx]["action"]
        action = {}
        for i, name in enumerate(dataset.features["action"]["names"]):
            action[name] = action_array[i]

        #     # 将 action 转换为 torch.Tensor
        #     try:
        #         action_tensor = torch.tensor([action[name].item() for name in dataset.features["action"]["names"] if name in action])
        #     except KeyError as e:
        #         print(f"KeyError: {e} not found in action")
        #         continue
        # robot.send_action(action_tensor)

        print(f"Replay action: {action}")

        robot.send_action(action)

        dt_s = time.perf_counter() - start_episode_t
        busy_wait(1 / dataset.fps - dt_s)

    # robot.disconnect()


def main():
    replay()


if __name__ == "__main__":
    main()
