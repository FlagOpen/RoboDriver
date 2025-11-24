from __future__ import annotations

from dataclasses import dataclass

import logging_mp
from deepdiff import DeepDiff

logger = logging_mp.get_logger(__name__)


from operating_platform.core.recorder import RecordConfig
from operating_platform.dataset.dorobot_dataset import *
from operating_platform.robot.robots.configs import RobotConfig
from operating_platform.utils import parser
from operating_platform.utils.utils import git_branch_log


def sanity_check_dataset_robot_compatibility(
    dataset: DoRobotDataset, robot: RobotConfig, fps: int, use_videos: bool
) -> None:
    fields = [
        ("robot_type", dataset.meta.robot_type, robot.robot_type),
        ("fps", dataset.fps, fps),
        # ("features", dataset.features, get_features_from_robot(robot, use_videos)),
    ]

    mismatches = []
    for field, dataset_value, present_value in fields:
        diff = DeepDiff(
            dataset_value, present_value, exclude_regex_paths=[r".*\['info'\]$"]
        )
        if diff:
            mismatches.append(f"{field}: expected {present_value}, got {dataset_value}")

    if mismatches:
        raise ValueError(
            "Dataset metadata compatibility check failed with mismatches:\n"
            + "\n".join(mismatches)
        )


@dataclass
class ControlPipelineConfig:
    robot: RobotConfig
    record: RecordConfig

    @classmethod
    def __get_path_fields__(cls) -> list[str]:
        """This enables the parser to load config from the policy using `--policy.path=local/dir`"""
        return ["control.policy"]


@parser.wrap()
def record(cfg: ControlPipelineConfig):
    logging_mp.basic_config(level=logging_mp.INFO)
    git_branch_log()

    # daemon = Daemon(fps=DEFAULT_FPS)
    # daemon.start(cfg.robot)

    # robot_daemon = Record(cfg.fps,cfg.)

    # robot_daemon.start()


def main():
    record()


if __name__ == "__main__":
    main()
