
import cv2
import json
import time
import draccus
import socketio
import requests
import traceback
import threading
import queue

from dataclasses import dataclass, asdict
from pathlib import Path
from pprint import pformat
from deepdiff import DeepDiff
from functools import cache
from termcolor import colored
from datetime import datetime


# from operating_platform.policy.config import PreTrainedConfig
from operating_platform.robot.robots.configs import RobotConfig
from operating_platform.robot.robots.utils import make_robot_from_config, Robot, busy_wait, safe_disconnect
from operating_platform.utils import parser
from operating_platform.utils.utils import has_method, init_logging, log_say, get_current_git_branch, git_branch_log, get_container_ip_from_hosts
from operating_platform.utils.data_file import find_epindex_from_dataid_json

from operating_platform.utils.constants import DOROBOT_DATASET
from operating_platform.dataset.dorobot_dataset import *
from operating_platform.dataset.visual.visual_dataset import visualize_dataset

# from operating_platform.core._client import Coordinator
from operating_platform.robot.daemon import Daemon
from operating_platform.core.recorder import Record, RecordConfig
from operating_platform.core.replayer import DatasetReplayConfig, ReplayConfig, replay

DEFAULT_FPS = 30

@cache
def is_headless():
    """Detects if python is running without a monitor."""
    try:
        import pynput  # noqa

        return False
    except Exception:
        print(
            "Error trying to import pynput. Switching to headless mode. "
            "As a result, the video stream from the cameras won't be shown, "
            "and you won't be able to change the control flow with keyboards. "
            "For more info, see traceback below.\n"
        )
        traceback.print_exc()
        print()
        return True


@dataclass
class ControlPipelineConfig:
    robot: RobotConfig
    record: RecordConfig
    # control: ControlConfig

    @classmethod
    def __get_path_fields__(cls) -> list[str]:
        """This enables the parser to load config from the policy using `--policy.path=local/dir`"""
        return ["control.policy"]
    

@parser.wrap()
def main(cfg: ControlPipelineConfig):

    init_logging()
    git_branch_log()
    logging.info(pformat(asdict(cfg)))

    daemon = Daemon(fps=DEFAULT_FPS)
    daemon.start(cfg.robot)
    daemon.update()


    # coordinator = Coordinator(daemon)
    # coordinator.start()

    # coordinator.stream_info(daemon.cameras_info)
    # coordinator.update_stream_info_to_server()


    repo_id = cfg.record.repo_id
    date_str = datetime.now().strftime("%Y%m%d")

    # 构建目标目录路径
    dataset_path = DOROBOT_DATASET

    git_branch_name = get_current_git_branch()
    if "release" in git_branch_name:
        target_dir = dataset_path / date_str / "user" / repo_id
    elif "dev"  in git_branch_name:
        target_dir = dataset_path / date_str / "dev" / repo_id
    else:
        target_dir = dataset_path / date_str / "dev" / repo_id


    # 判断是否存在对应文件夹以决定是否启用恢复模式
    resume = False

    # 检查数据集目录是否存在
    if not dataset_path.exists():
        logging.info(f"Dataset directory '{dataset_path}' does not exist. Cannot resume.")
    else:
        # 检查目标文件夹是否存在且为目录
        if target_dir.exists() and target_dir.is_dir():
            resume = True
            logging.info(f"Found existing directory for repo_id '{repo_id}'. Resuming operation.")
        else:
            logging.info(f"No directory found for repo_id '{repo_id}'. Starting fresh.")

    # resume 变量现在可用于后续逻辑
    print(f"Resume mode: {'Enabled' if resume else 'Disabled'}")

    msg = {
        "task_id": "001",
        "task_name": cfg.record.repo_id,
        "task_data_id": "001",
        "collector_id":"001",
        "machine_id":"SO101-001",
        "countdown_seconds": 3,
        "task_steps": [
            {
                "doruation": "10",
                "instruction": "put"
            },
            {
                "doruation": "2",
                "instruction": "close"
            },
            {
                "doruation": "5",
                "instruction": "clean"
            }
        ]
    }

    record_cfg = RecordConfig(fps=cfg.record.fps, repo_id=repo_id, single_task=cfg.record.single_task, video=daemon.robot.use_videos, resume=resume, root=target_dir)
    record = Record(fps=cfg.record.fps, robot=daemon.robot, daemon=daemon, record_cfg = record_cfg, record_cmd=msg)
            
    record.start()

    try:
        while True:
            daemon.update()
            observation = daemon.get_observation()
            # print("get observation")
            if observation is not None:
                image_keys = [key for key in observation if "image" in key]
                for i, key in enumerate(image_keys, start=1):
                    img = cv2.cvtColor(observation[key].numpy(), cv2.COLOR_RGB2BGR) 

                    name = key[len("observation.images."):]
                    # coordinator.update_stream(name, img)

                    if not is_headless():
                        # print(f"will show image, name:{name}")
                        cv2.imshow(name, img)
                        cv2.waitKey(1)
                        # print("show image succese")
                    
            else:
                print("observation is none")
            
    except KeyboardInterrupt:
        print("coordinator and daemon stop")

    finally:
        record.stop()
        record.save()
        daemon.stop()
        # coordinator.stop()
        cv2.destroyAllWindows()
    

if __name__ == "__main__":
    main()
