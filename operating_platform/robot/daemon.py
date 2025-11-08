import time
import torch
import threading
import logging_mp

from typing import Any, Union, Dict, Optional
from termcolor import colored

from operating_platform.robot.robots.configs import RobotConfig
from operating_platform.robot.robots.utils import make_robot_from_config, Robot, busy_wait, safe_update_status

logger = logging_mp.get_logger(__name__)


def log_control_info(robot: Robot, dt_s, episode_index=None, frame_index=None, fps=None):
    log_items = []
    if episode_index is not None:
        log_items.append(f"ep:{episode_index}")
    if frame_index is not None:
        log_items.append(f"frame:{frame_index}")

    def log_dt(shortname, dt_val_s):
        nonlocal log_items, fps
        info_str = f"{shortname}:{dt_val_s * 1000:5.2f} ({1 / dt_val_s:3.1f}hz)"
        if fps is not None:
            actual_fps = 1 / dt_val_s
            if actual_fps < fps - 1:
                info_str = colored(info_str, "yellow")
        log_items.append(info_str)

    # total step time displayed in milliseconds and its frequency
    log_dt("dt", dt_s)

    if not robot.robot_type.startswith("stretch"):
        # for name in robot.leader_arms:
        #     key = f"read_leader_{name}_pos_dt_s"
        #     if key in robot.logs:
        #         log_dt(f"dt_R_leader_{name}", robot.logs[key])
        if hasattr(robot, 'follower_arms') and robot.follower_arms is not None:
            for name in robot.follower_arms:
                key = f"read_follower_{name}_pos_dt_s"
                if key in robot.logs:
                    log_dt(f"dt_R_foll_{name}", robot.logs[key])
        
        for name in robot.cameras:
            key = f"read_camera_{name}_dt_s"
            if key in robot.logs:
                log_dt(f"dt_R_camera_{name}", robot.logs[key])

    info_str = " ".join(log_items)
    logger.debug(info_str)


class Daemon:
    def __init__(self, fps: int | None = None):
        self.fps = fps

        self.running = True

        self.data_lock = threading.Lock()
        self.pre_action: Union[Any, Dict[str, torch.Tensor]] = None
        self.obs_action: Union[Any, Dict[str, torch.Tensor]] = None
        self.observation: Union[Any, Dict[str, torch.Tensor]] = None
        self.status: Optional[str] = None

        self.robot = None

    @property
    def cameras_info(self):
        cameras = {}
        for name, camera in self.robot.cameras.items():
            cameras[name] = camera.camera_index
        return cameras

    def start(self, config: RobotConfig):
        try:
            self.robot = make_robot_from_config(config)
        except Exception as e:
            logger.critical(f"Failed to create robot: {e}")
            raise

        logger.info("Make robot success")
        logger.info(f"robot.type: {self.robot.robot_type}")

        if not self.robot.is_connected:
            self.robot.connect()
        logger.info("Connect robot success")
    
    def stop(self):
        pass

    def update(self):
        start_loop_t = time.perf_counter()

        observation, action = self.robot.teleop_step(record_data=True)
        status = safe_update_status(self.robot)

        self.set_observation(observation)
        self.set_obs_action(action)
        self.set_status(status)
        
        pre_action = self.get_pre_action()
        if pre_action is not None:
            action = self.robot.send_action(pre_action["action"])
            action = {"action": action}
        
        dt_s = time.perf_counter() - start_loop_t
        if self.fps is not None:
            busy_wait(1 / self.fps - dt_s)

        log_control_info(self.robot, dt_s, fps=self.fps)

    def set_pre_action(self, value: Union[Any, Dict[str, torch.Tensor]]):
        with self.data_lock:
            if value is None:
                return
            self.pre_action = value.copy()
    
    def set_obs_action(self, value: Union[Any, Dict[str, torch.Tensor]]):
        with self.data_lock:
            if value is None:
                return
            self.obs_action = value.copy()

    def set_observation(self, value: Union[Any, Dict[str, torch.Tensor]]):
        with self.data_lock:
            if value is None:
                return
            self.observation = value.copy()

    def set_status(self, value: Optional[str]):
        with self.data_lock:
            if value is None:
                return
            self.status = value

    def get_pre_action(self) -> Union[Any, Dict[str, torch.Tensor]]:
        with self.data_lock:
            if self.pre_action is None:
                return None
            return self.pre_action.copy()

    def get_obs_action(self) -> Union[Any, Dict[str, torch.Tensor]]:
        with self.data_lock:
            if self.obs_action is None:
                return None
            return self.obs_action.copy()
    
    def get_observation(self) -> Union[Any, Dict[str, torch.Tensor]]:
        with self.data_lock:
            if self.observation is None:
                return None
            return self.observation.copy()
        
    def get_status(self) -> Optional[str]:
        with self.data_lock:
            if self.status is None:
                return None
            return self.status
