import threading
import time
from dataclasses import dataclass, field
from typing import Optional

import logging_mp
import numpy as np
import rerun as rr

from lerobot.datasets.pipeline_features import (
    aggregate_pipeline_dataset_features,
    create_initial_features,
)
from lerobot.datasets.utils import build_dataset_frame, combine_feature_dicts
from lerobot.processor import make_default_processors
from lerobot.robots import Robot
from lerobot.teleoperators import Teleoperator
from lerobot.utils.constants import ACTION, OBS_STR

from robodriver.robots.daemon import Daemon

logger = logging_mp.get_logger(__name__)


@dataclass
class InferenceConfig:
    """推理配置"""
    # Policy server 配置
    policy_host: str = "localhost"
    policy_port: int = 8087
    policy_path: str = "/inference"
    api_key: Optional[str] = None
    
    # 推理参数
    prompt: str = "default_task"
    max_iterations: int = -1  # -1表示无限循环
    fps: int = 30
    action_chunk_sleep: float = 0.03  # 动作块执行间隔
    
    # Rerun 可视化
    use_rerun: bool = True


class Inferencer:
    """推理器类，负责连接策略服务器并执行推理"""
    
    def __init__(
        self,
        robot: Robot,
        daemon: Daemon,
        teleop: Optional[Teleoperator] = None,
        infer_cfg: InferenceConfig = None,
    ):
        self.robot = robot
        self.daemon = daemon
        self.teleop = teleop
        self.infer_cfg = infer_cfg or InferenceConfig()
        
        # 内部状态
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.policy_client = None
        self.dataset_features = None
        self.iteration = 0
        
        # 初始化 processors 和 dataset_features
        self._init_processors_and_features()
        
    def _init_processors_and_features(self):
        """初始化 processors 和 dataset features"""
        teleop_action_processor, robot_action_processor, robot_observation_processor = (
            make_default_processors()
        )

        action_features = (
            self.teleop.action_features 
            if self.teleop is not None 
            else self.robot.action_features
        )

        self.dataset_features = combine_feature_dicts(
            aggregate_pipeline_dataset_features(
                pipeline=teleop_action_processor,
                initial_features=create_initial_features(action=action_features),
                use_videos=False,
            ),
            aggregate_pipeline_dataset_features(
                pipeline=robot_observation_processor,
                initial_features=create_initial_features(
                    observation=self.robot.observation_features
                ),
                use_videos=False,
            ),
        )
        logger.info(f"Dataset features initialized: {list(self.dataset_features.keys())}")

    def connect(self):
        """主动连接策略服务器"""
        from openpi_client import websocket_client_policy
        
        # 构建完整 URL: ws://host:port/path
        host = self.infer_cfg.policy_host
        port = self.infer_cfg.policy_port
        path = self.infer_cfg.policy_path
        
        # 如果 host 已经是完整 URL 格式，直接使用
        if host.startswith("ws"):
            url = host
        else:
            url = f"ws://{host}:{port}{path}"
        
        logger.info(f"Connecting to policy server: {url}")
        self.policy_client = websocket_client_policy.WebsocketClientPolicy(
            host=url,
            port=None,  # URL 已包含端口
            api_key=self.infer_cfg.api_key
        )
        logger.info(f"Policy client connected: {url}")

    def start(self):
        """启动推理循环"""
        if self.running:
            logger.warning("Inferencer is already running")
            return
        
        self.running = True
        self.iteration = 0
        self.thread = threading.Thread(
            target=self._inference_loop,
            name="InferencerThread",
            daemon=True,
        )
        self.thread.start()
        logger.info("Inferencer started")

    def stop(self):
        """停止推理"""
        if not self.running:
            logger.warning("Inferencer is not running")
            return
        
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=5.0)
            if self.thread.is_alive():
                logger.warning("Warning: Inferencer thread did not exit cleanly")
        logger.info("Inferencer stopped")

    def _inference_loop(self):
        """核心推理循环"""
        logger.info("Starting inference loop...")
        
        while self.running:
            try:
                # 检查是否达到最大迭代次数
                if 0 < self.infer_cfg.max_iterations <= self.iteration:
                    logger.info(f"Reached max iterations ({self.infer_cfg.max_iterations})")
                    break
                
                logger.info(f"Inference iteration: {self.iteration}")
                
                # 获取观察
                observation = self._get_observation()
                observation_frame = build_dataset_frame(
                    self.dataset_features, observation, prefix=OBS_STR
                )
                observation_frame["prompt"] = self.infer_cfg.prompt
                
                # 记录 observation 到 rerun
                if self.infer_cfg.use_rerun:
                    self._log_observation_to_rerun(observation_frame, self.iteration)
                
                # 推理获取动作块
                actions = self._infer(observation_frame)
                
                if actions is not None:
                    # 执行动作块
                    self._execute_action_chunk(actions, self.iteration)
                
                self.iteration += 1
                
                # 控制推理频率
                time.sleep(1.0 / self.infer_cfg.fps)
                
            except Exception as e:
                logger.error(f"Error during inference at iteration {self.iteration}: {e}")
                time.sleep(0.5)  # 错误后短暂等待
        
        logger.info("Inference loop ended")

    def _get_observation(self) -> dict:
        """获取观察数据"""
        if self.daemon is not None:
            return self.daemon.get_observation()
        else:
            return self.robot.get_observation()

    def _infer(self, observation_frame: dict) -> Optional[dict]:
        """执行推理"""
        if self.policy_client is None:
            logger.error("Policy client not connected")
            return None
        
        try:
            actions = self.policy_client.infer(observation_frame)
            return actions
        except Exception as e:
            logger.error(f"Policy inference failed: {e}")
            return None

    def _execute_action_chunk(self, action_dict: dict, step: int = 0):
        """执行一个动作块（action chunk）"""
        if action_dict is None:
            logger.warning("Received None action dict")
            return
        
        # 从字典中提取 actions
        actions = action_dict.get("actions")
        if actions is None:
            logger.warning("No 'actions' key in returned dict")
            return
        
        # 转换为 numpy 数组
        if hasattr(actions, "numpy"):
            actions = actions.numpy()
        
        actions = np.asarray(actions)
        
        # 确保是 2D 数组
        if actions.ndim == 1:
            actions = actions.reshape(1, -1)
        
        action_features = self.robot.action_features
        chunk_size = actions.shape[0]
        logger.info(f"Executing action chunk with {chunk_size} steps")
        
        for i in range(chunk_size):
            if not self.running:
                logger.info("Stop signal received, aborting action chunk execution")
                break
            
            # 将 numpy 数组转换为 dict 格式
            action_array = actions[i]
            action_for_robot = self._numpy_to_action_dict(action_array, action_features)
            
            # 记录到 rerun
            if self.infer_cfg.use_rerun:
                self._log_action_to_rerun(action_for_robot, step + i)
            
            if action_for_robot:
                self.robot.send_action(action_for_robot)

            time.sleep(self.infer_cfg.action_chunk_sleep)

    @staticmethod
    def _numpy_to_action_dict(action_array, action_features) -> Optional[dict]:
        """将 numpy 动作数组转换为 robot.send_action 需要的 dict 格式"""
        if action_array is None:
            return None
        
        # 转换为 numpy 数组
        if hasattr(action_array, "numpy"):
            action_array = action_array.numpy()
        
        action_array = np.asarray(action_array)
        
        # 确保是 1D 数组
        if action_array.ndim == 2:
            action_array = action_array.flatten()
        
        # 转换为 dict
        action_keys = list(action_features.keys())
        if len(action_keys) != len(action_array):
            logger.warning(
                f"Action array length ({len(action_array)}) doesn't match "
                f"action_features ({len(action_keys)})"
            )
        
        action_dict = {}
        for key, val in zip(action_keys, action_array):
            action_dict[key] = float(val)
        
        return action_dict

    @staticmethod
    def _log_observation_to_rerun(observation_frame: dict, step: int):
        """将 observation 记录到 rerun"""
        rr.set_time("frame_idx", sequence=step)
        
        # 记录 prompt
        if "prompt" in observation_frame:
            rr.log("observation/prompt", rr.TextLog(str(observation_frame["prompt"])))
        
        # 记录图像
        for key, img in observation_frame.items():
            if key.startswith("observation.images."):
                if hasattr(img, "numpy"):
                    img = img.numpy()
                img = np.asarray(img)
                # 确保图像是 uint8
                if img.dtype != np.uint8:
                    img = (img * 255).astype(np.uint8)
                cam_name = key.replace("observation.images.", "")
                rr.log(f"observation/camera/{cam_name}", rr.Image(img))

    @staticmethod
    def _log_action_to_rerun(action_dict: dict, step: int):
        """将 action 记录到 rerun"""
        if action_dict is None:
            return
        
        rr.set_time("frame_idx", sequence=step)
        
        for key, val in action_dict.items():
            # 将 key 中的特殊字符替换为 /
            rr_key = key.replace(".", "/")
            rr.log(f"action/{rr_key}", rr.Scalars(float(val)))