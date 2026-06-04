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
from robodriver.core.policies import create_policy, BasePolicy

logger = logging_mp.get_logger(__name__)


@dataclass
class InferenceConfig:
    """推理配置"""
    # Policy server 配置
    policy_host: str = "localhost"
    policy_port: int = 8087
    policy_path: str = "/inference"
    api_key: Optional[str] = None

    # 推理服务类型: "flagscale" (默认), "openpi", 或 "bc"
    # "bc" 使用 bc_robodriver 风格的推理路径：直接从 robot.get_observation()
    # 提取图像和状态，使用异步 WebSocket + msgpack 通信，不经过 build_dataset_frame
    policy_type: str = "flagscale"

    # 机器人类型，policy_type="bc" 时透传给 BCPolicy 用于观察提取
    robot_type: Optional[str] = None

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
        logger.info(f"Creating policy client of type: {self.infer_cfg.policy_type}")

        kwargs = dict(
            host=self.infer_cfg.policy_host,
            port=self.infer_cfg.policy_port,
            path=self.infer_cfg.policy_path,
            api_key=self.infer_cfg.api_key,
        )
        # "bc" 策略需要 robot_type 用于观察提取
        if self.infer_cfg.policy_type == "bc":
            robot_type = self.infer_cfg.robot_type or getattr(self.robot, "robot_type", None)
            kwargs["robot_type"] = robot_type

        self.policy_client = create_policy(
            policy_type=self.infer_cfg.policy_type,
            **kwargs,
        )
        logger.info(f"Policy client created: {self.infer_cfg.policy_type}")

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
        
        # 关闭 policy client
        self.disconnect()
        
        logger.info("Inferencer stopped")

    def disconnect(self):
        """断开策略服务器连接"""
        if self.policy_client is not None:
            if isinstance(self.policy_client, BasePolicy):
                self.policy_client.close()
            elif hasattr(self.policy_client, "reset"):
                try:
                    self.policy_client.reset()
                except Exception:
                    pass
            self.policy_client = None
            logger.info("Policy client disconnected")

    def _inference_loop(self):
        """核心推理循环

        根据 policy_type 选择观察提取方式：
        - "bc"：使用 BCRobotUtil 直接从 robot.get_observation() 提取图像和状态，
          不经过 LeRobot 的 build_dataset_frame，动作通过 BCRobotUtil.zip_action 还原为 dict。
        - 其他（"flagscale", "openpi"）：使用 build_dataset_frame 构建结构化观察帧。
        """
        logger.info("Starting inference loop...")

        use_bc = self.infer_cfg.policy_type == "bc"
        if use_bc:
            from robodriver.core.policies.bc_policy import BCRobotUtil, format_image_for_bc
            logger.info("BC inference path: using BCRobotUtil for observation extraction")

        while self.running:
            try:
                # 检查是否达到最大迭代次数
                if 0 < self.infer_cfg.max_iterations <= self.iteration:
                    logger.info(f"Reached max iterations ({self.infer_cfg.max_iterations})")
                    break

                logger.info(f"Inference iteration: {self.iteration}")

                # 获取观察
                observation = self._get_observation()

                if use_bc:
                    # BC 路径：直接提取图像+状态，构造请求字典
                    actions = self._infer_bc(observation, self.infer_cfg.prompt)
                else:
                    # 默认路径：通过 build_dataset_frame 构建结构化观察帧
                    observation_frame = build_dataset_frame(
                        self.dataset_features, observation, prefix=OBS_STR
                    )
                    observation_frame["prompt"] = self.infer_cfg.prompt

                    # 记录 observation 到 rerun
                    if self.infer_cfg.use_rerun:
                        self._log_observation_to_rerun(observation_frame, self.iteration)

                    actions = self._infer(observation_frame)

                if actions is not None:
                    if use_bc:
                        self._execute_action_chunk_bc(actions, self.iteration)
                    else:
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

    def _infer_bc(self, observation: dict, prompt: str) -> Optional[dict]:
        """BC 路径推理

        从原始观察中提取图像和状态，按 FlagScale 服务器期望的格式构造请求：
        {
            "images": {
                "cam_top_left":   [[R_channel], [G_channel], [B_channel]],  # (C, H, W) 列表
                "cam_top_right":  ...,
                "cam_wrist_left": ...,
                "cam_wrist_right":...,
            },
            "state":  [float, ...],   # 14 个关节值
            "prompt": str,
        }

        Args:
            observation: robot.get_observation() 返回的原始观察字典
            prompt: 任务描述字符串

        Returns:
            策略服务器返回的动作字典，通常包含 "actions" 键；失败返回 None
        """
        if self.policy_client is None:
            logger.error("Policy client not connected")
            return None

        from robodriver.core.policies.bc_policy import BCRobotUtil

        try:
            state = BCRobotUtil.extract_state(observation)
            logger.debug(f"BC state shape: {state.shape}, values: {state}")

            # 图像键映射：obs key → FlagScale 相机名
            image_key_to_flagscale = {
                "image_top_left":    "cam_top_left",
                "image_top_right":   "cam_top_right",
                "image_wrist_left":  "cam_wrist_left",
                "image_wrist_right": "cam_wrist_right",
            }

            images = {}
            for obs_key, cam_name in image_key_to_flagscale.items():
                if obs_key not in observation:
                    logger.warning(f"BC: image key '{obs_key}' not found in observation")
                    continue
                img = observation[obs_key]
                if hasattr(img, "numpy"):
                    img = img.numpy()
                img = np.asarray(img)
                # uint8 HWC → float32 CHW → [[R], [G], [B]] 列表（FlagScale 格式）
                if img.dtype == np.uint8:
                    img = img.astype(np.float32) / 255.0
                chw = np.transpose(img, (2, 0, 1))  # (C, H, W)
                images[cam_name] = [chw[0].tolist(), chw[1].tolist(), chw[2].tolist()]

            logger.debug(f"BC images: {list(images.keys())}")

            request_data = {
                "images": images,
                "state":  state.tolist(),
                "prompt": prompt,
            }

            return self.policy_client.infer(request_data)
        except Exception as e:
            logger.error(f"BC policy inference failed: {e}")
            return None

    def _execute_action_chunk_bc(self, action_dict: dict, step: int = 0):
        """执行 BC 策略返回的动作块

        BC 策略返回 {"actions": ndarray(ACTION_HORIZON, N)}，
        使用 BCRobotUtil.zip_action 将每行动作与 robot.action_features 的 key 对应，
        逐步发送给机器人。

        Args:
            action_dict: BCPolicy.infer() 返回的字典，包含 "actions" 键
            step: 当前推理步数（用于 rerun 记录）
        """
        if action_dict is None:
            logger.warning("Received None action dict (BC)")
            return

        from robodriver.core.policies.bc_policy import BCRobotUtil

        actions = action_dict.get("actions")
        if actions is None:
            logger.warning("No 'actions' key in BC response dict")
            return

        if not isinstance(actions, np.ndarray):
            actions = np.asarray(actions)

        # 确保是 2D 数组
        if actions.ndim == 1:
            actions = actions.reshape(1, -1)

        action_features = self.robot.action_features
        chunk_size = min(actions.shape[0], self.policy_client.ACTION_HORIZON)
        logger.info(f"BC: executing action chunk with {chunk_size} steps")

        for i in range(chunk_size):
            if not self.running:
                logger.info("Stop signal received, aborting BC action chunk execution")
                break

            action_dict_step = BCRobotUtil.zip_action(actions[i], action_features)
            logger.debug(f"BC action step {i}: {action_dict_step}")

            # 记录到 rerun
            if self.infer_cfg.use_rerun:
                self._log_action_to_rerun(action_dict_step, step + i)

            if action_dict_step:
                self.robot.send_action(action_dict_step)

            time.sleep(self.infer_cfg.action_chunk_sleep)

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

    @staticmethod
    def _decode_serialized_ndarray(data: dict) -> np.ndarray:
        """解码序列化的 ndarray 字典格式 {'__ndarray__': True, 'dtype': '<f4', 'shape': [...], 'data': b'...'}"""
        if isinstance(data, dict) and data.get("__ndarray__"):
            dtype = np.dtype(data["dtype"])
            shape = tuple(data["shape"])
            raw_data = data["data"]
            if isinstance(raw_data, str):
                # Base64 encoded string
                import base64
                raw_bytes = base64.b64decode(raw_data)
            else:
                raw_bytes = raw_data
            array = np.frombuffer(raw_bytes, dtype=dtype)
            return array.reshape(shape)
        return np.asarray(data)

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
        
        # 转换为 numpy 数组，处理序列化的 ndarray 格式
        if isinstance(actions, dict) and actions.get("__ndarray__"):
            actions = self._decode_serialized_ndarray(actions)
        elif hasattr(actions, "numpy"):
            actions = actions.numpy()
        else:
            actions = np.asarray(actions)

        logger.info(f"inference get actions:{actions}")
        
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
            
            logger.info(f"inference get action_for_robot:{action_for_robot}")

            # 记录到 rerun
            if self.infer_cfg.use_rerun:
                self._log_action_to_rerun(action_for_robot, step + i)
            
            logger.info(f"will send action_for_robot.")

            if action_for_robot:
                self.robot.send_action(action_for_robot)

            logger.info(f"send action_for_robot success.")

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
            # Use rr.Scalar for single scalar values (accepts float directly)
            rr.log(f"action/{rr_key}", rr.Scalars(float(val)))


    @staticmethod
    def _log_action_list_to_rerun(action_list: list, step: int):
        """将 action 记录到 rerun"""
        if action_list is None:
            return
        
        rr.set_time("frame_idx", sequence=step)
        
        for i, val in enumerate(action_list):
            # 将 key 中的特殊字符替换为 /
            # rr_key = key.replace(".", "/")
            # Use rr.Scalar for single scalar values (accepts float directly)
            rr.log(f"action/{i}", rr.Scalars(float(val)))
