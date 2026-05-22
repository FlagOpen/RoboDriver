import asyncio
from dataclasses import asdict, dataclass
from pprint import pformat
from typing import List
import logging_mp
import time

import rerun as rr
import numpy as np

from lerobot.robots import RobotConfig
from lerobot.datasets.utils import build_dataset_frame, combine_feature_dicts
from lerobot.processor import make_default_processors
from lerobot.robots import Robot
from lerobot.teleoperators import Teleoperator
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.datasets.pipeline_features import (
    aggregate_pipeline_dataset_features,
    create_initial_features,
)

from robodriver.utils import parser
from robodriver.utils.import_utils import register_third_party_devices
from robodriver.utils.utils import git_branch_log

from robodriver.robots.utils import (
    make_robot_from_config,
)

from openpi_client import websocket_client_policy


logging_mp.basic_config(level=logging_mp.INFO)
logger = logging_mp.get_logger(__name__)

# 初始化 Rerun
rr.init("robodriver_inference", spawn=True)
# rr.connect()  # 连接到 rerun viewer

# 全局变量，用于控制推理循环
running = True


@dataclass
class ControlPipelineConfig:
    robot: RobotConfig
    @classmethod
    def __get_path_fields__(cls) -> List[str]:
        """This enables the parser to load config from the policy using `--policy.path=local/dir`"""
        return ["control.policy"]


@dataclass
class InferenceConfig:
    """推理配置"""
    prompt: str = "franka_A1_collision_sim_test"
    max_iterations: int = 5  # 最大推理周期数，-1表示无限循环


def numpy_to_action_dict(action_array, action_features):
    """将 numpy 动作数组转换为 robot.send_action 需要的 dict 格式"""
    import numpy as np
    
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
        logger.warning(f"Action array length ({len(action_array)}) doesn't match action_features ({len(action_keys)})")
    
    action_dict = {}
    for key, val in zip(action_keys, action_array):
        action_dict[key] = float(val)
    
    return action_dict


def log_observation_to_rerun(observation_frame, step):
    """将 observation 记录到 rerun"""

    rr.set_time("frame_idx", sequence=step)
    # 记录 prompt
    if "prompt" in observation_frame:
        rr.log("observation/prompt", rr.TextLog(str(observation_frame["prompt"])))
    
    # # 记录 state
    # if "observation.state" in observation_frame:
    #     state = observation_frame["observation.state"]
    #     if hasattr(state, "numpy"):
    #         state = state.numpy()
    #     state = np.asarray(state)
    #     rr.log("observation/state", rr.Scalar(float(np.mean(state))))  # 简化记录
    
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


def log_action_to_rerun(action_dict, step):
    """将 action 记录到 rerun"""
    if action_dict is None:
        return
    
    rr.set_time("frame_idx", sequence=step)
    
    for key, val in action_dict.items():
        # 将 key 中的特殊字符替换为 /
        rr_key = key.replace(".", "/")
        rr.log(f"action/{rr_key}", rr.Scalars(float(val)))


def execute_action_chunk(robot, action_dict, running_flag, step=0):
    """执行一个动作块（action chunk）
    
    pi0.5 模型返回的是一个字典，包含 'actions' 键
    actions 形状: (chunk_size, action_dim)，需要逐帧执行
    """
    import numpy as np
    
    # 从字典中提取 actions
    if action_dict is None:
        logger.warning("Received None action dict")
        return
    
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
    
    # 获取 action_features 用于转换
    action_features = robot.action_features
    
    # 逐帧执行动作块
    chunk_size = actions.shape[0]
    logger.info(f"Executing action chunk with {chunk_size} steps")
    
    for i in range(chunk_size):
        if not running_flag:
            logger.info("Stop signal received, aborting action chunk execution")
            break
        
        # 将 numpy 数组转换为 dict 格式
        action_array = actions[i]
        action_for_robot = numpy_to_action_dict(action_array, action_features)

        print(f"action_for_robot: {action_for_robot}")
        
        # 记录到 rerun
        log_action_to_rerun(action_for_robot, step+i)
        
        if action_for_robot:
            robot.send_action(action_for_robot)

        time.sleep(0.3)


@parser.wrap()
async def async_main(cfg: ControlPipelineConfig):
    global running
    logger.info(pformat(asdict(cfg)))
    
    inference_cfg = InferenceConfig()

    try:
        robot = make_robot_from_config(cfg.robot)
    except Exception as e:
        logger.critical(f"Failed to create robot: {e}")
        raise

    logger.info("Make robot success")
    logger.info(f"robot.type: {robot.robot_type}")

    if not robot.is_connected:
        robot.connect()
    logger.info("Connect robot success")

    teleop_action_processor, robot_action_processor, robot_observation_processor = (
        make_default_processors()
    )


    dataset_features = combine_feature_dicts(
        aggregate_pipeline_dataset_features(
            pipeline=teleop_action_processor,
            initial_features=create_initial_features(action=robot.action_features),
            use_videos=False,
        ),
        aggregate_pipeline_dataset_features(
            pipeline=robot_observation_processor,
            initial_features=create_initial_features(observation=robot.observation_features),
            use_videos=False,
        ),
    )

    policy = websocket_client_policy.WebsocketClientPolicy(
        host="localhost",
        port=8000,
        api_key=None
    )
    
    logger.info("Policy client created successfully")
    logger.info("Starting inference loop... Press Ctrl+C to stop.")
    
    iteration = 0
    
    try:
        while running:
            
            logger.info(f"Inference iteration: {iteration}")
            
            # 检查是否达到最大迭代次数
            if inference_cfg.max_iterations > 0 and iteration >= inference_cfg.max_iterations:
                logger.info(f"Reached max iterations ({inference_cfg.max_iterations}), stopping...")
                break
            
            try:
                # 获取观察
                observation = robot.get_observation()
                observation_frame = build_dataset_frame(
                    dataset_features, observation, prefix=OBS_STR
                )
                observation_frame["prompt"] = inference_cfg.prompt

                # print(f"observation_frame: {observation_frame.keys()}")
                
                # 记录 observation 到 rerun
                log_observation_to_rerun(observation_frame, iteration*10)
                
                # 推理获取动作块 (action chunk)
                actions = policy.infer(observation_frame)
                
                # 执行动作块
                execute_action_chunk(robot, actions, running, iteration*10)
                
            except KeyboardInterrupt:
                logger.info("Keyboard interrupt received")
                running = False
                break
            except Exception as e:
                logger.error(f"Error during inference at iteration {iteration}: {e}")
                # 询问用户是否继续
                continue

            iteration += 1

            # break
            
            # 等待推理循环继续
            await asyncio.sleep(0.01)
            
    except Exception as e:
        logger.error(f"Fatal error: {e}")
    finally:
        running = False
        logger.info("Inference loop ended")

def main():
    git_branch_log()

    register_third_party_devices()
    logger.info(f"Registered robot types: {list(RobotConfig._choice_registry.keys())}")

    asyncio.run(async_main())


if __name__ == "__main__":
    main()

# 启动命令：python -m robodriver.scripts.evaluate --robot.type=galaxealite-aio-ros2