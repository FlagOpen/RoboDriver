"""
简化版推理脚本

使用方式:
    python -m robodriver.scripts.inference --robot.type=galaxealite-aio-ros2 --inference.policy_port 8000

或者从配置文件加载:
    python -m robodriver.scripts.inference --config.path=path/to/config.yaml
"""
from dataclasses import asdict, dataclass, field
from pprint import pformat
from typing import List

import logging_mp
import rerun as rr

from robodriver.core.inferencer import InferenceConfig, Inferencer
from lerobot.robots.config import RobotConfig
from robodriver.robots.utils import make_robot_from_config
from robodriver.utils import parser
from robodriver.utils.import_utils import register_third_party_devices
from robodriver.utils.utils import git_branch_log

logging_mp.basic_config(level=logging_mp.INFO)
logger = logging_mp.get_logger(__name__)


@dataclass
class InferenceScriptConfig:
    """推理脚本配置"""
    # 机器人配置
    robot: RobotConfig
    
    # 推理配置
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    
    @classmethod
    def __get_path_fields__(cls) -> List[str]:
        """This enables the parser to load config from the policy using `--policy.path=local/dir`"""
        return ["inference"]


def init_rerun():
    """初始化 Rerun 可视化"""
    rr.init("robodriver_inference", spawn=True)


@parser.wrap()
def inference(cfg: InferenceScriptConfig):
    # 初始化 Rerun（如果需要）
    if cfg.inference.use_rerun:
        init_rerun()
    
    # 创建机器人
    try:
        robot = make_robot_from_config(cfg.robot)
    except Exception as e:
        logger.critical(f"Failed to create robot: {e}")
        raise
    
    logger.info(f"Robot created: {robot.robot_type}")
    
    # 确保机器人连接
    if not robot.is_connected:
        robot.connect()
    logger.info("Robot connected")
    
    # 创建 inferencer
    inferencer = Inferencer(
        robot=robot,
        daemon=None,  # 独立脚本模式不使用 daemon
        infer_cfg=cfg.inference,
    )
    
    try:
        # 连接数据通道
        inferencer.connect()
        logger.info("Data channel connected")
        
        # 启动推理
        inferencer.start()
        logger.info("Inference started")
        
        # 等待推理完成（由 inferencer 内部循环控制）
        # 这里阻塞等待，直到 inferencer 停止
        while inferencer.running:
            import time
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
    except Exception as e:
        logger.error(f"Error during inference: {e}")
        raise
    finally:
        # 确保停止推理
        inferencer.stop()
        logger.info("Inference complete")


def main():
    """主函数"""
    git_branch_log()
    register_third_party_devices()
    logger.info(f"Registered robot types: {list(RobotConfig._choice_registry.keys())}")

    inference()

if __name__ == "__main__":
    main()
