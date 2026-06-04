"""推理服务模块"""
from robodriver.core.policies.base_policy import BasePolicy
from robodriver.core.policies.openpi_policy import OpenPIPolicy
from robodriver.core.policies.flagscale_policy import FlagScalePolicy
from robodriver.core.policies.bc_policy import BCPolicy

# 策略注册表，用于根据名称创建策略
POLICY_REGISTRY = {
    "openpi": OpenPIPolicy,
    "flagscale": FlagScalePolicy,
    "bc": BCPolicy,
}

__all__ = ["BasePolicy", "OpenPIPolicy", "FlagScalePolicy", "BCPolicy", "POLICY_REGISTRY", "create_policy"]


def create_policy(policy_type: str = "flagscale", **kwargs):
    """根据类型创建策略客户端
    
    Args:
        policy_type: 策略类型，支持 "openpi" 和 "flagscale"
        **kwargs: 传递给策略构造函数的参数
        
    Returns:
        BasePolicy 实例
        
    Raises:
        ValueError: 如果 policy_type 不支持
    """
    policy_type = policy_type.lower()
    if policy_type not in POLICY_REGISTRY:
        raise ValueError(
            f"Unknown policy type: {policy_type}. "
            f"Supported types: {list(POLICY_REGISTRY.keys())}"
        )
    
    policy_class = POLICY_REGISTRY[policy_type]
    return policy_class(**kwargs)