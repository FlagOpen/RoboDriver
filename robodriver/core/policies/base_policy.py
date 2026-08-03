"""基础策略接口定义"""
from abc import ABC, abstractmethod
from typing import Dict, Optional


class BasePolicy(ABC):
    """推理策略基类，定义统一的推理接口"""
    
    @abstractmethod
    def infer(self, obs: Dict) -> Dict:
        """执行推理，获取动作
        
        Args:
            obs: 观察数据字典
            
        Returns:
            动作字典，通常包含 "actions" 键
        """
        pass
    
    @abstractmethod
    def reset(self) -> None:
        """重置策略状态"""
        pass
    
    @abstractmethod
    def close(self) -> None:
        """关闭连接，清理资源"""
        pass