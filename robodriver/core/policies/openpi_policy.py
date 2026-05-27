"""OpenPI 推理策略实现

封装 OpenPI 的 WebSocket 客户端，使用 LeRobot 格式的 observation。
"""
import logging
from typing import Dict, Optional

import numpy as np

from robodriver.core.policies.base_policy import BasePolicy

logger = logging.getLogger(__name__)


class OpenPIPolicy(BasePolicy):
    """OpenPI WebSocket 推理策略
    
    使用 OpenPI 的 msgpack 格式进行通信，observation 格式为：
    - observation.state
    - observation.images.{camera_name}
    - prompt
    """
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 8087,
        path: str = "/inference",
        api_key: Optional[str] = None,
    ):
        """初始化 OpenPI 策略客户端
        
        Args:
            host: 服务器地址，支持完整 URL 格式 (ws://...) 或仅主机名
            port: 服务器端口
            path: API 路径
            api_key: API 认证密钥
        """
        self._host = host
        self._port = port
        self._path = path
        self._api_key = api_key
        self._client = None
        
        # 延迟连接，在首次 infer 时连接
        self._connected = False
        
    def _ensure_connected(self):
        """确保已连接到服务器"""
        if self._connected and self._client is not None:
            return
        
        from openpi_client import websocket_client_policy
        
        # 构建完整 URL
        if self._host.startswith("ws"):
            url = self._host
        else:
            url = f"ws://{self._host}:{self._port}{self._path}"
        
        logger.info(f"Connecting to OpenPI server: {url}")
        self._client = websocket_client_policy.WebsocketClientPolicy(
            host=url,
            port=None,  # URL 已包含端口
            api_key=self._api_key
        )
        self._connected = True
        logger.info(f"OpenPI client connected: {url}")
        
    def infer(self, obs: Dict) -> Dict:
        """执行推理
        
        Args:
            obs: 观察数据字典，包含 observation.state, observation.images.*, prompt
            
        Returns:
            动作字典，包含 "actions" 键
        """
        self._ensure_connected()
        
        try:
            actions = self._client.infer(obs)
            return actions
        except Exception as e:
            logger.error(f"OpenPI inference failed: {e}")
            raise
    
    def reset(self) -> None:
        """重置策略状态"""
        pass
    
    def close(self) -> None:
        """关闭连接"""
        if self._client is not None:
            try:
                self._client.reset()
            except Exception:
                pass
        self._connected = False
        self._client = None
        logger.info("OpenPI client closed")