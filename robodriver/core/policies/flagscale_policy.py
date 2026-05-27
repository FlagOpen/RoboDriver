"""FlagScale 推理策略实现

封装 FlagScale 的 WebSocket 客户端，自动将 LeRobot 格式的 observation 转换为 FlagScale 格式。

FlagScale 输入格式:
{
    "images": {
        "cam_top_left": [[[R通道], [G通道], [B通道]]],  // shape (3, H, W)
        "cam_top_right": [...],
        "cam_wrist_left": [...],
        "cam_wrist_right": [...]
    },
    "state": [...],
    "prompt": "..."
}
"""
import logging
import time
from typing import Dict, Optional, Tuple

import numpy as np
import websockets.sync.client

from robodriver.core.policies.base_policy import BasePolicy
from openpi_client import msgpack_numpy

logger = logging.getLogger(__name__)

# 相机名称映射：LeRobot 格式 -> FlagScale 格式
CAMERA_NAME_MAPPING = {
    "observation.images.image_top_left": "cam_top_left",
    "observation.images.image_top_right": "cam_top_right",
    "observation.images.image_wrist_left": "cam_wrist_left",
    "observation.images.image_wrist_right": "cam_wrist_right",
    # 兼容可能的其他命名
    "observation.images.cam_top_left": "cam_top_left",
    "observation.images.cam_top_right": "cam_top_right",
    "observation.images.cam_wrist_left": "cam_wrist_left",
    "observation.images.cam_wrist_right": "cam_wrist_right",
}


class FlagScalePolicy(BasePolicy):
    """FlagScale WebSocket 推理策略
    
    自动将 LeRobot 格式的 observation 转换为 FlagScale 格式进行通信。
    """
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 8087,
        path: str = "/inference",
        api_key: Optional[str] = None,
        camera_mapping: Optional[Dict[str, str]] = None,
    ):
        """初始化 FlagScale 策略客户端
        
        Args:
            host: 服务器地址，支持完整 URL 格式 (ws://...) 或仅主机名
            port: 服务器端口
            path: API 路径
            api_key: API 认证密钥
            camera_mapping: 自定义相机名称映射，如果为 None 则使用默认映射
        """
        self._host = host
        self._port = port
        self._path = path
        self._api_key = api_key
        self._camera_mapping = camera_mapping or CAMERA_NAME_MAPPING
        self._client = None
        self._packer = None
        self._server_metadata = None
        
        # 延迟连接
        self._connected = False
        
    def _ensure_connected(self):
        """确保已连接到服务器"""
        if self._connected and self._client is not None:
            return
        
        # 构建完整 URL
        if self._host.startswith("ws"):
            url = self._host
        else:
            url = f"ws://{self._host}:{self._port}{self._path}"
        
        logger.info(f"Connecting to FlagScale server: {url}")
        
        self._packer = msgpack_numpy.Packer()
        
        # 等待服务器连接
        self._wait_for_server(url)
        
        self._connected = True
        logger.info(f"FlagScale client connected: {url}")
    
    def _wait_for_server(self, url: str) -> None:
        """等待服务器可用并建立连接"""
        while True:
            try:
                headers = {"Authorization": f"Api-Key {self._api_key}"} if self._api_key else None
                
                logger.info(f"Trying to connect server at {url}...")
                self._client = websockets.sync.client.connect(
                    url,
                    compression=None,
                    max_size=None,
                    additional_headers=headers
                )
                logger.info(f"Connect server {url} success.")
                
                # 接收服务器元数据
                response = self._client.recv()
                if isinstance(response, bytes):
                    self._server_metadata = msgpack_numpy.unpackb(response)
                logger.info(f"Received server metadata: {self._server_metadata}")
                return
                
            except ConnectionRefusedError:
                logger.info("Still waiting for server...")
                time.sleep(5)
            except Exception as e:
                logger.warning(f"Connection attempt failed: {e}")
                time.sleep(5)
    
    @staticmethod
    def _image_hwc_to_chw(image: np.ndarray) -> list:
        """将 HWC 格式的图像 (H, W, 3) 转换为 CHW 列表格式 [[R], [G], [B]]
        
        Args:
            image: numpy 数组，shape (H, W, 3) 或 (H, W)
            
        Returns:
            三个通道的列表: [[R_channel], [G_channel], [B_channel]]
            每个通道是 (H, W) 的二维数组
        """
        if image.ndim == 2:
            # 灰度图，复制三个通道
            image = np.stack([image, image, image], axis=-1)
        
        # 确保是 float 类型，值在 [0, 1] 范围
        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0
        
        # 从 HWC (H, W, C) 转换为 CHW (C, H, W)
        # 然后转为列表
        if hasattr(image, "numpy"):
            image = image.numpy()
        image = np.asarray(image)
        
        # 转换为 CHW 格式
        chw = np.transpose(image, (2, 0, 1))  # (C, H, W)
        
        # 转为列表格式 [[R], [G], [B]]
        return [chw[0].tolist(), chw[1].tolist(), chw[2].tolist()]
    
    def _convert_observation(self, obs: Dict) -> Dict:
        """将 LeRobot 格式的 observation 转换为 FlagScale 格式
        
        Args:
            obs: LeRobot 格式的 observation
                - observation.state
                - observation.images.{camera_name}
                - prompt
                
        Returns:
            FlagScale 格式的 observation:
                - images: {cam_name: [[R], [G], [B]]}
                - state: [...]
                - prompt: "..."
        """
        flagscale_obs = {
            "images": {},
            "state": None,
            "prompt": obs.get("prompt", "")
        }
        
        # 提取 state
        state_key = "observation.state"
        if state_key in obs:
            state = obs[state_key]
            if hasattr(state, "numpy"):
                state = state.numpy()
            flagscale_obs["state"] = np.asarray(state).tolist()
        
        # 提取并转换图像
        for key, value in obs.items():
            if key.startswith("observation.images."):
                # 获取 FlagScale 相机名称
                flagscale_cam_name = self._camera_mapping.get(key)
                if flagscale_cam_name is None:
                    # 如果没有映射，使用原始名称去掉前缀
                    flagscale_cam_name = key.replace("observation.images.", "")
                
                # 转换图像格式
                if hasattr(value, "numpy"):
                    value = value.numpy()
                image = np.asarray(value)
                
                # HWC -> CHW 列表格式
                flagscale_obs["images"][flagscale_cam_name] = self._image_hwc_to_chw(image)
        
        return flagscale_obs
    
    @staticmethod
    def _decode_nested_msgpack(obj):
        """递归解码嵌套的 msgpack bytes 数据
        
        FlagScale 服务器可能返回嵌套的 msgpack 编码数据，
        需要递归解码所有 bytes 类型的值。
        """
        if isinstance(obj, dict):
            return {k: FlagScalePolicy._decode_nested_msgpack(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [FlagScalePolicy._decode_nested_msgpack(v) for v in obj]
        elif isinstance(obj, bytes):
            try:
                decoded = msgpack_numpy.unpackb(obj)
                return FlagScalePolicy._decode_nested_msgpack(decoded)
            except Exception:
                return obj
        else:
            return obj

    def infer(self, obs: Dict) -> Dict:
        """执行推理
        
        Args:
            obs: LeRobot 格式的 observation 字典
            
        Returns:
            动作字典，包含 "actions" 键
        """
        self._ensure_connected()
        
        try:
            # 转换为 FlagScale 格式
            flagscale_obs = self._convert_observation(obs)
            
            # 序列化并发送
            data = self._packer.pack(flagscale_obs)
            self._client.send(data)
            
            # 接收响应
            response = self._client.recv()
            if isinstance(response, str):
                raise RuntimeError(f"Error in inference server:\n{response}")
            
            # 反序列化响应
            result = msgpack_numpy.unpackb(response)
            
            # 递归解码嵌套的 msgpack 数据
            result = self._decode_nested_msgpack(result)
            
            # 日志记录返回结果的键信息
            if isinstance(result, dict):
                logger.debug(f"FlagScale response keys: {list(result.keys())}")
                if "actions" in result:
                    actions = result["actions"]
                    logger.debug(f"Actions type: {type(actions)}, shape/len: {getattr(actions, 'shape', len(actions) if hasattr(actions, '__len__') else 'N/A')}")
            
            return result
            
        except Exception as e:
            logger.error(f"FlagScale inference failed: {e}")
            raise
    
    def reset(self) -> None:
        """重置策略状态"""
        pass
    
    def close(self) -> None:
        """关闭连接"""
        if self._client is not None:
            try:
                self._client.close()
            except Exception:
                pass
        self._connected = False
        self._client = None
        self._packer = None
        logger.info("FlagScale client closed")