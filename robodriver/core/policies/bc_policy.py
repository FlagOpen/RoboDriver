"""BC (Behavior Cloning) 推理策略实现

参考 bc_robodriver/RoboDriver/robodriver/infer_script/policy_pi_flag_platform.py
使用 BCRobotUtil 进行机器人特定的观察提取，使用异步 WebSocket + msgpack 通信。
"""
import asyncio
import logging
from typing import Any, Dict, List, Optional

import msgpack
import numpy as np
import websockets

from robodriver.core.policies.base_policy import BasePolicy

logger = logging.getLogger(__name__)


def _encode_ndarray(obj):
    """编码 numpy 数组为 msgpack 格式"""
    if isinstance(obj, np.ndarray):
        return {
            "__ndarray__": True,
            "dtype": obj.dtype.str,
            "shape": list(obj.shape),
            "data": obj.tobytes(),
        }
    raise TypeError(f"Cannot encode {type(obj)}")


def _decode_ndarray(obj):
    """解码 msgpack 格式的 numpy 数组"""
    if isinstance(obj, dict) and obj.get("__ndarray__"):
        return np.frombuffer(obj["data"], dtype=np.dtype(obj["dtype"])).reshape(obj["shape"])
    return obj


def _pack(data):
    return msgpack.packb(data, default=_encode_ndarray, use_bin_type=True)


def _unpack(data):
    return msgpack.unpackb(data, object_hook=_decode_ndarray, raw=False)


def format_image_for_bc(img: np.ndarray) -> np.ndarray:
    """将图像从 (H, W, C) uint8 转换为 (C, H, W) float32，并 resize 到 224x224."""
    import cv2
    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_LINEAR)
    img = np.transpose(img, (2, 0, 1))
    img = img.astype(np.float32) / 255.0
    return img


class BCRobotUtil:
    """机器人工具类，用于提取观察数据

    参考 bc_robodriver 的 RobotUtil，但简化为支持通用机器人类型。
    """

    CAM_TOP = "top"
    CAM_TOP_LEFT = "top_left"
    CAM_TOP_RIGHT = "top_right"
    CAM_WRIST_LEFT = "wrist_left"
    CAM_WRIST_RIGHT = "wrist_right"

    @staticmethod
    def extract_state(obs: dict) -> np.ndarray:
        """从观察中提取关节状态数据

        参考 bc_robodriver 原始实现：直接取 obs.values() 的前 14 个值，
        对应 follower 左臂 7 个关节 + 右臂 7 个关节（关节角 + 夹爪）。

        get_observation() 的字段顺序为：
          follower_left_arm_joint_1_rad.pos ~ follower_right_gripper_degree_mm.pos (14个)
          image_top_left, image_top_right, image_wrist_left, image_wrist_right

        Args:
            obs: robot.get_observation() 返回的原始观察字典

        Returns:
            float32 状态数组，shape (14,)
        """
        vals = list(obs.values())
        # 前 14 个值为关节状态（左臂7 + 右臂7），后面是图像
        state = [float(v) for v in vals[:14]]
        return np.asarray(state, dtype=np.float32)

    @staticmethod
    def extract_images(obs: dict, transform=None) -> dict:
        """从观察中提取图像数据，映射到标准相机位置名称

        直接按 obs 中的 image_* 键查找，映射规则：
          image_top_left   → top_left
          image_top_right  → top_right
          image_wrist_left → wrist_left
          image_wrist_right→ wrist_right

        Args:
            obs: robot.get_observation() 返回的原始观察字典
            transform: 可选的图像转换函数，例如 format_image_for_bc

        Returns:
            标准相机位置名称 -> 图像的映射字典
        """
        # image key → 标准相机名映射
        key_to_cam = {
            "image_top_left":    BCRobotUtil.CAM_TOP_LEFT,
            "image_top_right":   BCRobotUtil.CAM_TOP_RIGHT,
            "image_wrist_left":  BCRobotUtil.CAM_WRIST_LEFT,
            "image_wrist_right": BCRobotUtil.CAM_WRIST_RIGHT,
        }
        images = {}
        for key, cam_name in key_to_cam.items():
            if key in obs:
                images[cam_name] = transform(obs[key]) if transform else obs[key]
        return images

    @staticmethod
    def zip_action(action: np.ndarray, action_features: dict) -> dict:
        """将动作数组与动作特征 key 打包为字典

        Args:
            action: 动作数组
            action_features: robot.action_features 字典，提供 key 列表

        Returns:
            动作字典
        """
        action_keys = list(action_features.keys())
        return dict(zip(action_keys, action))


class BCPolicy(BasePolicy):
    """BC (Behavior Cloning) WebSocket 推理策略

    参考 bc_robodriver 的 PolicyPi 实现，使用异步 WebSocket + msgpack 通信。
    观察数据由 BCRobotUtil 从原始 robot.get_observation() 提取，
    不经过 LeRobot 的 build_dataset_frame 处理。

    请求格式::

        {
            "images": {
                "top_left": <ndarray (C, H, W) float32>,
                "top_right": <ndarray (C, H, W) float32>,
                "wrist_left": <ndarray (C, H, W) float32>,
                "wrist_right": <ndarray (C, H, W) float32>,
            },
            "state": <ndarray (N,) float32>,
            "prompt": "<task description>",
        }

    响应格式::

        {
            "actions": <ndarray (ACTION_HORIZON, N) float32>
        }
    """

    ACTION_HORIZON = 50

    def __init__(
        self,
        host: str = "localhost",
        port: int = 8087,
        path: str = "",
        api_key: Optional[str] = None,
        robot_type: Optional[str] = None,
        timeout: float = 15.0,
    ):
        """初始化 BC 策略客户端

        Args:
            host: 服务器地址
            port: 服务器端口
            path: API 路径（通常为空字符串）
            api_key: 认证密钥（目前未使用，保留参数兼容性）
            robot_type: 机器人类型字符串，透传给 BCRobotUtil（目前未用于分支逻辑）
            timeout: WebSocket 连接超时（秒）
        """
        self._host = host
        self._port = port
        self._path = path
        self._timeout = timeout
        self._robot_type = robot_type

        self._ws = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._connected = False

    # ------------------------------------------------------------------ #
    #  连接管理
    # ------------------------------------------------------------------ #

    def _ensure_connected(self):
        """首次调用 infer 时懒连接到策略服务器"""
        if self._connected and self._ws is not None:
            return

        uri = f"ws://{self._host}:{self._port}{self._path}"
        logger.info(f"Connecting to BC policy server: {uri}")

        if self._loop is None:
            self._loop = asyncio.new_event_loop()

        self._loop.run_until_complete(self._connect_async(uri))
        self._connected = True
        logger.info(f"BC policy client connected: {uri}")

    async def _connect_async(self, uri: str):
        self._ws = await websockets.connect(
            uri, open_timeout=self._timeout, max_size=None
        )
        # 读取服务器元数据（握手消息）
        metadata_raw = await self._ws.recv()
        metadata = _unpack(metadata_raw)
        logger.info(f"BC policy server metadata: {metadata}")

    def _run(self, coro):
        return self._loop.run_until_complete(coro)

    # ------------------------------------------------------------------ #
    #  BasePolicy 接口
    # ------------------------------------------------------------------ #

    def infer(self, request_data: Dict) -> Dict:
        """执行推理

        Args:
            request_data: 已由 Inferencer._infer_bc 构造好的请求字典，格式为::

                {
                    "images": {"top_left": ndarray, "top_right": ndarray, ...},
                    "state":  ndarray (N,) float32,
                    "prompt": str,
                }

        Returns:
            {"actions": ndarray (ACTION_HORIZON, N) float32}
        """
        self._ensure_connected()

        state_log = request_data.get("state", [])
        logger.debug(
            f"BC infer: state len={len(state_log)}, "
            f"images={list(request_data.get('images', {}).keys())}"
        )

        response = self._run(self._infer_async(request_data))

        if "error" in response:
            raise RuntimeError(f"BC policy server error: {response['error']}")

        all_actions = response["actions"]
        if isinstance(all_actions, np.ndarray):
            acts = all_actions[: self.ACTION_HORIZON]
        else:
            acts = np.asarray(all_actions[: self.ACTION_HORIZON])
        logger.debug(f"BC policy returned actions shape: {acts.shape}")
        return {"actions": acts}

    async def _infer_async(self, request_data: dict) -> dict:
        await self._ws.send(_pack(request_data))
        response_raw = await self._ws.recv()
        return _unpack(response_raw)

    def reset(self) -> None:
        """重置策略状态（当前无需操作）"""
        pass

    def close(self) -> None:
        """关闭 WebSocket 连接"""
        if self._ws is not None and self._loop is not None:
            try:
                self._run(self._ws.close())
            except Exception:
                pass
            self._ws = None
        self._connected = False
        logger.info("BC policy client closed")
