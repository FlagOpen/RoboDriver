"""WebSocket client for LeRobot policy inference servers."""

import logging
import time
from typing import Dict, Optional

import msgpack
import numpy as np
import websockets.sync.client

from robodriver.core.policies.base_policy import BasePolicy

logger = logging.getLogger(__name__)


def _pack_default(obj):
    if isinstance(obj, np.ndarray):
        return {
            "__ndarray__": True,
            "dtype": str(obj.dtype),
            "shape": list(obj.shape),
            "data": obj.tobytes(),
        }
    if isinstance(obj, np.generic):
        return obj.item()
    raise TypeError(f"Cannot serialize {type(obj).__name__}")


def _unpack_hook(mapping):
    if set(mapping) == {"__ndarray__", "dtype", "shape", "data"}:
        return np.frombuffer(mapping["data"], dtype=mapping["dtype"]).reshape(
            mapping["shape"]
        )
    return mapping


def _pack(obj) -> bytes:
    return msgpack.packb(obj, default=_pack_default, use_bin_type=True)


def _unpack(raw: bytes):
    return msgpack.unpackb(raw, object_hook=_unpack_hook, raw=False)


DEFAULT_CAMERA_MAPPING = {
    "observation.images.image_top_rgb": "observation.images.image_top",
}


class LeRobotPolicy(BasePolicy):
    """Client for the LeRobot WebSocket protocol using raw HWC RGB images."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 8003,
        path: str = "",
        headers: Optional[Dict[str, str]] = None,
        api_key: Optional[str] = None,
        timeout: float = 15.0,
        camera_mapping: Optional[Dict[str, str]] = None,
    ):
        self._host = host
        self._port = port
        self._path = path
        self._headers = dict(headers) if headers else {}
        if api_key and "Authorization" not in self._headers:
            self._headers["Authorization"] = f"Bearer {api_key}"
        self._timeout = timeout
        self._camera_mapping = (
            dict(camera_mapping)
            if camera_mapping is not None
            else dict(DEFAULT_CAMERA_MAPPING)
        )
        self._client = None
        self._packer = None
        self._server_metadata = None
        self._sequence = 0

    @property
    def server_metadata(self) -> Optional[Dict]:
        return self._server_metadata

    def _url(self) -> str:
        if self._host.startswith(("ws://", "wss://")):
            return self._host
        return f"ws://{self._host}:{self._port}{self._path}"

    def _ensure_connected(self) -> None:
        if self._client is not None:
            return

        url = self._url()
        logger.info("Connecting to LeRobot server: %s", url)
        self._client = websockets.sync.client.connect(
            url,
            compression=None,
            max_size=None,
            additional_headers=self._headers or None,
            open_timeout=self._timeout,
            ping_interval=20,
            ping_timeout=20,
        )
        self._packer = msgpack.Packer(default=_pack_default, use_bin_type=True)
        metadata_raw = self._client.recv()
        if isinstance(metadata_raw, str):
            self.close()
            raise RuntimeError(f"LeRobot inference server error:\n{metadata_raw}")
        self._server_metadata = _unpack(metadata_raw)
        logger.info("LeRobot server metadata: %s", self._server_metadata)

    @staticmethod
    def _as_numpy(value) -> np.ndarray:
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "cpu"):
            value = value.cpu()
        if hasattr(value, "numpy"):
            value = value.numpy()
        return np.asarray(value)

    @classmethod
    def _format_image(cls, image) -> np.ndarray:
        array = cls._as_numpy(image)
        if array.ndim != 3:
            raise ValueError(f"Expected a 3D image, got shape {array.shape}")
        if array.shape[-1] != 3:
            if array.shape[0] == 3:
                array = np.transpose(array, (1, 2, 0))
            else:
                raise ValueError(f"Expected an RGB image, got shape {array.shape}")
        if array.size == 0:
            raise ValueError("Image must not be empty")
        if array.dtype != np.uint8:
            if np.issubdtype(array.dtype, np.floating) and array.max() <= 1.0:
                array = array * 255.0
            array = np.clip(array, 0, 255).astype(np.uint8)
        return np.ascontiguousarray(array)

    def _convert_observation(self, obs: Dict) -> Dict:
        if "observation.state" not in obs:
            raise ValueError("Missing required observation key 'observation.state'")
        state = self._as_numpy(obs["observation.state"]).astype(np.float32)
        if state.ndim != 1:
            raise ValueError(
                f"Expected a 1D observation state, got shape {state.shape}"
            )

        images = {
            self._camera_mapping.get(key, key): self._format_image(value)
            for key, value in obs.items()
            if key.startswith("observation.images.")
        }
        if not images:
            raise ValueError("No observation images found")

        request = {
            "prompt": str(obs.get("prompt", "")),
            "images": images,
            "state": np.ascontiguousarray(state),
            "encoding": "raw",
            "timestamp": time.time(),
            "sequence": self._sequence,
        }
        self._sequence += 1
        return request

    def infer(self, obs: Dict) -> Dict:
        self._ensure_connected()
        try:
            request = self._convert_observation(obs)
            self._client.send(self._packer.pack(request))
            response_raw = self._client.recv()
            if isinstance(response_raw, str):
                raise RuntimeError(f"LeRobot inference server error:\n{response_raw}")

            result = _unpack(response_raw)
            if "error" in result:
                error = result["error"]
                if isinstance(error, dict):
                    message = (
                        f"{error.get('type', 'Error')}: {error.get('message', '')}"
                    )
                else:
                    message = str(error)
                raise RuntimeError(f"LeRobot inference failed: {message}")
            if "actions" not in result:
                raise RuntimeError("LeRobot response is missing the 'actions' key")

            actions = np.asarray(result["actions"], dtype=np.float32)
            if actions.ndim == 1:
                actions = actions.reshape(1, -1)
            if actions.ndim != 2 or actions.shape[0] == 0:
                raise RuntimeError(f"Invalid LeRobot action shape: {actions.shape}")
            result["actions"] = actions
            logger.debug(
                "LeRobot returned actions=%s actual_steps=%s action_fps=%s",
                actions.shape,
                result.get("actual_steps"),
                result.get("action_fps"),
            )
            return result
        except Exception:
            self.close()
            raise

    def reset(self) -> None:
        self.close()

    def close(self) -> None:
        if self._client is not None:
            try:
                self._client.close()
            except Exception:
                pass
        self._client = None
        self._packer = None
        self._server_metadata = None
