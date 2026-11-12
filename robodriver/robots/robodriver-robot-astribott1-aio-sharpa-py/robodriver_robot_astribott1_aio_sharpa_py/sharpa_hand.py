"""Sharpa Wave 灵巧手接入。

厂商 SDK 通过以太网/UDP 心跳发现设备，不走 ROS。SDK 可以：
  * 用 deb 安装到 /opt/sharpa-wave-sdk（推荐，postinst 会配置 ldconfig）
  * 放在用户目录，并通过 SHARPA_SDK_ROOT 指向解包根目录
  * 使用 ~/.local/opt/sharpa-wave-sdk（无 sudo 的用户级安装）

本模块只负责手部；与 AstriBot 的连接由当前包 robot.py 里的组合类完成。
"""

from __future__ import annotations

import ctypes
import os
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import logging_mp
import numpy as np


logger = logging_mp.get_logger(__name__)

LEFT = "left"
RIGHT = "right"
VALID_SIDES = (LEFT, RIGHT)

FINGERS = ("thumb", "index", "middle", "ring", "pinky")
TACTILE_CONTENTS = ("auto", "deform", "raw")

_DEFAULT_SDK_ROOT = Path.home() / ".local" / "opt" / "sharpa-wave-sdk"
_SYSTEM_SDK_ROOT = Path("/opt/sharpa-wave-sdk")

_sharpa_module: Any = None
_sharpa_root: Optional[Path] = None
_sharpa_lock = threading.Lock()


def _candidate_sdk_roots() -> List[Path]:
    roots: List[Path] = []
    env_root = os.environ.get("SHARPA_SDK_ROOT")
    if env_root:
        roots.append(Path(env_root).expanduser())
    roots.extend([_SYSTEM_SDK_ROOT, _DEFAULT_SDK_ROOT])
    return roots


def load_sharpa_sdk() -> Any:
    """Import the vendor SDK, falling back to an uninstalled SDK tree."""
    global _sharpa_module, _sharpa_root
    with _sharpa_lock:
        if _sharpa_module is not None:
            return _sharpa_module

        try:
            import sharpa

            _sharpa_module = sharpa
            return sharpa
        except ImportError:
            pass

        errors: List[str] = []
        for root in _candidate_sdk_roots():
            py_dir = root / "python"
            if not (py_dir / "sharpa" / "__init__.py").is_file():
                continue

            lib_path = root / "lib" / "libsharpa-wave-sdk.so"
            if lib_path.is_file():
                try:
                    ctypes.CDLL(str(lib_path), mode=ctypes.RTLD_GLOBAL)
                except OSError as exc:
                    errors.append(f"{lib_path}: {exc}")

            if str(py_dir) not in sys.path:
                sys.path.insert(0, str(py_dir))
            try:
                import sharpa

                _sharpa_module = sharpa
                _sharpa_root = root
                logger.info(f"使用 Sharpa SDK: {root}")
                return sharpa
            except ImportError as exc:
                errors.append(f"{py_dir}: {exc}")

        detail = f" 尝试记录: {'; '.join(errors)}" if errors else ""
        raise ImportError(
            "找不到 Sharpa Wave SDK。请安装 sharpa-wave-sdk deb"
            "（默认 /opt/sharpa-wave-sdk），或设置 SHARPA_SDK_ROOT 指向 SDK 根目录。"
            + detail
        )


@dataclass
class SharpaHandsConfig:
    enabled: bool = True
    # left / right / both
    side: str = "both"
    # 可选：两只手用 SN 固定，避免同侧设备被自动分配错
    left_sn: Optional[str] = None
    right_sn: Optional[str] = None
    dof: int = 22
    speed_coeff: float = 0.3
    current_coeff: float = 0.6
    interpolate: bool = True
    discovery_timeout_s: float = 10.0
    # 触觉：双手默认开启，每个指尖一路图像
    tactile_enabled: bool = True
    # auto: DEFORM -> DEFORM3_SE -> DEFORM3 -> RAW -> DIST_FORCE
    # deform/raw: 指定优先使用 DEFORM/RAW，缺失时自动回退
    tactile_content: str = "auto"
    tactile_fps: float = 30.0
    tactile_timeout_s: float = 3.0
    # 连接时是否先回零位（真机会动，谨慎开启）
    move_to_zero_on_connect: bool = False
    # 断开时是否先回零位；默认不回零，直接 stop()
    park_on_disconnect: bool = False

    def __post_init__(self):
        if self.side not in (LEFT, RIGHT, "both"):
            raise ValueError(f"sharpa_hands.side must be left/right/both, got {self.side!r}")
        if self.dof <= 0:
            raise ValueError(f"sharpa_hands.dof must be positive, got {self.dof}")
        if not 0.0 < self.speed_coeff <= 1.0:
            raise ValueError("sharpa_hands.speed_coeff must be in (0, 1]")
        if not 0.0 < self.current_coeff <= 1.0:
            raise ValueError("sharpa_hands.current_coeff must be in (0, 1]")
        if self.discovery_timeout_s <= 0:
            raise ValueError("sharpa_hands.discovery_timeout_s must be positive")
        if self.tactile_content not in TACTILE_CONTENTS:
            raise ValueError(
                f"sharpa_hands.tactile_content must be one of {TACTILE_CONTENTS}"
            )
        if self.tactile_fps <= 0:
            raise ValueError("sharpa_hands.tactile_fps must be positive")
        if self.tactile_timeout_s <= 0:
            raise ValueError("sharpa_hands.tactile_timeout_s must be positive")

    @property
    def sides(self) -> List[str]:
        return [LEFT, RIGHT] if self.side == "both" else [self.side]

    def sn_for(self, side: str) -> Optional[str]:
        return self.left_sn if side == LEFT else self.right_sn


class SharpaHands:
    """Owns the SharpaWaveManager and the connected left/right hands."""

    def __init__(self, config: SharpaHandsConfig):
        self.config = config
        self._sharpa: Any = None
        self._manager: Any = None
        self._waves: Dict[str, Any] = {}
        self._device_info: Dict[str, Any] = {}
        self._measured: Dict[str, List[float]] = {}
        self._tactile_sources: Dict[str, str] = {}
        self._tactile_shapes: Dict[Tuple[str, int], Tuple[int, int, int]] = {}
        self._tactile_cache: Dict[Tuple[str, int], np.ndarray] = {}
        self._tactile_last_poll: Dict[str, float] = {}
        self._shutdown_done = False

    @property
    def connected(self) -> bool:
        return bool(self._waves)

    @property
    def sides(self) -> List[str]:
        return self.config.sides

    @property
    def device_info(self) -> Dict[str, Any]:
        return dict(self._device_info)

    def follower_features(self) -> Dict[str, Any]:
        if not self.config.enabled:
            return {}
        features: Dict[str, Any] = {
            self._joint_key(side, index, "follower"): float
            for side in self.sides
            for index in range(self.config.dof)
        }
        if self.config.tactile_enabled:
            for side in self.sides:
                for channel in self._tactile_channels(side):
                    shape = self._tactile_shapes.get((side, channel))
                    features[self._tactile_key(side, channel)] = (
                        shape if shape is not None else self._default_tactile_shape()
                    )
        return features

    def leader_features(self) -> Dict[str, type]:
        if not self.config.enabled:
            return {}
        return {
            self._joint_key(side, index, "leader"): float
            for side in self.sides
            for index in range(self.config.dof)
        }

    def _joint_key(self, side: str, index: int, prefix: str) -> str:
        return f"{prefix}_hand_{side}_joint{index + 1}.pos"

    @staticmethod
    def _tactile_channels(side: str) -> range:
        return range(5, 10) if side == LEFT else range(0, 5)

    def _tactile_key(self, side: str, channel: int) -> str:
        offset = 5 if side == LEFT else 0
        return f"tactile_{side}_{FINGERS[channel - offset]}"

    def _default_tactile_shape(self) -> Tuple[int, int, int]:
        return (240, 320, 3) if self.config.tactile_content == "raw" else (240, 240, 3)

    def _wave(self, side: str) -> Any:
        wave = self._waves.get(side)
        if wave is None:
            raise RuntimeError(f"Sharpa {side} 手未连接")
        return wave

    def _missing_sides(self, infos: List[Any]) -> List[str]:
        sharpa = self._sharpa
        have = set()
        for info in infos:
            side = LEFT if info.hand_side == sharpa.HandSide.LEFT else RIGHT
            if side not in self.sides:
                continue
            wanted = self.config.sn_for(side)
            if wanted is None or info.sn == wanted:
                have.add(side)
        return [side for side in self.sides if side not in have]

    def _wait_for_devices(self) -> List[Any]:
        sharpa = self._sharpa
        manager = self._manager
        deadline = time.monotonic() + self.config.discovery_timeout_s
        last_log = 0.0
        last_error: Optional[Exception] = None
        missing: List[str] = list(self.sides)
        while time.monotonic() < deadline:
            try:
                infos = [
                    info
                    for info in manager.get_all_devices()
                    if info.device_type == sharpa.DeviceType.HAND
                ]
            except Exception as exc:
                last_error = exc
                infos = []
            # 心跳是异步到达的：必须等所有需要的 side 都出现，不能看到一只就返回
            missing = self._missing_sides(infos) if infos else list(self.sides)
            if infos and not missing:
                return infos
            now = time.monotonic()
            if now - last_log >= 2.0:
                found = sorted(
                    LEFT if info.hand_side == sharpa.HandSide.LEFT else RIGHT for info in infos
                )
                logger.info(
                    f"等待 Sharpa Wave 心跳（已发现: {found or '无'}，缺少: {missing}）..."
                    "确认网线/同网段/防火墙"
                )
                last_log = now
            time.sleep(0.2)
        detail = f"；最近一次错误: {last_error}" if last_error else ""
        raise TimeoutError(
            f"{self.config.discovery_timeout_s:.1f} 秒内未发现所需的 Sharpa Wave（缺少: {missing}）。"
            "确认手已上电、与主机同网段且 UDP 心跳未被防火墙拦截。"
            + detail
        )

    def _select_devices(self, infos: List[Any]) -> Dict[str, Any]:
        sharpa = self._sharpa
        by_side: Dict[str, List[Any]] = {}
        for info in infos:
            side = LEFT if info.hand_side == sharpa.HandSide.LEFT else RIGHT
            by_side.setdefault(side, []).append(info)

        selected: Dict[str, Any] = {}
        for side in self.sides:
            wanted_sn = self.config.sn_for(side)
            candidates = by_side.get(side, [])
            if wanted_sn:
                match = [info for info in infos if info.sn == wanted_sn]
                if not match:
                    found = ", ".join(f"{i.sn}({i.hand_side})" for i in infos)
                    raise ValueError(f"未找到 {side} 手 SN={wanted_sn}，当前发现: {found}")
                selected[side] = match[0]
            else:
                if not candidates:
                    found = ", ".join(f"{i.sn}({i.hand_side})" for i in infos)
                    raise ValueError(f"未发现 {side} 手，当前发现: {found}")
                selected[side] = candidates[0]
        return selected

    def _configure_wave(self, wave: Any) -> None:
        sharpa = self._sharpa
        steps = [
            ("set_control_mode", lambda: wave.set_control_mode(sharpa.ControlMode.POSITION)),
            ("set_speed_coeff", lambda: wave.set_speed_coeff(self.config.speed_coeff)),
            ("set_current_coeff", lambda: wave.set_current_coeff(self.config.current_coeff)),
            ("set_control_source", lambda: wave.set_control_source(sharpa.ControlSource.SDK)),
        ]
        for name, fn in steps:
            error = fn()
            if error.code != 0:
                raise RuntimeError(f"Sharpa {name} 失败: {error.message}")

        if wave.start():
            pass
        elif (
            self.config.tactile_enabled
            and wave.retry_tactile_alternate_port()
            and wave.is_hand_ready()
            and wave.is_tactile_ready()
        ):
            logger.info("Sharpa 触觉端口冲突，已切换到备用端口")
        else:
            raise RuntimeError(
                "Sharpa 手启动失败；确认设备在线、没有被 Sharpa Pilot/其他进程占用，"
                "并检查双手触觉 UDP 端口。"
            )

        if self.config.tactile_enabled and not wave.is_tactile_ready():
            if wave.retry_tactile_alternate_port() and wave.is_tactile_ready():
                logger.info("Sharpa 触觉端口冲突，已切换到备用端口")
            else:
                raise RuntimeError(
                    "Sharpa 触觉未就绪；检查触觉端口/网线，"
                    "或临时设置 sharpa_hands.tactile_enabled=false。"
                )

    def _read_positions(self, side: str) -> List[float]:
        wave = self._wave(side)
        error, values = wave.get_joint_position_rad()
        if error.code != 0:
            raise RuntimeError(f"读取 Sharpa {side} 关节失败: {error.message}")

        positions = [float(value) for value in values]
        if len(positions) != self.config.dof:
            raise ValueError(
                f"Sharpa {side} 手上报 {len(positions)} 个关节，"
                f"配置 sharpa_hands.dof={self.config.dof}。请按实际型号修改 dof。"
            )
        return positions

    def _choose_tactile_source(self, content: Dict[str, Any]) -> Optional[str]:
        order = {
            "auto": ("DEFORM", "DEFORM3_SE", "DEFORM3", "RAW", "DIST_FORCE"),
            "deform": ("DEFORM", "DEFORM3_SE", "RAW", "DEFORM3", "DIST_FORCE"),
            "raw": ("RAW", "DEFORM", "DEFORM3_SE", "DEFORM3", "DIST_FORCE"),
        }[self.config.tactile_content]
        for key in order:
            data = content.get(key)
            if data is not None and np.asarray(data).size > 0:
                return key
        return None

    @staticmethod
    def _reshape_content(data: Any, shape: Any) -> Optional[np.ndarray]:
        if data is None:
            return None
        arr = np.asarray(data)
        if arr.size == 0:
            return None
        dims = [int(value) for value in (shape or []) if int(value) > 0]
        if dims and int(np.prod(dims)) == arr.size:
            arr = arr.reshape(dims)
        else:
            arr = np.squeeze(arr)
        if (
            arr.ndim == 3
            and arr.shape[0] in (1, 3, 4)
            and arr.shape[2] not in (1, 3, 4)
        ):
            arr = np.transpose(arr, (1, 2, 0))
        return arr

    @staticmethod
    def _to_rgb_uint8(arr: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if arr is None:
            return None
        if arr.ndim == 2:
            arr = arr[:, :, None]
        if arr.ndim != 3 or arr.shape[2] not in (1, 3, 4):
            return None

        if arr.dtype != np.uint8:
            arr = np.nan_to_num(
                arr.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0
            )
            low = float(arr.min())
            high = float(arr.max())
            if high > low:
                arr = (arr - low) / (high - low) * 255.0
            else:
                arr = np.zeros_like(arr)
            arr = np.clip(arr, 0, 255).astype(np.uint8)

        if arr.shape[2] == 1:
            arr = np.repeat(arr, 3, axis=2)
        elif arr.shape[2] == 4:
            arr = arr[:, :, :3]
        # SDK 可能复用底层帧缓冲，拷贝一份再交给采集线程
        return np.ascontiguousarray(arr).copy()

    @staticmethod
    def _resize_rgb(
        image: np.ndarray, expected_shape: Optional[Tuple[int, int, int]]
    ) -> Optional[np.ndarray]:
        if expected_shape is None or image.shape == expected_shape:
            return image
        try:
            import cv2
        except ImportError:
            return None
        return np.ascontiguousarray(
            cv2.resize(
                image,
                (expected_shape[1], expected_shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
        )

    def _extract_tactile_image(
        self,
        frame: Dict[str, Any],
        side: str,
        expected_shape: Optional[Tuple[int, int, int]] = None,
    ) -> Tuple[Optional[np.ndarray], Optional[str]]:
        content = frame.get("content") or {}
        shapes = frame.get("shape") or {}
        source = self._tactile_sources.get(side)
        if source is None or source not in content:
            source = self._choose_tactile_source(content)
            if source is None:
                return None, None
            if side not in self._tactile_sources:
                self._tactile_sources[side] = source

        arr = self._reshape_content(content.get(source), shapes.get(source))
        image = self._to_rgb_uint8(arr)
        if image is None:
            return None, source
        return self._resize_rgb(image, expected_shape), source

    def _prime_tactile(self, side: str, wave: Any) -> None:
        channels = self._tactile_channels(side)
        pending = set(channels)
        deadline = time.monotonic() + self.config.tactile_timeout_s
        while pending and time.monotonic() < deadline:
            for channel in sorted(pending):
                frame = wave.fetch_tactile_frame(channel, 0.0)
                if frame is None:
                    continue
                image, _ = self._extract_tactile_image(frame, side)
                if image is None:
                    continue
                self._tactile_shapes[(side, channel)] = image.shape
                self._tactile_cache[(side, channel)] = image
                pending.remove(channel)
            if pending:
                time.sleep(0.02)

        if pending:
            raise RuntimeError(
                f"Sharpa {side} 触觉在 {self.config.tactile_timeout_s:.1f}s 内未就绪，"
                f"缺少通道 {sorted(pending)}。检查触觉端口/网线，"
                "或临时设置 sharpa_hands.tactile_enabled=false。"
            )

        first_channel = min(channels)
        logger.info(
            f"Sharpa {side} 触觉就绪: 5 指, "
            f"source={self._tactile_sources.get(side)}, "
            f"shape={self._tactile_shapes[(side, first_channel)]}"
        )

    def connect(self) -> None:
        if not self.config.enabled:
            return
        if self._waves:
            raise RuntimeError("Sharpa 手已经连接")

        self._sharpa = load_sharpa_sdk()
        try:
            self._manager = self._sharpa.SharpaWaveManager.get_instance()
        except Exception as exc:
            raise RuntimeError(f"初始化 SharpaWaveManager 失败: {exc}") from exc
        self._shutdown_done = False
        self._tactile_sources.clear()
        self._tactile_shapes.clear()
        self._tactile_cache.clear()
        self._tactile_last_poll.clear()

        try:
            infos = self._wait_for_devices()
            selected = self._select_devices(infos)
            for side in self.sides:
                info = selected[side]
                if self.config.tactile_enabled and not info.has_fingertip_tactile():
                    raise RuntimeError(
                        f"Sharpa {side} 手 SN={info.sn} 固件不支持触觉；"
                        "请升级固件，或设置 sharpa_hands.tactile_enabled=false。"
                    )
                logger.info(f"连接 Sharpa {side} 手: SN={info.sn}")
                wave = self._manager.connect(info.sn, not self.config.tactile_enabled)
                self._configure_wave(wave)
                self._waves[side] = wave
                self._device_info[side] = info
                positions = self._read_positions(side)
                self._measured[side] = positions
                if self.config.tactile_enabled:
                    self._prime_tactile(side, wave)
                logger.info(
                    f"Sharpa {side} 手就绪: SN={info.sn} firmware={info.firmware_version} "
                    f"dof={len(positions)} tactile={self.config.tactile_enabled}"
                )

            if self.config.move_to_zero_on_connect:
                for side in self.sides:
                    self._wave(side).set_joint_position(
                        [0.0] * self.config.dof, self.config.interpolate
                    )
        except Exception:
            self.disconnect()
            raise

    def get_observation(self) -> Dict[str, Any]:
        observation: Dict[str, Any] = {}
        for side in self.sides:
            positions = self._read_positions(side)
            self._measured[side] = positions
            for index, value in enumerate(positions):
                observation[self._joint_key(side, index, "follower")] = value

            if not self.config.tactile_enabled:
                continue

            now = time.monotonic()
            if (
                now - self._tactile_last_poll.get(side, 0.0)
                >= 1.0 / self.config.tactile_fps
            ):
                wave = self._wave(side)
                for channel in self._tactile_channels(side):
                    frame = wave.fetch_tactile_frame(channel, 0.0)
                    if frame is None:
                        continue
                    expected = self._tactile_shapes.get((side, channel))
                    image, _ = self._extract_tactile_image(frame, side, expected)
                    if image is not None:
                        self._tactile_cache[(side, channel)] = image
                self._tactile_last_poll[side] = now

            for channel in self._tactile_channels(side):
                image = self._tactile_cache.get((side, channel))
                if image is None:
                    shape = self._tactile_shapes.get(
                        (side, channel), self._default_tactile_shape()
                    )
                    image = np.zeros(shape, dtype=np.uint8)
                observation[self._tactile_key(side, channel)] = image
        return observation

    def get_action(self) -> Dict[str, float]:
        """AIO 没有独立手部 leader 时，诚实记录手当前实测位置。"""
        action: Dict[str, float] = {}
        for side in self.sides:
            positions = self._measured.get(side)
            if positions is None:
                positions = self._read_positions(side)
                self._measured[side] = positions
            for index, value in enumerate(positions):
                action[self._joint_key(side, index, "leader")] = float(value)
        return action

    def get_joint_snapshot(self) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Read measured hand joints for the recorder without polling tactile images."""
        observation: Dict[str, float] = {}
        action: Dict[str, float] = {}
        for side in self.sides:
            positions = self._read_positions(side)
            for index, value in enumerate(positions):
                observation[self._joint_key(side, index, "follower")] = float(value)
                action[self._joint_key(side, index, "leader")] = float(value)
        return observation, action

    def send_action(self, action: Dict[str, Any]) -> Dict[str, float]:
        if not self.connected:
            raise RuntimeError("Sharpa 手未连接，不能下发 action")

        sent: Dict[str, float] = {}
        for side in self.sides:
            keys = [self._joint_key(side, index, "leader") for index in range(self.config.dof)]
            missing = [key for key in keys if key not in action]
            if missing:
                raise KeyError(f"action 缺少 Sharpa {side} 手字段: {missing[:3]}...")

            positions = [float(action[key]) for key in keys]
            error = self._wave(side).set_joint_position(
                positions, self.config.interpolate
            )
            if error.code != 0:
                raise RuntimeError(
                    f"下发 Sharpa {side} 手 action 失败: {error.message}"
                )
            for key, value in zip(keys, positions):
                sent[key] = value
        return sent

    def disconnect(self) -> None:
        if self._shutdown_done:
            return
        self._shutdown_done = True

        waves = list(self._waves.items())
        self._waves.clear()
        self._measured.clear()
        self._device_info.clear()
        self._tactile_sources.clear()
        self._tactile_shapes.clear()
        self._tactile_cache.clear()
        self._tactile_last_poll.clear()

        try:
            if self.config.park_on_disconnect:
                for side, wave in waves:
                    try:
                        wave.set_joint_position(
                            [0.0] * self.config.dof, self.config.interpolate
                        )
                    except Exception as exc:
                        logger.warning(f"Sharpa {side} 手回零失败: {exc}")
                if waves:
                    time.sleep(0.2)

            for side, wave in waves:
                try:
                    wave.stop()
                    wave.destroy()
                except Exception as exc:
                    logger.warning(f"关闭 Sharpa {side} 手失败: {exc}")
        finally:
            if self._manager is not None:
                try:
                    self._manager.disconnect_all()
                except Exception as exc:
                    logger.warning(f"Sharpa manager 断开失败: {exc}")
            self._manager = None
