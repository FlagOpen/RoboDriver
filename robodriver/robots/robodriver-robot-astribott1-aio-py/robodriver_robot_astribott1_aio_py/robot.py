import atexit
import time
from functools import cached_property
from typing import Any, Dict, List

import logging_mp
import numpy as np

from lerobot.cameras import make_cameras_from_configs
from lerobot.robots.robot import Robot
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from .config import AstribotT1AioPyRobotConfig


logger = logging_mp.get_logger(__name__)


def load_astribot_sdk():
    try:
        from astribot_sdk import Astribot, AstribotCamera
    except ImportError as e:
        raise ImportError(
            "找不到 astribot_sdk。请先按本包 README 的「环境准备」安装厂商 wheel，"
            "并在启动 robodriver-run 前 source ~/astribot_examples/astribot_env.sh。"
        ) from e
    return Astribot, AstribotCamera


def _keep_frame_cached(topic_name, msg, width, height, array):
    """SDK 只缓存已订阅相机的图像，注册回调本身即为目的，不处理返回值。"""
    return None


class AstribotT1AioPyRobot(Robot):
    config_class = AstribotT1AioPyRobotConfig
    name = "astribott1_aio_py"

    def __init__(self, config: AstribotT1AioPyRobotConfig):
        super().__init__(config)
        self.config = config
        self.robot_type = config.type
        self.use_videos = config.use_videos
        self.microphones = config.microphones
        self.cameras = make_cameras_from_configs(config.cameras)
        self.connect_excluded_cameras: List[str] = []

        self._astribot = None
        self._camera = None
        self._cam_handles: Dict[str, Any] = {}
        self._joint_names: List[str] = []
        self._dof: Dict[str, int] = {}
        self._shutdown_done = False
        self.connected = False
        self.logs = {}

    # ========= 部件与关节名 =========

    def _sdk_part_names(self) -> List[str]:
        return [getattr(self._astribot, f"{part}_name") for part in self.config.record_parts]

    def _build_joint_index(self, positions: List[List[float]]) -> None:
        self._joint_names = []
        self._dof = {}
        for part, joints in zip(self.config.record_parts, positions):
            self._dof[part] = len(joints)
            self._joint_names += [f"{part}_joint{i + 1}" for i in range(len(joints))]
        logger.info(f"AstriBot 部件自由度: {self._dof}")

    def _require_connected(self) -> None:
        if not self.connected:
            raise DeviceNotConnectedError(f"{self} is not connected. You need to run `robot.connect()`.")

    # ========= features =========

    @property
    def _follower_motors_ft(self) -> Dict[str, type]:
        return {f"follower_{name}.pos": float for name in self._joint_names}

    @property
    def _leader_motors_ft(self) -> Dict[str, type]:
        return {f"leader_{name}.pos": float for name in self._joint_names}

    @property
    def _cameras_ft(self) -> Dict[str, tuple]:
        return {
            cam: (self.config.cameras[cam].height, self.config.cameras[cam].width, 3)
            for cam in self.cameras
            if cam in self.config.camera_name_map
        }

    @cached_property
    def observation_features(self) -> Dict[str, Any]:
        if not self._joint_names:
            raise RuntimeError("observation_features 需要在 connect() 之后访问（关节数量由机器人上报）")
        return {**self._follower_motors_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> Dict[str, Any]:
        if not self._joint_names:
            raise RuntimeError("action_features 需要在 connect() 之后访问（关节数量由机器人上报）")
        return self._leader_motors_ft

    # ========= connect / disconnect =========

    @property
    def is_connected(self) -> bool:
        return self.connected

    def connect(self, calibrate: bool = True) -> None:
        if self.connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        Astribot, AstribotCamera = load_astribot_sdk()
        start_time = time.perf_counter()

        self._astribot = Astribot(
            freq=self.config.freq,
            ip=self.config.ip,
            node_name=self.config.node_name,
            remote_wbc=self.config.remote_wbc,
        )
        self._build_joint_index(
            self._astribot.get_current_joints_position(names=self._sdk_part_names())
        )

        if self.config.move_to_home_on_connect:
            logger.info("移动 AstriBot 到 home 位")
            self._astribot.move_to_home(duration=self.config.home_duration)

        if self.config.camera_name_map:
            self._camera = AstribotCamera(ip=self.config.ip, timeout=self.config.camera_timeout)
            if not self._camera.activate_camera():
                raise RuntimeError("激活 AstriBot 相机服务失败，请确认 meta_camera 是否在机器人端运行")
            if not self._camera.wait_for_frame(timeout=self.config.camera_timeout):
                raise TimeoutError(
                    f"{self.config.camera_timeout} 秒内未收到相机数据，机器人可用相机: {self._camera.camera_names()}"
                )

            available = self._camera.camera_names()
            missing = [name for name in self.config.camera_name_map.values() if name not in available]
            if missing:
                raise RuntimeError(
                    f"配置的相机名 {missing} 在机器人上不存在，实际可用相机: {available}"
                )
            logger.info(f"AstriBot 相机就绪: {available}")

            for camera_name in self.config.camera_name_map.values():
                self._cam_handles[camera_name] = self._camera.register_image_callback(
                    camera_name, "color", _keep_frame_cached
                )

        self.connected = True
        self._shutdown_done = False
        # run.py 不会调用 robot.disconnect()，而 SDK 只靠 __del__ 收尾时会在进程退出处 SIGABRT
        atexit.register(self._shutdown_for_exit)
        logger.info(f"{self} 连接成功，耗时 {time.perf_counter() - start_time:.2f} 秒")

    def _shutdown_for_exit(self) -> None:
        if self.connected:
            self.disconnect()

    def disconnect(self) -> None:
        if self._shutdown_done:
            return
        self._shutdown_done = True

        if not self.connected:
            return

        if self._camera is not None:
            for handle in self._cam_handles.values():
                self._camera.unregister_callback(handle)
        self._cam_handles = {}

        if self._camera is not None:
            # 取过帧后必须先关相机再关 Astribot，否则 SDK 退出时 SIGABRT
            self._camera.deactivate_camera()
            self._camera.shutdown()
        if self._astribot is not None:
            self._astribot.release_control()
            self._astribot.shutdown()

        self._camera = None
        self._astribot = None
        self.connected = False

    # ========= 校准与配置 =========

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    # ========= 数据读写 =========

    def get_observation(self) -> Dict[str, Any]:
        self._require_connected()

        start = time.perf_counter()
        positions = self._astribot.get_current_joints_position(names=self._sdk_part_names())
        obs = {
            f"follower_{name}.pos": float(value)
            for name, value in zip(self._joint_names, np.concatenate([np.asarray(p, dtype=float) for p in positions]))
        }
        self.logs["read_joints_dt_s"] = time.perf_counter() - start

        if self._camera is not None:
            start = time.perf_counter()
            colors = self._camera.get_all_color()
            for image_key, camera_name in self.config.camera_name_map.items():
                if image_key in self.connect_excluded_cameras:
                    continue
                frame = colors.get(camera_name)
                if frame is None:
                    logger.warning(f"相机 {camera_name} 本帧无数据")
                    continue
                # 实测 AstriBot SDK 返回 BGR，而 LeRobot/推流按 RGB 约定，需翻转并保持连续内存
                obs[image_key] = np.ascontiguousarray(np.asarray(frame)[:, :, ::-1])
            self.logs["read_cameras_dt_s"] = time.perf_counter() - start

        return obs

    def get_action(self) -> Dict[str, Any]:
        """AIO 机型自带遥操作，desired 关节角即为本次采集要记录的 action。"""
        self._require_connected()

        positions = self._astribot.get_desired_joints_position(names=self._sdk_part_names())
        return {
            f"leader_{name}.pos": float(value)
            for name, value in zip(self._joint_names, np.concatenate([np.asarray(p, dtype=float) for p in positions]))
        }

    def send_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        self._require_connected()

        grouped: Dict[str, List[float]] = {part: [] for part in self.config.record_parts}
        for name in self._joint_names:
            key = f"leader_{name}.pos"
            if key not in action:
                raise KeyError(f"action 缺少 {key}")
            part = name.rsplit("_joint", 1)[0]
            grouped[part].append(float(action[key]))

        names = self._sdk_part_names()
        commands = [grouped[part] for part in self.config.record_parts]
        self._astribot.set_joints_position(names=names, position=commands, dt=1.0 / self.config.freq)

        return {name.replace("leader_", ""): value for name, value in action.items()}

    def __del__(self):
        if getattr(self, "connected", False):
            self.disconnect()
