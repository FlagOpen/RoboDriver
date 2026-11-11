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
from .ros_joints import AstribotJointNode


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
        self._joint_node = None
        self._shutdown_done = False
        self._control_lease = None
        self._control_warned = False
        self._crop_degraded: set = set()
        self.connected = False
        self.logs = {}

    # ========= 部件与关节名 =========

    def _sdk_part_names(self) -> List[str]:
        return [getattr(self._astribot, f"{part}_name") for part in self.config.record_parts]

    def _build_joint_index(self, dof: Dict[str, int]) -> None:
        self._joint_names = []
        self._dof = {}
        for part in self.config.record_parts:
            count = dof[part]
            self._dof[part] = count
            self._joint_names += [f"{part}_joint{i + 1}" for i in range(count)]
        logger.info(f"AstriBot 部件自由度: {self._dof}")

    def _sdk_dof(self) -> Dict[str, int]:
        positions = self._astribot.get_current_joints_position(names=self._sdk_part_names())
        return {part: len(p) for part, p in zip(self.config.record_parts, positions)}

    def _start_ros_joints(self) -> None:
        self._joint_node = AstribotJointNode(
            parts=self.config.record_parts,
            domain_id=self.config.ros_domain_id,
            obs_suffix=self.config.ros_obs_suffix,
            action_suffix=self.config.ros_action_suffix,
            node_name=self.config.ros_node_name,
            topic_prefix=self.config.ros_topic_prefix,
        )
        self._joint_node.start(wait_first_frame_s=self.config.ros_wait_first_frame_s)
        self._build_joint_index(self._joint_node.get_dof())

    def _require_connected(self) -> None:
        if not self.connected:
            raise DeviceNotConnectedError(f"{self} is not connected. You need to run `robot.connect()`.")

    # ========= 控制权 =========

    def _control_sdk_names(self) -> List[str]:
        parts = [p for p in self.config.record_parts if p in self.config.control_parts]
        return [getattr(self._astribot, f"{part}_name") for part in parts]

    def _control_owners(self) -> Dict[str, str]:
        try:
            state = self._astribot.get_control_rights_status().get("server_state") or {}
        except Exception as e:
            logger.warning(f"读取控制权状态失败: {e}")
            return {}
        return {
            name: f"{info.get('owner_name')}(idle={info.get('idle')},prio={info.get('priority_level')})"
            for name, info in state.items()
            if info.get("owner_name")
        }

    def _ensure_control(self) -> bool:
        """确保可以下发动作。takeover 模式下抢占控制权，passive 模式只提示。"""
        if self._control_lease is not None:
            return True

        names = self._control_sdk_names()
        if not names:
            logger.warning("control_parts 为空，send_action 不会下发任何部件")
            return False

        if self.config.control_mode != "takeover":
            if not self._control_warned:
                self._control_warned = True
                owners = self._control_owners()
                logger.warning(
                    f"未持有控制权，动作可能被忽略。当前占用: {owners}。"
                    f"如需回放/外部控制，请用 --robot.control_mode=takeover 启动（会抢占 VR 遥操作的控制权）"
                )
            return False

        owners = self._control_owners()
        busy = {k: v for k, v in owners.items() if k in names and "idle=False" in v}
        if busy and not self.config.force_takeover:
            raise RuntimeError(
                f"目标部件仍被非空闲租约占用: {busy}。确认现场无人操作后加 --robot.force_takeover=true 再试"
            )

        result = self._astribot.acquire_control(
            objects=names,
            priority=self.config.takeover_priority,
            sub_priority=self.config.takeover_sub_priority,
            reason="RoboDriver external control (replay/disturbance)",
        )
        if not result.get("granted"):
            raise RuntimeError(f"获取控制权失败: {result}")
        self._control_lease = result.get("lease_id")
        logger.warning(f"已接管控制权 {names}，lease={self._control_lease}（VR 遥操作将被顶掉）")
        return True

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
        # 关节来源：ROS 话题或 SDK。相机、控制权、send_action 始终走 SDK。
        if self.config.use_ros_joints:
            self._start_ros_joints()
        else:
            self._build_joint_index(self._sdk_dof())

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

            for camera_name in sorted(set(self.config.camera_name_map.values())):
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

        if self._joint_node is not None:
            self._joint_node.stop()
            self._joint_node = None

        if self._camera is not None:
            for handle in self._cam_handles.values():
                self._camera.unregister_callback(handle)
        self._cam_handles = {}

        if self._camera is not None:
            # 取过帧后必须先关相机再关 Astribot，否则 SDK 退出时 SIGABRT
            self._camera.deactivate_camera()
            self._camera.shutdown()
        if self._astribot is not None:
            if self._control_lease is not None:
                try:
                    self._astribot.release_control(objects=self._control_sdk_names())
                    logger.info(f"已释放控制权 lease={self._control_lease}")
                except Exception as e:
                    logger.warning(f"释放控制权失败: {e}")
                self._control_lease = None
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

    def _flatten_ros(self, per_part: Dict[str, List[float]]) -> List[float]:
        flat: List[float] = []
        for part in self.config.record_parts:
            values = per_part.get(part, [])
            if len(values) != self._dof.get(part, len(values)):
                raise RuntimeError(
                    f"{part} 的 ROS 关节数 {len(values)} 与索引 {self._dof.get(part)} 不一致，"
                    f"话题可能中途变了或部件未上报"
                )
            flat.extend(values)
        return flat

    def _crop_frame(self, image_key: str, rgb: np.ndarray) -> np.ndarray:
        """按 camera_crops 裁子图。

        头部相机双目拼接(mode=merge)时原始帧是两倍宽，需切出左右目；机器人端若
        没有下发 merge，帧就只有单目宽度，此时右目的裁剪起点会越界。帧尺寸恰好
        等于目标尺寸说明拿到的就是单目图，直接整帧返回（左右目内容相同）并告警一次，
        让采集能继续，而不是启动即崩。
        """
        crop = self.config.camera_crops.get(image_key)
        if crop is None:
            return rgb

        x, y, w, h = crop
        frame_h, frame_w = rgb.shape[:2]
        if y + h <= frame_h and x + w <= frame_w:
            return np.ascontiguousarray(rgb[y:y + h, x:x + w])

        if frame_w == w and frame_h == h:
            if image_key not in self._crop_degraded:
                self._crop_degraded.add(image_key)
                logger.warning(
                    f"{image_key} 裁剪区域 {crop} 超出原始帧 {(frame_h, frame_w)}，"
                    f"该相机当前未按双目拼接(mode=merge)输出，退化为整帧使用"
                )
            return rgb

        raise RuntimeError(
            f"{image_key} 裁剪区域 {crop} 超出原始帧 {(frame_h, frame_w)}，请检查 camera_crops"
        )

    def get_observation(self) -> Dict[str, Any]:
        self._require_connected()

        start = time.perf_counter()
        if self._joint_node is not None:
            positions = self._flatten_ros(self._joint_node.get_observation())
        else:
            positions = list(np.concatenate(
                [np.asarray(p, dtype=float) for p in
                 self._astribot.get_current_joints_position(names=self._sdk_part_names())]
            ))
        obs = {
            f"follower_{name}.pos": float(value)
            for name, value in zip(self._joint_names, positions)
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
                rgb = np.ascontiguousarray(np.asarray(frame)[:, :, ::-1])
                obs[image_key] = self._crop_frame(image_key, rgb)
            self.logs["read_cameras_dt_s"] = time.perf_counter() - start

        return obs

    def get_action(self) -> Dict[str, Any]:
        """AIO 机型自带遥操作，desired 关节角即为本次采集要记录的 action。"""
        self._require_connected()

        if self._joint_node is not None:
            positions = self._flatten_ros(self._joint_node.get_action())
        else:
            positions = list(np.concatenate(
                [np.asarray(p, dtype=float) for p in
                 self._astribot.get_desired_joints_position(names=self._sdk_part_names())]
            ))
        return {
            f"leader_{name}.pos": float(value)
            for name, value in zip(self._joint_names, positions)
        }

    def send_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        self._require_connected()

        parts = [p for p in self.config.record_parts if p in self.config.control_parts]
        grouped: Dict[str, List[float]] = {part: [] for part in parts}
        for name in self._joint_names:
            part = name.rsplit("_joint", 1)[0]
            if part not in grouped:
                continue
            key = f"leader_{name}.pos"
            if key not in action:
                raise KeyError(f"action 缺少 {key}")
            grouped[part].append(float(action[key]))

        if not parts:
            return {}

        self._ensure_control()
        names = [getattr(self._astribot, f"{part}_name") for part in parts]
        commands = [grouped[part] for part in parts]
        self._astribot.set_joints_position(names=names, position=commands, dt=1.0 / self.config.freq)

        return {name.replace("leader_", ""): value for name, value in action.items()}

    def __del__(self):
        if getattr(self, "connected", False):
            self.disconnect()
