import os
import threading
import time
from typing import Any, Dict, List, Optional, Set

import logging_mp
import numpy as np

logger = logging_mp.get_logger(__name__)

CONNECT_TIMEOUT_FRAME = 10
# action 话题无数据时的填充值；-1 与真实指令可区分
NO_ACTION_FILL = -1.0


def _load_sdk():
    try:
        from astribot_sdk import Astribot, AstribotCamera
    except ImportError as e:
        raise ImportError(
            "找不到 astribot_sdk。请先按 README 安装厂商 wheel，"
            "并在启动前 source ~/astribot_examples/astribot_env.sh。"
        ) from e
    return Astribot, AstribotCamera


class AstribotT1AioPyNode:
    """
    AstriBot T1 通信节点。

    封装 astribot_sdk 和 ROS2 订阅，对外暴露轮询缓存：
      recv_images   : Dict[str, np.ndarray]    image_key -> RGB 图像
      recv_follower : Dict[str, List[float]]   part -> 当前关节角
      recv_leader   : Dict[str, List[float]]   part -> 目标关节角
    """

    # 属于该节点订阅的所有话题（写死，不用配置）
    # obs: RobotJointState.position, BEST_EFFORT
    OBS_TOPICS = {
        "arm_left":       "/astribot_arm_left/joint_space_states",
        "arm_right":      "/astribot_arm_right/joint_space_states",
        "effector_left":  "/astribot_gripper_left/joint_space_states",
        "effector_right": "/astribot_gripper_right/joint_space_states",
        "head":           "/astribot_head/joint_space_states",
        "torso":          "/astribot_torso/joint_space_states",
        "chassis":        "/astribot_chassis/joint_space_states",
    }
    # action: RobotJointController.command, RELIABLE
    ACTION_TOPICS = {
        "arm_left":       "/astribot_arm_left/joint_space_command_forward",
        "arm_right":      "/astribot_arm_right/joint_space_command_forward",
        "effector_left":  "/astribot_gripper_left/joint_space_command_forward",
        "effector_right": "/astribot_gripper_right/joint_space_command_forward",
        "head":           "/astribot_head/joint_space_command_forward",
        "torso":          "/astribot_torso/joint_space_command_forward",
        "chassis":        "/astribot_chassis/joint_space_command_forward",
    }

    def __init__(
        self,
        record_parts: List[str],
        camera_name_map: Dict[str, str],
        camera_crops: Dict[str, tuple],
        ip: Optional[str] = None,
        node_name: str = "robodriver",
        ros_domain_id: int = 25,
        ros_wait_first_frame_s: float = 10.0,
        camera_timeout: float = 10.0,
        control_mode: str = "takeover",
        takeover_priority: int = 0,
        takeover_sub_priority: int = 0,
        force_takeover: bool = False,
        control_parts: Optional[List[str]] = None,
    ):
        self.record_parts = list(record_parts)
        self.camera_name_map = dict(camera_name_map)
        self.camera_crops = dict(camera_crops)
        self.ip = ip
        self.node_name = node_name
        self.ros_domain_id = ros_domain_id
        self.ros_wait_first_frame_s = ros_wait_first_frame_s
        self.camera_timeout = camera_timeout
        self.control_mode = control_mode
        self.takeover_priority = takeover_priority
        self.takeover_sub_priority = takeover_sub_priority
        self.force_takeover = force_takeover
        self.control_parts: List[str] = list(control_parts or [])

        self.lock = threading.Lock()

        # 轮询缓存
        self.recv_images: Dict[str, np.ndarray] = {}
        self.recv_images_status: Dict[str, int] = {}
        self.recv_follower: Dict[str, List[float]] = {}
        self.recv_follower_status: Dict[str, int] = {}
        self.recv_leader: Dict[str, List[float]] = {}
        self.recv_leader_status: Dict[str, int] = {}

        # 关节元数据
        self.dof: Dict[str, int] = {}

        # 内部句柄
        self._astribot = None
        self._camera = None
        self._ros_node = None
        self._spin_thread: Optional[threading.Thread] = None
        self._camera_thread: Optional[threading.Thread] = None

        self._ros_running = False
        self._camera_running = False
        self._control_lease = None
        self._control_warned = False
        self._crop_degraded: Set[str] = set()
        self._shutdown_done = False
        self.running = False

    # ==================== 生命周期 ====================

    def start(self) -> None:
        if self.running:
            return
        self.running = True

        Astribot, AstribotCamera = _load_sdk()

        # 1. SDK 机器人对象
        sdk_kwargs: Dict[str, Any] = {"node_name": self.node_name}
        if self.ip:
            sdk_kwargs["ip"] = self.ip
        self._astribot = Astribot(**sdk_kwargs)
        logger.info("AstriBot SDK 已连接")

        # 2. ROS2 关节订阅
        self._start_ros()

        # 3. 相机
        self._camera = AstribotCamera()
        if not self._camera.activate_camera():
            raise RuntimeError("AstribotCamera.activate_camera() 返回 False，请确认机器人端 meta_camera 服务正在运行")
        if not self._camera.wait_for_frame(timeout=self.camera_timeout):
            raise RuntimeError(f"{self.camera_timeout}s 内未收到相机数据，请检查网络连接")
        for camera_name in set(self.camera_name_map.values()):
            self._camera.register_image_callback(camera_name, "color", self._noop_image_cb)
        logger.info(f"相机已激活，可用: {self._camera.camera_names()}")

        self._camera_running = True
        self._camera_thread = threading.Thread(target=self._camera_loop, daemon=True, name="astribot_camera")
        self._camera_thread.start()

    def stop(self) -> None:
        if self._shutdown_done:
            return
        self._shutdown_done = True
        self.running = False

        self._camera_running = False
        if self._camera_thread is not None:
            self._camera_thread.join(timeout=2.0)
            self._camera_thread = None

        self._ros_running = False
        if self._spin_thread is not None:
            self._spin_thread.join(timeout=1.0)
            self._spin_thread = None
        if self._ros_node is not None:
            try:
                self._ros_node.destroy_node()
            except Exception as e:
                logger.warning(f"destroy_node 失败: {e}")
            self._ros_node = None

        if self._control_lease is not None and self._astribot is not None:
            try:
                self._astribot.release_control()
            except Exception as e:
                logger.warning(f"释放控制权失败: {e}")
            self._control_lease = None

        if self._camera is not None:
            try:
                self._camera.deactivate_camera()
                self._camera.shutdown()
            except Exception as e:
                logger.warning(f"关闭相机失败: {e}")
            self._camera = None

        if self._astribot is not None:
            try:
                self._astribot.shutdown()
            except Exception as e:
                logger.warning(f"关闭 Astribot SDK 失败: {e}")
            self._astribot = None

        logger.info("AstribotT1AioPyNode 已停止")


    # ==================== ROS2 订阅 ====================

    def _start_ros(self) -> None:
        os.environ["ROS_DOMAIN_ID"] = str(self.ros_domain_id)

        import rclpy
        from rclpy.node import Node as ROS2Node
        from rclpy.qos import (
            qos_profile_sensor_data,
            QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy,
        )
        from astribot_msgs.msg import RobotJointState, RobotJointController

        if not rclpy.ok():
            rclpy.init()

        self._ros_node = ROS2Node("robodriver_astribot")

        best_effort = qos_profile_sensor_data
        reliable = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        for part in self.record_parts:
            obs_topic = self.OBS_TOPICS.get(part)
            act_topic = self.ACTION_TOPICS.get(part)
            if obs_topic is None:
                logger.warning(f"[ROS] 未知部件 {part}，跳过订阅")
                continue
            self._ros_node.create_subscription(RobotJointState, obs_topic, self._make_obs_cb(part), best_effort)
            self._ros_node.create_subscription(RobotJointController, act_topic, self._make_action_cb(part), reliable)
            logger.info(f"[ROS] {part}: obs={obs_topic}  action={act_topic}")

        self._ros_running = True
        self._spin_thread = threading.Thread(target=self._spin_loop, daemon=True, name="astribot_ros_spin")
        self._spin_thread.start()
        self._wait_first_obs(self.ros_wait_first_frame_s)

    def _spin_loop(self) -> None:
        import rclpy
        try:
            while self._ros_running and rclpy.ok():
                rclpy.spin_once(self._ros_node, timeout_sec=0.001)
        except Exception as e:
            logger.error(f"[ROS] spin 异常: {e}")

    def _wait_first_obs(self, timeout_s: float) -> None:
        deadline = time.perf_counter() + timeout_s
        while time.perf_counter() < deadline:
            with self.lock:
                got = set(self.recv_follower.keys())
            if got >= set(self.record_parts):
                break
            time.sleep(0.02)
        with self.lock:
            missing = sorted(set(self.record_parts) - set(self.recv_follower.keys()))
        if missing:
            raise TimeoutError(
                f"{timeout_s}s 内未收齐 obs 首帧，缺: {missing}。"
                f"请确认 ROS_DOMAIN_ID={self.ros_domain_id} 与机器人在同一网络。"
            )
        logger.info(f"[ROS] 各部件自由度: {self.dof}")

    def _make_obs_cb(self, part: str):
        def obs_cb(msg):
            self._on_obs(part, msg)
        obs_cb.__name__ = f"_obs_cb_{part}"
        return obs_cb

    def _make_action_cb(self, part: str):
        def action_cb(msg):
            self._on_action(part, msg)
        action_cb.__name__ = f"_action_cb_{part}"
        return action_cb

    def _on_obs(self, part: str, msg) -> None:
        positions = list(msg.position)
        with self.lock:
            self.recv_follower[part] = positions
            self.recv_follower_status[part] = CONNECT_TIMEOUT_FRAME
            self.dof[part] = len(positions)

    def _on_action(self, part: str, msg) -> None:
        # command_forward.command = [position, velocity, torque] 拼接，取前 1/3
        command = list(msg.command)
        n = len(command) // 3 if len(command) % 3 == 0 else len(command)
        with self.lock:
            self.recv_leader[part] = command[:n]
            self.recv_leader_status[part] = CONNECT_TIMEOUT_FRAME


    # ==================== 相机 ====================

    @staticmethod
    def _noop_image_cb(*args) -> None:
        """register_image_callback 要求一个回调，实际取帧走 get_all_color()。"""
        pass

    def _crop_frame(self, image_key: str, rgb: np.ndarray) -> np.ndarray:
        crop = self.camera_crops.get(image_key)
        if crop is None:
            return rgb
        x, y, w, h = crop
        fh, fw = rgb.shape[:2]
        if y + h <= fh and x + w <= fw:
            return np.ascontiguousarray(rgb[y:y + h, x:x + w])
        if fw == w and fh == h:
            if image_key not in self._crop_degraded:
                self._crop_degraded.add(image_key)
                logger.warning(f"{image_key} 裁剪区域 {crop} 超出帧 {(fh, fw)}，退化为整帧")
            return rgb
        raise RuntimeError(f"{image_key} 裁剪区域 {crop} 超出帧 {(fh, fw)}")

    def _camera_loop(self) -> None:
        while self._camera_running:
            try:
                if self._camera is None:
                    break
                colors = self._camera.get_all_color()
                for image_key, camera_name in self.camera_name_map.items():
                    frame = colors.get(camera_name)
                    if frame is None:
                        continue
                    # SDK 返回 BGR，转为 RGB
                    rgb = np.ascontiguousarray(np.asarray(frame)[:, :, ::-1])
                    cropped = self._crop_frame(image_key, rgb)
                    with self.lock:
                        self.recv_images[image_key] = cropped
                        self.recv_images_status[image_key] = CONNECT_TIMEOUT_FRAME
            except Exception as e:
                logger.warning(f"[相机] 取帧异常: {e}")
            time.sleep(0.001)

    # ==================== 控制权 ====================

    def ensure_control(self) -> bool:
        if self._control_lease is not None:
            return True
        names = [
            getattr(self._astribot, f"{p}_name")
            for p in self.record_parts
            if p in self.control_parts
        ]
        if not names:
            logger.warning("control_parts 为空，send_action 不会下发任何部件")
            return False
        if self.control_mode != "takeover":
            if not self._control_warned:
                self._control_warned = True
                logger.warning("未持有控制权，动作可能被忽略。如需回放请用 --robot.control_mode=takeover")
            return False
        result = self._astribot.acquire_control(
            objects=names,
            priority=self.takeover_priority,
            sub_priority=self.takeover_sub_priority,
            reason="RoboDriver external control",
        )
        if not result.get("granted"):
            raise RuntimeError(f"获取控制权失败: {result}")
        self._control_lease = result.get("lease_id")
        logger.warning(f"已接管控制权 {names}，lease={self._control_lease}")
        return True

    # ==================== 下发动作 ====================

    def send_joints(self, grouped: Dict[str, List[float]], control_parts: List[str], freq: float) -> None:
        if self._astribot is None:
            raise RuntimeError("AstribotT1AioPyNode 未启动")
        self.ensure_control()
        names = [getattr(self._astribot, f"{p}_name") for p in control_parts]
        commands = [grouped[p] for p in control_parts]
        self._astribot.set_joints_position(names=names, position=commands, dt=1.0 / freq)

