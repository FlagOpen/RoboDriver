import os
from pathlib import Path
import json


# ===========================================================
# Utils
# ===========================================================
def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def make_output_dir(root: str, robot_name: str) -> Path:
    """Create output folder like ../../robots/{robot_name}"""
    base = Path(root).resolve() / robot_name
    ensure_dir(base)
    return base


def load_json(path: Path):
    with open(path, "r") as f:
        return json.load(f)


# ===========================================================
# CamelCase
# ===========================================================
def to_camel(name: str) -> str:
    parts = name.replace("-", "_").split("_")
    return "".join(p.capitalize() for p in parts)

# ===========================================================
# Dict
# ===========================================================
def build_config_dicts(robot_json: dict):
    """
    从 JSON 里解析出：
        leader_motors_dict: { comp_id: { joint_name: [id, type], ... }, ... }
        follower_motors_dict: 同上
        cameras_dict: { cam_id: {width, height, fps}, ... }
        use_videos: bool

    规则：
      - arm/ros2_joints_or_pose 组件：
          根据 params.group:
              "action"      -> leader_motors[组件id] = motors
              "observation" -> follower_motors[组件id] = motors
          每个组件单独成一个子 dict，不再 merge 成一个大 dict
      - camera/image 组件统一放到 cameras 里
    """

    leader_motors: dict[str, dict] = {}
    follower_motors: dict[str, dict] = {}
    cameras: dict[str, dict] = {}

    for comp in robot_json.get("components", []):
        ctype = comp.get("type")
        params = comp.get("params", {})
        cid = comp.get("id")

        # --------- 关节 / 电机类组件（用 group 区分 leader/follower）---------
        if ctype == "arm/ros2_joints_or_pose":
            group = (params.get("group") or "").lower()
            motors = comp.get("outputs_info", {}).get("joints", {}).get("motors", {})

            if not cid:
                continue

            if group == "action":
                # 每个 action 组件单独存成一个子 dict
                leader_motors[cid] = motors
            elif group == "observation":
                # 每个 observation 组件单独存成一个子 dict
                follower_motors[cid] = motors

        # --------------------- 相机组件 ---------------------
        if ctype == "camera/image":
            cam_id = comp.get("id")
            p = comp.get("params", {})
            if cam_id is not None:
                cameras[cam_id] = {
                    "width": p.get("width"),
                    "height": p.get("height"),
                    # JSON 里 period=30，这里直接当作 fps 用
                    "fps": p.get("period", 30),
                }

    if not leader_motors:
        raise ValueError("未在 JSON 中找到 group='action' 的 arm/ros2_joints_or_pose 组件")
    if not follower_motors:
        raise ValueError("未在 JSON 中找到 group='observation' 的 arm/ros2_joints_or_pose 组件")

    use_videos = robot_json.get("use_videos", False)

    return leader_motors, follower_motors, cameras, use_videos


def build_node_config_dict(robot_json: dict) -> dict:
    """
    从 JSON 中提取 node.py 需要的配置，生成 NODE_CONFIG：

    {
        "leader_joint_topics": {
            "<comp_id>": {"topic": "...", "msg": "..."},
            ...
        },
        "follower_joint_topics": {
            "<comp_id>": {"topic": "...", "msg": "..."},
            ...
        },
        "camera_topics": {
            "<cam_id>": {"topic": "...", "msg": "..."},
            ...
        },
    }

    规则：
      - 相机组件：type == "camera/image" -> camera_topics
      - 其它组件（非 camera），根据 params.group：
          group == "action"      -> leader_joint_topics
          group == "observation" -> follower_joint_topics
      - msg 字段最终保存类名，比如 "JointState" / "Pose" / "Odometry" / "Image"
        如果 JSON 里是 "sensor_msgs/JointState" 这种，会自动取最后一段。
    """

    leader_joint_topics: dict[str, dict] = {}
    follower_joint_topics: dict[str, dict] = {}
    camera_topics: dict[str, dict] = {}

    for comp in robot_json.get("components", []):
        ctype = comp.get("type")
        params = comp.get("params", {}) or {}
        cid = comp.get("id")

        # ---------------- 相机组件 ----------------
        if ctype == "camera/image":
            if not cid:
                continue
            topic = (params.get("topic") or "").strip()
            raw_msg = (params.get("msgs") or "").strip()
            if not topic or not raw_msg:
                continue

            # "sensor_msgs/Image" -> "Image"
            if "/" in raw_msg:
                msg_type = raw_msg.split("/")[-1]
            else:
                msg_type = raw_msg

            camera_topics[cid] = {"topic": topic, "msg": msg_type}
            continue

        # ---------------- 非相机组件：按 group 分 leader / follower ----------------
        group = (params.get("group") or "").lower()
        topic = (params.get("topic") or "").strip()
        raw_msg = (params.get("msgs") or "").strip()

        if not cid or not group or not topic or not raw_msg:
            continue

        if "/" in raw_msg:
            msg_type = raw_msg.split("/")[-1]
        else:
            msg_type = raw_msg

        entry = {"topic": topic, "msg": msg_type}

        if group == "action":
            leader_joint_topics[cid] = entry
        elif group == "observation":
            follower_joint_topics[cid] = entry
        # 其它 group 忽略

    if not leader_joint_topics:
        raise ValueError("JSON 中没有找到 group='action' 的组件，无法生成 leader_joint_topics")
    if not follower_joint_topics:
        raise ValueError("JSON 中没有找到 group='observation' 的组件，无法生成 follower_joint_topics")

    return {
        "leader_joint_topics": leader_joint_topics,
        "follower_joint_topics": follower_joint_topics,
        "camera_topics": camera_topics,
    }







# ===========================================================
# Generators (7 functions)
# ===========================================================
def generate_lerobot_init(base_dir: Path, robot_name: str):
    target_dir = base_dir / f"lerobot_robot_{robot_name}_ros"
    ensure_dir(target_dir)

    content = f"from robodriver_robot_{robot_name} import *\n"

    with open(target_dir / "__init__.py", "w") as f:
        f.write(content)


def generate_pyproject(base_dir: Path, robot_name: str):
    content = f"""[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.build_meta"

[project]
name = "robodriver_robot_{robot_name}"
version = "0.1.0"
readme = "README.md"
requires-python = ">=3.10"
license = {{ text = "Apache-2.0" }}
keywords = ["robotics", "lerobot", "{robot_name}"]

dependencies = [
    "logging_mp",
    "rclpy",
    "numpy",
    "pyzmq",
    "opencv-python",
    "cv_bridge",
]

[tool.setuptools.packages.find]
include = ["robodriver_robot_{robot_name}"]
"""

    with open(base_dir / "pyproject.toml", "w") as f:
        f.write(content)



def generate_robot_pkg_init(base_dir: Path, robot_name: str):
    target_dir = base_dir / f"robodriver_robot_{robot_name}"
    ensure_dir(target_dir)

    CamelName = to_camel(robot_name)

    content = (
        f"from .robot import {CamelName}Robot\n"
        f"from .config import {CamelName}RobotConfig\n"
    )

    with open(target_dir / "__init__.py", "w") as f:
        f.write(content)


def generate_robot_py(base_dir: Path, robot_name: str):
    """
    生成 robodriver_robot_{robot_name}/robot.py

    注意：这里假设 config.py 里有 {CamelName}RobotConfig，
          node.py 里有 {CamelName}Node，
          且 node 使用的是最终版接口（recv_follower / recv_leader / recv_images）。
    """

    CamelName = to_camel(robot_name)

    template = '''import time
from functools import cached_property
from typing import Any

import numpy as np
import rclpy
import logging_mp

from lerobot.cameras import make_cameras_from_configs
from lerobot.utils.errors import DeviceNotConnectedError, DeviceAlreadyConnectedError
from lerobot.robots.robot import Robot

from .config import __CAMEL_NAME__RobotConfig
from .node import __CAMEL_NAME__Node


logger = logging_mp.get_logger(__name__)


class __CAMEL_NAME__Robot(Robot):
    config_class = __CAMEL_NAME__RobotConfig
    name = "__ROBOT_NAME__"

    def __init__(self, config: __CAMEL_NAME__RobotConfig):
        rclpy.init()
        super().__init__(config)
        self.config = config
        self.robot_type = self.config.type
        self.use_videos = self.config.use_videos
        self.microphones = self.config.microphones

        # 这里的 leader_motors / follower_motors 可以是按组件分组的 dict
        # （比如 {"leader_arm": {...}, "left_arm": {...}}）
        self.leader_motors = config.leader_motors
        self.follower_motors = config.follower_motors
        self.cameras = make_cameras_from_configs(self.config.cameras)

        self.connect_excluded_cameras = ["image_pika_pose"]

        self.robot_ros2_node = __CAMEL_NAME__Node()
        self.robot_ros2_node.start()

        self.connected = False
        self.logs = {}

    # ========= features =========

    @property
    def _follower_motors_ft(self) -> dict[str, type]:
        return {
            f"follower_{joint_name}.pos": float
            for comp_name, joints in self.follower_motors.items()
            for joint_name in joints.keys()
        }
    
    @property
    def _leader_motors_ft(self) -> dict[str, type]:
        return {
            f"leader_{joint_name}.pos": float
            for comp_name, joints in self.leader_motors.items()
            for joint_name in joints.keys()
        }

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {
            cam: (
                self.config.cameras[cam].height,
                self.config.cameras[cam].width,
                3,
            )
            for cam in self.cameras
        }

    @cached_property
    def observation_features(self) -> dict[str, Any]:
        return {**self._follower_motors_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, Any]:
        return self._leader_motors_ft

    @property
    def is_connected(self) -> bool:
        return self.connected

    # ========= connect / disconnect =========

    def connect(self):
        timeout = 20
        start_time = time.perf_counter()

        if self.connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        # 约定：node 里有 recv_images / recv_follower / recv_leader
        conditions = [
            # 摄像头图像
            (
                lambda: all(
                    name in self.robot_ros2_node.recv_images
                    for name in self.cameras
                    if name not in self.connect_excluded_cameras
                ),
                lambda: [
                    name
                    for name in self.cameras
                    if name not in self.robot_ros2_node.recv_images
                    and name not in self.connect_excluded_cameras
                ],
                "等待摄像头图像超时",
            ),
            # 主臂
            (
                lambda: all(
                    any(name in key for key in self.robot_ros2_node.recv_leader)
                    for name in self.leader_motors
                ),
                lambda: [
                    name
                    for name in self.leader_motors
                    if not any(name in key for key in self.robot_ros2_node.recv_leader)
                ],
                "等待主臂数据超时",
            ),
            # 从臂
            (
                lambda: all(
                    any(name in key for key in self.robot_ros2_node.recv_follower)
                    for name in self.follower_motors
                ),
                lambda: [
                    name
                    for name in self.follower_motors
                    if not any(name in key for key in self.robot_ros2_node.recv_follower)
                ],
                "等待从臂数据超时",
            ),
        ]

        completed = [False] * len(conditions)

        while True:
            for i, (cond, _get_missing, _msg) in enumerate(conditions):
                if not completed[i] and cond():
                    completed[i] = True

            if all(completed):
                break

            if time.perf_counter() - start_time > timeout:
                failed_messages = []
                for i, (cond, get_missing, base_msg) in enumerate(conditions):
                    if completed[i]:
                        continue

                    missing = get_missing()
                    if cond() or not missing:
                        completed[i] = True
                        continue

                    if i == 0:
                        received = [
                            name
                            for name in self.cameras
                            if name not in missing
                        ]
                    elif i == 1:
                        received = [
                            name
                            for name in self.leader_motors
                            if name not in missing
                        ]
                    else:
                        received = [
                            name
                            for name in self.follower_motors
                            if name not in missing
                        ]

                    msg = (
                        f"{base_msg}: 未收到 [{', '.join(missing)}]; "
                        f"已收到 [{', '.join(received)}]"
                    )
                    failed_messages.append(msg)

                if not failed_messages:
                    break

                raise TimeoutError(
                    f"连接超时，未满足的条件: {{'; '.join(failed_messages)}}"
                )

            time.sleep(0.01)

        # 成功日志
        success_messages = []

        if conditions[0][0]():
            cam_received = [
                name
                for name in self.cameras
                if name in self.robot_ros2_node.recv_images
                and name not in self.connect_excluded_cameras
            ]
            success_messages.append(f"摄像头: {', '.join(cam_received)}")

        if conditions[1][0]():
            leader_received = [
                name
                for name in self.leader_motors
                if any(name in key for key in self.robot_ros2_node.recv_leader)
            ]
            success_messages.append(f"主臂数据: {', '.join(leader_received)}")

        if conditions[2][0]():
            follower_received = [
                name
                for name in self.follower_motors
                if any(name in key for key in self.robot_ros2_node.recv_follower)
            ]
            success_messages.append(f"从臂数据: {', '.join(follower_received)}")

        log_message = "\\n[连接成功] 所有设备已就绪:\\n"
        log_message += "\\n".join(f"  - {msg}" for msg in success_messages)
        log_message += f"\\n  总耗时: {time.perf_counter() - start_time:.2f} 秒\\n"
        logger.info(log_message)

        self.connected = True

    def disconnect(self):
        if not self.connected:
            raise DeviceNotConnectedError()
        self.connected = False

    def __del__(self):
        if getattr(self, "connected", False):
            self.disconnect()

    # ========= calibrate / configure =========

    def calibrate(self):
        pass

    def configure(self):
        pass

    @property
    def is_calibrated(self):
        return True

    # ========= obs / action =========

    def get_observation(self) -> dict[str, Any]:
        if not self.connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        start = time.perf_counter()
        obs_dict: dict[str, Any] = {}

        # ---- 逐组件展开，然后逐 joint 填入 ----
        for comp_name, joints in self.follower_motors.items():

            # node 中按组件名存放关节数组，例如：
            # self.recv_follower["follower_arm"] = np.array([... 6 joints ...])
            vec = self.robot_ros2_node.recv_follower.get(comp_name)
            if vec is None:
                continue

            # joints.keys() = ["joint_0", ..., "joint_5"]
            joint_names = list(joints.keys())

            for idx, joint in enumerate(joint_names):
                if idx >= len(vec):
                    break
                obs_dict[f"follower_{joint}.pos"] = float(vec[idx])

        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read follower state: {dt_ms:.1f} ms")

        # ---- 摄像头图像保持不变 ----
        for cam_key, _cam in self.cameras.items():
            start = time.perf_counter()
            for name, val in self.robot_ros2_node.recv_images.items():
                if cam_key == name or cam_key in name:
                    obs_dict[cam_key] = val
                    break
            dt_ms = (time.perf_counter() - start) * 1e3
            logger.debug(f"{self} read {cam_key}: {dt_ms:.1f} ms")

        return obs_dict
    
    def get_action(self) -> dict[str, Any]:
        if not self.connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        start = time.perf_counter()
        act_dict: dict[str, Any] = {}

        # ---- 逐组件展开，然后逐 joint 填入 ----
        for comp_name, joints in self.leader_motors.items():

            # node 中按组件名存放关节数组
            vec = self.robot_ros2_node.recv_leader.get(comp_name)
            if vec is None:
                continue

            joint_names = list(joints.keys())

            for idx, joint in enumerate(joint_names):
                if idx >= len(vec):
                    break
                act_dict[f"leader_{joint}.pos"] = float(vec[idx])

        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read action: {dt_ms:.1f} ms")

        return act_dict

    # ========= send_action =========

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        """The provided action is expected to be a vector."""
        if not self.is_connected:
            raise DeviceNotConnectedError(
                f"{self} is not connected. You need to run `robot.connect()`."
            )

        goal_joint = [val for _key, val in action.items()]
        goal_joint_numpy = np.array(goal_joint, dtype=np.float32)

        # node 需要实现 ros2_send("action_joint", numpy_array)
        self.robot_ros2_node.ros2_send("action_joint", goal_joint_numpy)

        return {f"{arm_motor}.pos": val for arm_motor, val in action.items()}
'''

    content = (
        template
        .replace("__CAMEL_NAME__", CamelName)
        .replace("__ROBOT_NAME__", robot_name)
    )

    target_dir = base_dir / f"robodriver_robot_{robot_name}"
    ensure_dir(target_dir)
    (target_dir / "robot.py").write_text(content, encoding="utf-8")






def generate_node_py(base_dir: Path, robot_name: str, robot_json: dict):
    """
    综合函数：
      - 基于 JSON 构建 NODE_CONFIG 字典
      - 生成 node.py 源码
      - 写入 robodriver_robot_{robot_name}/node.py
    """

    # 1. 构造 NODE_CONFIG
    node_config_dict = build_node_config_dict(robot_json)
    node_config_code = json.dumps(node_config_dict, indent=4)

    # 2. 计算 CamelCase 类名，例如 "so101_aio_ros2" -> "So101AioRos2"
    def _to_camel(name: str) -> str:
        parts = name.replace("-", "_").split("_")
        return "".join(p.capitalize() for p in parts)

    CamelName = _to_camel(robot_name)

    # 3. 模板（就是你给的最终版本，把几个地方换成占位符）
    template = '''# robodriver_robot___ROBOT_NAME__/node.py

import threading
from typing import Dict

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, JointState
from geometry_msgs.msg import Pose
from nav_msgs.msg import Odometry

import logging_mp

logger = logging_mp.get_logger(__name__)

CONNECT_TIMEOUT_FRAME = 10


# 由生成脚本根据 JSON 自动生成
NODE_CONFIG = __NODE_CONFIG__


class __CAMEL_NAME__Node(Node):
    """
    ROS2 → 本地数据存储（无 ZMQ，无 Dora）
    leader / follower / camera 多 topic，按 JSON 配置自动订阅。
    """

    def __init__(
        self,
        leader_joint_topics: Dict[str, Dict[str, str]] = NODE_CONFIG["leader_joint_topics"],
        follower_joint_topics: Dict[str, Dict[str, str]] = NODE_CONFIG["follower_joint_topics"],
        camera_topics: Dict[str, Dict[str, str]] = NODE_CONFIG["camera_topics"],
    ):
        super().__init__("__ROBOT_NAME___direct")

        # ---- 从参数 / NODE_CONFIG 中拿配置 ----
        self.leader_joint_cfgs = leader_joint_topics or {}
        self.follower_joint_cfgs = follower_joint_topics or {}
        self.camera_cfgs = camera_topics or {}

        if not self.leader_joint_cfgs:
            raise RuntimeError("leader_joint_topics is empty")
        if not self.follower_joint_cfgs:
            raise RuntimeError("follower_joint_topics is empty")

        # 相机 topic 简化一个 name -> topic 的 dict
        self.camera_topics: Dict[str, str] = {
            name: info["topic"] for name, info in self.camera_cfgs.items()
        }

        # ---- 各种缓存 ----
        self.recv_images: Dict[str, np.ndarray] = {}
        self.recv_images_status: Dict[str, int] = {}

        self.recv_follower: Dict[str, float] = {}
        self.recv_follower_status: Dict[str, int] = {}

        self.recv_leader: Dict[str, float] = {}
        self.recv_leader_status: Dict[str, int] = {}

        self.lock = threading.Lock()
        self.running = False


        # ---- follower side: 订阅所有 follower_joint_topics ----
        for comp_name, cfg in self.follower_joint_cfgs.items():
            topic = cfg["topic"]
            msg_name = cfg.get("msg", "JointState")

            if msg_name == "JointState":
                msg_cls = JointState
                callback = lambda msg, cname=comp_name: self._joint_callback_follower(
                    cname, msg
                )
            elif msg_name == "Pose":
                msg_cls = Pose
                callback = lambda msg, cname=comp_name: self._pose_callback_follower(
                    cname, msg
                )
            elif msg_name == "Odometry":
                msg_cls = Odometry
                callback = lambda msg, cname=comp_name: self._odom_callback_follower(
                    cname, msg
                )
            else:
                raise RuntimeError(f"Unsupported follower msg type: {msg_name}")

            self.create_subscription(
                msg_cls,
                topic,
                callback,
                10,
            )
            logger.info(
                f"[Direct] Follower subscriber '{comp_name}' at {topic} ({msg_name})"
            )

        # ---- leader side: 订阅所有 leader_joint_topics ----
        for comp_name, cfg in self.leader_joint_cfgs.items():
            topic = cfg["topic"]
            msg_name = cfg.get("msg", "JointState")

            if msg_name == "JointState":
                msg_cls = JointState
                callback = lambda msg, cname=comp_name: self._joint_callback_leader(
                    cname, msg
                )
            elif msg_name == "Pose":
                msg_cls = Pose
                callback = lambda msg, cname=comp_name: self._pose_callback_leader(
                    cname, msg
                )
            elif msg_name == "Odometry":
                msg_cls = Odometry
                callback = lambda msg, cname=comp_name: self._odom_callback_leader(
                    cname, msg
                )
            else:
                raise RuntimeError(f"Unsupported leader msg type: {msg_name}")

            self.create_subscription(
                msg_cls,
                topic,
                callback,
                10,
            )
            logger.info(
                f"[Direct] Leader subscriber '{comp_name}' at {topic} ({msg_name})"
            )

        # ---- cameras: 订阅所有 camera_topics（目前只支持 Image）----
        self.camera_subs = []
        for cam_name, cfg in self.camera_cfgs.items():
            topic = cfg["topic"]
            msg_name = cfg.get("msg", "Image")

            if msg_name != "Image":
                raise RuntimeError(f"Unsupported camera msg type: {msg_name}")

            sub = self.create_subscription(
                Image,
                topic,
                lambda msg, cname=cam_name: self._image_callback(cname, msg),
                10,
            )
            self.camera_subs.append(sub)
            logger.info(f"[Direct] Camera '{cam_name}' subscribed: {topic} ({msg_name})")

        logger.info("[Direct] READY (ROS2 callbacks active).")

    # ======================
    # callbacks
    # ======================

    def _image_callback(self, cam_name: str, msg: Image):
        try:
            with self.lock:
                event_id = f"{cam_name}"

                data = np.frombuffer(msg.data, dtype=np.uint8)
                h, w = msg.height, msg.width
                encoding = msg.encoding.lower()

                frame = None
                try:
                    if encoding == "bgr8":
                        frame = data.reshape((h, w, 3))
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    elif encoding == "rgb8":
                        frame = data.reshape((h, w, 3))
                    elif encoding in ["jpeg", "jpg", "png", "bmp", "webp"]:
                        frame = cv2.imdecode(data, cv2.IMREAD_COLOR)
                    
                except Exception as e:
                    logger.error(f"Image decode error ({encoding}): {e}")

                if frame is not None:
                    self.recv_images[event_id] = frame
                    self.recv_images_status[event_id] = CONNECT_TIMEOUT_FRAME

        except Exception as e:
            logger.error(f"Image callback error ({cam_name}): {e}")

    # ---------- JointState ----------

    def _joint_callback_follower(self, comp_name: str, msg: JointState):
        try:
            with self.lock:
                event_id = comp_name
                values = np.array(msg.position, dtype=float)  # shape (6,)
                self.recv_follower[event_id] = values
                self.recv_follower_status[event_id] = CONNECT_TIMEOUT_FRAME
        except Exception as e:
            logger.error(f"Joint callback error (follower:{comp_name}): {e}")

    def _joint_callback_leader(self, comp_name: str, msg: JointState):
        try:
            with self.lock:
                event_id = comp_name
                values = np.array(msg.position, dtype=float)
                self.recv_leader[event_id] = values
                self.recv_leader_status[event_id] = CONNECT_TIMEOUT_FRAME
        except Exception as e:
            logger.error(f"Joint callback error (leader:{comp_name}): {e}")

        # ---------- Pose ----------

    def _pose_callback_follower(self, comp_name: str, msg: Pose):
        """
        follower 侧 Pose 回调
        合并 position + orientation -> [px, py, pz, qx, qy, qz, qw]
        """
        try:
            with self.lock:
                vec = np.array(
                    [
                        msg.position.x,
                        msg.position.y,
                        msg.position.z,
                        msg.orientation.x,
                        msg.orientation.y,
                        msg.orientation.z,
                        msg.orientation.w,
                    ],
                    dtype=float,
                )
                event_id = f"{comp_name}"
                self.recv_follower[event_id] = vec
                self.recv_follower_status[event_id] = CONNECT_TIMEOUT_FRAME
        except Exception as e:
            logger.error(f"Pose callback error (follower:{comp_name}): {e}")

    def _pose_callback_leader(self, comp_name: str, msg: Pose):
        """
        leader 侧 Pose 回调
        合并 position + orientation -> [px, py, pz, qx, qy, qz, qw]
        """
        try:
            with self.lock:
                vec = np.array(
                    [
                        msg.position.x,
                        msg.position.y,
                        msg.position.z,
                        msg.orientation.x,
                        msg.orientation.y,
                        msg.orientation.z,
                        msg.orientation.w,
                    ],
                    dtype=float,
                )
                event_id = f"{comp_name}"
                self.recv_leader[event_id] = vec
                self.recv_leader_status[event_id] = CONNECT_TIMEOUT_FRAME
        except Exception as e:
            logger.error(f"Pose callback error (leader:{comp_name}): {e}")

    # ---------- Odometry ----------

    def _odom_callback_follower(self, comp_name: str, msg: Odometry):
        """
        follower 侧 Odometry 回调
        合并:
          - pose.position        (3)
          - pose.orientation     (4)
          - twist.linear         (3)
          - twist.angular        (3)
        -> 13 维向量
        """
        try:
            with self.lock:
                vec = np.array(
                    [
                        # position
                        msg.pose.pose.position.x,
                        msg.pose.pose.position.y,
                        msg.pose.pose.position.z,
                        # orientation
                        msg.pose.pose.orientation.x,
                        msg.pose.pose.orientation.y,
                        msg.pose.pose.orientation.z,
                        msg.pose.pose.orientation.w,
                        # linear velocity
                        msg.twist.twist.linear.x,
                        msg.twist.twist.linear.y,
                        msg.twist.twist.linear.z,
                        # angular velocity
                        msg.twist.twist.angular.x,
                        msg.twist.twist.angular.y,
                        msg.twist.twist.angular.z,
                    ],
                    dtype=float,
                )
                event_id = f"{comp_name}"
                self.recv_follower[event_id] = vec
                self.recv_follower_status[event_id] = CONNECT_TIMEOUT_FRAME
        except Exception as e:
            logger.error(f"Odometry callback error (follower:{comp_name}): {e}")

    def _odom_callback_leader(self, comp_name: str, msg: Odometry):
        """
        leader 侧 Odometry 回调
        同上，合成 13 维向量
        """
        try:
            with self.lock:
                vec = np.array(
                    [
                        # position
                        msg.pose.pose.position.x,
                        msg.pose.pose.position.y,
                        msg.pose.pose.position.z,
                        # orientation
                        msg.pose.pose.orientation.x,
                        msg.pose.pose.orientation.y,
                        msg.pose.pose.orientation.z,
                        msg.pose.pose.orientation.w,
                        # linear velocity
                        msg.twist.twist.linear.x,
                        msg.twist.twist.linear.y,
                        msg.twist.twist.linear.z,
                        # angular velocity
                        msg.twist.twist.angular.x,
                        msg.twist.twist.angular.y,
                        msg.twist.twist.angular.z,
                    ],
                    dtype=float,
                )
                event_id = f"{comp_name}"
                self.recv_leader[event_id] = vec
                self.recv_leader_status[event_id] = CONNECT_TIMEOUT_FRAME
        except Exception as e:
            logger.error(f"Odometry callback error (leader:{comp_name}): {e}")

    # ======================
    # spin 线程控制
    # ======================

    def start(self):
        """启动 ROS2 spin 线程"""
        if self.running:
            return

        self.running = True
        self.spin_thread = threading.Thread(target=self._spin_loop, daemon=True)
        self.spin_thread.start()

        logger.info("[ROS2] Node started (spin thread running)")

    def _spin_loop(self):
        """独立线程执行 ROS2 spin"""
        try:
            rclpy.spin(self)
        except Exception as e:
            logger.error(f"[ROS2] Spin error: {e}")

    def stop(self):
        """停止 ROS2"""
        if not self.running:
            return

        self.running = False
        rclpy.shutdown()

        if getattr(self, "spin_thread", None):
            self.spin_thread.join(timeout=1.0)

        logger.info("[ROS2] Node stopped.")
'''

    # 4. 做占位符替换
    content = (
        template
        .replace("__NODE_CONFIG__", node_config_code)
        .replace("__CAMEL_NAME__", CamelName)
        .replace("__ROBOT_NAME__", robot_name)
    )

    # 5. 写文件
    target_dir = base_dir / f"robodriver_robot_{robot_name}"
    ensure_dir(target_dir)
    (target_dir / "node.py").write_text(content, encoding="utf-8")




def generate_config_py_code(
    robot_name: str,
    leader_motors: dict,
    follower_motors: dict,
    cameras: dict,
    use_videos: bool,
):
    """
    根据已经从 JSON 提取好的字典，生成 config.py 的完整源码字符串。
    现在 leader_motors / follower_motors 是“按组件分组”的嵌套 dict：
        { comp_id: { joint_name: [id, type], ... }, ... }
    """

    CamelName = to_camel(robot_name)

    # --------- 构造嵌套 motors 的代码块 ---------
    def motors_block(grouped_motors: dict) -> str:
        """
        grouped_motors: { comp_id: { joint: [id, type], ... }, ... }
        生成形如：
            "leader_arm": {
                "joint_0": Motor(1, "robot_motor", norm_mode_body),
                ...
            },
            "left_arm": {
                ...
            },
        """
        lines: list[str] = []
        for comp_id, motors in grouped_motors.items():
            lines.append(f'            "{comp_id}": {{')
            for joint, value in motors.items():
                if isinstance(value, (list, tuple)) and len(value) >= 2:
                    mid, mtype = value[0], value[1]
                else:
                    continue
                lines.append(
                    f'                "{joint}": Motor({mid}, "{mtype}", norm_mode_body),'
                )
            lines.append("            },")
        return "\n".join(lines)

    leader_block = motors_block(leader_motors)
    follower_block = motors_block(follower_motors)

    # --------- 构造 cameras 的代码块 ---------
    def cameras_block_func(cameras_dict: dict) -> str:
        lines = []
        for name, info in cameras_dict.items():
            width = info.get("width", 640)
            height = info.get("height", 480)
            fps = info.get("fps", 30)

            line = (
                f'            "{name}": OpenCVCameraConfig('
                f"index_or_path=0, fps={fps}, width={width}, height={height}),"
            )
            lines.append(line)
        return "\n".join(lines)

    cameras_block = cameras_block_func(cameras)

    use_videos_literal = "True" if use_videos else "False"

    # --------- 生成完整 config.py 源码 ---------
    config_code = f"""from typing import Dict
from dataclasses import dataclass, field

from lerobot.robots.config import RobotConfig
from lerobot.cameras import CameraConfig
from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.motors import Motor, MotorNormMode


@RobotConfig.register_subclass("{robot_name}")
@dataclass
class {CamelName}RobotConfig(RobotConfig):
    use_degrees = True
    norm_mode_body = (
        MotorNormMode.DEGREES if use_degrees else MotorNormMode.RANGE_M100_100
    )

    # 按组件分组：{{ comp_id: {{ joint_name: Motor, ... }}, ... }}
    leader_motors: Dict[str, Dict[str, Motor]] = field(
        default_factory=lambda norm_mode_body=norm_mode_body: {{
{leader_block}
        }}
    )

    follower_motors: Dict[str, Dict[str, Motor]] = field(
        default_factory=lambda norm_mode_body=norm_mode_body: {{
{follower_block}
        }}
    )

    cameras: Dict[str, CameraConfig] = field(
        default_factory=lambda: {{
{cameras_block}
        }}
    )

    use_videos: bool = {use_videos_literal}

    microphones: Dict[str, int] = field(default_factory=lambda: {{}}
    )
"""

    return config_code


def generate_config_py(base_dir: Path, robot_name: str, robot_json: dict):
    """
    综合函数：
      - 从 JSON 提取字典（按组件分组）
      - 生成 config.py 源码
      - 写入 robodriver_robot_{robot_name}/config.py
    """

    # 1. 从 JSON 提取字典
    leader_motors, follower_motors, cameras, use_videos = build_config_dicts(robot_json)

    # 2. 生成源码字符串
    config_code = generate_config_py_code(
        robot_name,
        leader_motors,
        follower_motors,
        cameras,
        use_videos,
    )

    # 3. 写文件
    target_dir = base_dir / f"robodriver_robot_{robot_name}"
    ensure_dir(target_dir)
    (target_dir / "config.py").write_text(config_code, encoding="utf-8")




def generate_calibrate_py(base_dir: Path, robot_name: str):
    target_dir = base_dir / f"robodriver_robot_{robot_name}"
    ensure_dir(target_dir)

    content = """def calibrate():
    pass
"""

    with open(target_dir / "calibrate.py", "w") as f:
        f.write(content)


# ===========================================================
# Main (Robot_name provided by caller)
# ===========================================================
def main(robot_name: str):
    """
    Main entry for external program.
    Example:
        main("so101_aio_ros2")
    """
    print(f"[INFO] Generating ROS2 AIO package for robot: {robot_name}")

    # 1. Load JSON
    json_path = Path(f"../config/{robot_name}.json")
    if not json_path.exists():
        raise FileNotFoundError(f"JSON not found: {json_path}")

    robot_config = load_json(json_path)

    # 2. Create output folder
    base_dir = make_output_dir("../../robots", f"robodriver-robot-{robot_name}")
    print(f"[INFO] Output directory: {base_dir}")

    # 3. Generate 7 files
    generate_lerobot_init(base_dir, robot_name)
    generate_pyproject(base_dir, robot_name)
    generate_robot_pkg_init(base_dir, robot_name)
    generate_robot_py(base_dir, robot_name)
    generate_node_py(base_dir, robot_name, robot_config)
    generate_config_py(base_dir, robot_name, robot_config)
    generate_calibrate_py(base_dir, robot_name)

    print("[OK] Package generated successfully.")


# Standard CLI hook
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot_name", type=str, required=True)
    args = parser.parse_args()
    main(args.robot_name)
