"""从 ROS2 话题读取 AstriBot T1 关节数据。

观测走 `/astribot_<part>/joint_space_states`（RobotJointState.position, BEST_EFFORT），
动作走 `/astribot_<part>/joint_space_command_forward`（RobotJointController.command, RELIABLE）。

command_forward 在机器人静止/无外部指令时不发布，此时 get_action 回退到 obs
（AIO 机型 desired≈current），避免因缺帧让上游展平逻辑崩溃。

实测两种消息的 `name` 字段恒为空，只按位置排列，因此关节名由本模块按
「部件 + 序号」生成，与 SDK 版本的 `follower_<part>_joint<N>.pos` 命名一致。
"""

import os
import threading
import time
from typing import Dict, List, Optional

import logging_mp

logger = logging_mp.get_logger(__name__)

# record_parts 用 effector_*，但 ROS 话题是 astribot_gripper_*
_PART_TOPIC_ALIASES = {
    "effector_left": "gripper_left",
    "effector_right": "gripper_right",
}


class AstribotJointNode:
    """独立的 rclpy 节点，缓存各部件最新一帧 obs/action。

    rclpy 与 astribot_msgs 均懒加载：只有在真正启用 ROS 关节时才 import，
    这样未装 ROS 的环境仍能导入本包（例如只用 SDK 关节时）。
    """

    def __init__(
        self,
        parts: List[str],
        domain_id: int = 25,
        obs_suffix: str = "joint_space_states",
        action_suffix: str = "joint_space_command_forward",
        node_name: str = "robodriver_astribot_joints",
        topic_prefix: str = "astribot",
    ):
        self.parts = list(parts)
        self.domain_id = domain_id
        self.obs_suffix = obs_suffix
        self.action_suffix = action_suffix
        self.node_name = node_name
        self.topic_prefix = topic_prefix

        self._lock = threading.Lock()
        self._obs: Dict[str, List[float]] = {}
        self._action: Dict[str, List[float]] = {}
        self._dof: Dict[str, int] = {}
        self._node = None
        self._spin_thread: Optional[threading.Thread] = None
        self._running = False
        self._shutdown_done = False

        # part -> 话题里的部件名（处理 effector/gripper 别名）
        self._topic_part = {
            part: _PART_TOPIC_ALIASES.get(part, part) for part in self.parts
        }

    # ========= 话题名 =========

    def obs_topic(self, part: str) -> str:
        return f"/{self.topic_prefix}_{self._topic_part[part]}/{self.obs_suffix}"

    def action_topic(self, part: str) -> str:
        return f"/{self.topic_prefix}_{self._topic_part[part]}/{self.action_suffix}"

    # ========= 生命周期 =========

    def start(self, wait_first_frame_s: float = 10.0) -> None:
        # domain id 必须在 rclpy.init() 之前设置，context 初始化时读取
        os.environ["ROS_DOMAIN_ID"] = str(self.domain_id)

        import rclpy
        from rclpy.node import Node
        from rclpy.qos import qos_profile_sensor_data, QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
        from astribot_msgs.msg import RobotJointState, RobotJointController

        if not rclpy.ok():
            rclpy.init()

        self._node = Node(self.node_name)

        # joint_space_states 是 BEST_EFFORT，必须用 sensor QoS，否则静默收不到
        best_effort = qos_profile_sensor_data
        # command 类话题是 RELIABLE，用匹配的 reliable profile
        reliable = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        for part in self.parts:
            self._node.create_subscription(
                RobotJointState,
                self.obs_topic(part),
                lambda msg, p=part: self._obs_cb(p, msg),
                best_effort,
            )
            self._node.create_subscription(
                RobotJointController,
                self.action_topic(part),
                lambda msg, p=part: self._action_cb(p, msg),
                reliable,
            )
            logger.info(
                f"[ROS关节] {part}: obs={self.obs_topic(part)} (best_effort), "
                f"action={self.action_topic(part)} (reliable)"
            )

        self._running = True
        self._spin_thread = threading.Thread(target=self._spin_loop, daemon=True)
        self._spin_thread.start()

        try:
            self._wait_first_frame(wait_first_frame_s)
        except Exception:
            # 等首帧失败时回收 spin 线程与 node，避免泄漏
            self.stop()
            self._shutdown_done = False
            raise

    def _spin_loop(self) -> None:
        import rclpy
        try:
            while self._running and rclpy.ok():
                rclpy.spin_once(self._node, timeout_sec=0.05)
        except Exception as e:
            logger.error(f"[ROS关节] spin 异常: {e}")

    def _wait_first_frame(self, timeout_s: float) -> None:
        deadline = time.perf_counter() + timeout_s
        while time.perf_counter() < deadline:
            with self._lock:
                missing_obs = set(self.parts) - set(self._obs.keys())
            if not missing_obs:
                break
            time.sleep(0.02)
        with self._lock:
            missing_obs = sorted(set(self.parts) - set(self._obs.keys()))
            missing_action = sorted(set(self.parts) - set(self._action.keys()))
            dof = dict(self._dof)
        # obs 是采集必需的，且 joint_space_states 持续发布；收不到说明连接有问题
        if missing_obs:
            raise TimeoutError(
                f"{timeout_s}s 内未收齐 obs 首帧。缺: {missing_obs}。"
                f"请确认 ROS_DOMAIN_ID={self.domain_id}、话题 {self.obs_suffix} 与机器人是否在线。"
            )
        # action 走 command_forward，机器人静止时不发布，收不到属正常，只告警不崩
        if missing_action:
            logger.warning(
                f"[ROS关节] {timeout_s}s 内未收到 action({self.action_suffix}) 首帧，缺: {missing_action}。"
                f"机器人静止时该话题不发布属正常；开始遥操作后应恢复。"
                f"采集期间若 action 持续为空，记录的是 0 值。"
            )
        logger.info(f"[ROS关节] 各部件自由度: {dof}")

    def stop(self) -> None:
        if self._shutdown_done:
            return
        self._shutdown_done = True
        self._running = False
        if self._spin_thread is not None:
            self._spin_thread.join(timeout=1.0)
        if self._node is not None:
            try:
                self._node.destroy_node()
            except Exception as e:
                logger.warning(f"[ROS关节] destroy_node 失败: {e}")
            self._node = None

    # ========= 回调 =========

    def _obs_cb(self, part: str, msg) -> None:
        with self._lock:
            self._obs[part] = list(msg.position)
            self._dof[part] = len(msg.position)

    def _action_cb(self, part: str, msg) -> None:
        # command_forward 是 RobotJointController，其 .command 是
        # [position, velocity, torque] 三段拼接，长度为关节数的 3 倍
        # (实测 arm 7→21、head 2→6、gripper 1→3，且前 1/3 与 recv.position 逐值相同)。
        # RoboDriver 的 action 只记录位置目标，故仅取前 1/3。
        command = list(msg.command)
        n = len(command) // 3 if len(command) % 3 == 0 else len(command)
        with self._lock:
            self._action[part] = command[:n]

    # ========= 读取 =========

    def get_dof(self) -> Dict[str, int]:
        with self._lock:
            return dict(self._dof)

    def get_observation(self) -> Dict[str, List[float]]:
        with self._lock:
            return {part: list(self._obs.get(part, [])) for part in self.parts}

    def get_action(self) -> Dict[str, List[float]]:
        """返回各部件 action。

        command_forward 在机器人静止/无外部指令时不发布，缓存可能为空。此时回退到
        当前实测位置(obs)：AIO 机型没被指令时 desired≈current，这样既不会返回空值
        让上游 _flatten_ros 因长度不符崩溃，也避免记录无意义的 0。
        """
        with self._lock:
            out = {}
            for part in self.parts:
                act = self._action.get(part)
                out[part] = list(act) if act else list(self._obs.get(part, []))
            return out
