import time
from functools import cached_property
from typing import Any
import math
import numpy as np
import logging_mp

from lerobot.cameras import make_cameras_from_configs
from lerobot.utils.errors import DeviceNotConnectedError, DeviceAlreadyConnectedError
from lerobot.robots.robot import Robot

from .config import LinkerBotAioRos1RobotConfig
from .node import LinkerBotAioRos1Node
from .status import LinkerBotAioRos1RobotStatus

logger = logging_mp.get_logger(__name__)


class LinkerBotAioRos1Robot(Robot):
    config_class = LinkerBotAioRos1RobotConfig
    name = "linkerbot_aio_ros1"

    def __init__(self, config: LinkerBotAioRos1RobotConfig):
        super().__init__(config)
        self.config = config
        self.robot_type = self.config.type
        self.use_videos = self.config.use_videos
        self.microphones = self.config.microphones

        self.follower_motors = config.follower_motors
        self.leader_motors = config.leader_motors
        self.cameras = make_cameras_from_configs(self.config.cameras)

        self.connect_excluded_cameras = ["image_pika_pose"]

        self.status = LinkerBotAioRos1RobotStatus()
        self.robot_ros1_node = LinkerBotAioRos1Node()
        self.robot_ros1_node.start()

        self.connected = False
        self.logs = {}


    # ========= 数据处理核心逻辑=========

    def _get_hardware_info(self, comp_name: str):
        """辅助函数：获取硬件配置"""
        is_leader = "leader" in comp_name.lower()
        is_hand = "hand" in comp_name.lower()

        if is_hand:
            status_name = "leader_hand" if is_leader else "follower_hand"
            return next((h for h in self.status.specifications.hand.information if h.name == status_name), None)
        else:
            status_name = "leader_arm" if is_leader else "follower_arm"
            return next((a for a in self.status.specifications.arm.information if a.name == status_name), None)

    def _process_incoming(self, raw_val: float, comp_name: str, joint_idx: int) -> float:
        """接收时调用：Raw -> 0~1 / 弧度"""
        info = self._get_hardware_info(comp_name)
        if not info:
            return raw_val  # 查不到配置直接透传保平安

        is_hand = "hand" in comp_name.lower()

        if is_hand:
            v_max = float(info.joint_p_limit[joint_idx])
            v_min = float(info.joint_n_limit[joint_idx])
            
            if v_max == v_min:
                norm = 0.5
            else:
                norm = (raw_val - v_min) / (v_max - v_min)
            
            norm = float(np.clip(norm, 0.0, 1.0))
            
            # 使用配置变量决定是否反转 (1.0 - norm)
            return 1.0 - norm if info.invert_direction else norm

        else:
            # 机械臂：使用配置变量决定是否转弧度
            val_in_rad = raw_val if info.is_radian else math.radians(raw_val)
            
            # 使用配置变量决定是否反转符号
            return -val_in_rad if info.invert_direction else val_in_rad

    def _process_outgoing(self, ml_val: float, comp_name: str, joint_idx: int) -> float:
        """下发时调用：0~1 / 弧度 -> Raw"""
        info = self._get_hardware_info(comp_name)
        if not info:
            return ml_val

        is_hand = "hand" in comp_name.lower()

        if is_hand:
            v_max = float(info.joint_p_limit[joint_idx])
            v_min = float(info.joint_n_limit[joint_idx])
            
            norm = float(np.clip(ml_val, 0.0, 1.0))
            
            # 撤销反转
            if info.invert_direction:
                norm = 1.0 - norm
                
            return norm * (v_max - v_min) + v_min

        else:
            # 撤销反转
            rad_val = -ml_val if info.invert_direction else ml_val
            
            # 撤销弧度转换
            return rad_val if info.is_radian else math.degrees(rad_val)
    # ========= features =========
    def _generate_feature_name(self, role_prefix: str, comp_name: str, joint_name: str) -> str:
        """
        改进的特征键名生成器：保留完整关节名，防止重名覆盖
        """
        side = 'left' if 'left' in comp_name.lower() else 'right'
        
        # 1. 如果是位姿数据 (Pose)
        if 'pose' in comp_name.lower():
            # 保留 position_x 等完整描述
            suffix = joint_name.replace("pose_", "")
            return f"{role_prefix}_{side}_pose_{suffix}"
            
        # 2. 如果是灵巧手 (Hand)
        elif 'hand' in comp_name.lower():
            # 不要 split('_')[-1]，直接用 thumb_spread 这样的完整名字
            # 或者如果你想统一格式，可以去掉 joint_ 前缀
            clean_joint_name = joint_name.replace("joint_", "")
            return f"{role_prefix}_{side}_hand_{clean_joint_name}.pos"
            
        # 3. 默认认为是机械臂关节 (Arm)
        else:
            # 将 joint_1 转为 1
            clean_joint_name = joint_name.replace("joint_", "")
            return f"{role_prefix}_{side}_arm_{clean_joint_name}.pos"
    
    @property
    def _follower_motors_ft(self) -> dict[str, type]:
        return {
            self._generate_feature_name("follower", comp_name, joint_name): float
            for comp_name, joints in self.follower_motors.items()
            for joint_name in joints
        }

    @property
    def _leader_motors_ft(self) -> dict[str, type]:
        return {
            self._generate_feature_name("leader", comp_name, joint_name): float
            for comp_name, joints in self.leader_motors.items()
            for joint_name in joints
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
        return self._leader_motors_ft # 拿不到主臂数据暂时用从臂

    @property
    def is_connected(self) -> bool:
        return self.connected

    # ========= connect / disconnect =========

    def connect(self):
        timeout = 20
        start_time = time.perf_counter()

        if self.connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        node = self.robot_ros1_node
        conditions = [
            # 摄像头图像 (原样保留)
            (
                lambda: all(
                    name in node.recv_images
                    for name in self.cameras
                    if name not in self.connect_excluded_cameras
                ),
                lambda: [
                    name
                    for name in self.cameras
                    if name not in node.recv_images
                    and name not in self.connect_excluded_cameras
                ],
                "等待摄像头图像超时",
            ),
            # 从臂 (原样保留)
            (
                lambda: all(
                    any(name in key for key in node.recv_follower)
                    for name in self.follower_motors
                ),
                lambda: [
                    name
                    for name in self.follower_motors
                    if not any(name in key for key in node.recv_follower)
                ],
                "等待从臂数据超时",
            ),
        ]

        # === 新增：如果配置了主臂，则等待主臂数据 ===
        if self.leader_motors:
            conditions.append((
                lambda: all(
                    any(name in key for key in getattr(node, "recv_leader", {}))
                    for name in self.leader_motors
                ),
                lambda: [
                    name
                    for name in self.leader_motors
                    if not any(name in key for key in getattr(node, "recv_leader", {}))
                ],
                "等待主臂数据超时",
            ))

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
                        received = [name for name in self.cameras if name not in missing]
                    elif i == 1:
                        received = [name for name in self.follower_motors if name not in missing]
                    else:
                        received = [name for name in self.leader_motors if name not in missing]

                    msg = (
                        f"{base_msg}: 未收到 [{', '.join(missing)}]; "
                        f"已收到 [{', '.join(received)}]"
                    )
                    failed_messages.append(msg)

                if not failed_messages:
                    break

                raise TimeoutError(
                    f"连接超时，未满足的条件: {'; '.join(failed_messages)}"
                )

            time.sleep(0.01)

        # 成功日志
        success_messages = []

        if conditions[0][0]():
            cam_received = [
                name
                for name in self.cameras
                if name in node.recv_images
                and name not in self.connect_excluded_cameras
            ]
            success_messages.append(f"摄像头: {', '.join(cam_received)}")

        if conditions[1][0]():
            follower_received = [
                name
                for name in self.follower_motors
                if any(name in key for key in node.recv_follower)
            ]
            success_messages.append(f"从臂数据: {', '.join(follower_received)}")

        # === 新增：主臂成功日志 ===
        if self.leader_motors and conditions[2][0]():
            leader_received = [
                name
                for name in self.leader_motors
                if any(name in key for key in getattr(node, "recv_leader", {}))
            ]
            success_messages.append(f"主臂数据: {', '.join(leader_received)}")

        log_message = "\n[连接成功] 所有设备已就绪:\n"
        log_message += "\n".join(f"  - {msg}" for msg in success_messages)
        log_message += f"\n  总耗时: {time.perf_counter() - start_time:.2f} 秒\n"
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

        node = self.robot_ros1_node
        # follower joints
        for comp_name, joints in self.follower_motors.items():
            vec = node.recv_follower.get(comp_name)
            if vec is None:
                continue
            joint_names = list(joints.keys())
            for idx, joint in enumerate(joint_names):
                if idx >= len(vec):
                    break
                key_name = self._generate_feature_name("follower", comp_name, joint)
                obs_dict[key_name] = self._process_incoming(float(vec[idx]), comp_name, idx)

        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read follower state: {dt_ms:.1f} ms")
        # camera images
        for cam_key, _cam in self.cameras.items():
            start = time.perf_counter()
            for name, val in node.recv_images.items():
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

        node = self.robot_ros1_node

        for comp_name, joints in self.leader_motors.items():
            vec = node.recv_leader.get(comp_name)
            if vec is None:
                continue
            joint_names = list(joints.keys())
            for idx, joint in enumerate(joint_names):
                if idx >= len(vec):
                    break
                key_name = self._generate_feature_name("leader", comp_name, joint)
                act_dict[key_name] = self._process_incoming(float(vec[idx]), comp_name, idx)
                
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

        # 遍历下发目标
        for comp_name, joints in self.follower_motors.items():
            comp_positions = []
            joint_names = list(joints.keys())
            for idx, joint_name in enumerate(joint_names):
                feature_key = self._generate_feature_name("follower", comp_name, joint_name)
                
                if feature_key in action:
                    ml_val = float(action[feature_key])
                    # 【修改点】：将标准特征还原为底层 Raw 格式
                    real_val = self._process_outgoing(ml_val, comp_name, idx)
                    comp_positions.append(real_val)
            
            if comp_positions:
                self.robot_ros1_node.send_control_command(comp_name, comp_positions)

        return {f"{arm_motor}.pos": val for arm_motor, val in action.items()}