import time
import logging_mp
import numpy as np
import rclpy

from functools import cached_property
from typing import Any

from lerobot.cameras import make_cameras_from_configs
from lerobot.utils.errors import DeviceNotConnectedError, DeviceAlreadyConnectedError
from lerobot.robots.robot import Robot

from .config import A2DAioRos2RobotConfig
from .node import A2DAioRos2Node


logger = logging_mp.get_logger(__name__)


class A2DAioRos2Robot(Robot):
    config_class = A2DAioRos2RobotConfig
    name = "a2d_aio_ros2"

    def __init__(self, config: A2DAioRos2RobotConfig):
        rclpy.init()
        super().__init__(config)
        self.config = config
        self.robot_type = self.config.type
        self.use_videos = self.config.use_videos
        self.microphones = self.config.microphones

        # A2D 通常没有 leader（遥操作主手），所以 leader_motors 为空
        self.leader_motors = config.leader_motors  # 应为 {}
        self.follower_motors = config.follower_motors  # 包含 14 arm + 2 gripper
        self.cameras = make_cameras_from_configs(self.config.cameras)

        # 排除某些虚拟相机（如 pose 图）
        self.connect_excluded_cameras = ["image_pika_pose"]

        # 启动 ROS2 节点
        self.robot_ros2_node = A2DAioRos2Node()
        self.robot_ros2_node.start()

        self.connected = False
        self.logs = {}

    # ========= features =========

    @property
    def _follower_motors_ft(self) -> dict[str, type]:
        return {
            f"follower_{motor}.pos": float
            for motor in self.follower_motors
        }
    
    @property
    def _leader_motors_ft(self) -> dict[str, type]:
        # 如果无 leader，返回空
        return {
            f"leader_{motor}.pos": float
            for motor in self.leader_motors
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
        # A2D 是 follower-only 控制，action 即目标 follower joint
        # 所以 action_features 应与 follower 一致（用于模仿学习）
        return self._follower_motors_ft

    @property
    def is_connected(self) -> bool:
        return self.connected

    # ========= connect / disconnect =========

    def _log_recv_snapshot(self):
        """调试用：打印当前 recv_follower 里到底有谁"""
        all_keys = set()
        for comp_dict in self.robot_ros2_node.recv_follower.values():
            all_keys.update(comp_dict.keys())
        logger.warning("【快照】recv_follower 所有 key = %s", sorted(all_keys))

    def connect(self):
        try:
            timeout = 20
            start_time = time.perf_counter()

            if self.connected:
                raise DeviceAlreadyConnectedError(f"{self} already connected")

            # 条件检查：A2D 只需 follower + cameras
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
                # 从臂（A2D 只有 follower）
                (
                    lambda: all(
                        # any(name == key for comp in self.robot_ros2_node.recv_follower.values() for key in comp)
                        # for name in self.follower_motors
                        any(
                            name in comp
                            for comp in self.robot_ros2_node.recv_follower.values()
                        )
                        for name in self.follower_motors
                    ),
                    lambda: [
                        name
                        for name in self.follower_motors
                        if not any(name == key for comp in self.robot_ros2_node.recv_follower.values() for key in comp)
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
                    # self._log_recv_snapshot()
                    failed_messages = []
                    for i, (cond, get_missing, base_msg) in enumerate(conditions):
                        if completed[i]:
                            continue

                        missing = get_missing()
                        if not missing:
                            completed[i] = True
                            continue

                        if i == 0:
                            received = [name for name in self.cameras if name in self.robot_ros2_node.recv_images]
                        else:
                            received = [
                                name for name in self.follower_motors
                                if any(name == key for comp in self.robot_ros2_node.recv_follower.values() for key in comp)
                            ]

                        msg = (
                            f"{base_msg}: 未收到 [{', '.join(missing)}]; "
                            f"已收到 [{', '.join(received)}]"
                        )
                        failed_messages.append(msg)

                    if failed_messages:
                        raise TimeoutError(f"连接超时，未满足的条件: {'; '.join(failed_messages)}")

                time.sleep(0.01)

            # 成功日志
            success_messages = []

            if conditions[0][0]():
                cam_received = [name for name in self.cameras if name in self.robot_ros2_node.recv_images]
                success_messages.append(f"摄像头: {', '.join(cam_received)}")

            if conditions[1][0]():
                follower_received = [
                    name for name in self.follower_motors
                    if any(name == key for comp in self.robot_ros2_node.recv_follower.values() for key in comp)
                ]
                success_messages.append(f"从臂数据: {', '.join(follower_received)}")

            log_message = "\n[连接成功] 所有设备已就绪:\n"
            log_message += "\n".join(f"  - {msg}" for msg in success_messages)
            log_message += f"\n  总耗时: {time.perf_counter() - start_time:.2f} 秒\n"
            logger.info(log_message)

            self.connected = True
        except Exception as e:
            logger.exception("【CONNECT 异常提前抛出】%s", e)
            raise

    def disconnect(self):
        if not self.connected:
            raise DeviceNotConnectedError()
        self.robot_ros2_node.stop()
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
        try:
            # logger.info("Get observation")
            if not self.connected:
                raise DeviceNotConnectedError(f"{self} is not connected.")

            obs_dict: dict[str, Any] = {}

            # 从 recv_follower 中提取所有 follower 电机值
            for motor_name in self.follower_motors:
                found = False
                for comp_dict in self.robot_ros2_node.recv_follower.values():
                    if motor_name in comp_dict:
                        obs_dict[f"follower_{motor_name}.pos"] = float(comp_dict[motor_name])
                        found = True
                        break
                if not found:
                    # 可选：设为 0 或报 warning
                    obs_dict[f"follower_{motor_name}.pos"] = 0.0
                    logger.warning(f"Motor {motor_name} not found in follower data")

            # 相机图像
            for cam_key in self.cameras:
                if cam_key in self.robot_ros2_node.recv_images:
                    obs_dict[cam_key] = self.robot_ros2_node.recv_images[cam_key]
                else:
                    # 可选：填充空白图像或报错
                    h, w = self.config.cameras[cam_key].height, self.config.cameras[cam_key].width
                    obs_dict[cam_key] = np.zeros((h, w, 3), dtype=np.uint8)
                    logger.warning(f"Camera {cam_key} image not received")
            return obs_dict
        except Exception as e:
            # 关键：用 logger.exception 会把完整 traceback 打印出来
            logger.exception("get_observation 内部爆炸：%s", e)
            raise  # 继续抛出去，让上层继续报 "observation is none"

    
    def get_action(self) -> dict[str, Any]:
        """
        A2D 无 leader，所以 action 通常不使用。
        但为了兼容 lerobot pipeline，可返回当前 follower 状态作为“伪 action”
        """
        if not self.connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        act_dict = {}
        for motor_name in self.follower_motors:
            found = False
            for comp_dict in self.robot_ros2_node.recv_follower.values():
                if motor_name in comp_dict:
                    act_dict[f"follower_{motor_name}.pos"] = float(comp_dict[motor_name])  # 注意：这里用 leader_ 前缀是为了匹配 action_features
                    found = True
                    break
            if not found:
                act_dict[f"follower_{motor_name}.pos"] = 0.0
        # logger.warning("action_features 要求的键=%s", list(self.action_features.keys()))
        # logger.warning("get_action 实际返回的键=%s", list(act_dict.keys()))
        return act_dict

    # ========= send_action =========

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        """发送动作到 A2D 机器人"""
        if not self.is_connected:
            raise DeviceNotConnectedError(
                f"{self} is not connected. You need to run `robot.connect()`."
            )

        cleaned_action = {}
        for key, value in action.items():
            # 兼容两种格式：
            # 1. 'follower_left_arm_joint1.pos' （来自 dataset）
            # 2. 'left_arm_joint1' （直接控制）
            if key.startswith("follower_") and key.endswith(".pos"):
                motor = key[len("follower_"):-len(".pos")]
                cleaned_action[motor] = value
            elif key in self.follower_motors:
                cleaned_action[key] = value
            else:
                logger.warning(f"Ignoring unknown action key: {key}")

        # 发送给 ROS2 节点
        self.robot_ros2_node.ros2_send(cleaned_action)

        # 返回标准化的 action dict（用于记录）
        return {
            f"follower_{motor}.pos": float(val)
            for motor, val in cleaned_action.items()
        }