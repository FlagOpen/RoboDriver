import threading
import time
from typing import Any

import logging_mp
import numpy as np
from lerobot.cameras import make_cameras_from_configs
from lerobot.robots.robot import Robot
from lerobot.utils.errors import DeviceNotConnectedError, DeviceAlreadyConnectedError
from functools import cached_property

import rclpy

from .config import GALAXEALITEFollowerRos2RobotConfig
from .status import GALAXEALITEFollowerRos2RobotStatus
from .node import GALAXEALITEFollowerRos2RobotNode, ros_spin_thread


logger = logging_mp.get_logger(__name__)



class GALAXEALITEFollowerRos2Robot(Robot):
    config_class = GALAXEALITEFollowerRos2RobotConfig
    name = "galaxealite-follower-ros2"

    def __init__(self, config: GALAXEALITEFollowerRos2RobotConfig):
        super().__init__(config)
        self.config = config
        self.robot_type = self.config.type
        self.use_videos = self.config.use_videos
        self.microphones = self.config.microphones

        self.follower_motors = config.follower_motors
        self.cameras = make_cameras_from_configs(self.config.cameras)
        
        self.connect_excluded_cameras = ["image_pika_pose"]

        self.status = GALAXEALITEFollowerRos2RobotStatus()
        if not rclpy.ok():
            rclpy.init()
        self.robot_ros2_node = GALAXEALITEFollowerRos2RobotNode()  # 创建节点实例
        self.ros_spin_thread = threading.Thread(
            target=ros_spin_thread, 
            args=(self.robot_ros2_node,), 
            daemon=True
        )
        self.ros_spin_thread.start()

        self.connected = False
        self.logs = {}

    @property
    def _follower_motors_ft(self) -> dict[str, type]:
        return {
            f"follower_{joint_name}.pos": float
            for comp_name, joints in self.follower_motors.items()
            for joint_name in joints.keys()
        }

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {
            cam: (self.config.cameras[cam].height, self.config.cameras[cam].width, 3) for cam in self.cameras
        }

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._follower_motors_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return self._follower_motors_ft
    
    @property
    def is_connected(self) -> bool:
        return self.connected
    
    def connect(self):
        timeout = 20  # 统一的超时时间（秒）
        start_time = time.perf_counter()

        if self.connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        # 定义所有需要等待的条件及其错误信息
        conditions = [
            (
                lambda: all(
                    name in self.robot_ros2_node.recv_images
                    for name in self.cameras
                    if name not in self.connect_excluded_cameras
                ),
                lambda: [name for name in self.cameras if name not in self.robot_ros2_node.recv_images],
                "等待摄像头图像超时",
            ),
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
                "等待从臂关节角度超时",
            ),
        ]

        # 跟踪每个条件是否已完成
        completed = [False] * len(conditions)

        while True:
            # 检查每个未完成的条件
            for i in range(len(conditions)):
                if not completed[i]:
                    condition_func = conditions[i][0]
                    if condition_func():
                        completed[i] = True

            # 如果所有条件都已完成，退出循环
            if all(completed):
                break

            # 检查是否超时
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
                    f"连接超时，未满足的条件: {'; '.join(failed_messages)}"
                )

            # 减少 CPU 占用
            time.sleep(0.01)

        # ===== 新增成功打印逻辑 =====
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
            follower_received = [
                name
                for name in self.follower_motors
                if any(name in key for key in self.robot_ros2_node.recv_follower)
            ]
            success_messages.append(f"从臂数据: {', '.join(follower_received)}")


        log_message = "\n[连接成功] 所有设备已就绪:\n"
        log_message += "\n".join(f"  - {msg}" for msg in success_messages)
        log_message += f"\n  总耗时: {time.perf_counter() - start_time:.2f} 秒\n"
        logger.info(log_message)
        # ===========================

        for i in range(self.status.specifications.camera.number):
            self.status.specifications.camera.information[i].is_connect = True
        for i in range(self.status.specifications.arm.number):
            self.status.specifications.arm.information[i].is_connect = True

        self.connected = True

    @property
    def is_calibrated(self) -> bool:
        """Whether the robot is currently calibrated or not. Should be always `True` if not applicable"""
        return True

    def calibrate(self) -> None:
        """
        Calibrate the robot if applicable. If not, this should be a no-op.

        This method should collect any necessary data (e.g., motor offsets) and update the
        :pyattr:`calibration` dictionary accordingly.
        """
        pass

    def configure(self) -> None:
        """
        Apply any one-time or runtime configuration to the robot.
        This may include setting motor parameters, control modes, or initial state.
        """
        pass
    
    def get_observation(self) -> dict[str, Any]:
        if not self.connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        start = time.perf_counter()
        obs_dict: dict[str, Any] = {}
        for comp_name, joints in self.follower_motors.items():
            for follower_name, follower in self.robot_ros2_node.recv_follower.items():
                if follower_name == comp_name:
                    for i, joint_name in enumerate(joints.keys()):
                        obs_dict[f"follower_{joint_name}.pos"] = float(follower[i])

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
    

    def send_action(self, action: dict[str, Any]):
        """The provided action is expected to be a vector."""
        if not self.is_connected:
            raise DeviceNotConnectedError(
                "KochRobot is not connected. You need to run `robot.connect()`."
            )
        goal_joint = [ val for key, val in action.items()]
        goal_joint_numpy = np.array([
            t.item() if hasattr(t, 'item') else t
            for t in goal_joint
        ], dtype=np.float32)
        try:
            if goal_joint_numpy.shape != (14,):
                raise ValueError(f"Action vector must be 38-dimensional, got {goal_joint_numpy.shape[0]}")
            
            # 调用ROS2节点的ros_replay方法发布动作
            self.robot_ros2_node.ros_replay(goal_joint_numpy)
            
        except Exception as e:
            logger.error(f"Failed to send action: {e}")
            raise

    def update_status(self) -> str:
        for i in range(self.status.specifications.camera.number):
            match_name = self.status.specifications.camera.information[i].name
            for name in self.robot_ros2_node.recv_images_status:
                if match_name in name:
                    self.status.specifications.camera.information[i].is_connect = (
                        True if self.robot_ros2_node.recv_images_status[name] > 0 else False
                    )

        for i in range(self.status.specifications.arm.number):
            match_name = self.status.specifications.arm.information[i].name
            for name in self.robot_ros2_node.recv_leader_status:
                if match_name in name:
                    self.status.specifications.arm.information[i].is_connect = (
                        True if self.robot_ros2_node.recv_leader_status[name] > 0 else False
                    )

        for i in range(self.status.specifications.arm.number):
            match_name = self.status.specifications.arm.information[i].name
            for name in self.robot_ros2_node.recv_follower_status:
                if match_name in name:
                    self.status.specifications.arm.information[i].is_connect = (
                        True if self.robot_ros2_node.recv_follower_status[name] > 0 else False
                    )

        return self.status.to_json()

    def disconnect(self):
        if not self.is_connected:
            raise DeviceNotConnectedError(
                "robot is not connected. You need to run `robot.connect()` before disconnecting."
            )
        if hasattr(self, "ros_node"):
            self.robot_ros2_node.destroy()
        if rclpy.ok():
            rclpy.shutdown()

        self.connected = False

    def __del__(self):
        try:
            if getattr(self, "is_connected", False):
                self.disconnect()
        except Exception:
            pass

    def reset(self, 
                state = [
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 100, 
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 100, 
                    -0.8600000143051147, 1.6863000392913818, 0.7631000280380249
                    ], 
                duration: float = 3.0):
        if not self.is_connected:
            raise DeviceNotConnectedError(
                "Robot is not connected. You need to run `robot.connect()`."
            )

        # Get robot current states
        curr: list[float] = []
        for comp_name, joints in self.follower_motors.items():
            follower = self.robot_ros2_node.recv_follower.get(comp_name)
            if follower is not None:
                for i, _ in enumerate(joints):
                    curr.append(float(follower[i]))
        curr_arr = np.array(curr, dtype=np.float32)
        goal_arr = np.array(state, dtype=np.float32)
        
        # Interpolate actions within the given duration
        try:
            FPS = 30.0
            steps = max(1, int(round(duration * FPS))) # at least execute once
            for i in range(1, steps + 1):
                start_episode_t = time.perf_counter()
                ratio = i / steps
                action = curr_arr + ratio * (goal_arr - curr_arr)
                self.robot_ros2_node.ros_replay(action)
                dt_s = time.perf_counter() - start_episode_t
                time.sleep(1 / FPS - dt_s)
            
        except Exception as e:
            logger.error(f"Failed to send action: {e}")
            raise