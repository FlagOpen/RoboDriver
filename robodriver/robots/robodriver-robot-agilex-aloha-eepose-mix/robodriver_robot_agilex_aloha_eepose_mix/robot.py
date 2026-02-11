import threading
import time
from typing import Any, Tuple
import math
import logging_mp
import numpy as np
import torch
from lerobot.cameras import make_cameras_from_configs
from lerobot.robots.robot import Robot
from lerobot.utils.errors import DeviceNotConnectedError, DeviceAlreadyConnectedError
from functools import cached_property
from scipy.spatial.transform import Rotation

from .config import AgilexAlohaEEposeMixRobotConfig
from .node import AgilexAlohaHeadImageROS2RobotNode
from .node import AgilexAlohaEEposeDoraRobotNode
                  

logger = logging_mp.get_logger(__name__)


def euler_xyz_to_quaternion(array):
    """
    将欧拉角（XYZ顺序，即绕X、Y、Z轴旋转）转换为四元数
    
    参数:
        roll: 绕X轴旋转角度（弧度）
        pitch: 绕Y轴旋转角度（弧度）
        yaw: 绕Z轴旋转角度（弧度）
    
    返回:
        (w, x, y, z): 四元数，w为实部，x,y,z为虚部
    """
    # 计算半角
    cy = math.cos(array[2] * 0.5)
    sy = math.sin(array[2] * 0.5)
    cp = math.cos(array[1] * 0.5)
    sp = math.sin(array[1] * 0.5)
    cr = math.cos(array[0] * 0.5)
    sr = math.sin(array[0] * 0.5)
    
    # 计算四元数
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    
    return (w, x, y, z)

class AgilexAlohaEEposeMixRobot(Robot):
    config_class = AgilexAlohaEEposeMixRobotConfig
    name = "agilex_aloha_eepose_mix"

    def __init__(self, config: AgilexAlohaEEposeMixRobotConfig):
        super().__init__(config)
        self.config = config
        self.robot_type = self.config.type
        self.use_videos = self.config.use_videos
        self.microphones = self.config.microphones

        self.motors = config.motors
        self.actuators = config.actuators
        self.cameras = make_cameras_from_configs(self.config.cameras)

        self.connect_excluded_cameras_dora = ["image_head"]
        self.connect_excluded_cameras_ros2 = ["image_right", "image_left"]

        self.robot_ros2_node = AgilexAlohaHeadImageROS2RobotNode()
        self.robot_dora_node = AgilexAlohaEEposeDoraRobotNode()
        self.robot_dora_node.start()

        self.connected = False
        self.logs = {}

    @property
    def _actuator_ft(self) -> dict[str, type]:
        return {f"leader_{actuator}.pos": float for actuator in self.actuators}

    @property
    def _motors_ft(self) -> dict[str, type]:
        return {f"follower_{motor}.pos": float for motor in self.motors}

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {
            cam: (self.config.cameras[cam].height, self.config.cameras[cam].width, 3) for cam in self.cameras
        }

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._motors_ft, **self._cameras_ft}
    
    @cached_property
    def action_features(self) -> dict[str, type]:
        return {**self._actuator_ft}
    
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
                    name in self.robot_dora_node.recv_images
                    for name in self.cameras
                    if name not in self.connect_excluded_cameras_dora
                ),
                lambda: [name for name in self.cameras if name not in self.robot_dora_node.recv_images and name not in self.connect_excluded_cameras_dora],
                "等待dora摄像头图像超时",
            ),
            (
                lambda: all(
                    name in self.robot_ros2_node.recv_images
                    for name in self.cameras
                    if name not in self.connect_excluded_cameras_ros2
                ),
                lambda: [name for name in self.cameras if name not in self.robot_ros2_node.recv_images and name not in self.connect_excluded_cameras_ros2],
                "等待ros摄像头图像超时",
            ),
            (
                lambda: len(self.robot_dora_node.recv_follower_joint_right) > 0,
                lambda: [] if len(self.robot_dora_node.recv_follower_joint_right) > 0 else ["recv_leader_joint_right"],
                "等待右臂关节角度超时",
            ),
            (
                lambda: len(self.robot_dora_node.recv_follower_joint_left) > 0,
                lambda: [] if len(self.robot_dora_node.recv_follower_joint_left) > 0 else ["recv_leader_joint_left"],
                "等待左臂关节角度超时",
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
                for i in range(len(completed)):
                    if not completed[i]:
                        condition_func, get_missing, base_msg = conditions[i]
                        missing = get_missing()

                        # 重新检查条件是否满足（可能刚好在最后一次检查后满足）
                        if condition_func():
                            completed[i] = True
                            continue

                        # 如果没有 missing，也视为满足
                        if not missing:
                            completed[i] = True
                            continue

                        # 构造错误信息
                        if i == 0 or i == 1:  # 摄像头条件
                            received = [
                                name for name in self.cameras if name not in missing
                            ]
                            msg = f"{base_msg}: 未收到 [{', '.join(missing)}]; 已收到 [{', '.join(received)}]"
                        
                        else:  # 关节角度条件
                            # 对于关节角度条件，missing要么是空列表，要么是["recv_leader_joint_right"]或["recv_leader_joint_left"]
                            if i == 2:  # 右臂关节角度
                                data_source = self.robot_dora_node.recv_follower_joint_right
                            else:  # 左臂关节角度
                                data_source = self.robot_dora_node.recv_follower_joint_left
                            
                            received_count = len(data_source)
                            msg = f"{base_msg}: 未收到数据; 已收到 {received_count} 个数据点"

                        failed_messages.append(msg)

                # 如果所有条件都已完成，break
                if not failed_messages:
                    break

                # 抛出超时异常
                raise TimeoutError(
                    f"连接超时，未满足的条件: {'; '.join(failed_messages)}"
                )

            # 减少 CPU 占用
            time.sleep(0.01)

        # ===== 新增成功打印逻辑 =====
        success_messages = []
        # 摄像头连接状态
        if conditions[0][0]():
            cam_received = [
                name
                for name in self.cameras
                if name in self.robot_dora_node.recv_images and name not in self.connect_excluded_cameras_dora
            ]
            success_messages.append(f"摄像头dora: {', '.join(cam_received)}")
        if conditions[1][0]():
            cam_received = [
                name
                for name in self.cameras
                if name in self.robot_ros2_node.recv_images and name not in self.connect_excluded_cameras_ros2
            ]
            success_messages.append(f"摄像头ros2: {', '.join(cam_received)}")

        # 右臂关节角度状态
        if conditions[2][0]():
            # 检查recv_leader_joint_right字典中实际接收到的键
            success_messages.append(f"右臂关节角度: 已接收 ({len(self.robot_dora_node.recv_follower_joint_right)}个数据点)")

        # 左臂关节角度状态
        if conditions[3][0]():
            # 检查recv_leader_joint_left字典中实际接收到的键
            success_messages.append(f"左臂关节角度: 已接收 ({len(self.robot_dora_node.recv_follower_joint_left)}个数据点)")

        log_message = "\n[连接成功] 所有设备已就绪:\n"
        log_message += "\n".join(f"  - {msg}" for msg in success_messages)
        log_message += f"\n  总耗时: {time.perf_counter() - start_time:.2f} 秒\n"
        logger.info(log_message)
        # ===========================

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

        # Read arm position
        start = time.perf_counter()
        obs_dict = {}
        conv_recv_follower_endpose_left = np.zeros(7)
        conv_recv_follower_endpose_left[:3] = self.robot_dora_node.recv_follower_endpose_left[:3]
        conv_recv_follower_endpose_left[3:] = euler_xyz_to_quaternion(self.robot_dora_node.recv_follower_endpose_left[3:])
        conv_recv_follower_endpose_right = np.zeros(7)
        conv_recv_follower_endpose_right[:3] = self.robot_dora_node.recv_follower_endpose_right[:3]
        conv_recv_follower_endpose_right[3:] = euler_xyz_to_quaternion(self.robot_dora_node.recv_follower_endpose_right[3:])

        # Add right arm positions
        for i, motor in enumerate(self.motors):
            if "joint" in motor and "right" in motor:
                obs_dict[f"follower_{motor}.pos"] = self.robot_dora_node.recv_follower_joint_right[i]
            elif "gripper" in motor and "right" in motor:
                obs_dict[f"follower_{motor}.pos"] = self.robot_dora_node.recv_follower_joint_right[i]

            elif "joint" in motor and "left" in motor:
                obs_dict[f"follower_{motor}.pos"] = self.robot_dora_node.recv_follower_joint_left[i-7]
            elif "gripper" in motor and "left" in motor:
                obs_dict[f"follower_{motor}.pos"] = self.robot_dora_node.recv_follower_joint_left[i-7]

            elif "arm" in motor and "left" in motor and ("quat" in motor or "pos" in motor):
                obs_dict[f"follower_{motor}.pos"] = conv_recv_follower_endpose_left[i-14]
            elif "arm" in motor and "right" in motor and ("quat" in motor or "pos" in motor):
                obs_dict[f"follower_{motor}.pos"] = conv_recv_follower_endpose_right[i-21]
        
        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read state: {dt_ms:.1f} ms")

        # Capture images from cameras
        for cam_key, _cam in self.cameras.items():
            start = time.perf_counter()
            
            for name, val in self.robot_dora_node.recv_images.items():
                if cam_key in name:
                    obs_dict[cam_key] = val
            for name, val in self.robot_ros2_node.recv_images.items():
                if cam_key in name:
                    obs_dict[cam_key] = val
            
            dt_ms = (time.perf_counter() - start) * 1e3
            logger.debug(f"{self} read {cam_key}: {dt_ms:.1f} ms")

        return obs_dict

    def send_action(self, action: dict[str, Any]):
        """The provided action is expected to be a vector."""
        if not self.is_connected:
            raise DeviceNotConnectedError(
                f"{self} is not connected. You need to run `robot.connect()`."
            )

        right_arm_action = []
        left_arm_action = []

        right_gripper_action = []
        left_gripper_action = []

        for _i, motor in enumerate(self.actuators):
            if "arm" in motor and "right" in motor:
                right_arm_action.append(action[f"leader_{motor}.pos"])
            if "gripper" in motor and "right" in motor:
                right_gripper_action.append(action[f"leader_{motor}.pos"])

            if "arm" in motor and "left" in motor:
                left_arm_action.append(action[f"leader_{motor}.pos"])
            if "gripper" in motor and "left" in motor:
                left_gripper_action.append(action[f"leader_{motor}.pos"])

        # # Send right arm action
        # if right_arm_action:
        #     goal_numpy = np.array(right_arm_action, dtype=np.float32)
        #     self.robot_dora_node.dora_send(f"action_endpose_right", goal_numpy)
        
        # # Send left arm action
        # if left_arm_action:
        #     goal_numpy = np.array(left_arm_action, dtype=np.float32)
        #     self.robot_dora_node.dora_send(f"action_endpose_left", goal_numpy)

        # Send right arm action
        if right_arm_action:
            goal_numpy = np.array(right_arm_action, dtype=np.float32)
            
            # 假设 goal_numpy 的结构是 [x, y, z, qx, qy, qz, qw]
            if len(goal_numpy) == 7:
                # 提取位置和四元数
                position = goal_numpy[:3]  # x, y, z
                quaternion_wxyz = goal_numpy[3:]  # qw, qx, qy, qz
                
                # 重新排列四元数为 qx, qy, qz, qw
                quaternion_xyzw = np.array([
                    quaternion_wxyz[1],  # qx
                    quaternion_wxyz[2],  # qy
                    quaternion_wxyz[3],  # qz
                    quaternion_wxyz[0]   # qw
                ])
                
                # 将四元数转换为欧拉角 (弧度)
                rotation = Rotation.from_quat(quaternion_xyzw)
                euler_angles = rotation.as_euler('xyz', degrees=False)  # 使用 'xyz' 旋转顺序
                
                # 组合成新的数组 [x, y, z, rx, ry, rz]
                transformed_goal = np.concatenate([position, euler_angles], dtype=np.float32)
                
                self.robot_dora_node.dora_send(f"action_endpose_right", transformed_goal)
            else:
                print(f"Warning: Unexpected array length {len(goal_numpy)}. Expected 7.")
                self.robot_dora_node.dora_send(f"action_endpose_right", goal_numpy)

        # Send left arm action
        if left_arm_action:
            goal_numpy = np.array(left_arm_action, dtype=np.float32)
            
            if len(goal_numpy) == 7:
                # 提取位置和四元数
                position = goal_numpy[:3]  # x, y, z
                quaternion_wxyz = goal_numpy[3:]  # qw, qx, qy, qz
                
                # 重新排列四元数为 qx, qy, qz, qw
                quaternion_xyzw = np.array([
                    quaternion_wxyz[1],  # qx
                    quaternion_wxyz[2],  # qy
                    quaternion_wxyz[3],  # qz
                    quaternion_wxyz[0]   # qw
                ])
                
                # 将四元数转换为欧拉角 (弧度)
                rotation = Rotation.from_quat(quaternion_xyzw)
                euler_angles = rotation.as_euler('xyz', degrees=False)
                
                # 组合成新的数组 [x, y, z, rx, ry, rz]
                transformed_goal = np.concatenate([position, euler_angles], dtype=np.float32)
                
                self.robot_dora_node.dora_send(f"action_endpose_left", transformed_goal)
            else:
                print(f"Warning: Unexpected array length {len(goal_numpy)}. Expected 7.")
                self.robot_dora_node.dora_send(f"action_endpose_left", goal_numpy)


        # Send right arm action
        if right_gripper_action:
            goal_numpy = np.array(right_gripper_action, dtype=np.float32)
            goal_numpy = goal_numpy / 100
            self.robot_dora_node.dora_send(f"action_gripper_right", goal_numpy)
        
        # Send left arm action
        if left_gripper_action:
            goal_numpy = np.array(left_gripper_action, dtype=np.float32)
            goal_numpy = goal_numpy / 100
            self.robot_dora_node.dora_send(f"action_gripper_left", goal_numpy)

    def get_node(self):
        return self.robot_ros2_node

    def disconnect(self):
        if not self.is_connected:
            raise DeviceNotConnectedError(
                "Agilex Aloha is not connected. You need to run `robot.connect()` before disconnecting."
            )

        self.robot_dora_node.running = False
        self.robot_dora_node.stop()

        if hasattr(self, "robot_ros2_node"):
            self.robot_ros2_node.destroy()

        self.connected = False

    def __del__(self):
        if getattr(self, "is_connected", False):
            self.disconnect()
