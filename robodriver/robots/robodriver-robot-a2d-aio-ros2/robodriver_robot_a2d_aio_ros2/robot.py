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
        # self.config = config
        # self.robot_type = self.config.type
        # self.use_videos = self.config.use_videos
        # self.microphones = self.config.microphones

        # # 这里的 leader_motors / follower_motors 可以是按组件分组的 dict
        # # （比如 {"leader_arm": {...}, "left_arm": {...}}）
        # self.leader_motors = config.leader_motors
        # self.follower_motors = config.follower_motors
        # self.cameras = make_cameras_from_configs(self.config.cameras)

        # self.connect_excluded_cameras = ["image_pika_pose"]

        # self.robot_ros2_node = A2DAioRos2Node()
        # self.robot_ros2_node.start()

        # self.connected = False
        # self.logs = {}
        self.config = config
        self.leader_motors = config.leader_motors
        self.follower_motors = config.follower_motors
        self.cameras = make_cameras_from_configs(self.config.cameras)

        self.robot_node = A2DAioRos2Node()
        self.robot_node.start()

        self.connected = False

    # ========= features =========

    # @property
    # def _follower_motors_ft(self) -> dict[str, type]:
    #     return {
    #         f"follower_{motor}.pos": float
    #         for motor in self.follower_motors
    #     }
    
    # @property
    # def _leader_motors_ft(self) -> dict[str, type]:
    #     return {
    #         f"leader_{motor}.pos": float
    #         for motor in self.leader_motors
    #     }

    # @property
    # def _cameras_ft(self) -> dict[str, tuple]:
    #     return {
    #         cam: (
    #             self.config.cameras[cam].height,
    #             self.config.cameras[cam].width,
    #             3,
    #         )
    #         for cam in self.cameras
    #     }

    # @cached_property
    # def observation_features(self) -> dict[str, Any]:
    #     return {**self._follower_motors_ft, **self._cameras_ft}

    # @cached_property
    # def action_features(self) -> dict[str, Any]:
    #     return self._leader_motors_ft

    # @property
    # def is_connected(self) -> bool:
    #     return self.connected

    # ========= connect / disconnect =========

    # def connect(self):
    #     timeout = 20
    #     start_time = time.perf_counter()

    #     if self.connected:
    #         raise DeviceAlreadyConnectedError(f"{self} already connected")

    #     # 约定：node 里有 recv_images / recv_follower / recv_leader
    #     conditions = [
    #         # 摄像头图像
    #         (
    #             lambda: all(
    #                 name in self.robot_ros2_node.recv_images
    #                 for name in self.cameras
    #                 if name not in self.connect_excluded_cameras
    #             ),
    #             lambda: [
    #                 name
    #                 for name in self.cameras
    #                 if name not in self.robot_ros2_node.recv_images
    #                 and name not in self.connect_excluded_cameras
    #             ],
    #             "等待摄像头图像超时",
    #         ),
    #         # 主臂
    #         (
    #             lambda: all(
    #                 any(name in key for _leader_name, leader in self.robot_ros2_node.recv_leader.items() for key in leader)
    #                 for name in self.leader_motors
    #             ),
    #             lambda: [
    #                 name
    #                 for name in self.leader_motors
    #                 if not any(name in key for _leader_name, leader in self.robot_ros2_node.recv_leader.items() for key in leader)
    #             ],
    #             "等待主臂数据超时",
    #         ),
    #         # 从臂
    #         (
    #             lambda: all(
    #                 any(name in key for _follower_name, follower in self.robot_ros2_node.recv_follower.items() for key in follower)
    #                 for name in self.follower_motors
    #             ),
    #             lambda: [
    #                 name
    #                 for name in self.follower_motors
    #                 if not any(name in key for _follower_name, follower in self.robot_ros2_node.recv_follower.items() for key in follower)
    #             ],
    #             "等待从臂数据超时",
    #         ),
    #     ]

    #     completed = [False] * len(conditions)

    #     while True:
    #         for i, (cond, _get_missing, _msg) in enumerate(conditions):
    #             if not completed[i] and cond():
    #                 completed[i] = True

    #         if all(completed):
    #             break

    #         if time.perf_counter() - start_time > timeout:
    #             failed_messages = []
    #             for i, (cond, get_missing, base_msg) in enumerate(conditions):
    #                 if completed[i]:
    #                     continue

    #                 missing = get_missing()
    #                 if cond() or not missing:
    #                     completed[i] = True
    #                     continue

    #                 if i == 0:
    #                     received = [
    #                         name
    #                         for name in self.cameras
    #                         if name not in missing
    #                     ]
    #                 elif i == 1:
    #                     received = [
    #                         name
    #                         for name in self.leader_motors
    #                         if name not in missing
    #                     ]
    #                 else:
    #                     received = [
    #                         name
    #                         for name in self.follower_motors
    #                         if name not in missing
    #                     ]

    #                 msg = (
    #                     f"{base_msg}: 未收到 [{', '.join(missing)}]; "
    #                     f"已收到 [{', '.join(received)}]"
    #                 )
    #                 failed_messages.append(msg)

    #             if not failed_messages:
    #                 break

    #             raise TimeoutError(
    #                 f"连接超时，未满足的条件: {{'; '.join(failed_messages)}}"
    #             )

    #         time.sleep(0.01)

    #     # 成功日志
    #     success_messages = []

    #     if conditions[0][0]():
    #         cam_received = [
    #             name
    #             for name in self.cameras
    #             if name in self.robot_ros2_node.recv_images
    #             and name not in self.connect_excluded_cameras
    #         ]
    #         success_messages.append(f"摄像头: {', '.join(cam_received)}")

    #     if conditions[1][0]():
    #         leader_received = [
    #             name
    #             for name in self.leader_motors
    #             if any(name in key for _leader_name, leader in self.robot_ros2_node.recv_leader.items() for key in leader)
    #         ]
    #         success_messages.append(f"主臂数据: {', '.join(leader_received)}")

    #     if conditions[2][0]():
    #         follower_received = [
    #             name
    #             for name in self.follower_motors
    #             if any(name in key for _follower_name, follower in self.robot_ros2_node.recv_follower.items() for key in follower)
    #         ]
    #         success_messages.append(f"从臂数据: {', '.join(follower_received)}")

    #     log_message = "\n[连接成功] 所有设备已就绪:\n"
    #     log_message += "\n".join(f"  - {msg}" for msg in success_messages)
    #     log_message += f"\n  总耗时: {time.perf_counter() - start_time:.2f} 秒\n"
    #     logger.info(log_message)

    #     self.connected = True

    # def disconnect(self):
    #     if not self.connected:
    #         raise DeviceNotConnectedError()
    #     self.connected = False

    # def __del__(self):
    #     if getattr(self, "connected", False):
    #         self.disconnect()
    @property
    def is_connected(self) -> bool:
        return self.connected

    def connect(self):
        if self.connected: return

        logger.info("Connecting to A2D AIO Robot...")
        start_time = time.time()
        while time.time() - start_time < 10:
            # 检查是否有手臂数据更新
            has_arm = self.robot_node.recv_follower_status.get("arm", 0) > 0
            
            # 可以在此添加相机检查逻辑
            
            if has_arm:
                self.connected = True
                logger.info("A2D Robot Connected!")
                return
            time.sleep(0.1)
        
        raise TimeoutError("Failed to connect to A2D ROS2 topics (Timeout).")

    def disconnect(self):
        self.robot_node.stop()
        self.connected = False

    # ========= calibrate / configure =========

    def calibrate(self):
        pass

    def configure(self):
        pass

    @property
    def is_calibrated(self):
        return True

    # ========= obs / action =========

    # def get_observation(self) -> dict[str, Any]:
    #     if not self.connected:
    #         raise DeviceNotConnectedError(f"{self} is not connected.")

    #     start = time.perf_counter()
    #     obs_dict: dict[str, Any] = {}

    #     for name in self.follower_motors:
    #         for _follower_name, follower in self.robot_ros2_node.recv_follower.items():
    #             for key, value in follower.items():
    #                 if name == key:
    #                     obs_dict[f"follower_{name}.pos"] = float(value)

    #     # # ---- 逐组件展开，然后逐 joint 填入 ----
    #     # for comp_name, joints in self.follower_motors.items():

    #     #     # node 中按组件名存放关节数组，例如：
    #     #     # self.recv_follower["follower_arm"] = np.array([... 6 joints ...])
    #     #     vec = self.robot_ros2_node.recv_follower.get(comp_name)
    #     #     if vec is None:
    #     #         continue

    #     #     # joints.keys() = ["joint_0", ..., "joint_5"]
    #     #     joint_names = list(joints.keys())

    #     #     for idx, joint in enumerate(joint_names):
    #     #         if idx >= len(vec):
    #     #             break
    #     #         obs_dict[f"follower_{joint}.pos"] = float(vec[idx])

    #     dt_ms = (time.perf_counter() - start) * 1e3
    #     logger.debug(f"{self} read follower state: {dt_ms:.1f} ms")

    #     # ---- 摄像头图像保持不变 ----
    #     for cam_key, _cam in self.cameras.items():
    #         start = time.perf_counter()
    #         for name, val in self.robot_ros2_node.recv_images.items():
    #             if cam_key == name or cam_key in name:
    #                 obs_dict[cam_key] = val
    #                 break
    #         dt_ms = (time.perf_counter() - start) * 1e3
    #         logger.debug(f"{self} read {cam_key}: {dt_ms:.1f} ms")

    #     return obs_dict


    def get_observation(self) -> dict[str, Any]:
        if not self.connected:
            raise DeviceNotConnectedError()

        obs_dict = {}
        
        # 1. 组装电机数据 (Arm + Gripper)
        for name in self.follower_motors:
            # 映射 node 中的 key (例如 'left_arm_joint1', 'left_gripper')
            val = self.robot_node.recv_follower.get(name)
            if val is not None:
                obs_dict[f"follower_{name}.pos"] = float(val)

        # 2. 组装图像数据
        for cam_key in self.cameras:
            img = self.robot_node.recv_images.get(cam_key)
            if img is not None:
                obs_dict[cam_key] = img

        return obs_dict
    
    # def get_action(self) -> dict[str, Any]:
    #     if not self.connected:
    #         raise DeviceNotConnectedError(f"{self} is not connected.")

    #     start = time.perf_counter()
    #     act_dict: dict[str, Any] = {}

    #     for name in self.leader_motors:
    #         for _leader_name, leader in self.robot_ros2_node.recv_leader.items():
    #             for key, value in leader.items():
    #                 if name == key:
    #                     act_dict[f"leader_{name}.pos"] = float(value)

    #     # # ---- 逐组件展开，然后逐 joint 填入 ----
    #     # for comp_name, joints in self.leader_motors.items():

    #     #     # node 中按组件名存放关节数组
    #     #     vec = self.robot_ros2_node.recv_leader.get(comp_name)
    #     #     if vec is None:
    #     #         continue

    #     #     joint_names = list(joints.keys())

    #     #     for idx, joint in enumerate(joint_names):
    #     #         if idx >= len(vec):
    #     #             break
    #     #         act_dict[f"leader_{joint}.pos"] = float(vec[idx])

    #     dt_ms = (time.perf_counter() - start) * 1e3
    #     logger.debug(f"{self} read action: {dt_ms:.1f} ms")

    #     return act_dict



    def get_action(self) -> dict[str, Any]:
        """
        用于数据采集时的 Leader 状态获取
        """
        # 如果有 Leader 设备逻辑，在此实现
        return {}
    
    

    # ========= send_action =========

    # def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
    #     """The provided action is expected to be a vector."""
    #     if not self.is_connected:
    #         raise DeviceNotConnectedError(
    #             f"{self} is not connected. You need to run `robot.connect()`."
    #         )

    #     # goal_joint = [val for _key, val in action.items()]
    #     # goal_joint_numpy = np.array(goal_joint, dtype=np.float32)

    #     # Extract motor names from keys like 'leader_elbow.pos' -> 'elbow'
    #     cleaned_action = {}
    #     for key, value in action.items():
    #         if key.startswith("leader_") and key.endswith(".pos"):
    #             motor = key[len("leader_"):-len(".pos")]
    #             cleaned_action[motor] = value
    #         else:
    #             raise ValueError(f"Unexpected action key format: {key}. Expected 'leader_{{motor}}.pos'.")

    #     # Send the cleaned action to the ROS 2 node
    #     self.robot_ros2_node.ros2_send(cleaned_action)

    #     return {f"{arm_motor}.pos": val for arm_motor, val in action.items()}


    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        """
        将 Action 分发给 Node
        """
        if not self.connected:
            raise DeviceNotConnectedError()

        arm_joints = {}
        gripper_joints = {}
        
        for key, val in action.items():
            # key 格式预期: "follower_left_arm_joint1.pos" (训练回放) 或其他
            # 提取核心关节名
            raw_name = key.replace("follower_", "").replace("leader_", "").replace(".pos", "")
            
            if "gripper" in raw_name:
                gripper_joints[raw_name] = float(val)
            else:
                arm_joints[raw_name] = float(val)

        self.robot_node.send_command(arm_joints, gripper_joints)

        return action

