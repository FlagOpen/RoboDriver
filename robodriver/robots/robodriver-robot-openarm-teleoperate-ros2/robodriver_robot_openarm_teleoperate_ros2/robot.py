import time
from typing import Any

import logging_mp
import numpy as np
from lerobot.cameras import make_cameras_from_configs
from lerobot.robots.robot import Robot
from lerobot.utils.errors import DeviceNotConnectedError, DeviceAlreadyConnectedError
from functools import cached_property

import rclpy  # ROS2 Python客户端库

from .config import OPENARMRos2RobotConfig
from .status import OPENARMRos2RobotStatus
from .node import OPENARMRos2RobotNode


logger = logging_mp.get_logger(__name__)



class OPENARMRos2Robot(Robot):

    config_class = OPENARMRos2RobotConfig
    name = "openarm-teleop-ros2"

    def __init__(self, config: OPENARMRos2RobotConfig):
        super().__init__(config)
        # 基础配置
        self.config = config
        self.robot_type = self.config.type
        self.use_videos = self.config.use_videos
        self.microphones = self.config.microphones

        # 电机和相机配置
        #self.leader_motors = config.leader_motors  # 主臂电机（用于示教）
        self.follower_motors = config.follower_motors  # 从臂电机（执行动作）
        self.cameras = make_cameras_from_configs(self.config.cameras)
        # 连接时排除的相机列表（如姿态估计相机）
        self.connect_excluded_cameras = ["image_pika_pose"]

        # 初始化机器人状态对象
        self.status = OPENARMRos2RobotStatus()
        
        # 初始化ROS2
        if not rclpy.ok():
            rclpy.init()
        
        # 创建 ROS2 节点；spin 由 robodriver.scripts.run 中的 ROS2_NodeManager 统一调度
        self.robot_ros2_node = OPENARMRos2RobotNode()

        # 状态标志
        self.connected = False
        self.logs = {}  # 日志记录

    @staticmethod
    def _motor_key_to_ros_joint_name(motor_key: str) -> str:
        """将配置中的关节键映射为 /joint_states 里的关节名。"""
        if motor_key.startswith("left_arm_joint_") and motor_key.endswith("_rad"):
            joint_idx = motor_key.replace("left_arm_joint_", "").replace("_rad", "")
            return f"openarm_left_joint{joint_idx}"
        if motor_key.startswith("right_arm_joint_") and motor_key.endswith("_rad"):
            joint_idx = motor_key.replace("right_arm_joint_", "").replace("_rad", "")
            return f"openarm_right_joint{joint_idx}"
        if motor_key == "left_gripper_degree_mm":
            return "openarm_left_finger_joint1"
        if motor_key == "right_gripper_degree_mm":
            return "openarm_right_finger_joint1"
        return ""

    @property
    def _follower_motors_ft(self) -> dict[str, type]:
        return {
            f"follower_{joint_name}.pos": float
            for comp_name, joints in self.follower_motors.items()
            for joint_name in joints.keys()
        }
    
    #@property
    #def _leader_motors_ft(self) -> dict[str, type]:
    #    return {
    #        f"leader_{joint_name}.pos": float
    #        for comp_name, joints in self.leader_motors.items()
    #        for joint_name in joints.keys()
    #    }
    @property
    def _camera_ft(self) -> dict[str, tuple]:
        return {
            cam: (self.config.cameras[cam].height, self.config.cameras[cam].width, 3) for cam in self.cameras
        }

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
       
        return {**self._follower_motors_ft, **self._camera_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        # 参考其他机器人实现：action_features 需与 get_action 输出键保持一致，
        # 否则数据集不会生成 action 列，后续数据校验会因缺少 "action" 失败。
        return self._follower_motors_ft
    
    @property
    def is_connected(self) -> bool:
        return self.connected
    
    def connect(self):
        timeout = 20  # 统一的超时时间（秒）
        start_time = time.perf_counter()

        # 检查是否已连接
        if self.connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        conditions = [
            # 注释掉主臂检查
            #(
            #    lambda: all(
            #        any(name in key for key in self.robot_ros2_node.recv_leader)
            #        for name in self.leader_motors
            #    ),
            #    lambda: [
            #        name
            #        for name in self.leader_motors
            #        if not any(name in key for key in self.robot_ros2_node.recv_leader)
            #    ],
            #    "等待主臂关节角度超时",
            #),
            
            # 仅保留从臂检查
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


            (
                lambda: all(
                    name in self.robot_ros2_node.recv_images
                    for name in self.cameras
                    if name not in self.connect_excluded_cameras
                ),
                lambda: [name for name in self.cameras if name not in self.robot_ros2_node.recv_images],
                "等待摄像头图像超时",
            ),
        ]

       
        completed = [False] * len(conditions)

       
        while True:
            for i in range(len(conditions)):
                if not completed[i]:
                    condition_func = conditions[i][0]
                    if condition_func():
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

                    # 只有从臂数据，i始终为0
                    elif i == 0:
                        received = [
                            name
                            for name in self.follower_motors
                            if name not in missing
                        ]
                    elif i == 1:
                        received = [
                            name
                            for name in self.cameras
                            if name not in missing
                        ]
                    

                    msg = (
                        f"{base_msg}: 未收到 [{', '.join(missing)}]; "
                        f"已收到 [{', '.join(received)}]"
                    )
                    failed_messages.append(msg)

                # 如果所有条件都满足了，退出
                if not failed_messages:
                    break

                # 抛出超时异常，包含详细的失败信息
                raise TimeoutError(
                    f"连接超时，未满足的条件: {'; '.join(failed_messages)}"
                )

            # 减少 CPU 占用，避免忙等待
            time.sleep(0.01)

        # ===== 打印连接成功信息 =====
        success_messages = []
        """
         if conditions[0][0]():
            leader_received = [
                name
                for name in self.leader_motors
                if any(name in key for key in self.robot_ros2_node.recv_leader)
            ]
            success_messages.append(f"主臂数据: {', '.join(leader_received)}")

        """

        
        if conditions[0][0]():
            follower_received = [
                name
                for name in self.follower_motors
                if any(name in key for key in self.robot_ros2_node.recv_follower)
            ]
            success_messages.append(f"从臂数据: {', '.join(follower_received)}")
        if conditions[1][0]():
            cam_received = [
                name
                for name in self.cameras
                if name in self.robot_ros2_node.recv_images
                and name not in self.connect_excluded_cameras
            ]
            success_messages.append(f"摄像头: {', '.join(cam_received)}")



        # 构建并记录成功日志
        log_message = "\n[连接成功] 所有设备已就绪:\n"
        log_message += "\n".join(f"  - {msg}" for msg in success_messages)
        log_message += f"\n  总耗时: {time.perf_counter() - start_time:.2f} 秒\n"
        logger.info(log_message)
        # ===========================

        # 更新状态对象中的连接状态
        # 将所有相机标记为已连接（暂时注释掉，没有相机配置）
        for i in range(self.status.specifications.camera.number):
            self.status.specifications.camera.information[i].is_connect = True
        
        # 将所有机械臂标记为已连接（暂时注释掉，没有机械臂规格）
        for i in range(self.status.specifications.arm.number):
            self.status.specifications.arm.information[i].is_connect = True

        # 设置全局连接标志
        self.connected = True

    @property
    def is_calibrated(self) -> bool:
        return True

    def get_node(self) -> OPENARMRos2RobotNode:
        """返回 ROS2 节点，供 ROS2_NodeManager 注册并 spin。"""
        return self.robot_ros2_node

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass
    
    def get_observation(self) -> dict[str, Any]:
        if not self.connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        start = time.perf_counter()
        obs_dict: dict[str, Any] = {}
        
        # 读取从臂关节位置
        for comp_name, joints in self.follower_motors.items():
            for follower_name, follower in self.robot_ros2_node.recv_follower.items():
                if follower_name == comp_name:
                    joint_keys = list(joints.keys())
                    if isinstance(follower, dict):
                        for joint_name in joint_keys:
                            ros_joint_name = self._motor_key_to_ros_joint_name(joint_name)
                            if ros_joint_name and ros_joint_name in follower:
                                obs_dict[f"follower_{joint_name}.pos"] = float(
                                    follower[ros_joint_name]
                                )
                    else:
                        # 兼容旧格式：按顺序映射
                        num_joints = min(len(joint_keys), len(follower))
                        for i in range(num_joints):
                            joint_name = joint_keys[i]
                            obs_dict[f"follower_{joint_name}.pos"] = float(follower[i])

        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read follower state: {dt_ms:.1f} ms")
        
        # 读取相机图像
        for cam_key, _cam in self.cameras.items():
            start_cam = time.perf_counter()
            for name, val in self.robot_ros2_node.recv_images.items():
                if cam_key == name or cam_key in name:
                    obs_dict[cam_key] = val
                    break
            dt_ms = (time.perf_counter() - start_cam) * 1e3
            logger.debug(f"{self} read {cam_key}: {dt_ms:.1f} ms")
        
        return obs_dict

    def get_action(self) -> dict[str, Any]:
        if not self.connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        start = time.perf_counter()
        act_dict: dict[str, Any] = {}
         # 读取从臂关节位置
        for comp_name, joints in self.follower_motors.items():
            for follower_name, follower in self.robot_ros2_node.recv_follower.items():
                if follower_name == comp_name:
                    joint_keys = list(joints.keys())
                    if isinstance(follower, dict):
                        for joint_name in joint_keys:
                            ros_joint_name = self._motor_key_to_ros_joint_name(joint_name)
                            if ros_joint_name and ros_joint_name in follower:
                                act_dict[f"follower_{joint_name}.pos"] = float(
                                    follower[ros_joint_name]
                                )
                    else:
                        # 兼容旧格式：按顺序映射
                        num_joints = min(len(joint_keys), len(follower))
                        for i in range(num_joints):
                            joint_name = joint_keys[i]
                            act_dict[f"follower_{joint_name}.pos"] = float(follower[i])

        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read follower state: {dt_ms:.1f} ms")

        # 读取相机图像
        for cam_key, _cam in self.cameras.items():
            start = time.perf_counter()
            for name, val in self.robot_ros2_node.recv_images.items():
                if cam_key == name or cam_key in name:
                    act_dict[cam_key] = val
                    break
            dt_ms = (time.perf_counter() - start) * 1e3
            logger.debug(f"{self} read {cam_key}: {dt_ms:.1f} ms")

        return act_dict
        
   

    def send_action(self, action: dict[str, Any]):
        if not self.is_connected:
            raise DeviceNotConnectedError(
                "机器人未连接。需要先运行 `robot.connect()`。"
            )
        
        # 从动作字典提取值列表
        goal_joint = [val for key, val in action.items()]
        goal_joint_numpy = np.array(
            [
                float(t.item()) if hasattr(t, "item") else float(t)
                for t in goal_joint
            ],
            dtype=np.float32,
        )
        
        try:
            # 验证动作向量维度
            expected_dim = len(self._follower_motors_ft)
            if goal_joint_numpy.shape != (expected_dim,):
                raise ValueError(
                    f"动作向量必须是{expected_dim}维，实际为{goal_joint_numpy.shape[0]}维"
                )
            
            # 通过ROS2节点发布动作指令
            self.robot_ros2_node.ros_replay(goal_joint_numpy)
            
        except Exception as e:
            logger.error(f"发送动作失败: {e}")
            raise

    def update_status(self) -> str:

        # 跳过主臂状态更新（本系统不使用主臂）
        # 主臂相关代码已在node.py中注释掉
        
        # 更新从臂连接状态
        for i in range(self.status.specifications.arm.number):
            match_name = self.status.specifications.arm.information[i].name
            for name in self.robot_ros2_node.recv_follower_status:
                if match_name in name:
                    self.status.specifications.arm.information[i].is_connect = (
                        True if self.robot_ros2_node.recv_follower_status[name] > 0 else False
                    )
        # 更新相机连接状态
        for i in range(self.status.specifications.camera.number):
            match_name = self.status.specifications.camera.information[i].name
            for name in self.robot_ros2_node.recv_images_status:
                if match_name in name:
                    # 如果状态值>0则表示连接正常
                    self.status.specifications.camera.information[i].is_connect = (
                        True if self.robot_ros2_node.recv_images_status[name] > 0 else False
                    )


        # 返回JSON格式的状态信息
        return self.status.to_json()

    def disconnect(self):
        if not self.is_connected:
            raise DeviceNotConnectedError(
                "机器人未连接。需要先运行 `robot.connect()` 才能断开连接。"
            )
        
        # 销毁ROS2节点
        if hasattr(self, "robot_ros2_node"):
            self.robot_ros2_node.destroy()
        
        # 关闭ROS2
        if rclpy.ok():
            rclpy.shutdown()

        # 更新连接状态
        self.connected = False

    def __del__(self):
        try:
            if getattr(self, "is_connected", False):
                self.disconnect()
        except Exception:
            pass  # 忽略析构时的异常
