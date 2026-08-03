import threading
import time
from typing import Any

import logging_mp
import numpy as np
from lerobot.cameras import make_cameras_from_configs
from lerobot.robots.robot import Robot
from lerobot.utils.errors import DeviceNotConnectedError, DeviceAlreadyConnectedError
from functools import cached_property

import rospy  # 替换rclpy为rospy

# 导入ROS1版本的配置、状态和节点（与你提供的ROS1节点文件对应）
from .config import LEJUKuavoRos1Config  # 改为ROS1配置类
from .status import LEJUKuavoRos1RobotStatus  # 改为ROS1状态类（需确保status.py已适配ROS1）
from .node import LEJUKuavoRos1Node, ros_spin_thread  # 导入ROS1节点和spin函数


logger = logging_mp.get_logger(__name__)


class LEJUKuavoRos1Robot(Robot):  # 类名改为ROS1标识
    config_class = LEJUKuavoRos1Config  # 关联ROS1配置类
    name = "leju-kuavo-teleop-ros1"  # 名称改为ROS1版本标识

    def __init__(self, config: LEJUKuavoRos1Config):
        super().__init__(config)
        #init
        self.config = config
        self.robot_type = self.config.type
        self.use_videos = self.config.use_videos
        self.microphones = self.config.microphones

        self.follower_motors = config.follower_motors
        self.cameras = make_cameras_from_configs(self.config.cameras)
        
        self.connect_excluded_cameras = ["image_pika_pose"]

        self.status = LEJUKuavoRos1RobotStatus()
        if not rospy.core.is_initialized():
            rospy.init_node('ros1_recv_pub_driver', anonymous=True)
            logger.info(f"✅ 初始化ROS节点：{rospy.get_name()}")
        else:
            logger.info(f"✅ 复用已存在的ROS节点：{rospy.get_name()}")
        
        logger.info(f"✅ ROS节点状态：is_shutdown={rospy.is_shutdown()}")
        
        self.robot_ros1_node = LEJUKuavoRos1Node()  # 创建ROS1节点实例（替换ROS2节点）
        logger.info("✅ ROS1节点实例创建成功")
        
        self.ros_spin_thread = threading.Thread(
            target=ros_spin_thread, 
            args=(self.robot_ros1_node,),  # 传入ROS1节点实例
            daemon=True
        )
        self.ros_spin_thread.start()
        logger.info("✅ ROS spin线程启动成功")

        self.connected = False
        self.logs = {}
        self._last_valid_follower: dict[str, np.ndarray] = {}
        self.required_follower_keys = ["right_arm", "left_arm", "head"]

    @property
    def _follower_motors_ft(self) -> dict[str, type]:
        features: dict[str, type] = {}
        for i in range(7):
            features[f"follower_left_joint_{i}.pos"] = float
        for n in ["l_thumb", "l_thumb_aux", "l_index", "l_middle", "l_ring", "l_pinky"]:
            features[f"follower_{n}.pos"] = float
        for i in range(7):
            features[f"follower_right_joint_{i}.pos"] = float
        for n in ["r_thumb", "r_thumb_aux", "r_index", "r_middle", "r_ring", "r_pinky"]:
            features[f"follower_{n}.pos"] = float
        features["follower_head_yaw.pos"] = float
        features["follower_head_pitch.pos"] = float
        return features
    


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

        # 定义所有需要等待的条件及其错误信息（替换ROS2节点为ROS1节点）
        conditions = [
    (
        # 摄像头：只要 recv_images 非空（且排除项外至少有一个）
        lambda: len([
            name for name in self.cameras.keys()
            if name not in self.connect_excluded_cameras
        ]) == 0 or len(self.robot_ros1_node.recv_images) > 0,
        lambda: [] if len(self.robot_ros1_node.recv_images) > 0 else list(self.cameras.keys()),
        "等待摄像头图像超时",
    ),
    (
        # 从臂：至少需要双臂+头部到位，避免仅手部数据导致误判成功
        lambda: all(k in self.robot_ros1_node.recv_follower for k in self.required_follower_keys),
        lambda: [k for k in self.required_follower_keys if k not in self.robot_ros1_node.recv_follower],
        "等待从臂关节角度超时",
    ),
        ]      

        # 跟踪每个条件是否已完成
        completed = [False] * len(conditions)

        while True:
            # 增加ROS1节点状态检查（替换rclpy.ok()）
            if rospy.is_shutdown():
                raise RuntimeError("ROS1节点已关闭，无法完成机器人连接")

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
                    f"连接超时，未满足的条件: {'; '.join(failed_messages)}; "
                    f"从臂回调计数 body={getattr(self.robot_ros1_node, 'body_msg_count', 0)}, "
                    f"hand={getattr(self.robot_ros1_node, 'hand_msg_count', 0)}"
                )

            # 减少 CPU 占用
            time.sleep(0.01)

        # ===== 新增成功打印逻辑 =====
        success_messages = []

        if conditions[0][0]():
            cam_received = [
                name
                for name in self.cameras
                if name in self.robot_ros1_node.recv_images
                and name not in self.connect_excluded_cameras
            ]
            success_messages.append(f"摄像头: {', '.join(cam_received)}")

        if conditions[1][0]():
            follower_received = [
                name for name in self.required_follower_keys
                if name in self.robot_ros1_node.recv_follower
            ]
            success_messages.append(f"从臂数据: {', '.join(follower_received)}")
        else:
            success_messages.append(
                f"从臂回调计数: body={getattr(self.robot_ros1_node, 'body_msg_count', 0)}, "
                f"hand={getattr(self.robot_ros1_node, 'hand_msg_count', 0)}"
            )

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
        if self.robot_ros1_node.recv_follower:
            self._last_valid_follower = {
                k: np.array(v, dtype=np.float32)
                for k, v in self.robot_ros1_node.recv_follower.items()
            }
        follower_source = self._last_valid_follower if self._last_valid_follower else self.robot_ros1_node.recv_follower

        # 读取从臂数据（28维：左臂7 + 左手6 + 右臂7 + 右手6 + 头2）
        left_arm = follower_source.get("left_arm", np.zeros(7, dtype=np.float32))
        for i in range(7):
            obs_dict[f"follower_left_joint_{i}.pos"] = float(left_arm[i]) if i < len(left_arm) else 0.0

        left_hand = follower_source.get("left_dexhand", np.zeros(6, dtype=np.float32))
        left_names = ["l_thumb", "l_thumb_aux", "l_index", "l_middle", "l_ring", "l_pinky"]
        for i, name in enumerate(left_names):
            obs_dict[f"follower_{name}.pos"] = float(left_hand[i]) if i < len(left_hand) else 0.0

        right_arm = follower_source.get("right_arm", np.zeros(7, dtype=np.float32))
        for i in range(7):
            obs_dict[f"follower_right_joint_{i}.pos"] = float(right_arm[i]) if i < len(right_arm) else 0.0

        right_hand = follower_source.get("right_dexhand", np.zeros(6, dtype=np.float32))
        right_names = ["r_thumb", "r_thumb_aux", "r_index", "r_middle", "r_ring", "r_pinky"]
        for i, name in enumerate(right_names):
            obs_dict[f"follower_{name}.pos"] = float(right_hand[i]) if i < len(right_hand) else 0.0

        head = follower_source.get("head", np.zeros(2, dtype=np.float32))
        obs_dict["follower_head_yaw.pos"] = float(head[0]) if len(head) > 0 else 0.0
        obs_dict["follower_head_pitch.pos"] = float(head[1]) if len(head) > 1 else 0.0

        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read follower state: {dt_ms:.1f} ms")

        # 读取摄像头图像（替换ROS2节点为ROS1节点）
        for cam_key, _cam in self.cameras.items():
            start = time.perf_counter()
            for name, val in self.robot_ros1_node.recv_images.items():
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
        if self.robot_ros1_node.recv_follower:
            self._last_valid_follower = {
                k: np.array(v, dtype=np.float32)
                for k, v in self.robot_ros1_node.recv_follower.items()
            }
        follower_source = self._last_valid_follower if self._last_valid_follower else self.robot_ros1_node.recv_follower

        # 读取从臂关节位置（28维：左臂7 + 左手6 + 右臂7 + 右手6 + 头2）
        left_arm = follower_source.get("left_arm", np.zeros(7, dtype=np.float32))
        for i in range(7):
            act_dict[f"follower_left_joint_{i}.pos"] = float(left_arm[i]) if i < len(left_arm) else 0.0

        left_hand = follower_source.get("left_dexhand", np.zeros(6, dtype=np.float32))
        left_names = ["l_thumb", "l_thumb_aux", "l_index", "l_middle", "l_ring", "l_pinky"]
        for i, name in enumerate(left_names):
            act_dict[f"follower_{name}.pos"] = float(left_hand[i]) if i < len(left_hand) else 0.0

        right_arm = follower_source.get("right_arm", np.zeros(7, dtype=np.float32))
        for i in range(7):
            act_dict[f"follower_right_joint_{i}.pos"] = float(right_arm[i]) if i < len(right_arm) else 0.0

        right_hand = follower_source.get("right_dexhand", np.zeros(6, dtype=np.float32))
        right_names = ["r_thumb", "r_thumb_aux", "r_index", "r_middle", "r_ring", "r_pinky"]
        for i, name in enumerate(right_names):
            act_dict[f"follower_{name}.pos"] = float(right_hand[i]) if i < len(right_hand) else 0.0

        head = follower_source.get("head", np.zeros(2, dtype=np.float32))
        act_dict["follower_head_yaw.pos"] = float(head[0]) if len(head) > 0 else 0.0
        act_dict["follower_head_pitch.pos"] = float(head[1]) if len(head) > 1 else 0.0

        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read follower state: {dt_ms:.1f} ms")

        # 读取相机图像
        for cam_key, _cam in self.cameras.items():
            start = time.perf_counter()
            for name, val in self.robot_ros1_node.recv_images.items():
                if cam_key == name or cam_key in name:
                    act_dict[cam_key] = val
                    break
            dt_ms = (time.perf_counter() - start) * 1e3
            logger.debug(f"{self} read {cam_key}: {dt_ms:.1f} ms")

        return act_dict
        
   
    

    def send_action(self, action: dict[str, Any]):
        """The provided action is expected to be a vector."""
        if not self.is_connected:
            raise DeviceNotConnectedError(
                "LEJUKuavo is not connected. You need to run `robot.connect()`."
            )
        expected_action_keys = [
            "follower_left_joint_0.pos",
            "follower_left_joint_1.pos",
            "follower_left_joint_2.pos",
            "follower_left_joint_3.pos",
            "follower_left_joint_4.pos",
            "follower_left_joint_5.pos",
            "follower_left_joint_6.pos",
            "follower_l_thumb.pos",
            "follower_l_thumb_aux.pos",
            "follower_l_index.pos",
            "follower_l_middle.pos",
            "follower_l_ring.pos",
            "follower_l_pinky.pos",
            "follower_right_joint_0.pos",
            "follower_right_joint_1.pos",
            "follower_right_joint_2.pos",
            "follower_right_joint_3.pos",
            "follower_right_joint_4.pos",
            "follower_right_joint_5.pos",
            "follower_right_joint_6.pos",
            "follower_r_thumb.pos",
            "follower_r_thumb_aux.pos",
            "follower_r_index.pos",
            "follower_r_middle.pos",
            "follower_r_ring.pos",
            "follower_r_pinky.pos",
            "follower_head_yaw.pos",
            "follower_head_pitch.pos",
        ]

        goal_joint = []
        for key in expected_action_keys:
            val = action.get(key, 0.0)
            if hasattr(val, "item"):
                val = val.item()
            goal_joint.append(float(val))
        goal_joint_numpy = np.array(goal_joint, dtype=np.float32)
        try:
            if goal_joint_numpy.shape != (28,):
                raise ValueError(f"Action vector must be 28-dimensional, got {goal_joint_numpy.shape[0]}")
            
            # 调用ROS1节点的ros_replay方法发布动作（替换ROS2节点）
            self.robot_ros1_node.ros_replay(goal_joint_numpy)
            
        except Exception as e:
            logger.error(f"Failed to send action: {e}")
            raise

    def update_status(self) -> str:
        # 更新摄像头状态（替换ROS2节点为ROS1节点）
        for i in range(self.status.specifications.camera.number):
            match_name = self.status.specifications.camera.information[i].name
            for name in self.robot_ros1_node.recv_images_status:
                if match_name in name:
                    self.status.specifications.camera.information[i].is_connect = (
                        True if self.robot_ros1_node.recv_images_status[name] > 0 else False
                    )


        # 更新从臂状态（替换ROS2节点为ROS1节点）
        for i in range(self.status.specifications.arm.number):
            match_name = self.status.specifications.arm.information[i].name
            for name in self.robot_ros1_node.recv_follower_status:
                if match_name in name:
                    self.status.specifications.arm.information[i].is_connect = (
                        True if self.robot_ros1_node.recv_follower_status[name] > 0 else False
                    )

        return self.status.to_json()

    def disconnect(self):
        if not self.is_connected:
            raise DeviceNotConnectedError(
                "LEJUKuavo is not connected. You need to run `robot.connect()` before disconnecting."
            )
        # 销毁ROS1节点（替换ROS2节点的destroy）
        if hasattr(self, "robot_ros1_node"):
            self.robot_ros1_node.destroy()
        # 关闭ROS1（替换rclpy.shutdown()）
        if rospy.core.is_initialized():
            rospy.signal_shutdown("Robot disconnected, shutting down ROS1 node")

        self.connected = False

    def __del__(self):
        try:
            if getattr(self, "is_connected", False):
                self.disconnect()
        except Exception:
            pass
