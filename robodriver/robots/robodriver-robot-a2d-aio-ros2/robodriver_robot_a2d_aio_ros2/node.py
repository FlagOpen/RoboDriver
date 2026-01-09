# robodriver_robot_a2d_aio_ros2/node.py

import threading

import cv2
import numpy as np
import logging_mp
import rclpy

from rclpy.node import Node
from std_msgs.msg import Header
from sensor_msgs.msg import Image, JointState
from geometry_msgs.msg import Pose
from nav_msgs.msg import Odometry
from typing import Dict, Any
from genie_msgs.msg import ArmState

# 尝试导入厂家自定义消息
try:
    from genie_msgs.msg import EndState, ArmState
    GENIE_MSGS_AVAILABLE = True
except ImportError:
    GENIE_MSGS_AVAILABLE = False
    print("Warning: genie_msgs not found. End effector feedback may not work.")

logger = logging_mp.get_logger(__name__)

CONNECT_TIMEOUT_FRAME = 10


# 由生成脚本根据 JSON 自动生成
NODE_CONFIG = {
    # "leader_joint_topics": {
    #     "leader_arm": {
    #         "topic": "/joint_states",
    #         "msg": "JointState"
    #     }
    # },
    # "follower_joint_topics": {
    #     "follower_arm": {
    #         "topic": "/f_joint_states",
    #         "msg": "JointState"
    #     }
    # },
    # "camera_topics": {
    #     "image_top": {
    #         "topic": "/camera/camera/color/image_raw",
    #         "msg": "Image"
    #     }
    # }
    "follower_joint_topics": {
        "arm_states": {
            "topic": "/hal/arm_joint_state", # 双臂关节状态 
            "msg": "JointState"
        },
        "left_ee": {
            "topic": "/hal/left_ee_data",    # 左末端状态 
            "msg": "EndState"
        },
        "right_ee": {
            "topic": "/hal/right_ee_data",   # 右末端状态 
            "msg": "EndState"
        }
    },
    "camera_topics": {
        "head_color": {"topic": "/camera/head_color", "msg": "Image"},
        "hand_left_color": {"topic": "/camera/hand_left_color", "msg": "Image"},
        "hand_right_color": {"topic": "/camera/hand_right_color", "msg": "Image"}
    },
    "action_topics": {
        "arm_command": "/wbc/arm_command",          # 双臂控制 
        "left_ee_command": "/wbc/left_ee_command",  # 左末端控制 
        "right_ee_command": "/wbc/right_ee_command" # 右末端控制 
    }
}


class A2DAioRos2Node(Node):
    """
    ROS2 → 本地数据存储（无 ZMQ，无 Dora）
    leader / follower / camera 多 topic，按 JSON 配置自动订阅。

    订阅 A2D 官方 HAL 接口：
      /hal/left_arm_data   (ArmState)
      /hal/right_arm_data  (ArmState)
      /camera/head_color   (Image)
    发布：
      /wbc/arm_command     (JointState)  14 个位置

    """

    def __init__(
        self,
        follower_joint_topics: Dict[str, Dict[str, str]] = NODE_CONFIG["follower_joint_topics"],
        camera_topics: Dict[str, Dict[str, str]] = NODE_CONFIG["camera_topics"],
    ):
        super().__init__("a2d_aio_ros2_direct")

        # ---- 从参数 / NODE_CONFIG 中拿配置 ----
        self.follower_joint_cfgs = follower_joint_topics or {}
        self.camera_cfgs = camera_topics or {}


        if not self.follower_joint_cfgs:
            raise RuntimeError("follower_joint_topics is empty")

        # 相机 topic 简化一个 name -> topic 的 dict
        # self.camera_topics: Dict[str, str] = {
        #     name: info["topic"] for name, info in self.camera_cfgs.items()
        # }

        # ---- 各种缓存 ----
        # self.recv_images: Dict[str, np.ndarray] = {}
        # self.recv_images_status: Dict[str, int] = {}

        # self.recv_follower: Dict[str, Any] = {}
        # self.recv_follower_status: Dict[str, int] = {}

        # self.recv_leader: Dict[str, Any] = {}
        # self.recv_leader_status: Dict[str, int] = {}

        # self.lock = threading.Lock()
        # self.running = False
        self.recv_images: Dict[str, np.ndarray] = {}
        self.recv_images_status: Dict[str, int] = {}


        # 存储电机状态：{ joint_name: position }
        self.recv_follower: Dict[str, float] = {} 
        self.recv_follower_status: Dict[str, int] = {}

        self.lock = threading.Lock()
        self.running = False

        # # ---- follower side: 订阅所有 follower_joint_topics ----
        # for comp_name, cfg in self.follower_joint_cfgs.items():
        #     topic = cfg["topic"]
        #     msg_name = cfg.get("msg", "JointState")

        #     if msg_name == "JointState":
        #         msg_cls = JointState
        #         callback = lambda msg, cname=comp_name: self._joint_callback_follower(
        #             cname, msg
        #         )
        #     elif msg_name == "Pose":
        #         msg_cls = Pose
        #         callback = lambda msg, cname=comp_name: self._pose_callback_follower(
        #             cname, msg
        #         )
        #     elif msg_name == "Odometry":
        #         msg_cls = Odometry
        #         callback = lambda msg, cname=comp_name: self._odom_callback_follower(
        #             cname, msg
        #         )
        #     else:
        #         raise RuntimeError(f"Unsupported follower msg type: {msg_name}")

        #     self.create_subscription(
        #         msg_cls,
        #         topic,
        #         callback,
        #         10,
        #     )
        #     logger.info(
        #         f"[Direct] Follower subscriber '{comp_name}' at {topic} ({msg_name})"
        #     )


        # ---- 1. 订阅手臂关节状态 (/hal/arm_joint_state) ----
        # 这是一个标准的 JointState 
        self.create_subscription(
            JointState,
            NODE_CONFIG["follower_joint_topics"]["arm_states"]["topic"],
            self._arm_joint_callback,
            10,
        )


        # # ---- leader side: 订阅所有 leader_joint_topics ----
        # for comp_name, cfg in self.leader_joint_cfgs.items():
        #     topic = cfg["topic"]
        #     msg_name = cfg.get("msg", "JointState")

        #     if msg_name == "JointState":
        #         msg_cls = JointState
        #         callback = lambda msg, cname=comp_name: self._joint_callback_leader(
        #             cname, msg
        #         )
        #     elif msg_name == "Pose":
        #         msg_cls = Pose
        #         callback = lambda msg, cname=comp_name: self._pose_callback_leader(
        #             cname, msg
        #         )
        #     elif msg_name == "Odometry":
        #         msg_cls = Odometry
        #         callback = lambda msg, cname=comp_name: self._odom_callback_leader(
        #             cname, msg
        #         )
        #     else:
        #         raise RuntimeError(f"Unsupported leader msg type: {msg_name}")

        #     self.create_subscription(
        #         msg_cls,
        #         topic,
        #         callback,
        #         10,
        #     )
        #     logger.info(
        #         f"[Direct] Leader subscriber '{comp_name}' at {topic} ({msg_name})"
        #     )

        # self.pub_action_joint_states = self.create_publisher(
        #     JointState,
        #     topic="/joint_states",
        #     qos_profile=10,
        # )

        # ---- 2. 订阅末端执行器状态 (/hal/xx_ee_data) ----
        # 这是一个自定义消息 genie_msgs/EndState 
        if GENIE_MSGS_AVAILABLE:
            self.create_subscription(
                EndState,
                NODE_CONFIG["follower_joint_topics"]["left_ee"]["topic"],
                lambda msg: self._ee_callback("left", msg),
                10
            )
            self.create_subscription(
                EndState,
                NODE_CONFIG["follower_joint_topics"]["right_ee"]["topic"],
                lambda msg: self._ee_callback("right", msg),
                10
            )

        # # ---- cameras: 订阅所有 camera_topics（目前只支持 Image）----
        # self.camera_subs = []
        # for cam_name, cfg in self.camera_cfgs.items():
        #     topic = cfg["topic"]
        #     msg_name = cfg.get("msg", "Image")

        #     if msg_name != "Image":
        #         raise RuntimeError(f"Unsupported camera msg type: {msg_name}")

        #     sub = self.create_subscription(
        #         Image,
        #         topic,
        #         lambda msg, cname=cam_name: self._image_callback(cname, msg),
        #         10,
        #     )
        #     self.camera_subs.append(sub)
        #     logger.info(f"[Direct] Camera '{cam_name}' subscribed: {topic} ({msg_name})")

        # logger.info("[Direct] READY (ROS2 callbacks active).")

        # ---- 3. 订阅相机 ----
        for cam_key, cfg in self.camera_cfgs.items():
            self.create_subscription(
                Image,
                cfg["topic"],
                lambda msg, cname=cam_key: self._image_callback(cname, msg),
                10,
            )

        # ---- 4. 发布控制指令 ----
        # 手臂控制 (JointState) 
        self.pub_arm_cmd = self.create_publisher(
            JointState,
            NODE_CONFIG["action_topics"]["arm_command"],
            10
        )
        # 末端控制 (JointState, name=['left'/'right'], position=[0.0~1.0]) [cite: 242]
        self.pub_left_ee_cmd = self.create_publisher(
            JointState,
            NODE_CONFIG["action_topics"]["left_ee_command"],
            10
        )
        self.pub_right_ee_cmd = self.create_publisher(
            JointState,
            NODE_CONFIG["action_topics"]["right_ee_command"],
            10
        )

        logger.info("[A2D] Node READY.")

    # ======================
    # callbacks
    # ======================

    # def _image_callback(self, cam_name: str, msg: Image):
    #     try:
    #         with self.lock:
    #             event_id = f"{cam_name}"

    #             data = np.frombuffer(msg.data, dtype=np.uint8)
    #             h, w = msg.height, msg.width
    #             encoding = msg.encoding.lower()

    #             frame = None
    #             try:
    #                 if encoding == "bgr8":
    #                     frame = data.reshape((h, w, 3))
    #                     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #                 elif encoding == "rgb8":
    #                     frame = data.reshape((h, w, 3))
    #                 elif encoding in ["jpeg", "jpg", "png", "bmp", "webp"]:
    #                     frame = cv2.imdecode(data, cv2.IMREAD_COLOR)
                    
    #             except Exception as e:
    #                 logger.error(f"Image decode error ({encoding}): {e}")

    #             if frame is not None:
    #                 self.recv_images[event_id] = frame
    #                 self.recv_images_status[event_id] = CONNECT_TIMEOUT_FRAME

    #     except Exception as e:
    #         logger.error(f"Image callback error ({cam_name}): {e}")

    
    def _image_callback(self, cam_name: str, msg: Image):
        try:
            with self.lock:
                data = np.frombuffer(msg.data, dtype=np.uint8)
                h, w = msg.height, msg.width
                encoding = msg.encoding.lower()
                
                frame = None
                if encoding == "rgb8":
                    frame = data.reshape((h, w, 3))
                elif encoding == "bgr8":
                    frame = data.reshape((h, w, 3))
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                if frame is not None:
                    self.recv_images[cam_name] = frame
                    self.recv_images_status[cam_name] = CONNECT_TIMEOUT_FRAME
        except Exception as e:
            logger.error(f"Image decode error {cam_name}: {e}")

    

    # ---------- JointState ----------

    # def _joint_callback_follower(self, comp_name: str, msg: JointState):
    #     try:
    #         with self.lock:
    #             event_id = comp_name
    #             self.recv_follower[event_id] = {
    #                 name: position
    #                 for name, position in zip(msg.name, msg.position)
    #             }
    #             self.recv_follower_status[event_id] = CONNECT_TIMEOUT_FRAME
    #     except Exception as e:
    #         logger.error(f"Joint callback error (follower:{comp_name}): {e}")

    # def _joint_callback_leader(self, comp_name: str, msg: JointState):
    #     try:
    #         with self.lock:
    #             event_id = comp_name
    #             self.recv_leader[event_id] = {
    #                 name: position
    #                 for name, position in zip(msg.name, msg.position)
    #             }
    #             self.recv_leader_status[event_id] = CONNECT_TIMEOUT_FRAME
    #     except Exception as e:
    #         logger.error(f"Joint callback error (leader:{comp_name}): {e}")



    def _arm_joint_callback(self, msg: JointState):
        """
        处理 /hal/arm_joint_state 
        msg.position 包含 14 个关节数据 (左7 + 右7)
        """
        try:
            with self.lock:
                if len(msg.name) == len(msg.position):
                    for name, pos in zip(msg.name, msg.position):
                        self.recv_follower[name] = pos
                    self.recv_follower_status["arm"] = CONNECT_TIMEOUT_FRAME
        except Exception as e:
            logger.error(f"Arm joint callback error: {e}")

    
        # ---------- Pose ----------

    # def _pose_callback_follower(self, comp_name: str, msg: Pose):
    #     """
    #     follower 侧 Pose 回调
    #     合并 position + orientation -> [px, py, pz, qx, qy, qz, qw]
    #     """
    #     try:
    #         with self.lock:
    #             vec = np.array(
    #                 [
    #                     msg.position.x,
    #                     msg.position.y,
    #                     msg.position.z,
    #                     msg.orientation.x,
    #                     msg.orientation.y,
    #                     msg.orientation.z,
    #                     msg.orientation.w,
    #                 ],
    #                 dtype=float,
    #             )
    #             event_id = f"{comp_name}"
    #             self.recv_follower[event_id] = vec
    #             self.recv_follower_status[event_id] = CONNECT_TIMEOUT_FRAME
    #     except Exception as e:
    #         logger.error(f"Pose callback error (follower:{comp_name}): {e}")

    # def _pose_callback_leader(self, comp_name: str, msg: Pose):
    #     """
    #     leader 侧 Pose 回调
    #     合并 position + orientation -> [px, py, pz, qx, qy, qz, qw]
    #     """
    #     try:
    #         with self.lock:
    #             vec = np.array(
    #                 [
    #                     msg.position.x,
    #                     msg.position.y,
    #                     msg.position.z,
    #                     msg.orientation.x,
    #                     msg.orientation.y,
    #                     msg.orientation.z,
    #                     msg.orientation.w,
    #                 ],
    #                 dtype=float,
    #             )
    #             event_id = f"{comp_name}"
    #             self.recv_leader[event_id] = vec
    #             self.recv_leader_status[event_id] = CONNECT_TIMEOUT_FRAME
    #     except Exception as e:
    #         logger.error(f"Pose callback error (leader:{comp_name}): {e}")

    # ---------- Odometry ----------

    # def _odom_callback_follower(self, comp_name: str, msg: Odometry):
    #     """
    #     follower 侧 Odometry 回调
    #     合并:
    #       - pose.position        (3)
    #       - pose.orientation     (4)
    #       - twist.linear         (3)
    #       - twist.angular        (3)
    #     -> 13 维向量
    #     """
    #     try:
    #         with self.lock:
    #             vec = np.array(
    #                 [
    #                     # position
    #                     msg.pose.pose.position.x,
    #                     msg.pose.pose.position.y,
    #                     msg.pose.pose.position.z,
    #                     # orientation
    #                     msg.pose.pose.orientation.x,
    #                     msg.pose.pose.orientation.y,
    #                     msg.pose.pose.orientation.z,
    #                     msg.pose.pose.orientation.w,
    #                     # linear velocity
    #                     msg.twist.twist.linear.x,
    #                     msg.twist.twist.linear.y,
    #                     msg.twist.twist.linear.z,
    #                     # angular velocity
    #                     msg.twist.twist.angular.x,
    #                     msg.twist.twist.angular.y,
    #                     msg.twist.twist.angular.z,
    #                 ],
    #                 dtype=float,
    #             )
    #             event_id = f"{comp_name}"
    #             self.recv_follower[event_id] = vec
    #             self.recv_follower_status[event_id] = CONNECT_TIMEOUT_FRAME
    #     except Exception as e:
    #         logger.error(f"Odometry callback error (follower:{comp_name}): {e}")

    # def _odom_callback_leader(self, comp_name: str, msg: Odometry):
    #     """
    #     leader 侧 Odometry 回调
    #     同上，合成 13 维向量
    #     """
    #     try:
    #         with self.lock:
    #             vec = np.array(
    #                 [
    #                     # position
    #                     msg.pose.pose.position.x,
    #                     msg.pose.pose.position.y,
    #                     msg.pose.pose.position.z,
    #                     # orientation
    #                     msg.pose.pose.orientation.x,
    #                     msg.pose.pose.orientation.y,
    #                     msg.pose.pose.orientation.z,
    #                     msg.pose.pose.orientation.w,
    #                     # linear velocity
    #                     msg.twist.twist.linear.x,
    #                     msg.twist.twist.linear.y,
    #                     msg.twist.twist.linear.z,
    #                     # angular velocity
    #                     msg.twist.twist.angular.x,
    #                     msg.twist.twist.angular.y,
    #                     msg.twist.twist.angular.z,
    #                 ],
    #                 dtype=float,
    #             )
    #             event_id = f"{comp_name}"
    #             self.recv_leader[event_id] = vec
    #             self.recv_leader_status[event_id] = CONNECT_TIMEOUT_FRAME
    #     except Exception as e:
    #         logger.error(f"Odometry callback error (leader:{comp_name}): {e}")

    # def ros2_send(self, action: dict[str, Any]):

    #     msg = JointState()
    #     msg.header = Header()
    #     msg.header.stamp = self.get_clock().now().to_msg()
    #     msg.name = list(action.keys())
    #     msg.position = [float(value) for value in action.values()]
    #     msg.velocity = []
    #     msg.effort = []
    #     self.pub_action_joint_states.publish(msg)
    def _ee_callback(self, side: str, msg): # msg type: genie_msgs.msg.EndState
        """
        处理 /hal/left_ee_data 或 right_ee_data 
        文档未详细列出 EndState 的所有字段，但通常包含 position 或 state
        假设 msg.position 存在且为 float (或者根据实际 msg 结构调整)
        """
        try:
            with self.lock:
                # 假设 genie_msgs/EndState 有一个 current_position 字段或类似字段
                # 如果没有文档详解 EndState 结构，这里可能需要根据实际 `ros2 interface show` 调整
                # 假设它模拟了 JointState 的一部分或有一个 value
                # 这里暂时假设 msg.position 是一个列表或数值
                val = 0.0
                if hasattr(msg, 'position'):
                    # 有些末端可能是多指灵巧手，position 是数组
                    # 如果是夹爪，可能只是单一值
                    val = msg.position[0] if isinstance(msg.position, (list, tuple, np.ndarray)) else msg.position
                
                self.recv_follower[f"{side}_gripper"] = float(val)
        except Exception as e:
            logger.error(f"End effector callback error: {e}")

    # # ======================
    # # spin 线程控制
    # # ======================

    # def start(self):
    #     """启动 ROS2 spin 线程"""
    #     if self.running:
    #         return

    #     self.running = True
    #     self.spin_thread = threading.Thread(target=self._spin_loop, daemon=True)
    #     self.spin_thread.start()

    #     logger.info("[ROS2] Node started (spin thread running)")

    # def _spin_loop(self):
    #     """独立线程执行 ROS2 spin"""
    #     try:
    #         rclpy.spin(self)
    #     except Exception as e:
    #         logger.error(f"[ROS2] Spin error: {e}")

    # def stop(self):
    #     """停止 ROS2"""
    #     if not self.running:
    #         return

    #     self.running = False
    #     rclpy.shutdown()

    #     if getattr(self, "spin_thread", None):
    #         self.spin_thread.join(timeout=1.0)

    #     logger.info("[ROS2] Node stopped.")


    # ======================
    # Send Action
    # ======================

    def send_command(self, arm_joints: Dict[str, float], gripper_joints: Dict[str, float]):
        """
        发送控制指令
        """
        # 1. 发送手臂指令 (/wbc/arm_command) 
        if arm_joints:
            msg = JointState()
            msg.header = Header()
            msg.header.stamp = self.get_clock().now().to_msg()
            
            # 严格按照文档定义的顺序或名称
            target_order = [
                "left_arm_joint1", "left_arm_joint2", "left_arm_joint3", "left_arm_joint4", "left_arm_joint5", "left_arm_joint6", "left_arm_joint7",
                "right_arm_joint1", "right_arm_joint2", "right_arm_joint3", "right_arm_joint4", "right_arm_joint5", "right_arm_joint6", "right_arm_joint7"
            ]
            
            names = []
            positions = []
            for name in target_order:
                if name in arm_joints:
                    names.append(name)
                    positions.append(arm_joints[name])
            
            if names:
                msg.name = names
                msg.position = positions
                self.pub_arm_cmd.publish(msg)

        # 2. 发送末端指令 (/wbc/xx_ee_command) [cite: 242]
        # 文档说明：name[] 执行器种类, position[] 夹爪闭合距离或灵巧手角度
        for side in ["left", "right"]:
            key = f"{side}_gripper"
            if key in gripper_joints:
                val = gripper_joints[key]
                msg = JointState()
                msg.header = Header()
                msg.header.stamp = self.get_clock().now().to_msg()
                msg.name = [side] # 文档示例中 name 字段填的是 'left' 或 'right'
                msg.position = [val] 
                
                if side == "left":
                    self.pub_left_ee_cmd.publish(msg)
                else:
                    self.pub_right_ee_cmd.publish(msg)

    # ======================
    # Threading
    # ======================

    def start(self):
        if self.running: return
        self.running = True
        self.spin_thread = threading.Thread(target=self._spin_loop, daemon=True)
        self.spin_thread.start()

    def _spin_loop(self):
        try:
            rclpy.spin(self)
        except Exception:
            pass

    def stop(self):
        self.running = False
        rclpy.shutdown()
        if getattr(self, "spin_thread", None):
            self.spin_thread.join()