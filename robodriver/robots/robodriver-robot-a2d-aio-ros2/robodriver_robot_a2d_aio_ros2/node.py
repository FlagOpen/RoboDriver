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

logger = logging_mp.get_logger(__name__)

# 尝试导入厂家自定义消息
try:
    from genie_msgs.msg import EndState, ArmState
    GENIE_MSGS_AVAILABLE = True
except ImportError:
    GENIE_MSGS_AVAILABLE = False
    logger.info("Warning: genie_msgs not found. End effector feedback may not work.")


CONNECT_TIMEOUT_FRAME = 10


# 由生成脚本根据 JSON 自动生成
NODE_CONFIG = {
    "follower_joint_topics": {
        "arm_states": {
            "topic": "/hal/arm_joint_state", # 双臂关节状态 
            "msg": "JointState"
        }
    },
    "left_ee_topics":{
        "topic": "/hal/left_ee_data",    # 左末端状态 
        "msg": "EndState"
    },
    "right_ee_topics":{
        "topic": "/hal/right_ee_data",   # 右末端状态 
        "msg": "EndState"
    },
    "camera_topics": {
        "head_color_image": {"topic": "/camera/head_color", "msg": "Image"},
        "hand_left_color_image": {"topic": "/camera/hand_left_color", "msg": "Image"},
        "hand_right_color_image": {"topic": "/camera/hand_right_color", "msg": "Image"}
    },
    "action_topics": {
        "arm_command": "/wbc/arm_command",          # 双臂控制 
        "left_ee_command": "/wbc/left_ee_command",  # 左末端控制 
        "right_ee_command": "/wbc/right_ee_command" # 右末端控制 
    }
}


class A2DAioRos2Node(Node):

    def __init__(
        self,
        follower_joint_topics: Dict[str, Dict[str, str]] = NODE_CONFIG["follower_joint_topics"],
        lee_topic: Dict[str,str] = NODE_CONFIG["left_ee_topics"],
        ree_topic: Dict[str,str] = NODE_CONFIG["right_ee_topics"],
        camera_topics: Dict[str, Dict[str, str]] = NODE_CONFIG["camera_topics"],
    ):
        super().__init__("a2d_aio_ros2_direct")

        # ---- 从参数 / NODE_CONFIG 中拿配置 ----
        self.follower_joint_cfgs = follower_joint_topics or {}
        self.camera_cfgs = camera_topics or {}
        self.lee_cfg = lee_topic or {}
        self.ree_cfg = ree_topic or {}


        if not self.follower_joint_cfgs:
            raise RuntimeError("follower_joint_topics is empty")

        # 相机 topic 简化一个 name -> topic 的 dict
        self.camera_topics: Dict[str, str] = {
            name: info["topic"] for name, info in self.camera_cfgs.items()
        }

        # ---- 各种缓存 ----
        self.recv_images: Dict[str, np.ndarray] = {}
        self.recv_images_status: Dict[str, int] = {}

        # 存储电机状态：{ joint_name: position }
        self.recv_follower: Dict[str, Any] = {}
        self.recv_follower_status: Dict[str, int] = {}

        self.recv_leader: Dict[str, Any] = {}
        self.recv_leader_status: Dict[str, int] = {}

        self.lock = threading.Lock()
        self.running = False


        # ---- 1. 订阅手臂关节状态 (/hal/arm_joint_state) ----
        for comp_name, cfg in self.follower_joint_cfgs.items():
            topic = cfg["topic"]
            msg_name = cfg.get("msg", "JointState")
            # logger.info(comp_name)
            if msg_name == "JointState":
                msg_cls = JointState
                callback = lambda msg, cname=comp_name: self._joint_callback_follower(
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
            logger.info(f"【注册】comp_name={comp_name}, topic={topic}")
            logger.info(
                f"[Direct] Follower subscriber '{comp_name}' at {topic} ({msg_name})"
            )

        # ---- 2. 订阅末端执行器关节状态 (/hal/xx_ee_data) ----
        ltopic = self.lee_cfg["topic"]
        lmsg_name = self.lee_cfg.get("msg", "EndState")
        if lmsg_name == "EndState":
            lmsg_cls = EndState
            lcallback = lambda msg, cname="left_ee_topics": self._gripper_callback_follower(
                cname, msg
            )
        else:
            raise RuntimeError(f"Unsupported follower msg type: {lmsg_name}")

        self.create_subscription(
            lmsg_cls,
            ltopic,
            lcallback,
            10,
        )
        logger.info(f"[Direct] Follower subscriber 'left_ee_topics' at {ltopic} ({lmsg_name})")




        rtopic = self.ree_cfg["topic"]
        rmsg_name = self.ree_cfg.get("msg", "EndState")
        if rmsg_name == "EndState":
            rmsg_cls = EndState
            rcallback = lambda msg, cname="right_ee_topics": self._gripper_callback_follower(
                cname, msg
            )
        else:
            raise RuntimeError(f"Unsupported follower msg type: {rmsg_name}")

        self.create_subscription(
            rmsg_cls,
            rtopic,
            rcallback,
            10,
        )
        logger.info(f"[Direct] Follower subscriber 'right_ee_topics' at {rtopic} ({rmsg_name})")



        # ---- 3. cameras: 订阅所有 camera_topics（目前只支持 Image）----
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


        # ---- 4. 创建控制指令发布者 ----
        self.pub_arm_cmd = self.create_publisher(
            JointState,
            NODE_CONFIG["action_topics"]["arm_command"],
            10
        )
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
                self.recv_follower[event_id] = {
                    name: position
                    for name, position in zip(msg.name, msg.position)
                }
                self.recv_follower_status[event_id] = CONNECT_TIMEOUT_FRAME
        except Exception as e:
            logger.error(f"Joint callback error (follower:{comp_name}): {e}")

    def _gripper_callback_follower(self, comp_name: str, msg: EndState):
        try:
            with self.lock:
                
                # 1. 组装 {joint_name: position} 字典
                joint_dict = {}
                for name, st in zip(msg.name, msg.end_state):
                    joint_dict[name] = float(st.position)

                # 2. 写入缓存
                self.recv_follower[comp_name] = joint_dict
                self.recv_follower_status[comp_name] = CONNECT_TIMEOUT_FRAME

        except Exception as e:
            logger.error(f"Joint callback error (follower:{comp_name}): {e}")


    # ======================
    # Send Action
    # ======================

    def ros2_send(self, action: dict[str, Any]):

        """
        发送控制指令。
        支持字段：
          - 手臂关节: "left_arm_joint1" ~ "right_arm_joint7"
          - 末端夹爪: "left_gripper_joint1", "right_gripper_joint1" ∈ [0.0, 1.0]
        """
        now = self.get_clock().now().to_msg()
        arm_names = [
            "left_arm_joint1", "left_arm_joint2", "left_arm_joint3", "left_arm_joint4",
            "left_arm_joint5", "left_arm_joint6", "left_arm_joint7",
            "right_arm_joint1", "right_arm_joint2", "right_arm_joint3", "right_arm_joint4",
            "right_arm_joint5", "right_arm_joint6", "right_arm_joint7"
        ]

        # 手臂控制
        arm_positions = [float(action.get(name, 0.0)) for name in arm_names]
        if any(name in action for name in arm_names):
            arm_msg = JointState()
            arm_msg.header.stamp = now
            arm_msg.name = arm_names
            arm_msg.position = arm_positions
            self.pub_arm_cmd.publish(arm_msg)

        # 左末端
        if "left_gripper_joint1" in action:
            raw1 = float(action["left_gripper_joint1"])
            send1 = np.interp(np.clip(raw1, 0, 120), [35, 120], [0.0, 1.0])
            left_msg = EndState()
            left_msg.header.stamp = now
            left_msg.name = ["left_gripper_joint1"]
            left_msg.position = [send1]
            self.pub_left_ee_cmd.publish(left_msg)

        # 右末端
        if "right_gripper_joint1" in action:
            raw2 = float(action["left_gripper_joint1"])
            send2 = np.interp(np.clip(raw2, 0, 120), [35, 120], [0.0, 1.0])
            right_msg = EndState()
            right_msg.header.stamp = now
            right_msg.name = ["right_gripper_joint1"]
            right_msg.position = [send2]
            self.pub_right_ee_cmd.publish(right_msg)
    

    # # ======================
    # # spin 线程控制
    # # ======================

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
