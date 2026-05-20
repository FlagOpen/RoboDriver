import threading
import time
import re
import subprocess
from typing import Dict

import numpy as np
import cv2
import rclpy
from rclpy.node import Node as ROS2Node
from sensor_msgs.msg import JointState, Image
from std_msgs.msg import Float64MultiArray
from message_filters import Subscriber, ApproximateTimeSynchronizer
from geometry_msgs.msg import PoseStamped
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy, HistoryPolicy

import logging_mp

CONNECT_TIMEOUT_FRAME = 10 
logger = logging_mp.get_logger(__name__)

class OPENARMRos2RobotNode(ROS2Node):

    def __init__(self):
        super().__init__("ros2_recv_pub_driver")
        self.stop_spin = False
        self.qos = QoSProfile(
            durability=DurabilityPolicy.VOLATILE,
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        self.qos_best_effort = QoSProfile(
            durability=DurabilityPolicy.VOLATILE,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        # 发布回放控制命令（对接 ros2_control forward_position_controller）
        self.publisher_left_arm = self.create_publisher(
            Float64MultiArray, "/left_forward_position_controller/commands", self.qos
        )
        self.publisher_right_arm = self.create_publisher(
            Float64MultiArray, "/right_forward_position_controller/commands", self.qos
        )
        # 控制器关节顺序不做硬编码假设，运行时根据 /joint_states 的名称顺序自动学习。
        self.left_arm_joint_names: list[str] = []
        self.right_arm_joint_names: list[str] = []
        self.controller_joint_order_locked = False
        # 回放动作的“语义基准顺序”（来自采集特征定义）
        self.left_action_joint_names = [f"openarm_left_joint{i}" for i in range(1, 8)]
        self.right_action_joint_names = [f"openarm_right_joint{i}" for i in range(1, 8)]
        # 16维动作向量标准顺序：
        # [L1..L7, L_gripper, R1..R7, R_gripper]
        # 其中控制器只接收左右臂 7 维，夹爪值先保留给后续夹爪控制器使用。
        
     

        self.last_main_send_time_ns = 0 
        self.last_follow_send_time_ns = 0
        self.min_interval_ns = 1e9 / 30

        #self._init_message_main_filters()  # 注释掉主臂初始化
        self._init_message_follow_filters()
        self._init_image_message_filters()  # 启用图像订阅

       
        self.recv_images: Dict[str, float] = {}  
        #self.recv_leader: Dict[str, float] = {}  # 注释掉主臂数据
        self.recv_follower: Dict[str, float] = {}  
        
       
        self.recv_images_status: Dict[str, int] = {}  
        #self.recv_leader_status: Dict[str, int] = {}  # 注释掉主臂状态
        self.recv_follower_status: Dict[str, int] = {}  

        self.lock = threading.Lock()
        self._load_controller_joint_orders_from_params()

    def _init_message_follow_filters(self):
        self.joint_states_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_states_callback,
            10
        )
        self.get_logger().info('已订阅 /joint_states 话题')

    def joint_states_callback(self, msg):
        """关节状态回调函数"""
        try:
            # 频率控制：检查是否达到最小时间间隔
            current_time_ns = time.time_ns()
            if (current_time_ns - self.last_follow_send_time_ns) < self.min_interval_ns:
                return  # 距离上次更新时间太短，丢弃本次数据
            self.last_follow_send_time_ns = current_time_ns
 
            # 按关节名存储，避免依赖消息顺序
            joint_positions = np.array(msg.position, dtype=np.float32)
            joint_name_to_pos = {
                joint_name: float(joint_positions[idx])
                for idx, joint_name in enumerate(msg.name)
                if idx < len(joint_positions)
            }
            
            # 第一次接收到数据时打印日志
            if 'follower_arms' not in self.recv_follower:
                self.get_logger().info(f'首次接收到关节数据: {len(joint_positions)} 个关节')
                self.get_logger().info(f'关节名称: {msg.name}')
            self._update_controller_joint_orders(msg.name)
            
            # 线程安全地更新共享数据
            with self.lock:
                self.recv_follower['follower_arms'] = joint_name_to_pos
                # 重置连接状态计数器
                self.recv_follower_status['follower_arms'] = CONNECT_TIMEOUT_FRAME
        except Exception as e:
            self.get_logger().error(f"关节状态回调错误: {e}")

    def _update_controller_joint_orders(self, joint_names: list[str]):
        """根据 /joint_states 名称顺序动态更新左右臂控制器关节顺序（参数读取失败时兜底）。"""
        if self.controller_joint_order_locked:
            return

        def joint_index(name: str) -> int:
            m = re.search(r"joint(\d+)$", name)
            return int(m.group(1)) if m else 10**9

        left = [
            name for name in joint_names
            if re.match(r"^openarm_left_joint[1-7]$", name)
        ]
        right = [
            name for name in joint_names
            if re.match(r"^openarm_right_joint[1-7]$", name)
        ]
        left = sorted(left, key=joint_index)
        right = sorted(right, key=joint_index)
        if len(left) == 7 and left != self.left_arm_joint_names:
            self.left_arm_joint_names = left
            self.get_logger().info(
                f"回退到 /joint_states 推断左臂顺序: {self.left_arm_joint_names}"
            )
        if len(right) == 7 and right != self.right_arm_joint_names:
            self.right_arm_joint_names = right
            self.get_logger().info(
                f"回退到 /joint_states 推断右臂顺序: {self.right_arm_joint_names}"
            )

    def _parse_joint_names_from_param_output(self, text: str, side: str) -> list[str]:
        """解析 `ros2 param get ... joints` 输出，提取关节顺序。"""
        names: list[str] = []
        bracket_match = re.search(r"\[(.*?)\]", text, flags=re.S)
        if bracket_match:
            raw = bracket_match.group(1).strip()
            if raw:
                items = [item.strip().strip("'\"") for item in raw.split(",")]
                names = [n for n in items if n]
        else:
            for line in text.splitlines():
                line = line.strip()
                if line.startswith("- "):
                    names.append(line[2:].strip().strip("'\""))

        pattern = rf"^openarm_{side}_joint[1-7]$"
        return [n for n in names if re.match(pattern, n)]

    def _load_controller_joint_orders_from_params(self):
        """优先从 controller 参数读取关节顺序，避免任何顺序假设。"""
        controllers = {
            "left": "/left_forward_position_controller",
            "right": "/right_forward_position_controller",
        }
        parsed_orders: dict[str, list[str]] = {}

        for side, controller_node in controllers.items():
            try:
                result = subprocess.run(
                    ["ros2", "param", "get", controller_node, "joints"],
                    capture_output=True,
                    text=True,
                    timeout=2.0,
                    check=False,
                )
                output = (result.stdout or "") + "\n" + (result.stderr or "")
                names = self._parse_joint_names_from_param_output(output, side)
                if len(names) == 7:
                    parsed_orders[side] = names
                else:
                    self.get_logger().warning(
                        f"读取 {controller_node} joints 失败或不完整，输出为: {output.strip()}"
                    )
            except Exception as e:
                self.get_logger().warning(
                    f"调用 ros2 param 读取 {controller_node}.joints 失败: {e}"
                )

        if len(parsed_orders.get("left", [])) == 7 and len(parsed_orders.get("right", [])) == 7:
            self.left_arm_joint_names = parsed_orders["left"]
            self.right_arm_joint_names = parsed_orders["right"]
            self.controller_joint_order_locked = True
            self.get_logger().info(
                "已从 controller 参数锁定关节顺序: "
                f"left={self.left_arm_joint_names}, right={self.right_arm_joint_names}"
            )
           
   # def _init_message_main_filters(self):
        # 创建订阅者：订阅主臂的关节和姿态数据
        #sub_joint_left = Subscriber(self, JointState, '/joint_state', qos_profile=self.qos_best_effort)
        #sub_joint_right = Subscriber(self, JointState, '/joint_state', qos_profile=self.qos_best_effort)
        #sub_joint_torso = Subscriber(self, JointState, '/motion_target/target_joint_state_torso', qos_profile=self.qos_best_effort)
        #sub_pose_left = Subscriber(self, PoseStamped, '/motion_target/target_pose_arm_left', qos_profile=self.qos_best_effort)
        #sub_pose_right = Subscriber(self, PoseStamped, '/motion_target/target_pose_arm_right', qos_profile=self.qos_best_effort)
        #sub_torso = Subscriber(self, PoseStamped, '/motion_target/target_pose_torso', qos_profile=self.qos_best_effort)
        #sub_gripper_left = Subscriber(self, JointState, '/joint_state', qos_profile=self.qos_best_effort)
        #sub_gripper_right = Subscriber(self, JointState, '/joint_state', qos_profile=self.qos_best_effort)

        # 创建近似时间同步器：同步所有主臂数据
        #self.pose_sync = ApproximateTimeSynchronizer(
           # [sub_joint_left, sub_joint_right, sub_gripper_left, sub_gripper_right],
           ## slop=0.01  # 更严格的时间同步要求（0.01秒）
        #)
        # 注册同步回调函数
        #self.pose_sync.registerCallback(self.synchronized_main_callback)
    """
     def synchronized_main_callback(self, joint_left, joint_right, pose_left, pose_right, gripper_left, gripper_right):
        
        try:
            # 频率控制：检查是否达到最小时间间隔
            current_time_ns = time.time_ns()
            if (current_time_ns - self.last_main_send_time_ns) < self.min_interval_ns:
                return  # 距离上次更新时间太短，丢弃本次数据
            self.last_main_send_time_ns = current_time_ns

            # 提取关节角度
            left_joint = np.array(joint_left.position, dtype=np.float32)
            right_joint = np.array(joint_right.position, dtype=np.float32)
 
            # 提取夹爪位置
            gripper_left_pose = np.array(gripper_left.position, dtype=np.float32)
            gripper_right_pose = np.array(gripper_right.position, dtype=np.float32)
            #torso_joint = np.array(joint_torso.position, dtype=np.float32)

            # 提取左臂笛卡尔姿态：位置(x,y,z) + 四元数(x,y,z,w)
            left_pos = np.array([
                pose_left.pose.position.x, pose_left.pose.position.y, pose_left.pose.position.z,
                pose_left.pose.orientation.x, pose_left.pose.orientation.y, pose_left.pose.orientation.z, pose_left.pose.orientation.w
            ], dtype=np.float32)

            # 提取右臂笛卡尔姿态
            right_pos = np.array([
                pose_right.pose.position.x, pose_right.pose.position.y, pose_right.pose.position.z,
                pose_right.pose.orientation.x, pose_right.pose.orientation.y, pose_right.pose.orientation.z, pose_right.pose.orientation.w
            ], dtype=np.float32)

            # 提取躯干笛卡尔姿态
            #torso_pose = np.array([
            #    torso.pose.position.x, torso.pose.position.y, torso.pose.position.z,
            #    torso.pose.orientation.x, torso.pose.orientation.y, torso.pose.orientation.z, torso.pose.orientation.w
            #], dtype=np.float32)
            # 合并所有数据：关节 + 夹爪 + 姿态
            merged_data = np.concatenate([left_joint, gripper_left_pose, right_joint, gripper_right_pose, left_pos, right_pos])
            
            # 线程安全地更新共享数据
            with self.lock:
                self.recv_leader['leader_arms'] = merged_data
                # 重置连接状态计数器
                self.recv_leader_status['leader_arms'] = CONNECT_TIMEOUT_FRAME
        except Exception as e:
            self.get_logger().error(f"主臂姿态回调错误: {e}")
    """
    def _init_image_message_filters(self):
        sub_camera_top = Subscriber(self, Image, '/openarm/D435_top/color/image_raw')
        sub_camera_wrist_left = Subscriber(self, Image, '/openarm/D415_wrist_left/color/image_raw')
        sub_camera_wrist_right = Subscriber(self, Image, '/openarm/D415_wrist_right/color/image_raw')

        self.image_sync = ApproximateTimeSynchronizer(
            [sub_camera_top, sub_camera_wrist_left, sub_camera_wrist_right],
            queue_size=5,  # 较小的队列，图像数据量大
            slop=0.1  # 允许的时间误差
        )
        # 注册同步回调函数
        self.image_sync.registerCallback(self.image_synchronized_callback)

    def image_synchronized_callback(self, top, wrist_left, wrist_right):
        try:
            # 按消息自带分辨率/编码处理，避免硬编码
            self.images_recv(top, "image_top")
            self.images_recv(wrist_left, "image_wrist_left")
            self.images_recv(wrist_right, "image_wrist_right")
        except Exception as e:
            self.get_logger().error(f"图像同步回调错误: {e}")
    
    
    def images_recv(self, msg, event_id):
        try:
            if 'image' not in event_id:
                return

            encoding = (msg.encoding or "").lower()
            height, width = msg.height, msg.width
            data_len = len(msg.data)

            if data_len == 0:
                self.get_logger().error(f"接收图像错误: 数据为空 event={event_id} encoding={encoding}")
                return

            frame = None

            if encoding in ("bgr8", "rgb8", "bgra8", "rgba8", "mono8"):
                channels = 4 if "a8" in encoding else 1 if encoding == "mono8" else 3
                expected = height * width * channels
                if data_len < expected:
                    self.get_logger().error(
                        f"接收图像错误: 数据长度不足 event={event_id} encoding={encoding} len={data_len} expected={expected}"
                    )
                    return

                frame = np.frombuffer(msg.data, dtype=np.uint8).reshape((height, width, channels))

                if encoding == "bgr8":
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                elif encoding == "bgra8":
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
                elif encoding == "rgba8":
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
                elif encoding == "mono8":
                    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)

            else:
                compressed = np.frombuffer(msg.data, dtype=np.uint8)
                frame_bgr = cv2.imdecode(compressed, cv2.IMREAD_COLOR)
                if frame_bgr is None or frame_bgr.size == 0:
                    self.get_logger().error(
                        f"接收图像错误: 解码失败 event={event_id} encoding={encoding} len={data_len}"
                    )
                    return
                frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            if frame is None or frame.size == 0:
                self.get_logger().error(f"接收图像错误: 解码结果为空 event={event_id} encoding={encoding}")
                return

            with self.lock:
                self.recv_images[event_id] = frame
                self.recv_images_status[event_id] = CONNECT_TIMEOUT_FRAME
        except Exception as e:
            self.get_logger().error(f"接收图像错误: {e}")
 
   
    def ros_replay(self, array):
        try:
            def normalize_precision(val, decimals=3):
                val = float(val)
                # 检查并替换NaN或Inf值
                if np.isnan(val) or np.isinf(val):
                    self.get_logger().warning(f"检测到非法值 {val}，替换为0.0")
                    return 0.0
                return round(val, decimals)
            
            # openarm当前动作定义：左臂7 + 左夹爪1 + 右臂7 + 右夹爪1 = 16维
            if len(array) != 16:
                raise ValueError(f"回放动作维度必须是16维，实际为{len(array)}维")

            left_arm = [normalize_precision(v) for v in array[:7]]
            left_gripper = [normalize_precision(v) for v in array[7:8]]
            right_arm = [normalize_precision(v) for v in array[8:15]]
            right_gripper = [normalize_precision(v) for v in array[15:16]]

            if len(self.left_arm_joint_names) != 7 or len(self.right_arm_joint_names) != 7:
                self._load_controller_joint_orders_from_params()
            if len(self.left_arm_joint_names) != 7 or len(self.right_arm_joint_names) != 7:
                raise ValueError(
                    "尚未从 /joint_states 学习到完整左右臂关节顺序，"
                    f"left={self.left_arm_joint_names}, right={self.right_arm_joint_names}"
                )

            # 按动作语义名构建映射，再按控制器实际顺序重排，避免“假设顺序”。
            left_action_map = {
                self.left_action_joint_names[idx]: left_arm[idx] for idx in range(7)
            }
            right_action_map = {
                self.right_action_joint_names[idx]: right_arm[idx] for idx in range(7)
            }
            left_controller_cmd = [left_action_map[name] for name in self.left_arm_joint_names]
            right_controller_cmd = [
                right_action_map[name] for name in self.right_arm_joint_names
            ]

            # 发布左臂7关节目标（与 left_forward_position_controller joints 顺序一致）
            msg = Float64MultiArray()
            msg.data = left_controller_cmd
            self.publisher_left_arm.publish(msg)

            # 发布右臂7关节目标（与 right_forward_position_controller joints 顺序一致）
            msg = Float64MultiArray()
            msg.data = right_controller_cmd
            self.publisher_right_arm.publish(msg)
            # 夹爪值当前不下发到 arm controller，避免维度不匹配导致控制器拒收
            self.get_logger().debug(
                f"Replay gripper cached only: left={left_gripper[0]:.3f}, right={right_gripper[0]:.3f}"
            )


        except Exception as e:
            self.get_logger().error(f"发送动作指令时错误: {e}")
            raise

    def destroy(self):
        self.stop_spin = True
        super().destroy_node()


def ros_spin_thread(node):
    """在独立线程中运行ROS2节点的spin循环"""
    while rclpy.ok() and not getattr(node, "stop_spin", False):
        # 每次处理10ms的事件，然后检查是否需要停止
        rclpy.spin_once(node, timeout_sec=0.01)

 

        