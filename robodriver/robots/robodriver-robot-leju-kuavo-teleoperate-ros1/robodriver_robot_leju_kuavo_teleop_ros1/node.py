#!/usr/bin/env python
# -*- coding: utf-8 -*-
import threading
import time
from typing import Dict

import numpy as np
import cv2
import rospy
from sensor_msgs.msg import JointState, Image
from message_filters import Subscriber, ApproximateTimeSynchronizer
from geometry_msgs.msg import PoseStamped
try:
    # 优先使用ROS原生消息包，确保与rostopic type一致（kuavo_msgs/*）
    from kuavo_msgs.msg import robotHandPosition, robotHeadMotionData, sensorsData
except Exception:
    # 兼容旧SDK导入路径
    from kuavo_humanoid_sdk.msg.kuavo_msgs.msg import robotHandPosition, robotHeadMotionData, sensorsData
# 禁用kuavo-humanoid-sdk的日志记录，避免连接ws://localhost:8889失败



# ROS1没有logging_mp，替换为标准logging
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

CONNECT_TIMEOUT_FRAME = 10
MAX_SYNC_ERROR_SEC = 0.01


class LEJUKuavoRos1Node:
    def __init__(self):
        # ROS1节点初始化 - 检查是否已经初始化
        if not rospy.core.is_initialized():
            rospy.init_node('ros1_recv_pub_driver', anonymous=False, log_level=rospy.INFO)
            rospy.loginfo("ROS1节点初始化完成，开始设置话题订阅和发布")
        else:
            rospy.loginfo(f"ROS节点已存在，复用现有节点: {rospy.get_name()}")
        
        self.stop_spin = False  # 初始化停止标志
        
        # ROS1没有QoSProfile类，直接在订阅时指定队列大小，可靠性通过传输方式保证
        self.queue_size = 100
        self.best_effort_queue_size = 100

        # 打印发布者信息
        rospy.loginfo(f"创建发布者: /kuavo_arm_traj (队列大小: {self.queue_size})")
        self.arm_pub = rospy.Publisher('/kuavo_arm_traj', JointState, queue_size=self.queue_size)
        
        rospy.loginfo(f"创建发布者: /control_robot_hand_position (队列大小: {self.queue_size})")
        self.hand_pub = rospy.Publisher('/control_robot_hand_position', robotHandPosition, queue_size=self.queue_size)
        
        rospy.loginfo(f"创建发布者: /robot_head_motion_data (队列大小: {self.queue_size})")
        self.head_pub = rospy.Publisher('/robot_head_motion_data', robotHeadMotionData, queue_size=self.queue_size)
        
        self.last_main_send_time_ns = 0
        self.last_follow_send_time_ns = 0
        self.min_interval_ns = 1e9 / 30  # 30Hz
        self.lock = threading.Lock()
        self.recv_images: Dict[str, np.ndarray] = {}
        self.recv_follower: Dict[str, np.ndarray] = {}
        self.recv_images_status: Dict[str, int] = {}
        self.recv_follower_status: Dict[str, int] = {}
        self.latest_body_msg = None
        self.latest_hand_msg = None
        self.body_msg_count = 0
        self.hand_msg_count = 0
        self._warned_headerless_follow = False
        self._warned_headerless_image = False

        rospy.loginfo("初始化消息过滤器...")
        self._init_message_follow_filters()
        self._init_image_message_filters()
        rospy.loginfo("节点初始化完成，等待话题消息...")

    def _init_message_follow_filters(self):
        sub_body = Subscriber('/sensors_data_raw', sensorsData)
        sub_hand = Subscriber('/dexhand/state', JointState)

        self.follow_sync = ApproximateTimeSynchronizer(
            [sub_body, sub_hand],
            queue_size=50,
            slop=MAX_SYNC_ERROR_SEC,
            allow_headerless=True,
        )
        self.follow_sync.registerCallback(self.synchronized_follow_callback)
        rospy.loginfo(
            f"已启用从臂话题同步: /sensors_data_raw + /dexhand/state, 最大时间误差={MAX_SYNC_ERROR_SEC:.3f}s"
        )

    def _get_msg_stamp_sec(self, msg):
        header = getattr(msg, 'header', None)
        stamp = getattr(header, 'stamp', None)
        if stamp is None:
            return None
        try:
            return stamp.to_sec()
        except Exception:
            return None

    def _process_body_msg(self, body_msg):
        try:
            joint_data = getattr(body_msg, "joint_data", None)
            if joint_data is None:
                return

            if hasattr(joint_data, "position"):
                body_pos = joint_data.position
            elif hasattr(joint_data, "joint_q"):
                body_pos = joint_data.joint_q
            else:
                return

            if len(body_pos) < 26:
                rospy.logwarn(f"Body joint data too short for arms: {len(body_pos)}")
                return

            left_arm = np.array(body_pos[12:19], dtype=np.float32)
            right_arm = np.array(body_pos[19:26], dtype=np.float32)
            if len(body_pos) >= 28:
                head = np.array(body_pos[26:28], dtype=np.float32)
            else:
                head = np.zeros(2, dtype=np.float32)

            with self.lock:
                self.recv_follower['right_arm'] = right_arm
                self.recv_follower['left_arm'] = left_arm
                self.recv_follower['head'] = head
                self.recv_follower_status['right_arm'] = CONNECT_TIMEOUT_FRAME
                self.recv_follower_status['left_arm'] = CONNECT_TIMEOUT_FRAME
                self.recv_follower_status['head'] = CONNECT_TIMEOUT_FRAME
        except Exception as e:
            rospy.logwarn(f"_process_body_msg error: {e}")

    def _process_hand_msg(self, hand_msg):
        try:
            hand_pos = list(getattr(hand_msg, "position", []))
            if len(hand_pos) < 12:
                rospy.logwarn(f"Hand position length < 12: {len(hand_pos)}")
                hand_pos = hand_pos + [0.0] * (12 - len(hand_pos))

            left_dexhand = np.array(hand_pos[0:6], dtype=np.float32)
            right_dexhand = np.array(hand_pos[6:12], dtype=np.float32)

            with self.lock:
                self.recv_follower['right_dexhand'] = right_dexhand
                self.recv_follower['left_dexhand'] = left_dexhand
                self.recv_follower_status['right_dexhand'] = CONNECT_TIMEOUT_FRAME
                self.recv_follower_status['left_dexhand'] = CONNECT_TIMEOUT_FRAME
        except Exception as e:
            rospy.logwarn(f"_process_hand_msg error: {e}")

    def synchronized_follow_callback(self, body_msg, hand_msg):
        try:
            self.body_msg_count += 1
            self.hand_msg_count += 1
            if self.body_msg_count == 1:
                rospy.loginfo("首次收到 /sensors_data_raw")
            if self.hand_msg_count == 1:
                rospy.loginfo("首次收到 /dexhand/state")

            body_stamp = self._get_msg_stamp_sec(body_msg)
            hand_stamp = self._get_msg_stamp_sec(hand_msg)
            if body_stamp is not None and hand_stamp is not None:
                sync_error = abs(body_stamp - hand_stamp)
                if sync_error > MAX_SYNC_ERROR_SEC:
                    rospy.logwarn_throttle(
                        1.0,
                        f"从臂话题时间误差超限: {sync_error:.6f}s > {MAX_SYNC_ERROR_SEC:.3f}s"
                    )
                    return
            elif not self._warned_headerless_follow:
                rospy.logwarn("从臂话题缺少header.stamp，已使用接收时间近似同步，无法严格保证0.01s时间误差")
                self._warned_headerless_follow = True

            joint_data = getattr(body_msg, "joint_data", None)
            if joint_data is None:
                rospy.logwarn("body_msg 中不存在 joint_data 字段")
                return

            # 兼容不同消息定义：position 或 joint_q
            if hasattr(joint_data, "position"):
                body_pos = joint_data.position
            elif hasattr(joint_data, "joint_q"):
                body_pos = joint_data.joint_q
            else:
                rospy.logwarn("joint_data 不包含 position/joint_q 字段")
                return

            # 打印接收到的消息信息
            #rospy.loginfo(f"Received follow data: body_msg joints={len(body_pos)}, hand_msg positions={len(hand_msg.position)}")
            
            # 检测话题数据
            if len(body_pos) > 0:
                # 打印前几个关节位置作为示例
                sample_positions = body_pos[:5] if len(body_pos) >= 5 else body_pos
                #rospy.loginfo(f"Body joint sample positions: {sample_positions}")
                
                # 检测数据范围
                for i, pos in enumerate(body_pos):
                    if abs(pos) > 1000:  # 假设关节位置不应该超过1000
                        rospy.logwarn(f"Joint {i} position out of range: {pos}")
            
            if len(hand_msg.position) > 0:
                sample_hand = hand_msg.position[:3] if len(hand_msg.position) >= 3 else hand_msg.position
                #rospy.loginfo(f"Hand position sample: {sample_hand}")
                
                # 检测手部数据范围
                for i, pos in enumerate(hand_msg.position):
                    if pos < 0 or pos > 100:  # 假设手部位置在0-100范围内
                        rospy.logwarn(f"Hand position {i} out of range [0,100]: {pos}")
            
            # 独立更新，避免双话题严格同步导致无数据
            self._process_body_msg(body_msg)
            self._process_hand_msg(hand_msg)

            current_time_ns = time.time_ns()
            if (current_time_ns - self.last_follow_send_time_ns) >= self.min_interval_ns:
                self.last_follow_send_time_ns = current_time_ns
        except Exception as e:
            rospy.logerr(f"Synchronized follow callback error: {e}")

    def _init_image_message_filters(self):
        sub_camera_top = Subscriber('/camera/color/image_raw', Image)
        sub_camera_wrist_left = Subscriber('/left_wrist_camera/color/image_raw', Image)
        sub_camera_wrist_right = Subscriber('/right_wrist_camera/color/image_raw', Image)
 
        self.image_sync = ApproximateTimeSynchronizer(
            [sub_camera_top, sub_camera_wrist_left, sub_camera_wrist_right],
            queue_size=10,
            slop=MAX_SYNC_ERROR_SEC,
            allow_headerless=True,
        )
        self.image_sync.registerCallback(self.image_synchronized_callback)

    def image_synchronized_callback(self, top, wrist_left, wrist_right):
        try:
            top_stamp = self._get_msg_stamp_sec(top)
            left_stamp = self._get_msg_stamp_sec(wrist_left)
            right_stamp = self._get_msg_stamp_sec(wrist_right)
            stamps = [s for s in [top_stamp, left_stamp, right_stamp] if s is not None]
            if len(stamps) == 3:
                sync_error = max(stamps) - min(stamps)
                if sync_error > MAX_SYNC_ERROR_SEC:
                    rospy.logwarn_throttle(
                        1.0,
                        f"图像话题时间误差超限: {sync_error:.6f}s > {MAX_SYNC_ERROR_SEC:.3f}s"
                    )
                    return
            elif not self._warned_headerless_image:
                rospy.logwarn("图像话题缺少header.stamp，已使用接收时间近似同步，无法严格保证0.01s时间误差")
                self._warned_headerless_image = True

            # 打印接收到的图像消息信息
            #rospy.loginfo(f"Received synchronized images: top={top.header.seq if top.header.seq else 'N/A'}, "
                         #f"wrist_left={wrist_left.header.seq if wrist_left.header.seq else 'N/A'}, "
                         #f"wrist_right={wrist_right.header.seq if wrist_right.header.seq else 'N/A'}")
            
            # 检测图像数据
            #rospy.loginfo(f"Image data sizes: top={len(top.data)} bytes, "
                         #f"wrist_left={len(wrist_left.data)} bytes, "
                         #f"wrist_right={len(wrist_right.data)} bytes")
            
            # 检查图像数据是否为空
            if len(top.data) == 0:
                rospy.logwarn("Top camera image data is empty!")
            if len(wrist_left.data) == 0:
                rospy.logwarn("Left wrist camera image data is empty!")
            if len(wrist_right.data) == 0:
                rospy.logwarn("Right wrist camera image data is empty!")
            
            self.images_recv(top, "image_top", 424, 240)
            self.images_recv(wrist_left, "image_wrist_left", 848, 480)
            self.images_recv(wrist_right, "image_wrist_right", 848, 480)
        except Exception as e:
            rospy.logerr(f"Image synchronized callback error: {e}")
    
    def images_recv(self, msg, event_id, width, height, encoding="jpeg"):
        try:
           
            
            
           
            
            if 'image' in event_id:
                img_array = np.frombuffer(msg.data, dtype=np.uint8)
                
                # 检测图像数据大小是否合理
                expected_size = width * height * 3  # 假设RGB图像
                if len(img_array) < expected_size * 0.5:  # 至少50%的预期大小
                    rospy.logwarn(f"Image data size suspiciously small: {len(img_array)} bytes, "
                                 f"expected around {expected_size} bytes for {width}x{height} RGB image")
                
                if msg.encoding == "bgr8":
                    channels = 3
                    frame = img_array.reshape((height, width, channels)).copy()
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                elif msg.encoding == "rgb8":
                    channels = 3
                    frame = img_array.reshape((height, width, channels))
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                elif msg.encoding in ["jpeg", "jpg", "jpe", "bmp", "webp", "png"]:
                    channels = 3
                    frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                elif msg.encoding == "depth16":
                    frame = np.frombuffer(msg.data, dtype=np.uint16).reshape(height, width, 1)
                else:
                    # 尝试通用解码
                    frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                    if frame is not None:
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                if frame is not None:
                    # 修正 top 相机画面方向（当前倒置）
                    if event_id == "image_top":
                        frame = cv2.rotate(frame, cv2.ROTATE_180)

                    # 打印图像处理结果
                    #rospy.loginfo(f"Successfully decoded image {event_id}: shape={frame.shape}, dtype={frame.dtype}")
                    
                    # 检测图像质量
                    if frame.size > 0:
                        mean_val = np.mean(frame)
                        #rospy.loginfo(f"Image {event_id} mean pixel value: {mean_val:.2f}")
                        
                        # 检查图像是否全黑或全白
                        if mean_val < 10:
                            rospy.logwarn(f"Image {event_id} might be too dark (mean={mean_val:.2f})")
                        elif mean_val > 245:
                            rospy.logwarn(f"Image {event_id} might be too bright (mean={mean_val:.2f})")
                    
                    with self.lock:
                        self.recv_images[event_id] = frame
                        self.recv_images_status[event_id] = CONNECT_TIMEOUT_FRAME
                else:
                    rospy.logwarn(f"Failed to decode image for {event_id} with encoding {msg.encoding}")
        except Exception as e:
            logger.error(f"recv image error: {e}")
            rospy.logerr(f"Error processing image {event_id}: {e}")

    def ros_replay(self, array):
        try:
            def normalize_precision(val, decimals=3):
                val = float(val)
                if np.isnan(val) or np.isinf(val):
                    rospy.logwarn(f"检测到非法值 {val}，替换为0.0")
                    return 0.0
                return round(val, decimals)

            def to_uint_list(values, vmin=0, vmax=100):
                out = []
                for v in values:
                    fv = normalize_precision(v)
                    fv = max(vmin, min(vmax, fv))
                    out.append(int(round(fv)))
                return out

            # 兼容两种动作格式：
            # 21维: 7(arm) + 6(right_hand) + 6(left_hand) + 2(head)
            # 28维: 7(left_arm) + 6(left_hand) + 7(right_arm) + 6(right_hand) + 2(head)
            if len(array) >= 28:
                left_arm = [normalize_precision(v) for v in array[0:7]]
                left_dexhand = to_uint_list(array[7:13])
                right_arm = [normalize_precision(v) for v in array[13:20]]
                right_dexhand = to_uint_list(array[20:26])
                head = [normalize_precision(v) for v in array[26:28]]
            elif len(array) >= 21:
                arm = [normalize_precision(v) for v in array[0:7]]
                right_dexhand = to_uint_list(array[7:13])
                left_dexhand = to_uint_list(array[13:19])
                head = [normalize_precision(v) for v in array[19:21]]
                left_arm = arm
                right_arm = arm
            else:
                raise ValueError(f"Action vector too short: {len(array)}")

            # --- 手臂控制 ---
            arm_msg = JointState()
            arm_msg.name = ["arm_joint_" + str(i) for i in range(1, 15)]  # 1~14
            arm_msg.position = left_arm + right_arm  # 14维
            arm_msg.header.stamp = rospy.Time.now()
            self.arm_pub.publish(arm_msg)

            # --- 手部控制 ---
            hand_msg = robotHandPosition()
            hand_msg.left_hand_position = left_dexhand    # 6维 [0~100]
            hand_msg.right_hand_position = right_dexhand  # 6维 [0~100]
            self.hand_pub.publish(hand_msg)

            # --- 头部控制 ---
            head_msg = robotHeadMotionData()
            head_msg.joint_data = head  # [yaw, pitch] in degrees
            self.head_pub.publish(head_msg)

        except Exception as e:
            rospy.logerr(f"Error during replay at frame: {e}")
            raise

    def destroy(self):
        self.stop_spin = True
        rospy.signal_shutdown("Node shutdown requested")


# ROS1的spin线程函数
def ros_spin_thread(node):
    while not rospy.is_shutdown() and not getattr(node, "stop_spin", False):
        try:
            rospy.Rate(100).sleep()
        except rospy.ROSInterruptException:
            break


if __name__ == "__main__":
    try:
        node = LEJUKuavoRos1Node()
        spin_thread = threading.Thread(target=ros_spin_thread, args=(node,))
        spin_thread.start()
        
        while not rospy.is_shutdown() and not node.stop_spin:
            time.sleep(0.1)
        
        node.destroy()
        spin_thread.join()
    except rospy.ROSInterruptException:
        pass
