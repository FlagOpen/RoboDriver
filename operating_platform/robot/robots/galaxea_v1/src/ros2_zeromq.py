#!/usr/bin/python3
import rclpy
from rclpy.node import Node as ROS2Node
from sensor_msgs.msg import JointState, CompressedImage, Image
from std_msgs.msg import Float32MultiArray
import zmq
import threading
import numpy as np
import time
import json
from message_filters import Subscriber, ApproximateTimeSynchronizer

# IPC Address
ipc_address_joint = "ipc:///tmp/ros2-zeromq-galaxea-joint"
ipc_address_image = "ipc:///tmp/ros2-zeromq-galaxea-image"

class ROS2BridgeNode(ROS2Node):
    def __init__(self):
        super().__init__('ros2_zeromq_bridge')
        
        # ROS 2 订阅与发布
        self.publisher_left_arm = self.create_publisher(
            JointState, "/motion_target/target_joint_state_arm_left", 10
        )
        self.publisher_right_arm = self.create_publisher(
            JointState, "/motion_target/target_joint_state_arm_right", 10
        )
        self.publisher_left_gripper = self.create_publisher(
            JointState, "/motion_target/target_position_gripper_left", 10
        )
        self.publisher_right_gripper = self.create_publisher(
            JointState, "/motion_target/target_position_gripper_right", 10
        )
        
        # ZeroMQ 初始化
        self.galaxea_context = zmq.Context()
        self.socket_joint = self.galaxea_context.socket(zmq.PAIR)
        self.socket_joint.bind(ipc_address_joint)
        self.socket_joint.setsockopt(zmq.SNDHWM, 2000)
        self.socket_joint.setsockopt(zmq.SNDBUF, 2**25)
        self.socket_joint.setsockopt(zmq.SNDTIMEO, 2000)
        self.socket_joint.setsockopt(zmq.RCVTIMEO, 2000)
        self.socket_joint.setsockopt(zmq.LINGER, 0)
        
        self.socket_image = self.galaxea_context.socket(zmq.PAIR)
        self.socket_image.bind(ipc_address_image)
        self.socket_image.setsockopt(zmq.SNDHWM, 2000)
        self.socket_image.setsockopt(zmq.SNDBUF, 2**25)
        self.socket_image.setsockopt(zmq.SNDTIMEO, 2000)
        self.socket_image.setsockopt(zmq.RCVTIMEO, 2000)
        self.socket_image.setsockopt(zmq.LINGER, 0)
        # 运行标志
        self.running = True
        
        # 启动 ZeroMQ 接收线程
        self.zeromq_thread = threading.Thread(target=self.zeromq_receive_loop)
        self.zeromq_thread.daemon = True
        self.zeromq_thread.start()

        self.last_main_send_time_ns = 0
        self.last_follow_send_time_ns = 0
        self.min_interval_ns = 1e9 / 30  # 30Hz 对应的最小间隔时间(纳秒)
        # 使用message_filters同步左右臂和夹爪数据
        self._init_message_main_filters()
        self._init_message_follow_filters()
         # 为图片创建独立的同步器
        self._init_image_message_filters()
        self.get_logger().info(f"ROS2->ZeroMQ started, joint IPC: {ipc_address_joint}, image IPC: {ipc_address_image}")

    def _init_message_follow_filters(self):
        """初始化message_filters同步订阅器"""
        # 创建Subscriber对象（需指定消息类型和话题）
        sub_arm_left = Subscriber(self, JointState, '/hdas/feedback_arm_left')
        sub_arm_right = Subscriber(self, JointState, '/hdas/feedback_arm_right')
        sub_gripper_left = Subscriber(self, JointState, '/hdas/feedback_gripper_left')
        sub_gripper_right = Subscriber(self, JointState, '/hdas/feedback_gripper_right')

 
        # 创建同步器：队列大小=10，时间容差=100ms
        self.sync = ApproximateTimeSynchronizer(
            [sub_arm_left, sub_arm_right, sub_gripper_left, sub_gripper_right],
            queue_size=10,
            slop=0.01  # 时间容差（秒）
        )
        self.sync.registerCallback(self.synchronized_follow_callback)
 
    def synchronized_follow_callback(self, arm_left, arm_right, gripper_left, gripper_right):
        """同步后的回调函数"""
        try:
            current_time_ns = time.time_ns()
            # 频率控制：30Hz
            if (current_time_ns - self.last_follow_send_time_ns) < self.min_interval_ns:
                return
            self.last_follow_send_time_ns = current_time_ns
 
            # 提取左右臂的位置和速度
            left_pos = np.array(arm_left.position, dtype=np.float32)
            right_pos = np.array(arm_right.position, dtype=np.float32)
           
            # 合并左右臂数据
            left_arm_data = left_pos[:-1]
            right_arm_data = right_pos[:-1]
 
            # 提取夹爪位置（假设每个夹爪只有1个关节）
            gripper_left_pos = np.array(gripper_left.position, dtype=np.float32)
            gripper_right_pos = np.array(gripper_right.position, dtype=np.float32)
           
            # 最终合并所有数据
            merged_data = np.concatenate([left_arm_data, gripper_left_pos, right_arm_data, gripper_right_pos])
 
            # 通过ZeroMQ发送
            self.socket_joint.send_multipart([
                b"follower_arms",
                merged_data.tobytes()
            ], flags=zmq.NOBLOCK)
        except Exception as e:
            self.get_logger().error(f"Synchronized follow callback error: {e}")
            pass

    def _init_message_main_filters(self):
        """初始化message_filters同步订阅器"""
        # 创建Subscriber对象（需指定消息类型和话题）
        sub_arm_left = Subscriber(self, JointState, '/motion_target/target_joint_state_arm_left')
        sub_arm_right = Subscriber(self, JointState, '/motion_target/target_joint_state_arm_right')
        sub_gripper_left = Subscriber(self, JointState, '/motion_target/target_position_gripper_left')
        sub_gripper_right = Subscriber(self, JointState, '/motion_target/target_position_gripper_right')

 
        # 创建同步器：队列大小=10，时间容差=100ms
        self.sync = ApproximateTimeSynchronizer(
            [sub_arm_left, sub_arm_right, sub_gripper_left, sub_gripper_right],
            queue_size=10,
            slop=0.01  # 时间容差（秒）
        )
        self.sync.registerCallback(self.synchronized_main_callback)
 
    def synchronized_main_callback(self, arm_left, arm_right, gripper_left, gripper_right):
        """同步后的回调函数"""
        try:
            current_time_ns = time.time_ns()
            # 频率控制：30Hz
            if (current_time_ns - self.last_main_send_time_ns) < self.min_interval_ns:
                return
            self.last_main_send_time_ns = current_time_ns
 
            # 提取左右臂的位置和速度
            left_pos = np.array(arm_left.position, dtype=np.float32)
            right_pos = np.array(arm_right.position, dtype=np.float32)
 
            # 提取夹爪位置（假设每个夹爪只有1个关节）
            gripper_left_pos = np.array(gripper_left.position, dtype=np.float32)
            gripper_right_pos = np.array(gripper_right.position, dtype=np.float32)
           
            # 最终合并所有数据
            merged_data = np.concatenate([left_pos, gripper_left_pos, right_pos, gripper_right_pos])
 
            # 通过ZeroMQ发送
            self.socket_joint.send_multipart([
                b"leader_arms",
                merged_data.tobytes()
            ], flags=zmq.NOBLOCK)
        except Exception as e:
            self.get_logger().error(f"Synchronized main callback error: {e}")
            pass

    def _init_image_message_filters(self):
        """初始化图片数据的message_filters同步订阅器"""
        sub_camera_top_left = Subscriber(self, CompressedImage, '/hdas/camera_head/left_raw/image_raw_color/compressed')
        sub_camera_top_right = Subscriber(self, CompressedImage, '/hdas/camera_head/right_raw/image_raw_color/compressed')
        sub_camera_wrist_left = Subscriber(self, CompressedImage, '/hdas/camera_wrist_left/color/image_rect_raw/compressed')
        sub_camera_wrist_right = Subscriber(self, CompressedImage, '/hdas/camera_wrist_right/color/image_rect_raw/compressed')
        sub_camera_wrist_left_depth = Subscriber(self, Image, '/hdas/camera_wrist_left/aligned_depth_to_color/image_raw')
        sub_camera_wrist_right_depth = Subscriber(self, Image, '/hdas/camera_wrist_right/aligned_depth_to_color/image_raw')
 
        # 创建图片同步器：队列大小=5，时间容差=200ms（因为频率较低）
        self.image_sync = ApproximateTimeSynchronizer(
            [sub_camera_top_left, sub_camera_top_right, sub_camera_wrist_left, sub_camera_wrist_right,sub_camera_wrist_left_depth,sub_camera_wrist_right_depth],
            queue_size=5,
            slop=0.1  # 更大的时间容差
        )
        self.image_sync.registerCallback(self.image_synchronized_callback)

    def image_synchronized_callback(self, top_left, top_right, wrist_left, wrist_right,wrist_left_depth, wrist_right_depth):
        """图片数据同步后的回调函数"""
        try:
            # 发送各摄像头图片
            self.images_send_zeromq(top_left, "image_top_left", 1280, 720)
            self.images_send_zeromq(top_right, "image_top_right", 1280, 720)
            self.images_send_zeromq(wrist_left, "image_wrist_left", 640, 360)
            self.images_send_zeromq(wrist_right, "image_wrist_right", 640, 360)
            self.images_send_zeromq(wrist_left_depth, "image_depth_wrist_left", 640, 360, 'depth16')
            self.images_send_zeromq(wrist_right_depth, "image_depth_wrist_right", 640, 360, 'depth16')
            
        except Exception as e:
            # self.get_logger().error(f"Image synchronized callback error: {e}")
            pass


    def images_send_zeromq(self,msg,event_id,width,height,encoding="jpeg"):
        """ROS 2 -> ZeroMQ"""
        try:
            buffer_bytes = msg.data
            meta = {
                "encoding":encoding,
                "width": width,
                "height": height
            }
            meta_bytes = json.dumps(meta).encode('utf-8')
            self.socket_image.send_multipart([
                event_id.encode('utf-8'),
                buffer_bytes,
                meta_bytes,
            ], flags=zmq.NOBLOCK)
        except Exception as e:
            self.get_logger().error(f"ROS2->ZeroMQ error: {e}")
            pass

    
    def zeromq_receive_loop(self):
        """ZeroMQ -> ROS 2 (独立线程)"""
        while self.running:
            try:
                # 使用 poll 设置超时，避免完全阻塞
                if self.socket_joint.poll(timeout=100):  # 100ms 超时
                    event_id, buffer_bytes = self.socket_joint.recv_multipart()
                    event_id = event_id.decode('utf-8')
                    if 'action_joint' in event_id:
                        try:
                            array = np.frombuffer(buffer_bytes, dtype=np.float32)
                            print(array)
                            left_arm = array[:6]          # 左臂 6 关节
                            left_gripper = array[6:7]     # 左夹爪 1 关节
                            right_arm = array[7:13]       # 右臂 6 关节
                            right_gripper = array[13:14]  # 右夹爪 1 关节

                            # msg = JointState()
                            # msg.header.stamp = self.get_clock().now().to_msg()
                            # msg.name = self.left_arm_joint_names
                            # msg.position = left_arm
                            # self.publisher_left_arm.publish(msg)
                            # # 发布右臂
                            # msg = JointState()
                            # msg.header.stamp = self.get_clock().now().to_msg()
                            # msg.name = self.right_arm_joint_names
                            # msg.position = right_arm
                            # self.publisher_right_arm.publish(msg)
                            # # 发布左夹爪
                            # msg = JointState()
                            # msg.header.stamp = self.get_clock().now().to_msg()
                            # msg.name = self.left_gripper_joint_names
                            # msg.position = left_gripper
                            # self.publisher_left_gripper.publish(msg)
                            # # 发布右夹爪
                            # msg = JointState()
                            # msg.header.stamp = self.get_clock().now().to_msg()
                            # msg.name = self.right_gripper_joint_names
                            # msg.position = right_gripper
                            # self.publisher_right_gripper.publish(msg)
                        except Exception as e:
                            self.get_logger().error(f"Error during replay at frame: {e}")
            except zmq.Again:
                continue  # 超时后继续循环
            except Exception as e:
                self.get_logger().error(f"ZeroMQ->ROS2 error: {e}")
                time.sleep(1)  # 出错后稍作等待

    def destroy(self):
        """清理资源"""
        self.running = False
        if self.zeromq_thread.is_alive():
            self.zeromq_thread.join()
        self.socket_joint.close()
        self.socket_image.close()
        self.galaxea_context.term()
        super().destroy_node()

def main():
    rclpy.init()
    node = ROS2BridgeNode()
    
    try:
        rclpy.spin(node)  # 运行节点
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy()
        rclpy.shutdown()

if __name__ == "__main__":
    main()