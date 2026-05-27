#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, Image
from geometry_msgs.msg import PoseStamped
import math

class MockHardwareNode(Node):
    def __init__(self):
        super().__init__('mock_hardware_node')
        
        # --- 1. 发布者定义 ---
        # 主臂控制 (Master Control)
        self.pub_master_left = self.create_publisher(JointState, '/left_arm_joint_control', 10)
        self.pub_master_right = self.create_publisher(JointState, '/right_arm_joint_control', 10)
        self.pub_master_left_hand = self.create_publisher(JointState, '/cb_left_hand_raw_data', 10)
        self.pub_master_right_hand = self.create_publisher(JointState, '/cb_right_hand_raw_data', 10)
        # 从臂状态 (Slave States)
        self.pub_l_slave_joint = self.create_publisher(JointState, '/robot1/left_arm/joint_states', 10)
        self.pub_l_slave_pose = self.create_publisher(PoseStamped, '/robot1/left_arm/pose_states', 10)
        self.pub_r_slave_joint = self.create_publisher(JointState, '/robot1/right_arm/joint_states', 10)
        self.pub_r_slave_pose = self.create_publisher(PoseStamped, '/robot1/right_arm/pose_states', 10)
        
        # 灵巧手 (Hand)
        self.pub_hand_right = self.create_publisher(JointState, '/cb_right_hand_state', 10)
        self.pub_hand_left = self.create_publisher(JointState, '/cb_left_hand_state', 10)
        # 相机 (Camera)
        self.pub_camera = self.create_publisher(Image, '/camera/color/image_raw', 10)

        # 定时器
        self.timer = self.create_timer(1.0 / 30.0, self.timer_callback)
        self.step = 0.0
        self.get_logger().info("🚀 All systems GO! Mocking all topics...")

    def timer_callback(self):
        now = self.get_clock().now().to_msg()
        self.step += 0.05

        # --- A. 模拟主臂控制数据 (Master Arms) ---
        m_l_ctrl = JointState()
        m_l_ctrl.header.stamp = now
        m_l_ctrl.position = [math.sin(self.step) * 0.5] * 7
        self.pub_master_left.publish(m_l_ctrl)

        m_r_ctrl = JointState()
        m_r_ctrl.header.stamp = now
        m_r_ctrl.position = [math.cos(self.step) * 0.5] * 7
        self.pub_master_right.publish(m_r_ctrl)

        # --- B. 模拟从臂关节数据 (Slave Joints) ---
        # 左臂
        s_l_joint = JointState()
        s_l_joint.header.stamp = now
        s_l_joint.name = [f"L{i}_joint" for i in range(1, 8)]
        s_l_joint.position = [math.cos(self.step + i) for i in range(7)]
        self.pub_l_slave_joint.publish(s_l_joint)

        # 右臂
        s_r_joint = JointState()
        s_r_joint.header.stamp = now
        s_r_joint.name = [f"R{i}_joint" for i in range(1, 8)]
        s_r_joint.position = [math.sin(self.step + i) for i in range(7)]
        self.pub_r_slave_joint.publish(s_r_joint)

        # --- C. 模拟末端位姿数据 (Slave Poses) ---
        # 使用简单的正弦波模拟空间位置变化
        def create_mock_pose(offset):
            p = PoseStamped()
            p.header.stamp = now
            p.header.frame_id = "base_link"
            p.pose.position.x = 0.5 + 0.1 * math.sin(self.step + offset)
            p.pose.position.y = offset + 0.1 * math.cos(self.step)
            p.pose.position.z = 0.3
            p.pose.orientation.w = 1.0 # 保持姿态固定
            return p

        self.pub_l_slave_pose.publish(create_mock_pose(-0.2)) # 左手偏左
        self.pub_r_slave_pose.publish(create_mock_pose(0.2))  # 右手偏右

        # --- D. 模拟灵巧手 (Hand) ---
        msg_hand_l = JointState()
        msg_hand_l.header.stamp = now
        msg_hand_l.name = [f"left_finger_{i}" for i in range(10)]
        msg_hand_l.position = [abs(math.sin(self.step))] * 10
        self.pub_hand_left.publish(msg_hand_l)

        msg_hand_r = JointState()
        msg_hand_r.header.stamp = now
        msg_hand_r.name = [f"right_finger_{i}" for i in range(10)]
        msg_hand_r.position = [abs(math.cos(self.step))] * 10 # 使用 cos 让动作交替
        self.pub_hand_right.publish(msg_hand_r)
        # --- E. 模拟图像 (Camera) ---
        msg_img = Image()
        msg_img.header.stamp = now
        msg_img.height, msg_img.width = 480, 640
        msg_img.encoding = 'rgb8'
        msg_img.step = 640 * 3
        # 这里的图像会随时间变色（从纯黑到纯绿循环）
        green_val = int((math.sin(self.step) + 1) * 127)
        msg_img.data = bytes([0, green_val, 0] * (640 * 480))
        self.pub_camera.publish(msg_img)

def main(args=None):
    rclpy.init(args=args)
    node = MockHardwareNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()