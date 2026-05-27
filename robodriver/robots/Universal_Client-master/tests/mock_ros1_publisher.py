#!/usr/bin/env python3
import rospy
import math
from sensor_msgs.msg import JointState, Image
from geometry_msgs.msg import PoseStamped

class MockHardwareNodeROS1:
    def __init__(self):
        # 初始化 ROS 1 节点
        rospy.init_node('mock_hardware_node_ros1', anonymous=True)
        
        # --- 1. 发布者定义 (完全对齐 yaml 中的 sensors 话题) ---
        
        # 相机
        self.pub_camera = rospy.Publisher('/camera/color/image_raw', Image, queue_size=1)

        # 主臂 (Leader)
        self.pub_ld_l_joint = rospy.Publisher('/left_arm_joint_control', JointState, queue_size=10)
        self.pub_ld_r_joint = rospy.Publisher('/right_arm_joint_control', JointState, queue_size=10)
        self.pub_ld_l_hand = rospy.Publisher('/cb_left_hand_raw_data', JointState, queue_size=10)
        self.pub_ld_r_hand = rospy.Publisher('/cb_right_hand_raw_data', JointState, queue_size=10)

        # 从臂 (Follower)
        self.pub_fw_l_joint = rospy.Publisher('/robot1/left_arm/joint_states', JointState, queue_size=10)
        self.pub_fw_l_pose = rospy.Publisher('/robot1/left_arm/pose_states', PoseStamped, queue_size=10)
        self.pub_fw_r_joint = rospy.Publisher('/robot1/right_arm/joint_states', JointState, queue_size=10)
        self.pub_fw_r_pose = rospy.Publisher('/robot1/right_arm/pose_states', PoseStamped, queue_size=10) # 保持 right_armxi 拼写

        # 灵巧手 (Follower)
        self.pub_fw_l_hand = rospy.Publisher('/cb_left_hand_state', JointState, queue_size=10)
        self.pub_fw_r_hand = rospy.Publisher('/cb_right_hand_state', JointState, queue_size=10)

        self.step = 0.0
        rospy.loginfo("🚀 ROS 1 Mock Hardware Node started. Mocking all sensor topics at 30Hz...")

    def run(self):
        # 按照 yaml 中的 30 fps 进行模拟
        rate = rospy.Rate(30)
        
        while not rospy.is_shutdown():
            self.step += 0.05
            now = rospy.Time.now()

            # --- A. 模拟主臂数据 (Leader) ---
            msg = JointState(); msg.header.stamp = now
            msg.position = [math.sin(self.step) * 0.5] * 7
            self.pub_ld_l_joint.publish(msg)

            msg = JointState(); msg.header.stamp = now
            msg.position = [math.cos(self.step) * 0.5] * 7
            self.pub_ld_r_joint.publish(msg)

            msg = JointState(); msg.header.stamp = now
            msg.position = [abs(math.sin(self.step))] * 10
            self.pub_ld_l_hand.publish(msg)

            msg = JointState(); msg.header.stamp = now
            msg.position = [abs(math.cos(self.step))] * 10
            self.pub_ld_r_hand.publish(msg)

            # --- B. 模拟从臂关节数据 (Follower Joints) ---
            s_l_joint = JointState(); s_l_joint.header.stamp = now
            s_l_joint.name = [f"L{i}_joint" for i in range(1, 8)]
            s_l_joint.position = [math.cos(self.step + i) for i in range(7)]
            self.pub_fw_l_joint.publish(s_l_joint)

            s_r_joint = JointState(); s_r_joint.header.stamp = now
            s_r_joint.name = [f"R{i}_joint" for i in range(1, 8)]
            s_r_joint.position = [math.sin(self.step + i) for i in range(7)]
            self.pub_fw_r_joint.publish(s_r_joint)

            # --- C. 模拟末端位姿数据 (Follower Poses) ---
            def create_mock_pose(offset):
                p = PoseStamped(); p.header.stamp = now; p.header.frame_id = "base_link"
                p.pose.position.x = 0.5 + 0.1 * math.sin(self.step + offset)
                p.pose.position.y = offset + 0.1 * math.cos(self.step)
                p.pose.position.z = 0.3
                p.pose.orientation.w = 1.0
                return p

            self.pub_fw_l_pose.publish(create_mock_pose(-0.2))
            self.pub_fw_r_pose.publish(create_mock_pose(0.2))

            # --- D. 模拟灵巧手 (Follower Hands) ---
            msg = JointState(); msg.header.stamp = now
            msg.position = [abs(math.sin(self.step))] * 10
            self.pub_fw_l_hand.publish(msg)

            msg = JointState(); msg.header.stamp = now
            msg.position = [abs(math.cos(self.step))] * 10
            self.pub_fw_r_hand.publish(msg)

            # --- E. 模拟图像 (Camera) ---
            msg_img = Image(); msg_img.header.stamp = now
            msg_img.height, msg_img.width = 480, 640
            msg_img.encoding = 'rgb8'
            msg_img.step = 640 * 3
            # 让图像颜色随时间产生绿色的渐变呼吸效果
            green_val = int((math.sin(self.step) + 1) * 127)
            msg_img.data = bytes([0, green_val, 0] * (640 * 480))
            self.pub_camera.publish(msg_img)

            rate.sleep()

if __name__ == '__main__':
    try:
        node = MockHardwareNodeROS1()
        node.run()
    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        rospy.logerr(f"Error running mock node: {e}")