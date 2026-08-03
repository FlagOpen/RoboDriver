import rospy


# 初始化ROS节点
rospy.init_node('robot_head_controller')

# 创建发布者
pub_head_pose = rospy.Publisher('/robot_head_motion_data', robotHeadMotionData, queue_size=10)

# 创建消息对象
head_target_msg = robotHeadMotionData()
head_target_msg.joint_data = [0, 30]  # 偏航角和俯仰角

# 发布消息
pub_head_pose.publish(head_target_msg)