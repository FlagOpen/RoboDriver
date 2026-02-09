import rospy
from sensor_msgs.msg import JointState
import numpy as np

# 初始化ROS节点
rospy.init_node('sim_traj')

# 创建发布者
pub = rospy.Publisher("/kuavo_arm_traj", JointState, queue_size=10)

# 等待直到有订阅者连接
while pub.get_num_connections() == 0:
    rospy.sleep(0.1)  # 适当的睡眠时间，避免 CPU 占用过高

msg = JointState()
msg.name = ["arm_joint_" + str(i) for i in range(1, 15)]  # 关节名称列表
msg.header.stamp = rospy.Time.now()  # 当前时间戳
msg.position = np.array([50, 0, 0, 50, 0, 0, 0, 0, 0, 0, 0, 0, 0, 50])  # 关节位置列表

# 发布消息
pub.publish(msg)