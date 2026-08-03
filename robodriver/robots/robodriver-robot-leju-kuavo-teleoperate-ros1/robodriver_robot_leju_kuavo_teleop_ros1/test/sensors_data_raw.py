import rospy
from kuavo_humanoid_sdk.msg.kuavo_msgs.msg import sensorsData

def callback(data):
    rospy.loginfo(f"Received sensor data at time: {data.sensor_time}")

# 初始化ROS节点
rospy.init_node('sensor_data_listener')

# 订阅传感器数据话题
rospy.Subscriber('/sensors_data_raw', sensorsData, callback)

# 保持节点运行
rospy.spin()
