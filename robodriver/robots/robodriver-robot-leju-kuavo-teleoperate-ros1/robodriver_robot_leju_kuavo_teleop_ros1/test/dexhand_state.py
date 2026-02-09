import rospy
from sensor_msgs.msg import JointState

# 定义回调函数
def dexhand_state_callback(msg):
    dexhand_position = msg.position
    # 打印当前灵巧手状态
    rospy.loginfo(f"Current dexhand_position: left_position={dexhand_position[:6]}, right_position={dexhand_position[-6:]}")

if __name__ == '__main__':
    # 初始化ROS节点
    rospy.init_node('dexhand_state_listener')

    # 创建订阅者，监听 /leju_claw_state 话题
    rospy.Subscriber('/dexhand/state', JointState, dexhand_state_callback)

    # 保持节点运行
    rospy.spin()
