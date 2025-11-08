#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from kuavo_msgs.msg import sensorsData
import cv2

class ImageViewer:
    def __init__(self):
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/camera/color/image_raw", Image, self.image_callback)
        self.sensor_sub = rospy.Subscriber("/sensors_data_raw", sensorsData, self.sensor_callback)
        rospy.loginfo("Image viewer node started. Waiting for images and sensor data...")

    def image_callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            rospy.logerr("CvBridge Error: %s" % e)
            return

        cv2.imshow("Raw Image", cv_image)
        cv2.waitKey(1)

    def sensor_callback(self, msg):
        joint_q = msg.joint_data.joint_q

        if len(joint_q) < 28:
            rospy.logwarn("Received joint_q has fewer than 28 elements. Skipping.")
            return

        # 下半身关节索引和名称
        lower_body_indices = list(range(0, 12))
        lower_body_names = [
            "l_leg_roll", "l_leg_yaw", "l_leg_pitch", "l_knee", "l_foot_pitch", "l_foot_roll",
            "r_leg_roll", "r_leg_yaw", "r_leg_pitch", "r_knee", "r_foot_pitch", "r_foot_roll"
        ]

        # 上半身关节索引和名称
        upper_body_indices = list(range(12, 26)) + [26, 27]
        upper_body_names = [
            "l_arm_pitch", "l_arm_roll", "l_arm_yaw", "l_forearm_pitch",
            "l_hand_yaw", "l_hand_pitch", "l_hand_roll",
            "r_arm_pitch", "r_arm_roll", "r_arm_yaw", "r_forearm_pitch",
            "r_hand_yaw", "r_hand_pitch", "r_hand_roll",
            "head_yaw", "head_pitch"
        ]

        # 一次性构建日志字符串
        lines = [f"  {name}: {joint_q[idx]:.4f}" for name, idx in zip(upper_body_names, upper_body_indices)]
        log_message = "Upper body joint angles (rad):\n" + "\n".join(lines)

        lines = [f"  {name}: {joint_q[idx]:.4f}" for name, idx in zip(lower_body_names, lower_body_indices)]
        log_message = log_message + "\n\nLower body joint angles (rad):\n" + "\n".join(lines)

        rospy.loginfo(log_message)

def main():
    rospy.init_node('wanx_data_listener', anonymous=True)
    viewer = ImageViewer()
    
    try:
        rospy.spin()
    except KeyboardInterrupt:
        rospy.loginfo("Shutting down...")
    finally:
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()