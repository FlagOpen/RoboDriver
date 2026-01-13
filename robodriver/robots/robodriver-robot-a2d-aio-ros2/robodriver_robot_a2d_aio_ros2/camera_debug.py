#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2


def imgmsg_to_cv2_manual(msg):
    if msg.encoding == 'bgr8':
        dtype = np.uint8
        n_channels = 3
    elif msg.encoding == 'rgb8':
        dtype = np.uint8
        n_channels = 3
    elif msg.encoding == 'mono8':
        dtype = np.uint8
        n_channels = 1
    else:
        raise ValueError(f"Unsupported encoding: {msg.encoding}")
    
    # 将 data 转为 numpy 数组
    img = np.frombuffer(msg.data, dtype=dtype).reshape((msg.height, msg.width, n_channels))
    
    # 如果是 rgb8，转为 bgr8（OpenCV 默认）
    if msg.encoding == 'rgb8':
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    return img

class CameraDebugger(Node):
    def __init__(self):
        super().__init__('camera_debugger')
        self.bridge = CvBridge()

        # 订阅三个摄像头（可按需注释）
        self.create_subscription(Image, '/camera/head_color', self.front_cb, 10)
        self.create_subscription(Image, '/camera/hand_left_color', self.left_cb, 10)
        self.create_subscription(Image, '/camera/hand_right_color', self.right_cb, 10)

        self.get_logger().info("Camera Debugger Started. Showing front/left/right views...")
        self.windows_created = False

    def front_cb(self, msg):
        #cv_img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        #cv2.imshow("Front Camera", cv_img)
        #self.windows_created = True
        try:
            cv_img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            cv2.imshow("Front Camera", cv_img)
        except Exception as e:
            self.get_logger().error(f"Image decode failed: {e}")

    def left_cb(self, msg):
        cv_img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        cv2.imshow("Left Camera", cv_img)
        self.windows_created = True

    def right_cb(self, msg):
        cv_img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        cv2.imshow("Right Camera", cv_img)
        self.windows_created = True

    def check_quit(self):
        if self.windows_created:
            # 按 'q' 或 ESC 退出
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # ESC
                return True
        return False

def main(args=None):
    rclpy.init(args=args)
    node = CameraDebugger()

    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.01)  # 非阻塞
            if node.check_quit():
                break
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
