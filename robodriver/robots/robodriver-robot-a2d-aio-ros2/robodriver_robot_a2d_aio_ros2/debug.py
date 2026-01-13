#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from genie_msgs.msg import (
    HeadState,
    WaistState,
    BatteryStatus,
    Position,
    EndState  # <-- 注意：这里用 EndState
)

class A2DRobotDebugger(Node):
    def __init__(self):
        super().__init__('a2d_robot_debugger')

        # 订阅各状态 Topic
        self.create_subscription(JointState, '/hal/arm_joint_state', self.arm_cb, 10)
        self.create_subscription(HeadState, '/hal/neck_state', self.head_cb, 10)
        self.create_subscription(WaistState, '/hal/waist_state', self.waist_cb, 10)
        self.create_subscription(BatteryStatus, '/hal/batt_state', self.batt_cb, 10)
        self.create_subscription(Position, '/hal/position', self.pos_cb, 10)
        # 注意：左右末端实际是 ArmState 类型！
        self.create_subscription(EndState, '/hal/left_ee_data', self.left_ee_cb, 10)
        self.create_subscription(EndState, '/hal/right_ee_data', self.right_ee_cb, 10)

        self.get_logger().info("A2D Robot Debugger Started. Listening to key topics...")

    def arm_cb(self, msg):
        left = msg.position[:7]
        right = msg.position[7:]
        self.get_logger().info(f"[Arm] L: {[round(x, 3) for x in left]} | R: {[round(x, 3) for x in right]}")

    def head_cb(self, msg):
        yaw = msg.motor_states[0].position if len(msg.motor_states) > 0 else 0.0
        pitch = msg.motor_states[1].position if len(msg.motor_states) > 1 else 0.0
        self.get_logger().info(f"[Head] Yaw: {round(yaw, 3)} rad, Pitch: {round(pitch, 3)} rad")

    def waist_cb(self, msg):
        pitch = msg.motor_states[0].position if len(msg.motor_states) > 0 else 0.0
        lift = msg.motor_states[1].position if len(msg.motor_states) > 1 else 0.0
        self.get_logger().info(f"[Waist] Pitch: {round(pitch, 3)} rad, Lift: {round(lift * 100, 1)} cm")

    def batt_cb(self, msg):
        self.get_logger().info(
            f"[Battery] Level: {msg.energy}%, Voltage: {msg.voltage:.2f}V, Status: {msg.status}"
        )

    def pos_cb(self, msg):
        self.get_logger().info(
            f"[Chassis] Pose=({msg.agv_pos_x:.2f}, {msg.agv_pos_y:.2f})m, "
            f"Yaw={msg.agv_angle:.2f} rad, Conf={msg.position_conf}%"
        )

    def left_ee_cb(self, msg):
        # 从 motor_states 提取 position
        positions = [round(m.position, 3) for m in msg.end_state]
        self.get_logger().info(f"[Left EE] Motors: {positions}")

    def right_ee_cb(self, msg):
        positions = [round(m.position, 3) for m in msg.end_state]
        self.get_logger().info(f"[Right EE] Motors: {positions}")

def main(args=None):
    rclpy.init(args=args)
    node = A2DRobotDebugger()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down debugger...")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
