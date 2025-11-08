import pandas as pd
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
import time
import threading
class JointStateReplay(Node):
    def __init__(self, parquet_file):
        super().__init__("joint_state_replay")
        self.parquet_file = parquet_file
        self.running = False  # 控制回放线程的标志
        # 1. 读取 Parquet 文件
        try:
            df = pd.read_parquet(self.parquet_file)
            self.timestamps = df["timestamp"].tolist()  # 单位：秒
            self.actions = df["action"].tolist()       # 14 维位置数组
        except Exception as e:
            self.get_logger().error(f"Failed to read Parquet file: {e}")
            raise
        # 2. 创建发布者
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
        # 3. 定义关节名称
        self.left_arm_joint_names = [f"left_arm_joint_{i}" for i in range(1, 7)]
        self.right_arm_joint_names = [f"right_arm_joint_{i}" for i in range(1, 7)]
        self.left_gripper_joint_names = ["left_gripper_joint"]
        self.right_gripper_joint_names = ["right_gripper_joint"]
    def parse_action(self, action):
        """解析 14 维 action 数据，分割为左臂、右臂、左夹爪、右夹爪"""
        if isinstance(action, str):
            action = list(map(float, action.strip("[]").split(",")))
        action = [float(x) if x is not None else 0.0 for x in action]
        left_arm = action[:6]          # 左臂 6 关节
        left_gripper = action[6:7]     # 左夹爪 1 关节
        right_arm = action[7:13]       # 右臂 6 关节
        right_gripper = action[13:14]  # 右夹爪 1 关节
        return left_arm, left_gripper, right_arm, right_gripper
    def replay(self):
        """执行回放逻辑"""
        self.running = True
        self.get_logger().info("Starting joint state replay...")
        start_time = time.time()
        for i in range(len(self.timestamps)):
            if not self.running:
                break  # 提前终止
            # 同步到目标时间
            target_time = start_time + self.timestamps[i]
            while time.time() < target_time and self.running:
                time.sleep(0.001)
            # 解析并发布数据
            try:
                left_arm, left_gripper, right_arm, right_gripper = self.parse_action(self.actions[i])
                # 发布左臂
                msg = JointState()
                msg.header.stamp = self.get_clock().now().to_msg()
                msg.name = self.left_arm_joint_names
                msg.position = left_arm
                self.publisher_left_arm.publish(msg)
                # 发布右臂
                msg = JointState()
                msg.header.stamp = self.get_clock().now().to_msg()
                msg.name = self.right_arm_joint_names
                msg.position = right_arm
                self.publisher_right_arm.publish(msg)
                # 发布左夹爪
                msg = JointState()
                msg.header.stamp = self.get_clock().now().to_msg()
                msg.name = self.left_gripper_joint_names
                msg.position = left_gripper
                self.publisher_left_gripper.publish(msg)
                # 发布右夹爪
                msg = JointState()
                msg.header.stamp = self.get_clock().now().to_msg()
                msg.name = self.right_gripper_joint_names
                msg.position = right_gripper
                self.publisher_right_gripper.publish(msg)
            except Exception as e:
                self.get_logger().error(f"Error during replay at frame {i}: {e}")
                continue
        self.get_logger().info("Replay completed.")
        self.destroy_node()  # 直接销毁节点（需确保线程安全）
        rclpy.shutdown()
    def stop(self):
        """停止回放"""
        self.running = False
def replay_joint_states(parquet_path):
    """
    封装方法：在独立线程中运行回放
    :param parquet_path: Parquet 文件路径
    """
    rclpy.init()  # 初始化 ROS 2
    node = JointStateReplay(parquet_path)
    # 在独立线程中运行回放
    replay_thread = threading.Thread(target=node.replay)
    replay_thread.start()
    # 等待线程结束（或通过其他逻辑控制）
    try:
        while replay_thread.is_alive() and rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        node.get_logger().info("Interrupted by user.")
    finally:
        if node is not None:
            node.stop()
            if replay_thread is not None:
                replay_thread.join()
            node.destroy_node()
        rclpy.shutdown()

# file_path = '/home/litchi/DoRobot/dataset/20251014/dev/水果收纳_苹果_342/水果收纳_苹果_342_10866/data/chunk-000/episode_000000.parquet'
# replay_joint_states(file_path)