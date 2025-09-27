import time
import sys
from a2d_sdk.robot import RobotDds as Robot

# 初始化机器人
robot = Robot()
time.sleep(0.5)  # 等待资源初始化，确保能接收到消息

print("=== 开始获取机器人各部分状态 ===\n")

# 1. 获取整体姿态关节状态（body_pose_joint_states）
try:
    body_pose = robot.body_pose_joint_states()
    print("1. body_pose_joint_states() → list")
    print("   返回值:", body_pose)
    print("   类型:", type(body_pose))
    print("   长度:", len(body_pose) if isinstance(body_pose, (list, tuple)) else "N/A")
except Exception as e:
    print("   调用失败:", e)

print()

# 2. 获取头部关节状态 [yaw, pitch]
try:
    head_pos, head_ts = robot.head_joint_states()
    print("2. head_joint_states() → (list, int)")
    print("   关节角度 [yaw, pitch]:", head_pos)
    print("   时间戳 (ns):", head_ts)
    print("   角度类型:", type(head_pos), "| 时间戳类型:", type(head_ts))
except Exception as e:
    print("   调用失败:", e)

print()

# 3. 获取腰部关节状态 [pitch, height]
try:
    waist_pos, waist_ts = robot.waist_joint_states()
    print("3. waist_joint_states() → (list, int)")
    print("   腰部状态 [pitch, height]:", waist_pos)
    print("   时间戳 (ns):", waist_ts)
    print("   状态类型:", type(waist_pos), "| 时间戳类型:", type(waist_ts))
except Exception as e:
    print("   调用失败:", e)

print()

# 4. 获取双臂 14 个关节状态
try:
    arm_pos, arm_ts = robot.arm_joint_states()
    print("4. arm_joint_states() → (list, int)")
    print("   双臂关节角度（共14个）:", arm_pos)
    print("   时间戳 (ns):", arm_ts)
    print("   关节数量:", len(arm_pos) if isinstance(arm_pos, list) else "N/A")
    print("   类型:", type(arm_pos), "| 时间戳类型:", type(arm_ts))
except Exception as e:
    print("   调用失败:", e)

print()

# 5. 获取夹持器状态 [left, right]
try:
    gripper, gripper_ts = robot.gripper_states()
    print("5. gripper_states() → (list, int)")
    print("   夹持器开合度 [left, right]:", gripper)
    print("   时间戳 (ns):", gripper_ts)
    print("   类型:", type(gripper), "| 时间戳类型:", type(gripper_ts))
except Exception as e:
    print("   调用失败:", e)

print()

# 6. 获取手部 12 个关节状态
try:
    hand_pos, hand_ts = robot.hand_joint_states()
    print("6. hand_joint_states() → (list, int)")
    print("   手部关节角度（共12个）:", hand_pos)
    print("   时间戳 (ns):", hand_ts)
    print("   关节数量:", len(hand_pos) if isinstance(hand_pos, list) else "N/A")
    print("   类型:", type(hand_pos), "| 时间戳类型:", type(hand_ts))
except Exception as e:
    print("   调用失败:", e)

print()

# 7. 获取手部力传感器状态
try:
    hand_force = robot.hand_force_states()
    print("7. hand_force_states() → list")
    print("   手部力传感器读数:", hand_force)
    print("   类型:", type(hand_force))
    print("   传感器数量:", len(hand_force) if isinstance(hand_force, (list, tuple)) else "N/A")
except Exception as e:
    print("   调用失败:", e)

print()

# 8. 获取整机状态
try:
    whole_status = robot.whole_body_status()
    print("8. whole_body_status() → ()")
    print("   整机状态返回值:", whole_status)
    print("   类型:", type(whole_status))
    # 注意：根据文档描述返回空元组，但实际可能包含状态字典，此处按实际返回打印
except Exception as e:
    print("   调用失败:", e)

print("\n=== 状态获取完成 ===")

# 关闭连接
robot.shutdown()
sys.exit(0)