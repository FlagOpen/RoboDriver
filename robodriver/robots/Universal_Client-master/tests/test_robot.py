import sys
import os
import time
import math

sys.path.append(os.getcwd())
from src.core.factory import create_connector

def print_group_state(robot, device_key, label):
    data = robot.get_state(device_key)
    if data:
        pos = data.get('position', [])
        formatted_pos = [round(x, 3) for x in pos[:6]] 
        print(f"   {label} ({len(pos)}轴): {formatted_pos}{'...' if len(pos)>6 else ''}")
        return pos
    else:
        print(f"   ⚠️ {label} 数据尚未就绪")
        return None

def main():
    print("="*50)
    print("      LinkerBot 硬件接口测试 (HAL 架构)")
    print("="*50)

    try:
        robot = create_connector("config/settings.yaml")
        robot.start()
        print("✅ Robot Connector 初始化并启动成功")
        print("⏳ 等待接收传感器数据 (2秒)...")
        time.sleep(2)
    except Exception as e:
        print(f"❌ 初始化失败: {e}")
        return

    try:
        # ==========================================
        # 测试阶段 1: 读取状态
        # ==========================================
        print("\n[阶段 1/2] 测试状态读取...")
        for i in range(2):
            print(f"循环 {i+1}:")
            print_group_state(robot, "leader_left_joints", "左主臂")
            print_group_state(robot, "follower_left_joints", "左从臂关节")
            print_group_state(robot, "follower_left_hand", "左灵巧手")
            
            pose_data = robot.get_state("follower_left_pose")
            if pose_data:
                pos = pose_data['position']
                print(f"   左从臂位姿: x={pos['x']:.2f}, y={pos['y']:.2f}, z={pos['z']:.2f}")
                
            time.sleep(0.5)

        # ==========================================
        # 测试阶段 2: 增量式安全控制下发
        # ==========================================
        print("\n[阶段 2/2] 测试控制下发 (增量安全测试)...")
        input("👉 按回车键开始下发安全增量指令...")

        # --- 1. 机械臂增量控制 ---
        print("\n--- 测试机械臂控制 ---")
        current_arm_state = robot.get_state("follower_left_joints")
        
        if current_arm_state and 'position' in current_arm_state:
            current_pos = current_arm_state['position']
            # 安全增量：每个关节只增加 0.05 弧度
            safe_target = [pos + 0.05 for pos in current_pos]
            
            print(f"   当前关节角度: {[round(x, 3) for x in current_pos[:6]]}...")
            print(f"   目标安全角度: {[round(x, 3) for x in safe_target[:6]]}...")

            command_data = {"position": safe_target, "follow": True}
            success = robot.send_control("follower_left_arm_control", command_data)
            if success:
                print("   ✅ 机械臂控制指令已下发")
            else:
                print("   ❌ 机械臂控制指令发送失败")
        else:
            print("   ⚠️ 无法获取机械臂当前状态，跳过控制测试以确保硬件安全")


        # --- 2. 灵巧手增量控制 ---
        print("\n--- 测试灵巧手控制 ---")
        current_hand_state = robot.get_state("follower_left_hand")
        
        if current_hand_state and 'position' in current_hand_state:
            current_hand_pos = current_hand_state['position']
            # 安全增量：手指轻微变化 0.05
            safe_hand_target = [pos + 0.05 for pos in current_hand_pos]
            
            print(f"   当前手指位置: {[round(x, 3) for x in current_hand_pos[:6]]}...")
            print(f"   目标手指位置: {[round(x, 3) for x in safe_hand_target[:6]]}...")

            hand_command_data = {"position": safe_hand_target}
            success = robot.send_control("follower_left_hand_control", hand_command_data)
            if success:
                print("   ✅ 灵巧手控制指令已下发")
            else:
                print("   ❌ 灵巧手控制指令发送失败")
        else:
            print("   ⚠️ 无法获取灵巧手当前状态，跳过控制测试以确保硬件安全")

    except KeyboardInterrupt:
        print("\n🛑 测试被用户中断")
    finally:
        robot.stop()
        print("\n✅ 底层资源已安全释放")

if __name__ == "__main__":
    main()