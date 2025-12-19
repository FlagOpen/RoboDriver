#!/usr/bin/env python3
"""
测试 action_joint 处理功能
"""
import numpy as np
import sys
import os

# 添加src目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from main import FrankaSimulation

def test_handle_action_joint():
    """测试 handle_action_joint 方法"""
    print("测试 handle_action_joint 方法...")
    
    try:
        # 创建仿真实例
        simulation = FrankaSimulation("config/simulation_config.yaml")
        
        # 创建一个模拟的关节位置数组（9个关节）
        # 使用合理的关节位置值
        test_joint_positions = np.array([
            0.0,  # joint1
            -0.5, # joint2
            0.0,  # joint3
            -1.5, # joint4
            0.0,  # joint5
            1.0,  # joint6
            0.5,  # joint7
            0.0,  # gripper_joint1 (夹爪)
            0.0   # gripper_joint2 (夹爪)
        ], dtype=np.float32)
        
        print(f"测试关节位置: {test_joint_positions}")
        
        # 测试数据验证
        print("1. 测试数据验证...")
        simulation.handle_action_joint(test_joint_positions)
        print("   数据验证通过")
        
        # 测试无效数据
        print("\n2. 测试无效数据...")
        simulation.handle_action_joint(None)
        
        # 测试错误长度的数据
        print("\n3. 测试错误长度的数据...")
        simulation.handle_action_joint(np.array([1.0, 2.0, 3.0]))
        
        print("\n测试完成！")
        
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = test_handle_action_joint()
    if success:
        print("\n✅ 所有测试通过！")
        sys.exit(0)
    else:
        print("\n❌ 测试失败！")
        sys.exit(1)
