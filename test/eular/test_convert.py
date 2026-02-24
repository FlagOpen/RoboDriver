import numpy as np
from scipy.spatial.transform import Rotation as R

def convert_pose_A_to_B(pose_in_A, input_order='XYZ', output_order='XYZ', degrees=False):
    """
    将姿态从坐标系A转换到坐标系B
    
    参数:
    pose_in_A: 在A坐标系中的姿态（旋转对象或欧拉角）
    input_order: 输入姿态的欧拉角顺序
    output_order: 输出姿态的欧拉角顺序
    
    返回:
    在B坐标系中的姿态
    """
    # A到B的变换矩阵
    R_A_to_B = np.array([
        [0, 0, -1],
        [0, 1, 0],
        [1, 0, 0]
    ])
    
    # 如果是欧拉角列表，转换为旋转对象
    if isinstance(pose_in_A, (list, tuple, np.ndarray)):
        if len(pose_in_A) == 3:
            pose_in_A = R.from_euler(input_order, pose_in_A, degrees=degrees)
        else:
            raise ValueError("pose_in_A应为3个欧拉角")
    
    # 获取物体在A中的旋转矩阵
    R_pose_A = pose_in_A.as_matrix()
    
    # 转换到B中：R_pose_B = R_A_to_B @ R_pose_A
    R_pose_B = R_A_to_B @ R_pose_A
    # R_pose_B = R_pose_A
    
    # 创建旋转对象
    pose_in_B = R.from_matrix(R_pose_B)
    
    # 如果需要欧拉角输出
    if output_order:
        return pose_in_B.as_euler(output_order, degrees=degrees)
    return pose_in_B

def convert_pose_B_to_A(pose_in_B, input_order='XYZ', output_order='XYZ', degrees=False):
    """
    将姿态从坐标系B转换回坐标系A
    
    参数:
    pose_in_B: 在B坐标系中的姿态（旋转对象或欧拉角）
    input_order: 输入姿态的欧拉角顺序
    output_order: 输出姿态的欧拉角顺序
    degrees: 角度是否为度数
    
    返回:
    在A坐标系中的姿态
    """
    # A到B的变换矩阵
    R_A_to_B = np.array([
        [0, 0, -1],
        [0, 1, 0],
        [1, 0, 0]
    ])
    
    # B到A的变换矩阵（R_A_to_B的逆，对于旋转矩阵就是转置）
    R_B_to_A = R_A_to_B.T  # 或者使用 np.linalg.inv(R_A_to_B)
    
    # 如果是欧拉角列表，转换为旋转对象
    if isinstance(pose_in_B, (list, tuple, np.ndarray)):
        if len(pose_in_B) == 3:
            pose_in_B = R.from_euler(input_order, pose_in_B, degrees=degrees)
        else:
            raise ValueError("pose_in_B应为3个欧拉角")
    
    # 获取物体在B中的旋转矩阵
    R_pose_B = pose_in_B.as_matrix()
    
    # 转换回A中：R_pose_A = R_B_to_A @ R_pose_B
    R_pose_A = R_B_to_A @ R_pose_B
    
    # 创建旋转对象
    pose_in_A = R.from_matrix(R_pose_A)
    
    # 如果需要欧拉角输出
    if output_order:
        return pose_in_A.as_euler(output_order, degrees=degrees)
    return pose_in_A

# 测试 AtoB
test_pose = [0, 90, 20]
result = convert_pose_A_to_B(test_pose, input_order='XYZ', output_order='XYZ', degrees=True)
print(f"AtoB 输入(XYZ顺序): {test_pose}°")
print(f"AtoB 输出(XYZ顺序): [{result[0]:.1f}, {result[1]:.1f}, {result[2]:.1f}]°")

# 测试 BtoA
test_pose = [0, 0, 20]
result = convert_pose_B_to_A(test_pose, input_order='XYZ', output_order='XYZ', degrees=True)
print(f"BtoA 输入(XYZ顺序): {test_pose}°")
print(f"BtoA 输出(XYZ顺序): [{result[0]:.1f}, {result[1]:.1f}, {result[2]:.1f}]°")
