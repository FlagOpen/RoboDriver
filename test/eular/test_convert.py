import numpy as np
from scipy.spatial.transform import Rotation as R

# 定义常量
R_A_TO_B = np.array([
    [0, 0, -1],
    [0, 1, 0],
    [1, 0, 0]
])
R_B_TO_A = R_A_TO_B.T

def convert_pose(pose, direction='A_to_B', input_order='XYZ', output_order='XYZ', degrees=False):
    """
    在坐标系A和B之间转换姿态
    
    参数:
    pose: 输入的姿态（旋转对象或欧拉角）
    direction: 转换方向，'A_to_B' 或 'B_to_A'
    input_order: 输入姿态的欧拉角顺序
    output_order: 输出姿态的欧拉角顺序
    degrees: 角度是否为度数
    
    返回:
    转换后的姿态
    """
    # 选择变换矩阵
    transform = R_A_TO_B if direction == 'A_to_B' else R_B_TO_A
    
    # 转换为旋转对象
    if isinstance(pose, (list, tuple, np.ndarray)):
        if len(pose) != 3:
            raise ValueError("pose应为3个欧拉角")
        rot = R.from_euler(input_order, pose, degrees=degrees)
    else:
        rot = pose
    
    # 应用变换并转换结果
    result_rot = R.from_matrix(transform @ rot.as_matrix())
    
    # 返回所需格式
    return result_rot if not output_order else result_rot.as_euler(output_order, degrees=degrees)

# 使用示例
test_pose = [90, 70, 90]
result = convert_pose(test_pose, 'A_to_B', 'ZYX', 'ZYX', degrees=True)
print(f"AtoB 输入(ZYX顺序): {test_pose}°")
print(f"AtoB 输出(ZYX顺序): [{result[0]:.1f}, {result[1]:.1f}, {result[2]:.1f}]°")

test_pose = [20, 0, 0]
result = convert_pose(test_pose, 'B_to_A', 'ZYX', 'ZYX', degrees=True)
print(f"BtoA 输入(ZYX顺序): {test_pose}°")
print(f"BtoA 输出(ZYX顺序): [{result[0]:.1f}, {result[1]:.1f}, {result[2]:.1f}]°")
