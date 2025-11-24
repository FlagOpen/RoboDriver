import numpy as np
import torch

# 原始数值定义
right_value = 0.2
left_value = 0.4

# 创建初始numpy数组（显式指定 float32）
right_np_array = np.array([right_value], dtype=np.float32)
left_np_array = np.array([left_value], dtype=np.float32)

# 打印初始数组信息
print("=== 初始数组信息 ===")
print(
    f"right_np_array: {right_np_array} | 类型: {right_np_array.dtype} | 字节数: {right_np_array.nbytes}"
)
print(
    f"left_np_array:  {left_np_array} | 类型: {left_np_array.dtype} | 字节数: {left_np_array.nbytes}"
)

# 转换为字节流
r_buffer_bytes = right_np_array.tobytes()
l_buffer_bytes = left_np_array.tobytes()

# 打印字节流信息
print("\n=== 字节流信息 ===")
print(f"右夹爪字节流 (hex): {r_buffer_bytes.hex()}")
print(f"左夹爪字节流 (hex): {l_buffer_bytes.hex()}")

# 从字节流重建数组
right_gripper_array = np.frombuffer(r_buffer_bytes, dtype=np.float32)
left_gripper_array = np.frombuffer(l_buffer_bytes, dtype=np.float32)

# 打印重建数组信息
print("\n=== 重建数组信息 ===")
print(
    f"重建的右夹爪数组: {right_gripper_array} | 类型: {right_gripper_array.dtype} | 形状: {right_gripper_array.shape}"
)
print(
    f"重建的左夹爪数组: {left_gripper_array} | 类型: {left_gripper_array.dtype} | 形状: {left_gripper_array.shape}"
)

# 验证数值一致性
print("\n=== 数值验证 ===")
print(
    f"右值一致性: 原始={right_value:.6f} -> 重建={right_gripper_array[0]:.6f} | 相等: {np.isclose(right_value, right_gripper_array[0])}"
)
print(
    f"左值一致性: 原始={left_value:.6f} -> 重建={left_gripper_array[0]:.6f} | 相等: {np.isclose(left_value, left_gripper_array[0])}"
)

# 创建最终字典
recv_gripper = {
    "gripper_right": right_gripper_array,
    "gripper_left": left_gripper_array,
}

# 打印最终数据结构
print("\n=== recv_gripper 数据结构 ===")
for key, arr in recv_gripper.items():
    print(f"{key}: {arr} (类型: {type(arr)}, 形状: {arr.shape}, 数据类型: {arr.dtype})")


# ========================
# follower_arms 和 follower_gripper 映射处理
# ========================

follower_arms = {
    "right": 0,
    "left": 0,
}

follower_gripper = {}

print("\n=== 开始映射 recv_gripper -> follower_gripper ===")

# 注意：当前逻辑是每个 arm 都遍历所有 recv_gripper 键，这可能导致重复覆盖
# 实际上我们可能希望 'right' 匹配 'gripper_right', 'left' 匹配 'gripper_left'
# 下面按正确匹配方式改进 + 添加打印

# for name in follower_arms:
#     print(f"\n--- 处理 follower 机械臂: '{name}' ---")

#     # 构造匹配键名，例如 'right' -> 'gripper_right'
#     match_name = f"gripper_{name}"

#     if match_name not in recv_gripper:
#         print(f"  ⚠️ 警告: {match_name} 不存在于 recv_gripper 中！跳过...")
#         continue

#     gripper_read = recv_gripper[match_name]
#     print(f"  找到匹配数据: {match_name} = {gripper_read}")

#     # 创建目标数组并复制数据
#     byte_array = np.zeros(1, dtype=np.float32)
#     print(f"  初始化 byte_array: {byte_array}")

#     byte_array[:1] = gripper_read[:1]  # 复制第一个元素
#     print(f"  复制后 byte_array: {byte_array}")

#     # 四舍五入保留三位小数
#     byte_array = np.round(byte_array, 3)
#     print(f"  四舍五入后 byte_array: {byte_array}")

#     # 转换为 PyTorch Tensor
#     tensor_gripper = torch.from_numpy(byte_array)
#     follower_gripper[name] = tensor_gripper

#     print(f"  转换为 Torch Tensor: {tensor_gripper}")
#     print(f"  Tensor 属性: 值={tensor_gripper.item():.3f}, "
#           f"形状={tensor_gripper.shape}, "
#           f"数据类型={tensor_gripper.dtype}, "
#           f"设备={tensor_gripper.device}")

for name in follower_arms:
    for match_name in recv_gripper:
        if name in match_name:
            # now = time.perf_counter()

            byte_array = np.zeros(1, dtype=np.float32)
            gripper_read = recv_gripper[match_name]

            byte_array[:1] = gripper_read[:]
            byte_array = np.round(byte_array, 3)

            follower_gripper[name] = torch.from_numpy(byte_array)

            # self.logs[f"read_follower_{name}_gripper_dt_s"] = time.perf_counter() - now


# 最终输出
print("\n=== follower_gripper 最终结果 ===")
for key, tensor in follower_gripper.items():
    print(f"follower_gripper['{key}'] = {tensor} (值: {tensor.item():.3f})")

# 可选：检查是否在 GPU 上（目前是在 CPU）
print(
    f"\n提示: 当前张量位于 {follower_gripper['right'].device} 上。"
    f"如需迁移到 GPU，请使用 .to('cuda')"
)
