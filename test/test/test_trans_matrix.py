import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as R


def generate_random_pose():
    """生成随机的 6D 位姿（4x4 变换矩阵）"""
    euler_angles = np.random.uniform(-np.pi, np.pi, 3)
    rotation = R.from_euler("zyx", euler_angles).as_matrix()
    translation = np.random.uniform(-5, 5, 3)
    T = np.eye(4)
    T[:3, :3] = rotation
    T[:3, 3] = translation
    return T


def to_world_frame(T_world, T):
    return np.linalg.inv(T_world) @ T


def plot_transform(ax, T, axis_length=1.0, label=None):
    """绘制变换矩阵表示的坐标系，支持自定义轴线长度"""
    origin = T[:3, 3]
    x_axis = origin + T[:3, 0] * axis_length
    y_axis = origin + T[:3, 1] * axis_length
    z_axis = origin + T[:3, 2] * axis_length

    ax.plot(
        [origin[0], x_axis[0]],
        [origin[1], x_axis[1]],
        [origin[2], x_axis[2]],
        c="r",
        linewidth=2,
    )
    ax.plot(
        [origin[0], y_axis[0]],
        [origin[1], y_axis[1]],
        [origin[2], y_axis[2]],
        c="g",
        linewidth=2,
    )
    ax.plot(
        [origin[0], z_axis[0]],
        [origin[1], z_axis[1]],
        [origin[2], z_axis[2]],
        c="b",
        linewidth=2,
    )

    if label:
        ax.text(origin[0], origin[1], origin[2], label, color="k")


def plot_trajectory(ax, poses, color="gray", label="trajectory"):
    points = np.array([T[:3, 3] for T in poses])
    ax.plot(points[:, 0], points[:, 1], points[:, 2], color=color, label=label)


# 生成位姿
num_frames = 2
poses = [generate_random_pose() for _ in range(num_frames)]
T_world = poses[0]
world_poses = [to_world_frame(T_world, T) for T in poses]

# 创建图形
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")

# 绘制基座标系（世界坐标系）
base_T = np.eye(4)  # 原点处的单位变换矩阵
plot_transform(ax, base_T, axis_length=5.0, label="Base Frame")

# 绘制所有其他坐标系
for i, T in enumerate(world_poses, 1):
    plot_transform(ax, T, label=f"Frame {i}")

# 绘制轨迹
plot_trajectory(ax, world_poses)

# 设置图形属性
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("6D Visual")
ax.set_box_aspect([1, 1, 1])
ax.legend()

space_size = 10
ax.set_xlim(-space_size, space_size)
ax.set_ylim(-space_size, space_size)
ax.set_zlim(-space_size, space_size)

plt.tight_layout()
plt.show()
