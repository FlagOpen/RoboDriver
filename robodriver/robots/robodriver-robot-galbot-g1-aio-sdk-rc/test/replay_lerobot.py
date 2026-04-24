#!/usr/bin/env python3


"""Replay LeRobot dataset trajectories on Galbot robot.

This module provides functionality to replay robot trajectories stored in LeRobot
dataset format (parquet files) on a Galbot humanoid robot. It handles parsing
action data, converting it to robot joint commands, and executing trajectories.

The module supports:
- Reading LeRobot parquet dataset files
- Parsing action arrays into joint groups (legs, arms, grippers, head)
- Converting actions to robot trajectory points
- Executing trajectories on the Galbot robot
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from typing import Any

from galbot_sdk.g1 import ControlStatus, GalbotRobot, JointCommand, Trajectory, TrajectoryPoint
import numpy as np
import pandas as pd
import math


class FourOmniWheelKinematics:
    """
    四轮万向轮底盘运动学
    用于从四个轮子的速度计算机器人本体的运动速度
    
    轮子顺序固定为：
        0: 左前 (Front-Left, FL)
        1: 右前 (Front-Right, FR)
        2: 右后 (Rear-Right, RR)
        3: 左后 (Rear-Left, RL)
    """

    def __init__(self, layout_type='x45', L=0.2, W=0.2):
        """
        初始化运动学参数
        
        Args:
            layout_type: 布局类型
                - 'x45': X型45°布局（最常见）
                - 'plus': +型布局（前后左右）
                - 'custom': 自定义布局（需后续手动设置 wheel_directions）
            L: 底盘长度的一半（前后方向，单位：米）
            W: 底盘宽度的一半（左右方向，单位：米）
        """
        self.L = L  # 前后方向半长（从中心到前/后轮轴）
        self.W = W  # 左右方向半宽（从中心到左/右轮轴）
        self.layout_type = layout_type

        # 轮子位置（机器人坐标系：X向前，Y向左）
        # 顺序：[左前, 右前, 右后, 左后]
        self.wheel_positions = np.array([
            [ L,  W],   # 0: 左前 (FL)
            [ L, -W],   # 1: 右前 (FR)
            [-L, -W],   # 2: 右后 (RR)
            [-L,  W]    # 3: 左后 (RL)
        ])

        # 根据布局类型设置轮子滚动方向（单位向量，指向自由滚动方向）
        if layout_type == 'x45':
            s = np.sqrt(2) / 2
            self.wheel_directions = np.array([
                [-s,  s],   # 左前:（X负，Y正）
                [ s,  s],   # 右前:（X正，Y正）
                [ s, -s],   # 右后:（X正，Y负）
                [-s, -s]    # 左后:（X负，Y负）
            ])
        elif layout_type == 'plus':
            self.wheel_directions = np.array([
                [1, 0],     # 左前: 向前滚动（+X）
                [0, -1],    # 右前: 向左滚动（-Y）
                [-1, 0],    # 右后: 向后滚动（-X）
                [0, 1]      # 左后: 向右滚动（+Y）
            ])
        elif layout_type == 'custom':
            self.wheel_directions = None  # 需外部设置
        else:
            raise ValueError(f"未知的布局类型: {layout_type}")

        # 计算运动学雅可比矩阵 J 和其伪逆
        self.J = self._calculate_jacobian()
        self.J_pinv = np.linalg.pinv(self.J)

    def _calculate_jacobian(self):
        """计算运动学雅可比矩阵 J，满足 v_wheel = J @ [vx, vy, omega]^T"""
        if self.wheel_directions is None:
            directions = np.zeros((4, 2))
        else:
            directions = self.wheel_directions

        J = np.zeros((4, 3))
        for i in range(4):
            rx, ry = self.wheel_positions[i]
            dx, dy = directions[i]
            J[i, 0] = dx                    # vx 分量
            J[i, 1] = dy                    # vy 分量
            J[i, 2] = dy * rx - dx * ry     # omega 分量（绕Z轴）
        return J

    def wheels_to_chassis(self, wheel_speeds):
        """
        正运动学：由轮速计算底盘速度
        
        Args:
            wheel_speeds: [v_fl, v_fr, v_rr, v_rl]
                v_fl: 左前轮速度
                v_fr: 右前轮速度
                v_rr: 右后轮速度
                v_rl: 左后轮速度
                （正值表示沿 wheel_directions 定义的方向运动）
        
        Returns:
            [vx, vy, omega]: 机器人在本体坐标系下的速度
                vx: 前进速度 (m/s)
                vy: 横向速度 (m/s)，向左为正
                omega: 角速度 (rad/s)，逆时针为正
        """
        wheel_speeds = np.asarray(wheel_speeds).flatten()
        if wheel_speeds.size != 4:
            raise ValueError("wheel_speeds 必须包含4个元素：[左前, 右前, 右后, 左后]")
        return self.J_pinv @ wheel_speeds

    def chassis_to_wheels(self, vx, vy, omega):
        """
        逆运动学：由底盘速度计算所需轮速
        
        Args:
            vx: 前进速度 (m/s)
            vy: 横向速度 (m/s)
            omega: 角速度 (rad/s)
        
        Returns:
            [v_fl, v_fr, v_rr, v_rl]: 四个轮子的目标速度
        """
        chassis_vel = np.array([vx, vy, omega])
        return self.J @ chassis_vel

    def set_custom_directions(self, directions):
        """
        为 'custom' 布局设置轮子方向（必须按 [FL, FR, RR, RL] 顺序）
        
        Args:
            directions: shape (4, 2) 的数组，每行为 [dx, dy] 单位向量
        """
        directions = np.asarray(directions)
        if directions.shape != (4, 2):
            raise ValueError("directions 必须是 (4, 2) 形状的数组")
        self.wheel_directions = directions
        self.J = self._calculate_jacobian()
        self.J_pinv = np.linalg.pinv(self.J)

    def get_condition_number(self):
        """返回运动学矩阵 J 的条件数（越小越好）"""
        return np.linalg.cond(self.J)

    def print_configuration(self):
        """打印当前运动学配置"""
        print("四轮万向轮运动学配置")
        print(f"布局类型: {self.layout_type}")
        print(f"底盘尺寸: L={self.L:.3f} m, W={self.W:.3f} m")
        print("\n轮子顺序: [0]左前(FL), [1]右前(FR), [2]右后(RR), [3]左后(RL)")
        print("\n轮子位置 (X向前, Y向左):")
        names = ["左前", "右前", "右后", "左后"]
        for i in range(4):
            x, y = self.wheel_positions[i]
            print(f"  [{i}] {names[i]}: ({x:+.3f}, {y:+.3f}) m")

        if self.wheel_directions is not None:
            print("\n轮子滚动方向 (单位向量):")
            for i in range(4):
                dx, dy = self.wheel_directions[i]
                print(f"  [{i}] {names[i]}: [{dx:+.3f}, {dy:+.3f}]")
        else:
            print("\n轮子滚动方向: 未设置 (custom 模式)")

        print("\n运动学矩阵 J (4x3):")
        print(self.J)
        print(f"\n条件数: {self.get_condition_number():.3f}")

        det_val = np.linalg.det(self.J.T @ self.J)
        print(f"JᵀJ 行列式: {det_val:.6e}")
        if abs(det_val) < 1e-10:
            print("⚠️ 警告: JᵀJ 接近奇异，运动学可能不稳定！")


JOINT_GROUP_ORDER = [
    "leg",
    "right_arm",
    "right_gripper",
    "left_arm",
    "left_gripper",
    "head",
]
DEFAULT_PREPARE_DELTA_THRESHOLD_RAD = 0.001
DEFAULT_GRIPPER_FORMAT = "scaled"
GRIPPER_RAW_VALUE_THRESHOLD = 5.0
PREPARE_TIMEOUT_MARGIN_S = 5.0
DEFAULT_GRIPPER_FORCE_N = 10.0
GRIPPER_VALUE_TO_WIDTH_M = 0.001
GRIPPER_MIN_WIDTH_M = 0.001
GRIPPER_MAX_WIDTH_M = 0.12
GRIPPER_RAW_MAX_VALUE = 100.0
MIN_COMMAND_DT_S = 0.01
REPLAY_TAIL_WAIT_S = 1.0
DATA_FPS = 30.0

last_odom_pose_position_x = 0.0
last_odom_pose_position_y = 0.0
last_odom_pose_orientation_z = 0.0

def to_list(x: Any) -> list[Any]:
    """Convert action cell in parquet to Python list.

    This function handles various data types that might be stored in parquet
    action columns, including lists, numpy arrays, and JSON strings.

    Args:
        x: The action data to convert. Can be a list, numpy array, or JSON string.

    Returns:
        A Python list representation of the input data.

    Raises:
        TypeError: If the input type is not supported (not a list, numpy array,
            or JSON string).
    """
    if isinstance(x, list):
        return x
    if hasattr(x, "tolist"):
        return x.tolist()
    if isinstance(x, str):
        return json.loads(x)
    raise TypeError(f"Unsupported action type: {type(x)}")


def split_action(a: list[float] | np.ndarray, action_names: list[str] | None = None, use_diff_odom: bool = True) -> dict[str, list[float] | float]:
    """Split action array into parts based on the Galbot dataset structure.

    The action array is structured as:
    [leg(5), right_arm(7), right_gripper(1), left_arm(7), left_gripper(1), head(2), ...]

    This matches the flatten_scaled_device_robot_data function in
    galbot_mcap_to_lerobot.py. The gripper values are already scaled by 0.01
    in the dataset.

    Full 38-dimensional structure:
    [leg(5), right_arm(7), right_gripper(1), left_arm(7), left_gripper(1), head(2),
     chassis_pos(4), chassis_vel(4), odom_pose(4), odom_twist(3)]

    Args:
        a: Action array to split. Can be a list or numpy array. Expected length
            is at least 23 elements (5+7+1+7+1+2).
        action_names: Optional list of action names for backward compatibility.
            If provided and matches the action length, uses name-based parsing.
            Otherwise, uses positional parsing based on default structure.

    Returns:
        A dictionary containing the split action parts:
        - "leg": List of 5 leg joint values
        - "right_arm": List of 7 right arm joint values
        - "right_gripper": Single float value for right gripper
        - "left_arm": List of 7 left arm joint values
        - "left_gripper": Single float value for left gripper
        - "head": List of 2 head joint values
        - "odom_twist_linear": List of 2 values [linear_x, linear_y]
        - "odom_twist_angular": List of 1 value [angular_z]

    Raises:
        ValueError: If the action array is too short (less than 23 elements).
    """
    # Convert to numpy array for consistent indexing
    a = np.array(a) if not isinstance(a, np.ndarray) else a

    # Default structure: leg(5) + right_arm(7) + right_gripper(1) +
    # left_arm(7) + left_gripper(1) + head(2) = 23
    # The gripper values are already scaled by 0.01 in the dataset
    leg_joints = 5
    right_arm_joints = 7
    right_gripper_joints = 1
    left_arm_joints = 7
    left_gripper_joints = 1
    head_joints = 2

    ODOM_POSE_POSITION_LEN = 2
    ODOM_POSE_ORIENTATION_LEN = 2
    ODOM_TWIST_LINEAR_LEN = 2
    ODOM_TWIST_ANGULAR_LEN = 1

    CHASSIS_VEL_START = 27
    CHASSIS_VEL_LEN = 4
    ODOM_POSE_START = 31
    ODOM_TWIST_START = 35

    # If action_names is provided, try to use semantic parsing first so replay
    # can tolerate dataset-specific layouts and extra non-joint signals.
    if action_names is not None and len(action_names) == len(a):
        normalized_names = [name.lower() for name in action_names]

        def extract_joint_index(name: str) -> int | None:
            match = re.search(r"joint(\d+)", name)
            if match is None:
                return None
            return int(match.group(1))

        def collect_joint_indices(*required_tokens: str) -> list[int]:
            indices: list[tuple[int, int]] = []
            for i, name in enumerate(normalized_names):
                if all(token in name for token in required_tokens):
                    joint_idx = extract_joint_index(name)
                    if joint_idx is not None:
                        indices.append((joint_idx, i))
            return [i for _, i in sorted(indices)]

        def find_gripper_index(side: str) -> int | None:
            for i, name in enumerate(normalized_names):
                if side in name and "gripper" in name:
                    return i
            return None

        leg_indices = collect_joint_indices("leg", "joint")
        right_arm_indices = collect_joint_indices("right", "arm", "joint")
        left_arm_indices = collect_joint_indices("left", "arm", "joint")
        head_indices = collect_joint_indices("head", "joint")
        right_gripper_idx = find_gripper_index("right")
        left_gripper_idx = find_gripper_index("left")

        odom_twist_linear = []
        odom_twist_angular = []
        if use_diff_odom:
            # last_odom_pose_position_x = 0.0
            # last_odom_pose_position_y = 0.0
            # last_odom_pose_orientation_z = 0.0
            now_odom_pose_position_x = float(a[ODOM_POSE_START])
            now_odom_pose_position_y = float(a[ODOM_POSE_START+1])

            now_odom_pose_orientation_z = float(a[ODOM_POSE_START+2])
            now_odom_pose_orientation_w = float(a[ODOM_POSE_START+3])

            # 计算 Z 轴旋转角（Yaw），单位为弧度
            now_odom_pose_orientation_z = 2 * math.atan2(now_odom_pose_orientation_z, now_odom_pose_orientation_w)

            d_odom_pose_position_x = (now_odom_pose_position_x - last_odom_pose_position_x) * DATA_FPS
            d_odom_pose_position_y = (now_odom_pose_position_y - last_odom_pose_position_y) * DATA_FPS
            d_odom_pose_orientation_z = (now_odom_pose_orientation_z - last_odom_pose_orientation_z) * DATA_FPS

            last_odom_pose_position_x = now_odom_pose_position_x
            last_odom_pose_position_y = now_odom_pose_position_y
            last_odom_pose_orientation_z = now_odom_pose_orientation_z

            odom_twist_linear = [d_odom_pose_position_x, d_odom_pose_position_y]
            odom_twist_angular = [d_odom_pose_orientation_z]
        else:
            if len(a) >= ODOM_TWIST_START + ODOM_TWIST_LINEAR_LEN + ODOM_TWIST_ANGULAR_LEN:
                odom_twist_linear = a[ODOM_TWIST_START : ODOM_TWIST_START + ODOM_TWIST_LINEAR_LEN].tolist()
                odom_twist_angular = [float(a[ODOM_TWIST_START + ODOM_TWIST_LINEAR_LEN])]

        # If we found the main groups, use name-based parsing and include
        # chassis_vel (wheel speeds) for chassis control
        if right_arm_indices and left_arm_indices:
            result = {
                "leg": [a[i] for i in leg_indices] if leg_indices else [],
                "right_arm": [a[i] for i in right_arm_indices],
                "right_gripper": (a[right_gripper_idx] if right_gripper_idx is not None else 0.0),
                "left_arm": [a[i] for i in left_arm_indices],
                "left_gripper": (a[left_gripper_idx] if left_gripper_idx is not None else 0.0),
                "head": [a[i] for i in head_indices] if head_indices else [],
            }

            # Extract chassis_vel (wheel speeds) for chassis control
            if len(a) >= CHASSIS_VEL_START + CHASSIS_VEL_LEN:
                result["chassis_vel"] = a[CHASSIS_VEL_START : CHASSIS_VEL_START + CHASSIS_VEL_LEN].tolist()

            # Only add odom_twist if data is available
            if odom_twist_linear and odom_twist_angular:
                result["odom_twist_linear"] = odom_twist_linear
                result["odom_twist_angular"] = odom_twist_angular

        return result
    else:
        raise ValueError(f"Action length {len(a)} error, \"action_names is not None and len(action_names) == len(a)\"")


    # # Default parsing for galbot dataset structure
    # # Structure: [leg(5), right_arm(7), right_gripper(1), left_arm(7),
    # #            left_gripper(1), head(2), ...]
    # expected_min_length = (
    #     leg_joints + right_arm_joints + right_gripper_joints + left_arm_joints + left_gripper_joints + head_joints + ODOM_TWIST_LINEAR_LEN + ODOM_TWIST_ANGULAR_LEN
    # )
    # if len(a) < expected_min_length:
    #     raise ValueError(f"Action length {len(a)} is too short. Expected at least {expected_min_length} elements")

    # # Parse action array positionally
    # idx = 0
    # leg = a[idx : idx + leg_joints].tolist()
    # idx += leg_joints
    # right_arm = a[idx : idx + right_arm_joints].tolist()
    # idx += right_arm_joints
    # right_gripper = float(a[idx]) if len(a) > idx else 0.0
    # idx += right_gripper_joints
    # left_arm = a[idx : idx + left_arm_joints].tolist()
    # idx += left_arm_joints
    # left_gripper = float(a[idx]) if len(a) > idx else 0.0
    # idx += left_gripper_joints
    # head = a[idx : idx + head_joints].tolist()

    # # Parse odom twist (indices 35-38)
    # # Full structure: leg(5) + right_arm(7) + right_gripper(1) + left_arm(7) +
    # #                 left_gripper(1) + head(2) = 23
    # #                 chassis_pos(4) = 27
    # #                 chassis_vel(4) = 31
    # #                 odom_pose(4) = 35
    # #                 odom_twist(3) = 38
    # ODOM_TWIST_START = 35

    # odom_twist_linear = []
    # odom_twist_angular = []
    # if len(a) >= ODOM_TWIST_START + ODOM_TWIST_LINEAR_LEN + ODOM_TWIST_ANGULAR_LEN:
    #     odom_twist_linear = a[ODOM_TWIST_START : ODOM_TWIST_START + ODOM_TWIST_LINEAR_LEN].tolist()
    #     odom_twist_angular = [float(a[ODOM_TWIST_START + ODOM_TWIST_LINEAR_LEN])]

    # result = {
    #     "leg": leg,
    #     "right_arm": right_arm,
    #     "right_gripper": right_gripper,
    #     "left_arm": left_arm,
    #     "left_gripper": left_gripper,
    #     "head": head,
    # }

    # # Only add odom_twist if data is available
    # if odom_twist_linear and odom_twist_angular:
    #     result["odom_twist_linear"] = odom_twist_linear
    #     result["odom_twist_angular"] = odom_twist_angular

    # return result


def infer_gripper_format(
    parts_seq: list[dict[str, list[float] | float]],
    raw_value_threshold: float = GRIPPER_RAW_VALUE_THRESHOLD,
) -> tuple[str, float]:
    """Infer whether dataset gripper values are already in robot units."""
    max_abs_sample = 0.0
    for parts in parts_seq:
        for key in ("right_gripper", "left_gripper"):
            try:
                value = abs(float(parts.get(key, 0.0)))
            except (TypeError, ValueError):
                continue
            max_abs_sample = max(max_abs_sample, value)

    if max_abs_sample > raw_value_threshold:
        return "raw", max_abs_sample
    return "scaled", max_abs_sample


def resolve_gripper_format(
    requested_format: str,
    parts_seq: list[dict[str, list[float] | float]],
) -> str:
    """Resolve the gripper format, inferring it from sampled frames when needed."""
    if requested_format not in {"auto", "scaled", "raw"}:
        raise ValueError(f"Unsupported gripper format: {requested_format}")

    if requested_format != "auto":
        print(f"[gripper] Using user-specified gripper format: {requested_format}")
        return requested_format

    inferred_format, max_abs_sample = infer_gripper_format(parts_seq)
    print(
        f"[gripper] Auto-detected gripper format: {inferred_format} "
        f"(sample max abs value={max_abs_sample:.4f}, raw threshold={GRIPPER_RAW_VALUE_THRESHOLD:.4f})"
    )
    return inferred_format


def decode_gripper_position(
    value: float,
    gripper_scale: float,
    gripper_offset: float,
    gripper_format: str,
) -> float:
    """Convert dataset gripper value into SDK joint position."""
    value = float(value)
    if gripper_format == "raw":
        return value
    if gripper_format == "scaled":
        return (value + gripper_offset) * gripper_scale
    raise ValueError(f"Unsupported gripper format: {gripper_format}")


def decode_gripper_width_m(
    value: float,
    gripper_scale: float,
    gripper_offset: float,
    gripper_format: str,
) -> float:
    """Convert dataset gripper value into gripper width in meters [0.01, 0.1]."""
    decoded_value = decode_gripper_position(value, gripper_scale, gripper_offset, gripper_format)
    clipped_value = min(max(float(decoded_value), 0.0), GRIPPER_RAW_MAX_VALUE)
    ratio = clipped_value / GRIPPER_RAW_MAX_VALUE
    width_m = GRIPPER_MIN_WIDTH_M + ratio * (GRIPPER_MAX_WIDTH_M - GRIPPER_MIN_WIDTH_M)
    return float(min(max(width_m, GRIPPER_MIN_WIDTH_M), GRIPPER_MAX_WIDTH_M))


def command_grippers(
    robot: GalbotRobot,
    parts: dict[str, list[float] | float],
    gripper_scale: float,
    gripper_offset: float,
    gripper_format: str,
    gripper_speed: float | None,
) -> None:
    """Command left/right grippers through set_joint_commands."""
    gripper_speed_mps = 0.03 if gripper_speed is None else float(gripper_speed) * GRIPPER_VALUE_TO_WIDTH_M
    gripper_speed_mps = max(gripper_speed_mps, 1e-4)
    joint_commands = [JointCommand(), JointCommand()]
    joint_commands[0].position = decode_gripper_width_m(
        parts.get("right_gripper", 0.0), gripper_scale, gripper_offset, gripper_format
    )
    joint_commands[0].velocity = gripper_speed_mps
    joint_commands[0].effort = DEFAULT_GRIPPER_FORCE_N
    joint_commands[1].position = decode_gripper_width_m(
        parts.get("left_gripper", 0.0), gripper_scale, gripper_offset, gripper_format
    )
    joint_commands[1].velocity = gripper_speed_mps
    joint_commands[1].effort = DEFAULT_GRIPPER_FORCE_N

    status = robot.set_joint_commands(
        joint_commands,
        ["right_gripper", "left_gripper"],
        [],
        0.0,
    )
    if status != ControlStatus.SUCCESS:
        raise RuntimeError(
            "Failed to command grippers via set_joint_commands. "
            f"right_width_m={joint_commands[0].position:.6f}, "
            f"left_width_m={joint_commands[1].position:.6f}, "
            f"speed_mps={gripper_speed_mps:.6f}, status={status}"
        )


def build_replay_joint_commands(
    parts: dict[str, list[float] | float],
    gripper_scale: float,
    gripper_offset: float,
    gripper_format: str,
    gripper_speed: float | None,
) -> list[JointCommand]:
    """Build full-body JointCommand list for replay."""
    commands: list[JointCommand] = []
    gripper_speed_mps = 0.03 if gripper_speed is None else float(gripper_speed) * GRIPPER_VALUE_TO_WIDTH_M
    gripper_speed_mps = max(gripper_speed_mps, 1e-4)

    for value in parts.get("leg", []):
        cmd = JointCommand()
        cmd.position = float(value)
        commands.append(cmd)

    for value in parts.get("right_arm", []):
        cmd = JointCommand()
        cmd.position = float(value)
        commands.append(cmd)

    right_gripper_cmd = JointCommand()
    right_gripper_cmd.position = decode_gripper_width_m(
        parts.get("right_gripper", 0.0), gripper_scale, gripper_offset, gripper_format
    )
    right_gripper_cmd.velocity = gripper_speed_mps
    right_gripper_cmd.effort = DEFAULT_GRIPPER_FORCE_N
    commands.append(right_gripper_cmd)

    for value in parts.get("left_arm", []):
        cmd = JointCommand()
        cmd.position = float(value)
        commands.append(cmd)

    left_gripper_cmd = JointCommand()
    left_gripper_cmd.position = decode_gripper_width_m(
        parts.get("left_gripper", 0.0), gripper_scale, gripper_offset, gripper_format
    )
    left_gripper_cmd.velocity = gripper_speed_mps
    left_gripper_cmd.effort = DEFAULT_GRIPPER_FORCE_N
    commands.append(left_gripper_cmd)

    for value in parts.get("head", []):
        cmd = JointCommand()
        cmd.position = float(value)
        commands.append(cmd)

    return commands


def build_replay_trajectory_point(
    parts: dict[str, list[float] | float],
    time_from_start_second: float,
    gripper_scale: float,
    gripper_offset: float,
    gripper_format: str,
    gripper_speed: float | None,
) -> TrajectoryPoint:
    """Build a full-body trajectory point in JOINT_GROUP_ORDER."""
    point = TrajectoryPoint()
    point.time_from_start_second = time_from_start_second
    point.joint_command_vec = build_replay_joint_commands(
        parts,
        gripper_scale,
        gripper_offset,
        gripper_format,
        gripper_speed,
    )
    return point


def get_joint_positions_from_parts(
    parts: dict[str, list[float] | float],
    gripper_scale: float,
    gripper_offset: float,
    gripper_format: str,
) -> list[float]:
    """Convert parsed action parts into SDK joint position vector."""
    joint_pos_vec: list[float] = []
    joint_pos_vec.extend(parts.get("leg", []))
    joint_pos_vec.extend(parts.get("right_arm", []))
    joint_pos_vec.append(
        decode_gripper_position(parts.get("right_gripper", 0.0), gripper_scale, gripper_offset, gripper_format)
    )
    joint_pos_vec.extend(parts.get("left_arm", []))
    joint_pos_vec.append(
        decode_gripper_position(parts.get("left_gripper", 0.0), gripper_scale, gripper_offset, gripper_format)
    )
    joint_pos_vec.extend(parts.get("head", []))
    return joint_pos_vec


def get_stage_positions_from_parts(
    parts: dict[str, list[float] | float],
    joint_groups: list[str],
    gripper_scale: float,
    gripper_offset: float,
    gripper_format: str,
) -> list[float]:
    """Build SDK joint target vector for a subset of joint groups."""
    stage_positions: list[float] = []
    for group in joint_groups:
        if group == "leg":
            stage_positions.extend(parts.get("leg", []))
        elif group == "right_arm":
            stage_positions.extend(parts.get("right_arm", []))
        elif group == "left_arm":
            stage_positions.extend(parts.get("left_arm", []))
        elif group == "head":
            stage_positions.extend(parts.get("head", []))
        elif group == "right_gripper":
            stage_positions.append(
                decode_gripper_position(parts.get("right_gripper", 0.0), gripper_scale, gripper_offset, gripper_format)
            )
        elif group == "left_gripper":
            stage_positions.append(
                decode_gripper_position(parts.get("left_gripper", 0.0), gripper_scale, gripper_offset, gripper_format)
            )
        else:
            raise ValueError(f"Unsupported joint group for stage prepare: {group}")
    return stage_positions


def compute_stage_timeout_s(
    current_positions: list[float],
    target_positions: list[float],
    speed_rad_s: float,
    base_timeout_s: float,
) -> tuple[float, float, float]:
    """Compute delta summary and timeout that can accommodate the slowest joint."""
    if len(current_positions) != len(target_positions):
        return base_timeout_s, float("nan"), float("nan")

    diffs = [abs(c - t) for c, t in zip(current_positions, target_positions)]
    max_delta = max(diffs) if diffs else 0.0
    mean_delta = (sum(diffs) / len(diffs)) if diffs else 0.0

    if speed_rad_s <= 0:
        return base_timeout_s, max_delta, mean_delta

    required_timeout_s = (max_delta / speed_rad_s) + PREPARE_TIMEOUT_MARGIN_S
    return max(base_timeout_s, required_timeout_s), max_delta, mean_delta


def replay_with_execute_joint_trajectory(
    robot: GalbotRobot,
    parsed_frames: list[tuple[float, dict[str, list[float] | float]]],
    gripper_scale: float,
    gripper_offset: float,
    gripper_format: str,
    speed_scale: float,
    gripper_speed: float | None,
) -> None:
    """Replay full-body trajectory with execute_joint_trajectory and explicit joint_names."""
    if not parsed_frames:
        raise ValueError("Replay trajectory is empty.")

    expanded_joint_names: list[str] = []
    for group_name in JOINT_GROUP_ORDER:
        expanded_joint_names.extend(robot.get_joint_names(True, [group_name]))
    if not expanded_joint_names:
        raise RuntimeError("Failed to resolve joint_names from JOINT_GROUP_ORDER")

    traj = Trajectory()
    traj.joint_groups = []
    traj.joint_names = expanded_joint_names

    cumulative_time_s = 0.0
    points: list[TrajectoryPoint] = []
    for idx, (timestamp, parts) in enumerate(parsed_frames):
        if idx == 0:
            point_time_s = 0.0
        else:
            prev_timestamp = parsed_frames[idx - 1][0]
            raw_dt_s = max(timestamp - prev_timestamp, 0.0)
            scaled_dt_s = raw_dt_s / speed_scale
            step_s = max(scaled_dt_s, MIN_COMMAND_DT_S)
            cumulative_time_s += step_s
            point_time_s = cumulative_time_s
        points.append(
            build_replay_trajectory_point(
                parts,
                point_time_s,
                gripper_scale,
                gripper_offset,
                gripper_format,
                gripper_speed,
            )
        )
    traj.points = points

    if traj.points and len(traj.points[0].joint_command_vec) != len(expanded_joint_names):
        raise RuntimeError(
            "Command vector length does not match resolved joint_names length. "
            f"commands={len(traj.points[0].joint_command_vec)}, joint_names={len(expanded_joint_names)}"
        )

    total_duration_s = traj.points[-1].time_from_start_second if traj.points else 0.0
    print(
        f"[replay] Submit execute_joint_trajectory with {len(traj.points)} points, "
        f"duration={total_duration_s:.3f}s, speed_scale={speed_scale:.3f}"
    )
    status = robot.execute_joint_trajectory(traj, True)
    if status != ControlStatus.SUCCESS:
        raise RuntimeError(f"Failed to execute replay trajectory. status={status}")

    print(f"[replay] Waiting {REPLAY_TAIL_WAIT_S:.1f}s for final command execution before shutdown...")
    time.sleep(REPLAY_TAIL_WAIT_S)


def move_to_first_frame_joint_positions(
    robot: GalbotRobot,
    first_frame_parts: dict[str, list[float] | float],
    gripper_scale: float,
    gripper_offset: float,
    gripper_format: str,
    speed_rad_s: float,
    gripper_speed_rad_s: float | None,
    timeout_s: float,
    delta_threshold_rad: float = DEFAULT_PREPARE_DELTA_THRESHOLD_RAD,
    auto_confirm: bool = False,
    is_blocking: bool = True,
) -> None:
    """Move robot joints to first-frame target positions before replay when needed."""
    target_positions = get_joint_positions_from_parts(first_frame_parts, gripper_scale, gripper_offset, gripper_format)
    if not target_positions:
        raise ValueError("First frame target positions are empty.")

    current_positions = robot.get_joint_positions(joint_groups=JOINT_GROUP_ORDER, joint_names=[])
    needs_prepare_move = True
    needs_user_confirm = True
    if current_positions and len(current_positions) == len(target_positions):
        diffs = [abs(c - t) for c, t in zip(current_positions, target_positions)]
        max_delta = max(diffs)
        mean_delta = sum(diffs) / len(diffs)
        print(f"[prepare] max initial joint delta: {max_delta:.4f} rad, mean delta: {mean_delta:.4f} rad")
        if max_delta <= delta_threshold_rad:
            print(f"[prepare] All joints are already within threshold ({delta_threshold_rad:.4f} rad). Start replay.")
            needs_prepare_move = False
            needs_user_confirm = False
    else:
        print(
            "[prepare] unable to compute initial delta because current/target joint vector sizes differ: "
            f"{len(current_positions)} vs {len(target_positions)}"
        )

    if not needs_prepare_move:
        return

    if needs_user_confirm and not auto_confirm:
        print(
            "[confirm] Robot will move from current joint state to the replay first frame "
            f"(speed_rad_s={speed_rad_s}, timeout_s={timeout_s})."
        )
        confirm = input("[confirm] Type 'yes' or 'y' to continue: ").strip().lower()
        if confirm not in {"yes", "y"}:
            raise RuntimeError("User aborted replay before first-frame alignment.")
    elif needs_user_confirm and auto_confirm:
        print("[confirm] Auto-confirm enabled. Continue first-frame alignment.")

    prepare_stages = [
        ["leg"],
        ["right_arm", "left_arm", "head"],
        ["right_gripper", "left_gripper"],
    ]

    print("[prepare] Moving to replay initial frame in stages...")

    for stage_groups in prepare_stages:
        stage_target_positions = get_stage_positions_from_parts(
            first_frame_parts,
            stage_groups,
            gripper_scale,
            gripper_offset,
            gripper_format,
        )
        if not stage_target_positions:
            continue

        stage_current_positions = robot.get_joint_positions(joint_groups=stage_groups, joint_names=[])
        stage_speed_rad_s = speed_rad_s
        if stage_groups == ["right_gripper", "left_gripper"] and gripper_speed_rad_s is not None:
            stage_speed_rad_s = gripper_speed_rad_s
        stage_timeout_s, stage_max_delta, stage_mean_delta = compute_stage_timeout_s(
            stage_current_positions,
            stage_target_positions,
            stage_speed_rad_s,
            timeout_s,
        )
        if (
            stage_current_positions
            and len(stage_current_positions) == len(stage_target_positions)
            and stage_max_delta <= delta_threshold_rad
        ):
            print(
                f"[prepare] Skip stage {stage_groups}: max delta {stage_max_delta:.4f} rad "
                f"within threshold {delta_threshold_rad:.4f} rad."
            )
            continue

        if stage_current_positions and len(stage_current_positions) == len(stage_target_positions):
            print(
                f"[prepare] Stage {stage_groups}: max delta {stage_max_delta:.4f} rad, "
                f"mean delta {stage_mean_delta:.4f} rad, speed_rad_s={stage_speed_rad_s:.4f}, "
                f"timeout_s={stage_timeout_s:.1f}"
            )
        else:
            print(
                f"[prepare] Stage {stage_groups}: current/target size mismatch "
                f"{len(stage_current_positions)} vs {len(stage_target_positions)}, "
                f"speed_rad_s={stage_speed_rad_s:.4f}, timeout_s={stage_timeout_s:.1f}"
            )

        if stage_groups == ["right_gripper", "left_gripper"]:
            command_grippers(
                robot,
                first_frame_parts,
                gripper_scale,
                gripper_offset,
                gripper_format,
                gripper_speed_rad_s,
            )
        else:
            status = robot.set_joint_positions(
                stage_target_positions,
                stage_groups,
                [],
                is_blocking,
                stage_speed_rad_s,
                stage_timeout_s,
            )
            if status != ControlStatus.SUCCESS:
                raise RuntimeError(
                    f"Failed to move stage {stage_groups} to first frame. set_joint_positions status={status}"
                )
    print("[prepare] Reached replay initial frame. Start replay.")


def _safe_set_base_velocity(robot: GalbotRobot, velocity: list[float]) -> bool:
    """Safely send base velocity command with validation.

    Args:
        robot: GalbotRobot instance.
        velocity: [vx, vy, omega] velocity command.

    Returns:
        True if command was sent successfully, False otherwise.
    """
    if not isinstance(velocity, (list, tuple)) or len(velocity) != 3:
        print(f"[odom] Invalid velocity vector (wrong length): {velocity}")
        return False

    if any(np.isnan(v) or np.isinf(v) for v in velocity):
        print(f"[odom] Velocity contains invalid values: {velocity}")
        return False

    # Clamp to reasonable ranges
    max_linear = 2.0  # m/s
    max_angular = 3.0  # rad/s
    velocity = [
        float(np.clip(velocity[0], -max_linear, max_linear)),
        float(np.clip(velocity[1], -max_linear, max_linear)),
        float(np.clip(velocity[2], -max_angular, max_angular)),
    ]

    try:
        linear = [velocity[0], velocity[1], 0.0]
        angular = [0.0, 0.0, velocity[2]]

        if any(np.isnan(v) or np.isinf(v) for v in linear + angular):
            print(f"[odom] Computed velocity contains invalid values: {linear}, {angular}")
            return False

        status = robot.set_base_velocity(linear, angular)
        if status != ControlStatus.SUCCESS:
            print(f"[odom] Failed to set base velocity, status={status}")
        return status == ControlStatus.SUCCESS

    except Exception as e:
        print(f"[odom] Exception while sending velocity: {e}")
        return False


def replay_with_odom_twist(
    robot: GalbotRobot,
    parsed_frames: list[tuple[float, dict[str, list[float] | float]]],
    gripper_scale: float,
    gripper_offset: float,
    gripper_format: str,
    speed_scale: float,
    gripper_speed: float | None,
    odom_scale: float = 1.0,
) -> None:
    """Replay trajectory with synchronized odom twist chassis control.

    This function sends base velocity commands based on odom_twist data
    from the dataset while executing the joint trajectory.

    Args:
        robot: GalbotRobot instance.
        parsed_frames: List of (timestamp, parts) tuples.
        gripper_scale: Gripper scaling factor.
        gripper_offset: Gripper offset.
        gripper_format: Gripper format ("raw" or "scaled").
        speed_scale: Speed multiplier for replay.
        gripper_speed: Gripper speed.
        odom_scale: Scaling factor for odom twist velocities.
    """
    if not parsed_frames:
        raise ValueError("Replay trajectory is empty.")

    # Check if odom twist data is available
    has_odom = all(
        "odom_twist_linear" in parts and "odom_twist_angular" in parts
        for _, parts in parsed_frames
    )
    if not has_odom:
        print("[odom] No odom twist data found in frames, replaying without chassis control.")
        replay_with_execute_joint_trajectory(
            robot, parsed_frames, gripper_scale, gripper_offset,
            gripper_format, speed_scale, gripper_speed,
        )
        return

    expanded_joint_names: list[str] = []
    for group_name in JOINT_GROUP_ORDER:
        expanded_joint_names.extend(robot.get_joint_names(True, [group_name]))
    if not expanded_joint_names:
        raise RuntimeError("Failed to resolve joint_names from JOINT_GROUP_ORDER")

    traj = Trajectory()
    traj.joint_groups = []
    traj.joint_names = expanded_joint_names

    cumulative_time_s = 0.0
    points: list[TrajectoryPoint] = []
    odom_commands: list[tuple[float, list[float]]] = []

    for idx, (timestamp, parts) in enumerate(parsed_frames):
        if idx == 0:
            point_time_s = 0.0
        else:
            prev_timestamp = parsed_frames[idx - 1][0]
            raw_dt_s = max(timestamp - prev_timestamp, 0.0)
            scaled_dt_s = raw_dt_s / speed_scale
            step_s = max(scaled_dt_s, MIN_COMMAND_DT_S)
            cumulative_time_s += step_s
            point_time_s = cumulative_time_s

        points.append(
            build_replay_trajectory_point(
                parts, point_time_s, gripper_scale, gripper_offset,
                gripper_format, gripper_speed,
            )
        )

        # Extract odom twist and scale
        linear = parts.get("odom_twist_linear", [0.0, 0.0])
        angular = parts.get("odom_twist_angular", [0.0])
        vx = float(linear[0]) * odom_scale if len(linear) > 0 else 0.0
        vy = float(linear[1]) * odom_scale if len(linear) > 1 else 0.0
        omega = float(angular[0]) * odom_scale if len(angular) > 0 else 0.0
        odom_commands.append((point_time_s, [vx, vy, omega]))

    traj.points = points

    if traj.points and len(traj.points[0].joint_command_vec) != len(expanded_joint_names):
        raise RuntimeError(
            "Command vector length does not match resolved joint_names length. "
            f"commands={len(traj.points[0].joint_command_vec)}, joint_names={len(expanded_joint_names)}"
        )

    total_duration_s = traj.points[-1].time_from_start_second if traj.points else 0.0
    print(
        f"[replay] Submit execute_joint_trajectory with {len(traj.points)} points, "
        f"duration={total_duration_s:.3f}s, speed_scale={speed_scale:.3f}"
    )
    print(f"[odom] Sending {len(odom_commands)} odom twist commands")

    # Send zero velocity first for safety
    _safe_set_base_velocity(robot, [0.0, 0.0, 0.0])
    time.sleep(0.1)

    # Execute trajectory in non-blocking mode
    robot.execute_joint_trajectory(traj, is_blocking=False)

    # Main loop: synchronize odom twist commands
    start_time = time.monotonic()
    odom_idx = 0

    try:
        while odom_idx < len(odom_commands):
            current_time = time.monotonic() - start_time
            target_time, velocity = odom_commands[odom_idx]

            if current_time >= target_time:
                _safe_set_base_velocity(robot, velocity)
                odom_idx += 1

            time.sleep(0.005)  # 5ms control cycle

        print("[replay] Joint trajectory and odom twist execution completed")

    except Exception as e:
        print(f"[replay] Error during execution: {e}")
        raise

    finally:
        # Ensure chassis stops
        _safe_set_base_velocity(robot, [0.0, 0.0, 0.0])
        time.sleep(0.2)

    print(f"[replay] Waiting {REPLAY_TAIL_WAIT_S:.1f}s for final command execution before shutdown...")
    time.sleep(REPLAY_TAIL_WAIT_S)


def replay_with_chassis_wheel(
    robot: GalbotRobot,
    parsed_frames: list[tuple[float, dict[str, list[float] | float]]],
    gripper_scale: float,
    gripper_offset: float,
    gripper_format: str,
    speed_scale: float,
    gripper_speed: float | None,
    wheel_layout: str = "x45",
    wheel_L: float = 0.2,
    wheel_W: float = 0.2,
    wheel_scale: float = 1.0,
) -> None:
    """Replay trajectory with synchronized chassis control based on wheel speeds.

    This function uses FourOmniWheelKinematics to compute chassis velocity
    from wheel speeds (chassis_vel) in the dataset, then sends base velocity
    commands while executing the joint trajectory.

    Args:
        robot: GalbotRobot instance.
        parsed_frames: List of (timestamp, parts) tuples.
        gripper_scale: Gripper scaling factor.
        gripper_offset: Gripper offset.
        gripper_format: Gripper format ("raw" or "scaled").
        speed_scale: Speed multiplier for replay.
        gripper_speed: Gripper speed.
        wheel_layout: Wheel layout type ('x45', 'plus', or 'custom').
        wheel_L: Half length of chassis (front-back direction, meters).
        wheel_W: Half width of chassis (left-right direction, meters).
        wheel_scale: Scaling factor for wheel speeds.
    """
    if not parsed_frames:
        raise ValueError("Replay trajectory is empty.")

    # Check if chassis_vel data is available
    has_chassis_vel = all("chassis_vel" in parts for _, parts in parsed_frames)
    if not has_chassis_vel:
        print("[wheel] No chassis_vel data found in frames, replaying without chassis control.")
        replay_with_execute_joint_trajectory(
            robot, parsed_frames, gripper_scale, gripper_offset,
            gripper_format, speed_scale, gripper_speed,
        )
        return

    # Initialize kinematics
    kin = FourOmniWheelKinematics(layout_type=wheel_layout, L=wheel_L, W=wheel_W)
    print("[wheel] FourOmniWheelKinematics initialized:")
    kin.print_configuration()

    expanded_joint_names: list[str] = []
    for group_name in JOINT_GROUP_ORDER:
        expanded_joint_names.extend(robot.get_joint_names(True, [group_name]))
    if not expanded_joint_names:
        raise RuntimeError("Failed to resolve joint_names from JOINT_GROUP_ORDER")

    traj = Trajectory()
    traj.joint_groups = []
    traj.joint_names = expanded_joint_names

    cumulative_time_s = 0.0
    points: list[TrajectoryPoint] = []
    chassis_commands: list[tuple[float, list[float]]] = []

    for idx, (timestamp, parts) in enumerate(parsed_frames):
        if idx == 0:
            point_time_s = 0.0
        else:
            prev_timestamp = parsed_frames[idx - 1][0]
            raw_dt_s = max(timestamp - prev_timestamp, 0.0)
            scaled_dt_s = raw_dt_s / speed_scale
            step_s = max(scaled_dt_s, MIN_COMMAND_DT_S)
            cumulative_time_s += step_s
            point_time_s = cumulative_time_s

        points.append(
            build_replay_trajectory_point(
                parts, point_time_s, gripper_scale, gripper_offset,
                gripper_format, gripper_speed,
            )
        )

        # Extract wheel speeds and compute chassis velocity
        wheel_speeds = parts.get("chassis_vel", [0.0, 0.0, 0.0, 0.0])
        wheel_speeds_scaled = [float(v) * wheel_scale for v in wheel_speeds]

        # Compute chassis velocity from wheel speeds
        chassis_vel = kin.wheels_to_chassis(wheel_speeds_scaled)

        # Validate velocity
        if np.any(np.isnan(chassis_vel)) or np.any(np.isinf(chassis_vel)):
            print(f"[wheel] Invalid chassis velocity at time {point_time_s:.3f}s: {chassis_vel}, using zero")
            chassis_vel = np.zeros(3)

        # Clamp to reasonable ranges
        max_speed = 2.0  # m/s
        max_omega = 3.0  # rad/s
        chassis_vel = np.clip(chassis_vel, [-max_speed, -max_speed, -max_omega], [max_speed, max_speed, max_omega])

        chassis_commands.append((point_time_s, chassis_vel.tolist()))

    traj.points = points

    if traj.points and len(traj.points[0].joint_command_vec) != len(expanded_joint_names):
        raise RuntimeError(
            "Command vector length does not match resolved joint_names length. "
            f"commands={len(traj.points[0].joint_command_vec)}, joint_names={len(expanded_joint_names)}"
        )

    total_duration_s = traj.points[-1].time_from_start_second if traj.points else 0.0
    print(
        f"[replay] Submit execute_joint_trajectory with {len(traj.points)} points, "
        f"duration={total_duration_s:.3f}s, speed_scale={speed_scale:.3f}"
    )
    print(f"[wheel] Sending {len(chassis_commands)} chassis velocity commands")

    # Send zero velocity first for safety
    _safe_set_base_velocity(robot, [0.0, 0.0, 0.0])
    time.sleep(0.1)

    # Execute trajectory in non-blocking mode
    robot.execute_joint_trajectory(traj, is_blocking=False)

    # Main loop: synchronize chassis velocity commands
    start_time = time.monotonic()
    chassis_idx = 0

    try:
        while chassis_idx < len(chassis_commands):
            current_time = time.monotonic() - start_time
            target_time, velocity = chassis_commands[chassis_idx]

            if current_time >= target_time:
                _safe_set_base_velocity(robot, velocity)
                chassis_idx += 1

            time.sleep(0.005)  # 5ms control cycle

        print("[replay] Joint trajectory and chassis wheel execution completed")

    except Exception as e:
        print(f"[replay] Error during execution: {e}")
        raise

    finally:
        # Ensure chassis stops
        _safe_set_base_velocity(robot, [0.0, 0.0, 0.0])
        time.sleep(0.2)

    print(f"[replay] Waiting {REPLAY_TAIL_WAIT_S:.1f}s for final command execution before shutdown...")
    time.sleep(REPLAY_TAIL_WAIT_S)


def validate_required_joint_groups(robot: GalbotRobot) -> None:
    """Validate required joint groups exist on current robot/SDK."""
    available_groups = set(robot.get_joint_group_names())
    missing_groups = [g for g in JOINT_GROUP_ORDER if g not in available_groups]
    if missing_groups:
        raise RuntimeError(
            "Required joint groups are missing in current robot/SDK. "
            f"missing={missing_groups}, available={sorted(available_groups)}"
        )


def replay_parquet(
    parquet_path: str,
    action_names: list[str] | None,
    gripper_scale: float = 100.0,
    gripper_offset: float = 0.6,
    gripper_format: str = DEFAULT_GRIPPER_FORMAT,
    speed_scale: float = 1.0,
    prepare_first_frame: bool = True,
    joint_speed: float = 0.2,
    gripper_speed: float | None = 100.0,
    timeout_s: float = 15.0,
    delta_threshold_rad: float = DEFAULT_PREPARE_DELTA_THRESHOLD_RAD,
    auto_confirm: bool = False,
    wheel_control: bool = True,
    wheel_layout: str = "x45",
    wheel_L: float = 0.2,
    wheel_W: float = 0.2,
    wheel_scale: float = 1.0,
    odom_control: bool = False,
    odom_scale: float = 1.0,
    use_diff_odom: bool = True,
) -> None:
    """Replay a complete trajectory from a parquet dataset file.

    This function reads all frames from a parquet dataset file, converts them
    to trajectory points, and executes the complete trajectory on the robot.

    Args:
        parquet_path: Path to the parquet file containing the dataset.
        action_names: Optional list of action names for parsing. If None, uses
            default positional parsing.
        gripper_scale: Scaling factor for gripper values. Default is 100.0.
        gripper_offset: Offset for gripper values. Default is 0.6.
        wheel_control: Whether to enable wheel-based chassis control (default: True).
            Uses chassis_vel from dataset with FourOmniWheelKinematics.
        wheel_layout: Wheel layout type ('x45', 'plus', or 'custom').
        wheel_L: Half length of chassis (front-back direction, meters).
        wheel_W: Half width of chassis (left-right direction, meters).
        wheel_scale: Scaling factor for wheel speeds.
        odom_control: Whether to enable odom twist chassis control.
        odom_scale: Scaling factor for odom twist velocities.
    """
    if speed_scale <= 0:
        raise ValueError(f"speed_scale must be positive, got {speed_scale}")

    # Read parquet file
    df = pd.read_parquet(parquet_path)
    robot = GalbotRobot.get_instance()
    initialized = False
    try:
        # Initialize robot connection
        ok = robot.init()
        if not ok:
            raise RuntimeError("GalbotRobot.init() failed")
        initialized = True
        time.sleep(4.0)  # Wait for robot to fully initialize
        validate_required_joint_groups(robot)

        # Process all frames in the dataset
        parsed_frames: list[tuple[float, dict[str, list[float] | float]]] = []
        first_parts: dict[str, list[float] | float] | None = None
        for _, row in df.iterrows():
            # Parse action and split into parts
            action = to_list(row["action"])
            parts = split_action(action, action_names, use_diff_odom)

            if first_parts is None:
                first_parts = parts
            parsed_frames.append((float(row["timestamp"]), parts))

        resolved_gripper_format = resolve_gripper_format(
            gripper_format,
            [parts for _, parts in parsed_frames[: min(len(parsed_frames), 64)]],
        )

        if not parsed_frames:
            raise ValueError(f"Parquet file has no frames: {parquet_path}")

        # Safety prepare: move current robot state to first replay frame
        if prepare_first_frame and first_parts is not None:
            move_to_first_frame_joint_positions(
                robot=robot,
                first_frame_parts=first_parts,
                gripper_scale=gripper_scale,
                gripper_offset=gripper_offset,
                gripper_format=resolved_gripper_format,
                speed_rad_s=joint_speed,
                gripper_speed_rad_s=gripper_speed,
                timeout_s=timeout_s,
                delta_threshold_rad=delta_threshold_rad,
                auto_confirm=auto_confirm,
                is_blocking=True,
            )
            time.sleep(0.2)

        # Execute replay via execute_joint_trajectory with explicit joint_names.
        # Priority: wheel_control (default) > odom_control > no chassis control
        if wheel_control:
            replay_with_chassis_wheel(
                robot,
                parsed_frames,
                gripper_scale=gripper_scale,
                gripper_offset=gripper_offset,
                gripper_format=resolved_gripper_format,
                speed_scale=speed_scale,
                gripper_speed=gripper_speed,
                wheel_layout=wheel_layout,
                wheel_L=wheel_L,
                wheel_W=wheel_W,
                wheel_scale=wheel_scale,
            )
        elif odom_control:
            replay_with_odom_twist(
                robot,
                parsed_frames,
                gripper_scale=gripper_scale,
                gripper_offset=gripper_offset,
                gripper_format=resolved_gripper_format,
                speed_scale=speed_scale,
                gripper_speed=gripper_speed,
                odom_scale=odom_scale,
            )
        else:
            replay_with_execute_joint_trajectory(
                robot,
                parsed_frames,
                gripper_scale=gripper_scale,
                gripper_offset=gripper_offset,
                gripper_format=resolved_gripper_format,
                speed_scale=speed_scale,
                gripper_speed=gripper_speed,
            )
    finally:
        if initialized:
            # Cleanup robot connection
            robot.request_shutdown()
            robot.wait_for_shutdown()
            robot.destroy()

    print("[replay] finished")


def get_first_parquet_file(lerobot_data_dir: str) -> str:
    """Get the path to the first parquet file in a LeRobot dataset directory.

    This function constructs the path to the first episode parquet file in a
    standard LeRobot dataset structure. The expected structure is:
    lerobot_data_dir/data/chunk-000/episode_000000.parquet

    Args:
        lerobot_data_dir: Path to the LeRobot dataset directory.

    Returns:
        The absolute path to the first parquet file.

    Raises:
        ValueError: If the provided path is not a directory.
        FileNotFoundError: If the expected parquet file does not exist.
    """
    if not os.path.isdir(lerobot_data_dir):
        raise ValueError(f"path is not a directory: {lerobot_data_dir}")

    target_file = os.path.join(lerobot_data_dir, "data", "chunk-000", "episode_000000.parquet")
    if not os.path.isfile(target_file):
        raise FileNotFoundError(f"file not found: {target_file}")

    return target_file


def get_info_file(lerobot_data_dir: str) -> str:
    """Get the path to the info.json file in a LeRobot dataset directory.

    This function constructs the path to the info.json metadata file in a
    standard LeRobot dataset structure. The expected structure is:
    lerobot_data_dir/meta/info.json

    Args:
        lerobot_data_dir: Path to the LeRobot dataset directory.

    Returns:
        The absolute path to the info.json file.

    Raises:
        ValueError: If the provided path is not a directory.
        FileNotFoundError: If the expected info.json file does not exist.
    """
    if not os.path.isdir(lerobot_data_dir):
        raise ValueError(f"path is not a directory: {lerobot_data_dir}")

    target_file = os.path.join(lerobot_data_dir, "meta", "info.json")
    if not os.path.isfile(target_file):
        raise FileNotFoundError(f"file not found: {target_file}")
    return target_file


def get_action_names(info_file_path: str) -> list[str] | None:
    """Get action names from info.json file.

    This function reads the LeRobot dataset info.json file and extracts the
    action names if available. For Galbot datasets, action names may not be
    stored, in which case None is returned to use default positional parsing.

    Args:
        info_file_path: Path to the info.json file.

    Returns:
        A list of action names if available, or None if not found (which triggers
        default positional parsing).

    Raises:
        KeyError: If the required keys ('features' or 'features.action') are
            not found in the info.json file.
        FileNotFoundError: If the info.json file does not exist (handled by
            open()).
    """
    with open(info_file_path, encoding="utf-8") as f:
        info_data = json.load(f)

    if "features" not in info_data:
        raise KeyError(f"key 'features' not found in {info_file_path}")

    if "action" not in info_data["features"]:
        raise KeyError(f"key 'features.action' not found in {info_file_path}")

    # If names are available, return them; otherwise return None to use default parsing
    if "names" in info_data["features"]["action"]:
        return info_data["features"]["action"]["names"]
    # For galbot dataset, action names are not stored, so we'll use default parsing
    print(f"Warning: action names not found in {info_file_path}, using default galbot dataset structure")
    return None


if __name__ == "__main__":
    # Parse command-line arguments
    ap = argparse.ArgumentParser(description="Replay LeRobot dataset trajectories on Galbot robot")
    ap.add_argument("--lerobot-dir", required=True, help="Path to the LeRobot dataset directory")
    ap.add_argument(
        "--speed-scale",
        type=float,
        default=1.0,
        help="Replay speed multiplier. Values < 1 slow down replay, values > 1 speed it up.",
    )
    ap.add_argument(
        "--joint-speed",
        type=float,
        default=0.5,
        help="Maximum joint speed in rad/s used only by initial move-to-first-frame protection",
    )
    ap.add_argument(
        "--timeout",
        type=float,
        default=15.0,
        help="Blocking timeout in seconds used only by initial move-to-first-frame protection",
    )
    ap.add_argument(
        "--gripper-scale",
        type=float,
        default=1.0,
        help="Scaling factor applied only when --gripper-format=scaled.",
    )
    ap.add_argument(
        "--gripper-offset",
        type=float,
        default=-10.0,
        help="Offset applied only when --gripper-format=scaled. Default is 0.6",
    )
    ap.add_argument(
        "--gripper-format",
        choices=["auto", "scaled", "raw"],
        default=DEFAULT_GRIPPER_FORMAT,
        help=(
            "Gripper value encoding in the dataset. "
            "'auto' infers from sampled frames; 'scaled' applies legacy (value + offset) * scale; "
            "'raw' uses dataset gripper values directly."
        ),
    )
    ap.add_argument(
        "--gripper-speed",
        type=float,
        default=100.0,
        help="Dedicated gripper speed used for gripper prepare stage and replay trajectory points",
    )
    ap.add_argument(
        "--skip-prepare-first-frame",
        action="store_true",
        help="Skip initial safety move that aligns robot joint positions to the first replay frame",
    )
    ap.add_argument(
        "--prepare-delta-threshold",
        type=float,
        default=DEFAULT_PREPARE_DELTA_THRESHOLD_RAD,
        help="If max joint delta exceeds this threshold (rad), require confirmation and align to first frame",
    )
    ap.add_argument(
        "--yes",
        action="store_true",
        help="Skip confirmation prompt before moving to the first replay frame",
    )
    ap.add_argument(
        "--odom-control",
        action="store_true",
        help="Enable odom twist chassis control (uses odom_twist from dataset)",
    )
    ap.add_argument(
        "--odom-scale",
        type=float,
        default=1.0,
        help="Scaling factor for odom twist velocities (default: 1.0)",
    )
    ap.add_argument(
        "--use-diff-odom",
        action="store_true",
        help="Control odom with diff pose",
    )
    ap.add_argument(
        "--no-wheel-control",
        action="store_true",
        help="Disable wheel-based chassis control (wheel_control is enabled by default)",
    )
    ap.add_argument(
        "--wheel-layout",
        type=str,
        default="x45",
        choices=["x45", "plus", "custom"],
        help="Wheel layout type for FourOmniWheelKinematics (default: x45)",
    )
    ap.add_argument(
        "--wheel-L",
        type=float,
        default=0.2,
        help="Half length of chassis (front-back direction, meters) (default: 0.2)",
    )
    ap.add_argument(
        "--wheel-W",
        type=float,
        default=0.2,
        help="Half width of chassis (left-right direction, meters) (default: 0.2)",
    )
    ap.add_argument(
        "--wheel-scale",
        type=float,
        default=1.0,
        help="Scaling factor for wheel speeds (default: 1.0)",
    )
    args = ap.parse_args()

    # Get dataset file paths and metadata
    parquet_file = get_first_parquet_file(args.lerobot_dir)
    info_file = get_info_file(args.lerobot_dir)
    action_names = get_action_names(info_file)

    # Wheel control is enabled by default, can be disabled with --no-wheel-control
    wheel_control = not args.no_wheel_control

    # Replay the complete trajectory
    replay_parquet(
        parquet_path=parquet_file,
        action_names=action_names,
        gripper_scale=args.gripper_scale,
        gripper_offset=args.gripper_offset,
        gripper_format=args.gripper_format,
        speed_scale=args.speed_scale,
        prepare_first_frame=(not args.skip_prepare_first_frame),
        joint_speed=args.joint_speed,
        gripper_speed=args.gripper_speed,
        timeout_s=args.timeout,
        delta_threshold_rad=args.prepare_delta_threshold,
        auto_confirm=args.yes,
        wheel_control=wheel_control,
        wheel_layout=args.wheel_layout,
        wheel_L=args.wheel_L,
        wheel_W=args.wheel_W,
        wheel_scale=args.wheel_scale,
        odom_control=args.odom_control,
        odom_scale=args.odom_scale,
        use_diff_odom=args.use_diff_odom,
    )
