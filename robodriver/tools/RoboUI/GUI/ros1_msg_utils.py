#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""ROS 消息工具类（ROS1 Melodic 专用，通过 rosmsg 命令获取消息格式，兼容 Melodic 环境）"""
import sys
import subprocess
from PyQt5.QtWidgets import QMessageBox

ROS_ENV = "ros1"
ROS_DISTRO = "melodic"
ROS_AVAILABLE = False

# 检查 ROS1 Melodic 环境是否可用（通过 rosmsg --version + ROS_DISTRO 验证）
try:
    # 1. 先验证 rosmsg 命令是否存在（ROS1 消息查询核心命令）
    rosmsg_result = subprocess.run(
        ["rosmsg", "--version"],
        capture_output=True,
        text=True
    )
    if rosmsg_result.returncode != 0:
        ROS_AVAILABLE = False
    else:
        # 2. 验证 ROS_DISTRO 环境变量是否为 melodic
        import os
        current_distro = os.getenv("ROS_DISTRO", "").lower()
        if current_distro != "melodic":
            print(f"[警告] ROS_DISTRO 为 {current_distro}，当前工具仅适配 melodic")
            ROS_AVAILABLE = False
        else:
            ROS_AVAILABLE = True
except FileNotFoundError:
    ROS_AVAILABLE = False

print(f"[ROS 环境检测] 状态：{'可用' if ROS_AVAILABLE else '不可用'}，环境：{ROS_ENV} {ROS_DISTRO}")


class ROSJointMsgTool:
    @staticmethod
    def is_ros_available():
        """检查 ROS1 Melodic 环境是否可用"""
        return ROS_AVAILABLE

    @staticmethod
    def get_ros_env():
        """获取当前 ROS 环境（返回 ros1 + 版本）"""
        return f"{ROS_ENV} {ROS_DISTRO}"

    @staticmethod
    def validate_msg_type(msg_type, parent=None):
        """
        验证 ROS1 消息类型有效性（调用 rosmsg show 验证）
        :param msg_type: 消息类型（格式：包名/消息名，如 sensor_msgs/JointState）
        :param parent: 父窗口（用于弹窗）
        :return: 有效返回 True，无效返回 False
        """
        if not ROS_AVAILABLE:
            if parent:
                QMessageBox.warning(parent, "环境未就绪", f"未检测到 ROS1 Melodic 环境，无法使用自定义消息功能")
            else:
                print(f"[警告] 未检测到 ROS1 Melodic 环境，无法使用自定义消息功能")
            return False

        if not msg_type or '/' not in msg_type:
            if parent:
                QMessageBox.warning(parent, "格式错误", "消息类型格式应为：包名/消息名（如：my_package/MyJointMsg）")
            else:
                print("[警告] 消息类型格式应为：包名/消息名（如：my_package/MyJointMsg）")
            return False

        # ROS1 消息格式无需额外转换（直接用 包名/消息名）
        ros1_msg_type = msg_type

        # 调用 rosmsg show 验证消息是否存在（--raw 仅获取结构，屏蔽冗余输出）
        try:
            subprocess.run(
                ["rosmsg", "show", "--raw", ros1_msg_type],
                capture_output=True,
                check=True,
                text=True
            )
            return True
        except subprocess.CalledProcessError as e:
            err_msg = e.stderr.strip() or "消息类型不存在或未找到包"
            # ROS1 常见错误：包未在 ROS_PACKAGE_PATH 中
            if "Cannot locate message" in err_msg or "No such file or directory" in err_msg:
                err_msg += "\n提示：请确保消息包已编译，且 source 了工作空间（source devel/setup.bash）"
            if parent:
                QMessageBox.critical(parent, "验证失败", f"消息类型无效：{err_msg}\n环境：ROS1 Melodic")
            else:
                print(f"[错误] 消息类型无效：{err_msg}")
            return False

    @staticmethod
    def parse_msg_structure(msg_type):
        """
        解析 ROS1 消息结构（调用 rosmsg show 获取原始输出）
        :param msg_type: 消息类型（格式：包名/消息名，如 sensor_msgs/JointState）
        :return: 格式化后的消息结构字符串，失败返回 None
        """
        if not ROS_AVAILABLE or not msg_type or '/' not in msg_type:
            return None

        ros1_msg_type = msg_type

        try:
            # 调用 rosmsg show 获取消息结构（--raw 输出更简洁的结构）
            result = subprocess.run(
                ["rosmsg", "show", "--raw", ros1_msg_type],
                capture_output=True,
                check=True,
                text=True
            )
            raw_struct = result.stdout.strip()
            if not raw_struct:
                return "消息结构为空（可能是空消息类型）"
            
            # 格式化输出（增强可读性）
            formatted_struct = f"ROS1 Melodic 消息结构（{ros1_msg_type}）：\n{'-'*50}\n{raw_struct}"
            return formatted_struct
        except subprocess.CalledProcessError as e:
            err_detail = e.stderr.strip() or "未知错误"
            print(f"[错误] 解析消息结构失败：{err_detail}")
            return None


if __name__ == "__main__":
    # 命令行测试（ROS1 Melodic 环境下运行）
    if ROS_AVAILABLE:
        print("\n" + "="*60)
        # 测试1：系统自带消息（sensor_msgs/JointState，ROS1 内置）
        test_msg1 = "sensor_msgs/JointState"
        print(f"测试系统消息：{test_msg1}")
        if ROSJointMsgTool.validate_msg_type(test_msg1):
            struct1 = ROSJointMsgTool.parse_msg_structure(test_msg1)
            print(f"消息格式：\n{struct1}")

        print("\n" + "="*60)
        # 测试2：自定义消息（需替换为你的实际自定义消息）
        test_msg2 = "my_package/UUState"  # 示例：替换为你的 包名/消息名
        print(f"测试自定义消息：{test_msg2}")
        if ROSJointMsgTool.validate_msg_type(test_msg2):
            struct2 = ROSJointMsgTool.parse_msg_structure(test_msg2)
            print(f"消息格式：\n{struct2}")
        print("\n" + "="*60)
    else:
        print(f"未检测到 ROS1 Melodic 环境，跳过测试\n请检查：1. ROS1 是否安装 2. ROS_DISTRO 是否为 melodic 3. 是否 source 工作空间")