#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""ROS 消息工具类（ROS2 Humble 专用，通过 ros2 命令获取消息格式，简单稳定）"""
import sys
import subprocess
from PyQt5.QtWidgets import QMessageBox

ROS_ENV = "ros2"
ROS_AVAILABLE = False

# 检查 ROS2 环境是否可用（通过 ros2 --version 验证）
try:
    subprocess.run(["ros2", "-h"], capture_output=True, check=True, text=True)
    ROS_AVAILABLE = True
except (subprocess.CalledProcessError, FileNotFoundError):
    ROS_AVAILABLE = False

print(f"[ROS 环境检测] 状态：{'可用' if ROS_AVAILABLE else '不可用'}，环境：{ROS_ENV}")


class ROSJointMsgTool:
    @staticmethod
    def is_ros_available():
        """检查 ROS2 环境是否可用"""
        return ROS_AVAILABLE

    @staticmethod
    def get_ros_env():
        """获取当前 ROS 环境（固定返回 ros2）"""
        return ROS_ENV

    @staticmethod
    def validate_msg_type(msg_type, parent=None):
        """
        验证 ROS2 消息类型有效性（调用 ros2 interface show 验证）
        :param msg_type: 消息类型（格式：包名/消息名，如 sensor_msgs/JointState）
        :param parent: 父窗口（用于弹窗）
        :return: 有效返回 True，无效返回 False
        """
        if not ROS_AVAILABLE:
            if parent:
                QMessageBox.warning(parent, "环境未就绪", "未检测到 ROS2 Humble 环境，无法使用自定义消息功能")
            else:
                print("[警告] 未检测到 ROS2 Humble 环境，无法使用自定义消息功能")
            return False

        if not msg_type or '/' not in msg_type:
            if parent:
                QMessageBox.warning(parent, "格式错误", "消息类型格式应为：包名/消息名（如：my_package/MyJointMsg）")
            else:
                print("[警告] 消息类型格式应为：包名/消息名（如：my_package/MyJointMsg）")
            return False

        # 转换为 ROS2 标准格式（包名/msg/消息名）
        pkg_name, msg_name = msg_type.split('/', 1)
        ros2_msg_type = f"{pkg_name}/msg/{msg_name}"

        # 调用 ros2 interface show 验证消息是否存在
        try:
            subprocess.run(
                ["ros2", "interface", "show", ros2_msg_type],
                capture_output=True, check=True, text=True
            )
            return True
        except subprocess.CalledProcessError as e:
            err_msg = e.stderr.strip() or "消息类型不存在"
            if parent:
                QMessageBox.critical(parent, "验证失败", f"消息类型无效：{err_msg}\n环境：ROS2 Humble")
            else:
                print(f"[错误] 消息类型无效：{err_msg}")
            return False

    @staticmethod
    def parse_msg_structure(msg_type):
        """
        解析 ROS2 消息结构（调用 ros2 interface show 捕获原始输出）
        :param msg_type: 消息类型（格式：包名/消息名，如 sensor_msgs/JointState）
        :return: 原始消息格式字符串，失败返回 None
        """
        if not ROS_AVAILABLE or not msg_type or '/' not in msg_type:
            return None

        pkg_name, msg_name = msg_type.split('/', 1)
        ros2_msg_type = f"{pkg_name}/msg/{msg_name}"

        try:
            # 调用 ros2 interface show 获取原始输出
            result = subprocess.run(
                ["ros2", "interface", "show", ros2_msg_type],
                capture_output=True, check=True, text=True
            )
            return result.stdout.strip()  # 直接返回原始字符串
        except subprocess.CalledProcessError as e:
            err_detail = e.stderr.strip() or "未知错误"
            print(f"[错误] 解析消息结构失败：{err_detail}")
            return None


if __name__ == "__main__":
    # 命令行测试
    if ROS_AVAILABLE:
        print("\n" + "="*50)
        # 测试1：系统消息
        test_msg1 = "sensor_msgs/JointState"
        print(f"测试系统消息：{test_msg1}")
        if ROSJointMsgTool.validate_msg_type(test_msg1):
            struct1 = ROSJointMsgTool.parse_msg_structure(test_msg1)
            print(f"系统消息格式：\n{struct1}")

        print("\n" + "="*50)
        # 测试2：自定义消息
        test_msg2 = "my_package/UUState"
        print(f"测试自定义消息：{test_msg2}")
        if ROSJointMsgTool.validate_msg_type(test_msg2):
            struct2 = ROSJointMsgTool.parse_msg_structure(test_msg2)
            print(f"自定义消息格式：\n{struct2}")
        print("\n" + "="*50)
    else:
        print("未检测到 ROS2 Humble 环境，跳过测试")