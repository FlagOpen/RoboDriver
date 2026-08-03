"""
Pico Ultra4 遥操 Piper 机械臂的完整控制器

集成了 VR 数据获取、IK 求解和 Piper 硬件控制
"""

import time
import logging_mp
import numpy as np
from typing import Optional

from .interface import PiperInterface
from .node import PicoUltra4DoraTeleoperatorNode
from .config import PicoUltra4DoraTeleoperatorConfig


logger = logging_mp.getLogger(__name__)


class PicoPiperController:
    """
    Pico Ultra4 遥操 Piper 机械臂的完整控制器

    整合了：
    1. VR 数据获取（XRoboToolkit）
    2. IK 求解（Placo）
    3. Piper 硬件控制（piper_sdk）
    """

    def __init__(
        self,
        config: PicoUltra4DoraTeleoperatorConfig,
        can_port: str = "can0",
        control_rate_hz: int = 50,
    ):
        """
        初始化控制器

        Args:
            config: 遥操器配置
            can_port: Piper CAN 端口
            control_rate_hz: 控制频率
        """
        self.config = config
        self.can_port = can_port
        self.control_rate_hz = control_rate_hz
        self.dt = 1.0 / control_rate_hz

        # 创建节点配置
        node_config = {
            "vr_controller": config.vr_controller,
            "control_trigger": config.control_trigger,
            "gripper_trigger": config.gripper_trigger,
            "end_effector_link": config.end_effector_link,
            "scale_factor": config.scale_factor,
            "joint_names": config.joint_names,
        }

        # 初始化 VR + IK 节点
        self.teleop_node = PicoUltra4DoraTeleoperatorNode(
            robot_urdf_path=config.robot_urdf_path,
            config=node_config
        )

        # 初始化 Piper 接口
        self.piper = PiperInterface(
            can_port=can_port,
            dt=self.dt
        )

        self.running = False

        logger.info("PicoPiperController initialized")

    def start(self):
        """启动控制器"""
        logger.info("Starting PicoPiperController...")

        # 启动 VR 节点
        self.teleop_node.start()

        # Piper 移动到 Home 位置
        logger.info("Moving Piper to home position...")
        self.piper.go_home()
        time.sleep(2)

        self.running = True
        logger.info("PicoPiperController started successfully")

    def run(self):
        """运行控制循环"""
        if not self.running:
            logger.error("Controller not started. Call start() first.")
            return

        logger.info("Starting control loop...")
        logger.info("Press and hold the grip button to activate control")
        logger.info("Press Ctrl+C to stop")

        try:
            while self.running:
                loop_start = time.perf_counter()

                # 获取当前关节位置
                current_joints = self.piper.get_joint_positions()

                # 更新 IK 求解器的当前状态
                self.teleop_node.update_current_joints(current_joints)

                # 获取 VR 控制动作
                action = self.teleop_node.get_action()

                if action is not None:
                    # 提取关节角度
                    joint_positions = np.array([
                        action.get(f"{joint}.pos", 0.0)
                        for joint in self.config.joint_names
                    ])

                    # 提取夹爪位置
                    gripper_pos = action.get("gripper.pos", 0.0)

                    # 发送到 Piper
                    self.piper.set_joint_positions(joint_positions)
                    self.piper.set_gripper_position(gripper_pos)

                    logger.debug(f"Sent command - Joints: {joint_positions}, Gripper: {gripper_pos:.2f}")
                else:
                    logger.debug("Control not active (grip button not pressed)")

                # 控制频率
                elapsed = time.perf_counter() - loop_start
                sleep_time = max(0, self.dt - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)

        except KeyboardInterrupt:
            logger.info("User interrupted, stopping...")
        except Exception as e:
            logger.error(f"Error in control loop: {e}")
            raise
        finally:
            self.stop()

    def stop(self):
        """停止控制器"""
        logger.info("Stopping PicoPiperController...")

        self.running = False

        # 停止 VR 节点
        self.teleop_node.stop()

        # Piper 回到 Home 位置
        logger.info("Moving Piper to home position...")
        self.piper.go_home()
        time.sleep(1)

        # 失能 Piper
        self.piper.disable_robot()

        logger.info("PicoPiperController stopped")


def main():
    """主函数示例"""
    import os
    from pathlib import Path

    # 配置
    urdf_path = os.path.join(
        Path(__file__).parent.parent.parent.parent.parent,
        "assets", "piper", "piper.urdf"
    )

    config = PicoUltra4DoraTeleoperatorConfig(
        robot_urdf_path=urdf_path,
        vr_controller="right_controller",
        control_trigger="right_grip",
        gripper_trigger="right_trigger",
        scale_factor=1.5,
        control_rate_hz=50,
    )

    # 创建控制器
    controller = PicoPiperController(
        config=config,
        can_port="can0",
        control_rate_hz=50,
    )

    # 启动并运行
    controller.start()
    controller.run()


if __name__ == "__main__":
    main()
