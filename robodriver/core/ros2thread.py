import threading
import rclpy

import logging_mp


logger = logging_mp.get_logger(__name__)


class ROS2_Thread():
    def init(self):
        rclpy.init()
        self.spin_thread = threading.Thread(target=self._spin_loop, daemon=True)
        self.running = False

    def start(self):
        """启动 ROS2 spin 线程"""
        if self.running:
            return

        self.running = True

        self.spin_thread.start()

        logger.info("[ROS2] Node started (spin thread running)")

    def _spin_loop(self):
        """独立线程执行 ROS2 spin"""
        try:
            rclpy.spin(self)
        except Exception as e:
            logger.error(f"[ROS2] Spin error: {e}")
            rclpy.shutdown()

    def stop(self):
        """停止 ROS2"""
        if not self.running:
            return

        self.running = False
        rclpy.shutdown()

        if getattr(self, "spin_thread", None):
            self.spin_thread.join(timeout=1.0)

        logger.info("[ROS2] Node stopped.")
