import threading

from typing import List, Optional

import rclpy
from rclpy.node import Node
from rclpy.executors import Executor, MultiThreadedExecutor

import logging_mp

logger = logging_mp.get_logger(__name__)


class ROS2_NodeManager():
    def __init__(self):
        """初始化ROS2节点管理器"""
        self._lock = threading.Lock()
        self._nodes: List[Node] = []
        self._executor: Optional[Executor] = None
        self._spin_thread: Optional[threading.Thread] = None
        self.running = False
        self._initialized = False

    def _init_ros2(self):
        """初始化ROS2（线程安全）"""
        with self._lock:
            if not self._initialized:
                rclpy.init()
                self._executor = MultiThreadedExecutor()
                self._initialized = True
                logger.info("[ROS2] ROS2 initialized")

    def add_node(self, node: Node):
        """添加节点到管理器
        
        Args:
            node: ROS2节点实例
        """
        self._init_ros2()  # 确保ROS2已初始化
        
        with self._lock:
            if node not in self._nodes:
                self._nodes.append(node)
                if self._executor is not None:
                    self._executor.add_node(node)
                logger.info(f"[ROS2] Node '{node.get_name()}' added")

    def remove_node(self, node: Node):
        """从管理器移除节点
        
        Args:
            node: 要移除的节点
        """
        with self._lock:
            if node in self._nodes:
                if self._executor is not None:
                    self._executor.remove_node(node)
                self._nodes.remove(node)
                node.destroy_node()
                logger.info(f"[ROS2] Node '{node.get_name()}' removed")

    def create_node(self, node_name: str, **kwargs) -> Node:
        """创建并添加一个新节点
        
        Args:
            node_name: 节点名称
            **kwargs: 传递给节点的其他参数
            
        Returns:
            创建的节点实例
        """
        self._init_ros2()
        
        node = rclpy.create_node(node_name, **kwargs)
        self.add_node(node)
        return node

    def start(self):
        """启动ROS2 spin线程"""
        if self.running:
            logger.warning("[ROS2] Already running")
            return

        if not self._initialized:
            self._init_ros2()

        if not self._nodes:
            logger.warning("[ROS2] No nodes to spin")

        self.running = True
        self._spin_thread = threading.Thread(target=self._spin_loop, daemon=True)
        self._spin_thread.start()
        logger.info("[ROS2] Spin thread started")

    def _spin_loop(self):
        """独立线程执行ROS2 spin"""
        try:
            while self.running and self._executor is not None and rclpy.ok():
                self._executor.spin_once(timeout_sec=0.1)
        except Exception as e:
            logger.error(f"[ROS2] Spin error: {e}")
        finally:
            self._cleanup()

    def _cleanup(self):
        """清理资源"""
        with self._lock:
            # 移除所有节点
            for node in self._nodes[:]:  # 使用副本遍历
                self.remove_node(node)
            
            # 关闭ROS2
            if self._initialized:
                rclpy.shutdown()
                self._initialized = False
            
            self.running = False
            logger.info("[ROS2] Cleanup completed")

    def stop(self):
        """停止ROS2管理器"""
        if not self.running:
            return

        self.running = False
        
        # 等待spin线程结束
        if self._spin_thread is not None:
            self._spin_thread.join(timeout=2.0)
            if self._spin_thread.is_alive():
                logger.warning("[ROS2] Spin thread did not exit cleanly")

        logger.info("[ROS2] Node manager stopped")

    def get_nodes(self) -> List[Node]:
        """获取所有节点列表"""
        with self._lock:
            return self._nodes.copy()

    def __del__(self):
        """析构函数，确保资源清理"""
        if self.running:
            self.stop()
