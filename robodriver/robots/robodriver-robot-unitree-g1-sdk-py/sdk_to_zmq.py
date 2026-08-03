#!/usr/bin/env python3
"""
机器人数据发布器 - 共享端口版本
LowState和Camera都使用5555端口，通过不同主题区分
"""

import json
import logging
import signal
import sys
import traceback
import time
from datetime import datetime
from typing import Any, Dict
import zmq
import threading
from websocket import WebSocketApp
from queue import Queue


class RobotDataPublisher:
    """机器人数据发布器 - 主控制器，共享端口"""

    def __init__(self, host: str = "0.0.0.0", port: int = 5555):
        self.host = host
        self.port = port
        
        # 创建共享的ZeroMQ上下文和PUB socket
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        self.socket.setsockopt(zmq.SNDHWM, 5)  # 设置高水位，防止积压
        self.socket.setsockopt(zmq.LINGER, 0)
        
        # 初始化两个发布器
        self.lowstate_pub = LowStatePublisher(self.socket, host, port)
        self.camera_pub = CameraImagePublisher(self.socket, host, port)

        self.logger = logging.getLogger('RobotDataPublisher')
        self._setup_logging()

    def _setup_logging(self):
        """设置日志"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('robot_publisher.log')
            ]
        )

    def start(self):
        """启动所有发布器"""
        self.logger.info("Starting Robot Data Publisher...")
        
        try:
            # 绑定端口
            bind_address = f"tcp://{self.host}:{self.port}"
            self.socket.bind(bind_address)
            self.logger.info(f"ZeroMQ PUB socket bound to {bind_address}")

            # 启动LowState发布器
            self.lowstate_pub.start()
            time.sleep(0.5)  # 短暂延迟

            # 启动Camera发布器
            self.camera_pub.start()

            self.logger.info("=" * 60)
            self.logger.info("All publishers started on single port:")
            self.logger.info(f"Port: tcp://{self.host}:{self.port}")
            self.logger.info("Topics:")
            self.logger.info("  LowState: robot/lowstate/raw")
            self.logger.info("  Camera:   robot/camera")
            self.logger.info("=" * 60)

        except Exception as e:
            self.logger.error(f"Failed to start: {e}")
            self.stop()
            raise

    def stop(self):
        """停止所有发布器"""
        self.logger.info("Stopping Robot Data Publisher...")

        self.camera_pub.stop()
        self.lowstate_pub.stop()
        
        # 关闭socket
        if self.socket:
            self.socket.close()
        
        # 关闭context
        if self.context:
            self.context.term()

        self.logger.info("All publishers stopped")

    def run(self):
        """运行主循环（阻塞）"""
        self.start()

        try:
            # 主线程等待信号
            while (self.lowstate_pub.running or 
                   self.camera_pub.running):
                time.sleep(1)

        except KeyboardInterrupt:
            self.logger.info("Keyboard interrupt received")
        finally:
            self.stop()


class LowStatePublisher:
    """LowState发布器 - 使用共享socket"""

    def __init__(self, socket, host: str = "0.0.0.0", port: int = 5555):
        self.socket = socket
        self.host = host
        self.port = port
        
        # 状态
        self.running = False
        self.publish_thread = None

        # 统计
        self.frame_count = 0

        # 设置日志
        self.logger = logging.getLogger('LowStatePublisher')

        # Unitree订阅者
        self.subscriber = None

    def start(self):
        """启动发布器"""
        try:
            # 初始化Unitree订阅者
            self._init_unitree_subscriber()

            # 启动发布线程
            self.running = True
            self.publish_thread = threading.Thread(target=self._publish_loop, daemon=True)
            self.publish_thread.start()

            self.logger.info("LowState Publisher started")

        except Exception as e:
            self.logger.error(f"Failed to start LowState publisher: {e}")
            self.stop()
            raise

    def _init_unitree_subscriber(self):
        """初始化Unitree订阅者"""
        try:
            from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
            from unitree_sdk2py.idl.unitree_hg.msg.dds_._LowState_ import LowState_

            ChannelFactoryInitialize()
            self.subscriber = ChannelSubscriber("rt/lowstate", LowState_)
            self.subscriber.Init()

            self.logger.info("Unitree LowState subscriber initialized")

        except Exception as e:
            self.logger.error(f"Failed to init Unitree: {e}")
            raise

    def _publish_loop(self):
        """LowState发布循环"""
        self.logger.info("Starting LowState publish loop...")

        publish_interval = 1.0 / 30.0  # 30Hz

        while self.running:
            try:
                start_time = time.time()

                # 读取LowState消息
                msg = self.subscriber.Read()

                if msg is not None:
                    # 转换为字典
                    data = self._lowstate_to_dict(msg)
                    data["_timestamp"] = datetime.now().isoformat()
                    data["_frame_count"] = self.frame_count

                    # 发布 - 使用robot/lowstate/raw主题
                    json_data = json.dumps(data, default=self._json_serializer)
                    full_message = f"robot/lowstate/raw {json_data}"

                    try:
                        # 使用非阻塞发送，不阻塞其他发布器
                        self.socket.send_string(full_message, flags=zmq.NOBLOCK)
                        self.frame_count += 1

                        if self.frame_count % 100 == 0:
                            self.logger.debug(f"Published LowState frame {self.frame_count}")

                    except zmq.Again:
                        # 发送队列满，跳过此帧
                        pass
                    except Exception as e:
                        self.logger.error(f"Send error: {e}")

                # 控制帧率
                elapsed = time.time() - start_time
                sleep_time = max(0, publish_interval - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)
                else:
                    # 如果处理时间超过间隔，不睡眠
                    time.sleep(0.001)

            except Exception as e:
                self.logger.error(f"Error in LowState loop: {e}")
                time.sleep(0.1)

        self.logger.info("LowState publish loop stopped")

    def _lowstate_to_dict(self, msg) -> Dict[str, Any]:
        """
        将LowState_对象转换为字典
        """
        try:
            # 基本属性
            data = {
                "version": list(msg.version) if hasattr(msg, 'version') else [],
                "mode_pr": msg.mode_pr if hasattr(msg, 'mode_pr') else 0,
                "mode_machine": msg.mode_machine if hasattr(msg, 'mode_machine') else 0,
                "tick": msg.tick if hasattr(msg, 'tick') else 0,
                "crc": msg.crc if hasattr(msg, 'crc') else 0,
            }

            # IMU状态
            if hasattr(msg, 'imu_state'):
                imu = msg.imu_state
                data["imu_state"] = {
                    "quaternion": list(imu.quaternion) if hasattr(imu, 'quaternion') else [],
                    "gyroscope": list(imu.gyroscope) if hasattr(imu, 'gyroscope') else [],
                    "accelerometer": list(imu.accelerometer) if hasattr(imu, 'accelerometer') else [],
                    "rpy": list(imu.rpy) if hasattr(imu, 'rpy') else [],
                    "temperature": imu.temperature if hasattr(imu, 'temperature') else 0,
                }

            # 电机状态
            if hasattr(msg, 'motor_state'):
                motor_states = []
                for i, motor in enumerate(msg.motor_state):
                    motor_data = {
                        "index": i,
                        "mode": motor.mode if hasattr(motor, 'mode') else 0,
                        "q": float(motor.q) if hasattr(motor, 'q') else 0.0,
                        "dq": float(motor.dq) if hasattr(motor, 'dq') else 0.0,
                        "ddq": float(motor.ddq) if hasattr(motor, 'ddq') else 0.0,
                        "tau_est": float(motor.tau_est) if hasattr(motor, 'tau_est') else 0.0,
                        "temperature": list(motor.temperature) if hasattr(motor, 'temperature') else [],
                        "vol": float(motor.vol) if hasattr(motor, 'vol') else 0.0,
                        "sensor": list(motor.sensor) if hasattr(motor, 'sensor') else [],
                        "motorstate": motor.motorstate if hasattr(motor, 'motorstate') else 0,
                    }

                    # 如果有reserve字段
                    if hasattr(motor, 'reserve'):
                        motor_data["reserve"] = list(motor.reserve)

                    motor_states.append(motor_data)

                data["motor_state"] = motor_states

            # 无线遥控
            if hasattr(msg, 'wireless_remote'):
                if msg.wireless_remote:
                    data["wireless_remote"] = list(msg.wireless_remote)
                else:
                    data["wireless_remote"] = []

            # 保留字段
            if hasattr(msg, 'reserve'):
                data["reserve"] = list(msg.reserve)

            return data

        except Exception as e:
            self.logger.error(f"Error converting LowState to dict: {e}")
            return {"error": str(e), "timestamp": datetime.now().isoformat()}

    def _json_serializer(self, obj):
        """JSON序列化辅助函数"""
        if hasattr(obj, '__dict__'):
            return obj.__dict__
        elif hasattr(obj, 'tolist'):  # 处理numpy数组等
            return obj.tolist()
        else:
            try:
                return str(obj)
            except:
                return None

    def stop(self):
        """停止发布器"""
        self.logger.info("Stopping LowState Publisher...")
        self.running = False

        if self.publish_thread:
            self.publish_thread.join(timeout=2.0)

        self.logger.info(f"LowState Statistics: Frames={self.frame_count}")


class CameraImagePublisher:
    """优化版相机图像发布器 - 使用共享socket"""

    def __init__(self, socket, host: str = "0.0.0.0", port: int = 5555,
                 camera_url: str = "ws://localhost:8000/videos/ws-stream"):
        self.socket = socket
        self.host = host
        self.port = port
        self.camera_url = camera_url

        # 状态
        self.running = False
        self.publish_thread = None
        self.ws_thread = None

        # 使用队列代替锁，提高性能
        self.frame_queue = Queue(maxsize=2)  # 只保留最新2帧，丢弃旧帧
        self.frame_count = 0
        self.last_frame_time = 0

        # 统计信息
        self.stats = {
            'frames_received': 0,
            'frames_published': 0,
            'queue_drops': 0,
            'publish_errors': 0,
            'avg_latency': 0
        }

        # 日志
        self.logger = logging.getLogger('CameraImagePublisher')

        # WebSocket
        self.ws = None
        self.ws_connected = False

    def start(self):
        """启动发布器"""
        if self.running:
            return

        self.logger.info(f"Starting CameraImagePublisher on port {self.port}")

        # 启动发布线程（优先启动）
        self.running = True
        self.publish_thread = threading.Thread(target=self._publish_loop, daemon=True)
        self.publish_thread.start()

        # 启动WebSocket连接
        self._start_websocket()

    def _start_websocket(self):
        """启动WebSocket连接"""
        try:
            self.logger.info(f"Connecting to {self.camera_url}")
            
            # 创建WebSocketApp并在线程中运行
            self.ws = WebSocketApp(
                self.camera_url,
                on_open=self._on_ws_open,
                on_message=self._on_ws_message,
                on_error=self._on_ws_error,
                on_close=self._on_ws_close
            )

            # 运行WebSocket在独立线程
            self.ws_thread = threading.Thread(target=self.ws.run_forever, daemon=True)
            self.ws_thread.start()

        except Exception as e:
            self.logger.error(f"WebSocket start error: {e}")
            time.sleep(1)
            # 自动重连
            if self.running:
                threading.Thread(target=self._start_websocket, daemon=True).start()

    def _on_ws_open(self, ws):
        """WebSocket连接成功"""
        self.ws_connected = True
        self.logger.info("WebSocket connected successfully")

    def _on_ws_message(self, ws, message):
        """处理WebSocket消息 - 优化版本"""
        try:
            if not self.running:
                return

            current_time = time.time()
            
            # 快速验证JPEG
            if isinstance(message, bytes) and len(message) > 100:
                # 检查JPEG头部（快速检查）
                if message[:2] == b'\xff\xd8' and message[-2:] == b'\xff\xd9':
                    self.stats['frames_received'] += 1
                    
                    # 创建帧数据（最小化数据复制）
                    frame_data = {
                        'jpeg_data': message,  # 直接引用，不复制
                        'timestamp': current_time,
                        'frame_num': self.stats['frames_received']
                    }
                    
                    # 非阻塞方式放入队列（如果队列满则丢弃最旧的）
                    if self.frame_queue.full():
                        try:
                            self.frame_queue.get_nowait()  # 丢弃旧帧
                            self.stats['queue_drops'] += 1
                        except:
                            pass
                    
                    try:
                        self.frame_queue.put_nowait(frame_data)
                    except:
                        self.stats['queue_drops'] += 1
                        
        except Exception as e:
            self.logger.error(f"Message processing error: {e}")
            self.stats['publish_errors'] += 1

    def _on_ws_error(self, ws, error):
        """WebSocket错误"""
        self.logger.error(f"WebSocket error: {error}")
        self.ws_connected = False

    def _on_ws_close(self, ws, close_status_code, close_msg):
        """WebSocket关闭"""
        self.logger.warning(f"WebSocket closed: {close_status_code} - {close_msg}")
        self.ws_connected = False
        
        # 延迟重连
        if self.running:
            time.sleep(2)
            self.logger.info("Attempting to reconnect WebSocket...")
            threading.Thread(target=self._start_websocket, daemon=True).start()

    def _publish_loop(self):
        """发布循环 - 只在有数据时发布"""
        self.logger.info("Starting optimized publish loop...")
        
        while self.running:
            try:
                # 阻塞获取最新帧，最多等待1/30秒
                try:
                    frame_data = self.frame_queue.get(timeout=1.0/30.0)
                except:
                    # 超时，没有新帧，跳过本次发布
                    continue
                
                if frame_data:
                    current_time = time.time()
                    
                    # 计算延迟
                    latency = current_time - frame_data['timestamp']
                    self.stats['avg_latency'] = (self.stats['avg_latency'] * 0.9 + latency * 0.1)
                    
                    # 准备元数据
                    metadata = {
                        "timestamp": datetime.now().isoformat(),
                        "frame": frame_data['frame_num'],
                        "size": len(frame_data['jpeg_data']),
                        "latency_ms": round(latency * 1000, 2),
                        "queue_size": self.frame_queue.qsize(),
                        "fps_received": self.stats['frames_received']
                    }
                    
                    try:
                        # 发送元数据 - 使用robot/camera主题
                        self.socket.send_string(f"robot/camera {json.dumps(metadata)}", zmq.SNDMORE | zmq.NOBLOCK)
                        # 发送JPEG数据
                        self.socket.send(frame_data['jpeg_data'], copy=False, flags=zmq.NOBLOCK)
                        
                        self.stats['frames_published'] += 1
                        
                        # 每100帧记录一次
                        if self.stats['frames_published'] % 100 == 0:
                            self._log_stats()
                            
                    except zmq.Again:
                        # 发送超时（队列满）
                        self.stats['publish_errors'] += 1
                        if self.stats['publish_errors'] % 50 == 0:
                            self.logger.warning("ZMQ send queue full, dropping frames")
                    except Exception as e:
                        self.logger.error(f"Publish error: {e}")
                        self.stats['publish_errors'] += 1
                    
            except Exception as e:
                self.logger.error(f"Publish loop error: {e}")
                time.sleep(0.001)
        
        self.logger.info("Publish loop stopped")

    def _log_stats(self):
        """记录统计信息"""
        stats_str = (
            f"Stats: Recv={self.stats['frames_received']}, "
            f"Pub={self.stats['frames_published']}, "
            f"Latency={self.stats['avg_latency']*1000:.1f}ms, "
            f"Drops={self.stats['queue_drops']}, "
            f"Queue={self.frame_queue.qsize()}"
        )
        self.logger.info(stats_str)

    def stop(self):
        """停止发布器"""
        self.logger.info("Stopping CameraImagePublisher...")
        self.running = False
        
        # 关闭WebSocket
        if self.ws:
            try:
                self.ws.close()
            except:
                pass
        
        # 等待线程结束
        if self.publish_thread:
            self.publish_thread.join(timeout=2.0)
        
        if self.ws_thread:
            self.ws_thread.join(timeout=1.0)
        
        self._log_stats()
        self.logger.info(f"Total frames published: {self.stats['frames_published']}")


def signal_handler(signum, frame):
    """信号处理器"""
    print(f"\nReceived signal {signum}, shutting down...")
    sys.exit(0)


if __name__ == "__main__":
    # 注册信号处理
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    print("=" * 60)
    print("Robot Data Publisher - Shared Port Version")
    print("=" * 60)
    print("Single Port: tcp://0.0.0.0:5555")
    print("Topics:")
    print("  1. LowState Data: robot/lowstate/raw")
    print("  2. Camera Image:  robot/camera")
    print("=" * 60)
    print("Usage example:")
    print("  Subscriber: zmq.SUB.connect('tcp://localhost:5555')")
    print("  Then subscribe to: 'robot/lowstate/raw' or 'robot/camera'")
    print("=" * 60)
    print("Press Ctrl+C to stop")
    print("=" * 60)

    try:
        publisher = RobotDataPublisher(host="0.0.0.0", port=5555)
        publisher.run()

    except Exception as e:
        print(f"Fatal error: {e}")
        traceback.print_exc()
        sys.exit(1)