#!/usr/bin/python3
"""
ROS1到ZeroMQ桥接节点 - 共享端口版本
使用单个端口(5555)同时发布关节状态和RTSP图像数据
"""

import rospy
from sensor_msgs.msg import JointState
import zmq
import threading
import numpy as np
import time
import json
import cv2
from datetime import datetime
from queue import Queue, Full, Empty
import signal
import sys
import logging


class ROSZeroMQBridge:
    """ROS1到ZeroMQ桥接节点 - 共享端口设计"""

    def __init__(self, host="0.0.0.0", port=5555, rtsp_url="rtsp://192.168.1.105:8554/test"):
        # ROS初始化
        rospy.init_node('ros_zeromq_bridge_shared')
        
        # 配置参数
        self.host = host
        self.port = port
        self.rtsp_url = rtsp_url
        
        # 运行标志
        self.running = True
        
        # 创建共享的ZeroMQ PUB socket
        self._setup_zmq()
        
        # 初始化发布器
        self.joint_pub = JointStatePublisher(self.socket, self.host, self.port)
        self.camera_pub = RTSPCameraPublisher(self.socket, self.host, self.port, self.rtsp_url)
        
        # 设置日志
        self._setup_logging()
        
        rospy.loginfo("=" * 60)
        rospy.loginfo("ROS1->ZeroMQ桥接节点 (共享端口) 已启动")
        rospy.loginfo(f"绑定地址: tcp://{self.host}:{self.port}")
        rospy.loginfo("发布主题:")
        rospy.loginfo("  Joint State: robot/joint_states")
        rospy.loginfo("  Camera:       robot/camera")
        rospy.loginfo("=" * 60)
    
    def _setup_zmq(self):
        """初始化ZeroMQ"""
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        self.socket.setsockopt(zmq.SNDHWM, 5)  # 防止积压
        self.socket.setsockopt(zmq.LINGER, 0)  # 立即关闭
        
        # 绑定端口
        bind_address = f"tcp://{self.host}:{self.port}"
        self.socket.bind(bind_address)
        rospy.loginfo(f"ZeroMQ PUB socket绑定到 {bind_address}")
    
    def _setup_logging(self):
        """设置日志级别"""
        # 设置ROS日志级别
        rospy.logdebug("日志系统初始化完成")
    
    def start(self):
        """启动所有发布器"""
        try:
            # 启动关节状态发布器
            self.joint_pub.start()
            time.sleep(0.5)
            
            # 启动相机发布器
            self.camera_pub.start()
            
            rospy.loginfo("所有发布器已启动")
            
        except Exception as e:
            rospy.logerr(f"启动失败: {e}")
            self.stop()
            raise
    
    def stop(self):
        """停止所有发布器"""
        rospy.loginfo("正在停止所有发布器...")
        self.running = False
        
        self.joint_pub.stop()
        self.camera_pub.stop()
        
        # 关闭ZeroMQ
        if hasattr(self, 'socket') and self.socket:
            self.socket.close()
        if hasattr(self, 'context') and self.context:
            self.context.term()
        
        rospy.loginfo("所有发布器已停止")
    
    def run(self):
        """运行主循环"""
        self.start()
        
        # 设置信号处理
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        try:
            # ROS spin保持节点运行
            rospy.spin()
        except KeyboardInterrupt:
            rospy.loginfo("收到键盘中断信号")
        finally:
            self.stop()
    
    def _signal_handler(self, signum, frame):
        """信号处理器"""
        rospy.loginfo(f"收到信号 {signum}, 正在关闭...")
        self.stop()
        sys.exit(0)


class JointStatePublisher:
    """关节状态发布器 - 使用共享socket"""

    def __init__(self, socket, host, port, topic="/joint_states", rate=30):
        self.socket = socket
        self.host = host
        self.port = port
        self.topic = topic
        self.rate = rate
        
        # 状态
        self.running = False
        self.publish_thread = None
        
        # 统计
        self.frame_count = 0
        self.last_publish_time = 0
        self.min_interval = 1.0 / rate
        
        # 日志
        self.logger = logging.getLogger('JointStatePublisher')
        
        # ROS订阅者
        self.subscriber = None
        self.latest_msg = None
        self.msg_lock = threading.Lock()
    
    def start(self):
        """启动发布器"""
        try:
            # 订阅ROS话题
            self._init_ros_subscriber()
            
            # 启动发布线程
            self.running = True
            self.publish_thread = threading.Thread(target=self._publish_loop, daemon=True)
            self.publish_thread.start()
            
            rospy.loginfo(f"JointState发布器已启动 (主题: {self.topic}, 频率: {self.rate}Hz)")
            
        except Exception as e:
            rospy.logerr(f"启动JointState发布器失败: {e}")
            self.stop()
            raise
    
    def _init_ros_subscriber(self):
        """初始化ROS订阅者"""
        self.subscriber = rospy.Subscriber(
            self.topic, 
            JointState, 
            self._joint_callback,
            queue_size=1  # 只保留最新消息
        )
        rospy.loginfo(f"已订阅ROS话题: {self.topic}")
    
    def _joint_callback(self, msg):
        """关节数据回调 - 只保存最新数据"""
        with self.msg_lock:
            self.latest_msg = msg
    
    def _publish_loop(self):
        """发布循环"""
        rospy.logdebug("JointState发布循环已启动")
        
        while self.running and not rospy.is_shutdown():
            try:
                current_time = time.time()
                
                # 频率控制
                if current_time - self.last_publish_time < self.min_interval:
                    time.sleep(0.001)
                    continue
                
                # 获取最新消息
                with self.msg_lock:
                    msg = self.latest_msg
                    self.latest_msg = None  # 消费后清除
                
                if msg is not None:
                    self._publish_joint_state(msg)
                    self.last_publish_time = current_time
                
                time.sleep(0.001)
                
            except Exception as e:
                rospy.logerr(f"JointState发布循环错误: {e}")
                time.sleep(0.1)
        
        rospy.logdebug("JointState发布循环已停止")
    
    def _publish_joint_state(self, msg):
        """发布关节状态数据"""
        try:
            # 构建数据字典
            data = {
                "header": {
                    "seq": msg.header.seq,
                    "stamp": {
                        "secs": msg.header.stamp.secs,
                        "nsecs": msg.header.stamp.nsecs
                    },
                    "frame_id": msg.header.frame_id
                },
                "name": msg.name,
                "position": list(msg.position) if msg.position else [],
                "velocity": list(msg.velocity) if msg.velocity else [],
                "effort": list(msg.effort) if msg.effort else [],
                "_timestamp": datetime.now().isoformat(),
                "_frame_count": self.frame_count
            }
            
            # 转换为JSON
            json_data = json.dumps(data, default=self._json_serializer)
            
            # 发布 - 使用robot/joint_states主题
            full_message = f"robot/joint_states {json_data}"
            
            # 非阻塞发送
            try:
                self.socket.send_string(full_message, flags=zmq.NOBLOCK)
                self.frame_count += 1
                
                if self.frame_count % 100 == 0:
                    rospy.logdebug(f"已发布 {self.frame_count} 帧关节数据")
                    
            except zmq.Again:
                # 发送队列满，跳过
                rospy.logdebug("JointState发送队列满，跳过此帧")
            except Exception as e:
                rospy.logerr(f"JointState发送错误: {e}")
                
        except Exception as e:
            rospy.logerr(f"处理关节数据失败: {e}")
    
    def _json_serializer(self, obj):
        """JSON序列化辅助函数"""
        if hasattr(obj, '__dict__'):
            return obj.__dict__
        elif hasattr(obj, 'tolist'):  # numpy数组
            return obj.tolist()
        else:
            try:
                return str(obj)
            except:
                return None
    
    def stop(self):
        """停止发布器"""
        rospy.loginfo("正在停止JointState发布器...")
        self.running = False
        
        # 取消订阅
        if self.subscriber:
            self.subscriber.unregister()
        
        if self.publish_thread:
            self.publish_thread.join(timeout=2.0)
        
        rospy.loginfo(f"JointState统计: 总发布帧数={self.frame_count}")


class RTSPCameraPublisher:
    """RTSP摄像头发布器 - 使用共享socket"""

    def __init__(self, socket, host, port, rtsp_url, rate=30):
        self.socket = socket
        self.host = host
        self.port = port
        self.rtsp_url = rtsp_url
        self.rate = rate
        
        # 状态
        self.running = False
        self.capture_thread = None
        self.publish_thread = None
        
        # 帧队列（容量2，保持实时性）
        self.frame_queue = Queue(maxsize=2)
        
        # 统计
        self.stats = {
            'frames_captured': 0,
            'frames_published': 0,
            'queue_drops': 0,
            'publish_errors': 0,
            'reconnects': 0,
            'avg_latency': 0
        }
        self.stats_lock = threading.Lock()
        
        # 日志
        self.logger = logging.getLogger('RTSPCameraPublisher')
        
        # OpenCV捕获
        self.cap = None
    
    def start(self):
        """启动发布器"""
        try:
            # 启动捕获线程
            self.running = True
            self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
            self.capture_thread.start()
            
            # 启动发布线程
            self.publish_thread = threading.Thread(target=self._publish_loop, daemon=True)
            self.publish_thread.start()
            
            rospy.loginfo(f"RTSP摄像头发布器已启动 (URL: {self.rtsp_url}, 频率: {self.rate}Hz)")
            
        except Exception as e:
            rospy.logerr(f"启动RTSP摄像头发布器失败: {e}")
            self.stop()
            raise
    
    def _capture_loop(self):
        """RTSP捕获循环"""
        rospy.loginfo(f"RTSP捕获线程启动: {self.rtsp_url}")
        
        reconnect_delay = 1.0
        
        while self.running:
            try:
                # 打开RTSP流
                if self.cap is None:
                    self.cap = cv2.VideoCapture(self.rtsp_url)
                    self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # 最小缓冲
                    self.cap.set(cv2.CAP_PROP_FPS, self.rate)
                    
                    if not self.cap.isOpened():
                        raise Exception("无法打开RTSP流")
                    
                    rospy.loginfo("RTSP连接成功")
                    reconnect_delay = 1.0  # 重置重连延迟
                
                # 读取帧
                ret, frame = self.cap.read()
                
                if not ret:
                    rospy.logwarn("RTSP读取失败，准备重连...")
                    self._release_capture()
                    time.sleep(reconnect_delay)
                    reconnect_delay = min(reconnect_delay * 2, 10)  # 指数退避
                    with self.stats_lock:
                        self.stats['reconnects'] += 1
                    continue
                
                # 更新统计
                with self.stats_lock:
                    self.stats['frames_captured'] += 1
                
                # 放入队列（如果队列满则丢弃最旧的）
                frame_data = {
                    'frame': frame,
                    'timestamp': time.time(),
                    'frame_num': self.stats['frames_captured']
                }
                
                try:
                    # 非阻塞放入，如果队列满则丢弃旧帧
                    if self.frame_queue.full():
                        try:
                            self.frame_queue.get_nowait()  # 丢弃最旧的
                            with self.stats_lock:
                                self.stats['queue_drops'] += 1
                        except Empty:
                            pass
                    
                    self.frame_queue.put_nowait(frame_data)
                    
                except Full:
                    with self.stats_lock:
                        self.stats['queue_drops'] += 1
                
                # 控制捕获速率（可选）
                time.sleep(1.0 / (self.rate * 1.5))  # 略高于目标发布率
                
            except Exception as e:
                rospy.logerr(f"RTSP捕获错误: {e}")
                self._release_capture()
                time.sleep(reconnect_delay)
                reconnect_delay = min(reconnect_delay * 2, 10)
        
        self._release_capture()
        rospy.loginfo("RTSP捕获线程已退出")
    
    def _publish_loop(self):
        """发布循环"""
        rospy.logdebug("RTSP发布循环已启动")
        
        publish_interval = 1.0 / self.rate
        
        while self.running:
            try:
                loop_start = time.time()
                
                # 获取最新帧
                try:
                    frame_data = self.frame_queue.get(timeout=publish_interval)
                except Empty:
                    # 没有新帧，跳过本次发布
                    continue
                
                if frame_data:
                    self._publish_frame(frame_data)
                
                # 控制发布速率
                elapsed = time.time() - loop_start
                sleep_time = max(0, publish_interval - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)
                
            except Exception as e:
                rospy.logerr(f"RTSP发布循环错误: {e}")
                time.sleep(0.01)
        
        rospy.logdebug("RTSP发布循环已停止")
    
    def _publish_frame(self, frame_data):
        """发布单帧图像"""
        try:
            frame = frame_data['frame']
            capture_time = frame_data['timestamp']
            frame_num = frame_data['frame_num']
            
            # 计算延迟
            current_time = time.time()
            latency = current_time - capture_time
            
            # 更新平均延迟
            with self.stats_lock:
                self.stats['avg_latency'] = self.stats['avg_latency'] * 0.9 + latency * 0.1
            
            # 编码为JPEG（可调节质量平衡带宽和画质）
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, 80]
            success, buffer = cv2.imencode('.jpg', frame, encode_params)
            
            if not success:
                rospy.logwarn("图像编码失败")
                return
            
            # 准备元数据
            metadata = {
                "timestamp": datetime.now().isoformat(),
                "capture_time": capture_time,
                "frame_id": frame_num,
                "width": frame.shape[1],
                "height": frame.shape[0],
                "channels": frame.shape[2] if len(frame.shape) > 2 else 1,
                "encoding": "jpeg",
                "size_bytes": len(buffer),
                "latency_ms": round(latency * 1000, 2),
                "fps": self.rate
            }
            
            # 发送元数据 - 使用robot/camera主题
            try:
                self.socket.send_string(
                    f"robot/camera {json.dumps(metadata)}", 
                    flags=zmq.SNDMORE | zmq.NOBLOCK
                )
                # 发送图像数据
                self.socket.send(buffer.tobytes(), copy=False, flags=zmq.NOBLOCK)
                
                with self.stats_lock:
                    self.stats['frames_published'] += 1
                
                # 定期输出统计信息
                if frame_num % 100 == 0:
                    self._log_stats()
                    
            except zmq.Again:
                with self.stats_lock:
                    self.stats['publish_errors'] += 1
                if self.stats['publish_errors'] % 50 == 0:
                    rospy.logwarn("ZMQ发送队列满，正在丢帧")
            except Exception as e:
                rospy.logerr(f"发送图像失败: {e}")
                with self.stats_lock:
                    self.stats['publish_errors'] += 1
                    
        except Exception as e:
            rospy.logerr(f"处理图像帧失败: {e}")
    
    def _release_capture(self):
        """释放捕获资源"""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
    
    def _log_stats(self):
        """记录统计信息"""
        with self.stats_lock:
            stats = self.stats.copy()
        
        stats_str = (
            f"RTSP统计: "
            f"捕获={stats['frames_captured']}, "
            f"发布={stats['frames_published']}, "
            f"延迟={stats['avg_latency']*1000:.1f}ms, "
            f"丢帧={stats['queue_drops']}, "
            f"重连={stats['reconnects']}, "
            f"队列={self.frame_queue.qsize()}"
        )
        rospy.loginfo(stats_str)
    
    def stop(self):
        """停止发布器"""
        rospy.loginfo("正在停止RTSP摄像头发布器...")
        self.running = False
        
        # 释放捕获
        self._release_capture()
        
        # 清空队列
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except Empty:
                break
        
        # 等待线程
        if self.capture_thread:
            self.capture_thread.join(timeout=3.0)
        if self.publish_thread:
            self.publish_thread.join(timeout=2.0)
        
        self._log_stats()
        rospy.loginfo(f"RTSP摄像头发布器已停止，总发布帧数: {self.stats['frames_published']}")


def main():
    """主函数"""
    # 创建并运行桥接节点
    bridge = ROSZeroMQBridge(
        host="0.0.0.0",
        port=5555,
        rtsp_url="rtsp://192.168.1.105:8554/test"
    )
    
    try:
        bridge.run()
    except Exception as e:
        rospy.logerr(f"桥接节点运行失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()