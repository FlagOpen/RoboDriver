import logging
import threading
import queue
import time
import json
import os
from aliyun.log import LogClient, PutLogsRequest, LogItem, IndexConfig
from datetime import datetime
import platform
import socket
import uuid

class AliyunLogCollector:
    def __init__(self, endpoint, access_key_id, access_key_secret, project, logstore, 
                 machine_type, machine_id=None, batch_size=100, sync_interval=60,
                 local_log_enabled=True, local_log_dir="./logs", local_log_level=logging.INFO):
        """
        初始化阿里云日志收集器
        
        :param endpoint: 阿里云日志服务Endpoint
        :param access_key_id: AccessKey ID
        :param access_key_secret: AccessKey Secret
        :param project: 日志项目名称
        :param logstore: 日志库名称
        :param machine_type: 机器类型
        :param machine_id: 机器唯一标识(可选，默认为主机名)
        :param batch_size: 批量上传的日志条数
        :param sync_interval: 同步间隔(秒)
        :param local_log_enabled: 是否启用本地日志记录
        :param local_log_dir: 本地日志存储目录
        :param local_log_level: 本地日志记录级别
        """
        self.endpoint = endpoint # cn-beijing.log.aliyuncs.com
        # 从环境变量中获取 AccessKey ID 和 AccessKey Secret
        access_key_id = os.environ.get('ALIBABA_CLOUD_ACCESS_KEY_ID', '')
        access_key_secret = os.environ.get('ALIBABA_CLOUD_ACCESS_KEY_SECRET', '')
        self.access_key_id = access_key_id # 1560822971114422
        touch /etc/ilogtail/users/1560822971114422
        self.access_key_secret = access_key_secret
        self.project = project  # baai-dial
        self.logstore = logstore  # baai-eai-test
        self.machine_type = machine_type # aloha
        self.machine_id = machine_id or socket.gethostname()  # CcFWSaZzjZC_aloha
        self.batch_size = batch_size # 100
        self.sync_interval = sync_interval # 300
        self.local_log_enabled = local_log_enabled # True
        self.local_log_dir = local_log_dir 
        self.local_log_level = local_log_level
        
        # 初始化日志队列
        self.log_queue = queue.Queue(maxsize=10000)
        
        # 初始化阿里云客户端
        self.client = LogClient(endpoint, access_key_id, access_key_secret)
        
        # 初始化本地日志记录器
        self._init_local_logger()
        
        # 启动后台同步线程
        self.running = True
        self.sync_thread = threading.Thread(target=self._sync_worker)
        self.sync_thread.daemon = True # 设置成守护线程
        self.sync_thread.start()
        
        self.logger.info(f"AliyunLogCollector initialized. Machine Type: {machine_type}, ID: {self.machine_id}")

    def _init_local_logger(self):
        """初始化本地日志记录器"""
        self.local_logger = logging.getLogger('AliyunLogCollector_Local')
        self.local_logger.setLevel(logging.DEBUG)  # 设置记录器级别为DEBUG以捕获所有级别日志
        
        # 移除所有已有的处理器，避免重复记录
        self.local_logger.handlers.clear()
        
        if self.local_log_enabled:
            # 确保日志目录存在
            os.makedirs(self.local_log_dir, exist_ok=True)
            
            # 按时间滚动的文件处理器
            now = datetime.now()
            log_file = os.path.join(self.local_log_dir, f"{now.strftime('%Y-%m-%d')}.log")
            
            file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
            file_handler.setLevel(self.local_log_level)
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(formatter)
            self.local_logger.addHandler(file_handler)
        
        # 添加控制台处理器(可选)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        self.local_logger.addHandler(console_handler)

    def add_log(self, level, message, extra_fields=None):
        """
        添加日志到队列
        
        :param level: 日志级别(INFO, WARNING, ERROR等)
        :param message: 日志消息
        :param extra_fields: 额外字段字典
        """
        try:
            # 记录到本地日志
            if self.local_log_enabled:
                log_record = f"{level}: {message}"
                if extra_fields:
                    log_record += " - " + " ".join(f"{k}={v}" for k, v in extra_fields.items())
                
                # 根据级别调用不同的方法
                log_method = getattr(self.local_logger, level.lower())
                log_method(log_record)
            
            # 准备上传到阿里云的日志
            log_entry =[
                ('timestamp', datetime.now().strftime('[%Y-%m-%d %H:%M:%S]')),  
                ('level', level),
                ('message', message),
                ('machine_type', self.machine_type),
                ('machine_id', self.machine_id),
                ('hostname', socket.gethostname()),
                ('ip_address', socket.gethostbyname(socket.gethostname())),
            ]
            if extra_fields: # extra_fields = {"metadata": {"source": "web", "version": "1.0"}}
                log_entry.update(extra_fields)
            
            # 尝试放入队列，如果队列满则等待
            self.log_queue.put(log_entry, block=True, timeout=5)
        except queue.Full:
            self.local_logger.error("Log queue is full, dropping log entry")
        except Exception as e:
            self.local_logger.error(f"Error adding log to queue: {str(e)}")

    def _sync_worker(self):
        """后台同步线程工作函数"""
        while self.running:
            try:
                # 等待同步间隔或队列达到批量大小
                start_time = time.time()
                logs_to_sync = []
                
                while len(logs_to_sync) < self.batch_size and (time.time() - start_time) < self.sync_interval:
                    try:
                        # 设置超时以便定期检查是否需要退出
                        log_entry = self.log_queue.get(block=True, timeout=1)
                        logs_to_sync.append(log_entry)
                        self.log_queue.task_done()
                    except queue.Empty:
                        continue
                
                if logs_to_sync:
                    self._upload_logs(logs_to_sync)
                    logs_to_sync = []
                
                # 如果队列还有剩余日志且未达到批量大小，也进行同步
                if not self.log_queue.empty() and len(logs_to_sync) > 0:
                    self._upload_logs(logs_to_sync)
                    
            except Exception as e:
                self.local_logger.error(f"Error in sync worker: {str(e)}")
                time.sleep(5)  # 出错后等待一段时间再重试

    def _upload_logs(self, logs):
        """上传日志到阿里云日志服务"""
        try:
            
            # 转换为阿里云日志服务需要的格式
            log_group = []
            log_item = LogItem()
            for log in logs:
                log_item.set_contents(log)
                log_group.append(log_item)
            
            # 创建请求
            request = PutLogsRequest(
                project=self.project,
                logstore=self.logstore,
                topic=self.machine_type,
                source=self.machine_id,
                logitems=log_group
            )
            
            # 发送请求
            response = self.client.put_logs(request)
            
            if response.get_status() != 200:
                self.local_logger.error(f"Failed to upload logs to Aliyun. Status: {response.get_status()}, RequestID: {response.get_request_id()}")
                # 将失败的日志重新放回队列
                for log in logs:
                    self.log_queue.put(log)
            else:
                self.local_logger.debug(f"Successfully uploaded {len(logs)} logs to Aliyun")
                
        except Exception as e:
            self.local_logger.error(f"Error uploading logs to Aliyun: {str(e)}")
            # 将失败的日志重新放回队列
            for log in logs:
                try:
                    self.log_queue.put(log, block=False)
                except queue.Full:
                    self.local_logger.error("Log queue is full, cannot put back failed logs")
                    break

    def stop(self):
        """停止日志收集器"""
        self.running = False
        self.sync_thread.join()
        # 清空队列中的剩余日志
        remaining_logs = []
        while not self.log_queue.empty():
            try:
                remaining_logs.append(self.log_queue.get_nowait())
            except queue.Empty:
                break
        
        if remaining_logs:
            self._upload_logs(remaining_logs)
        
        self.local_logger.info("AliyunLogCollector stopped")

    def create_index(self):
        # 索引配置
        logstore_index = {
            'line': {
                'token': [',', ' ', "'", '"', ';', '=', '(', ')', '[', ']', '{', '}', '?', '@', '&', '<', '>', '/', ':', '\n', '\t', '\r'],
                'caseSensitive': False,
                'chn': False
            },
            'keys': {
                'timestamp': {
                    'type': 'text',
                    'token': [',', ' ', "'", '"', ';', '=', '(', ')', '[', ']', '{', '}', '?', '@', '&', '<', '>', '/', ':', '\n', '\t', '\r'],
                    'caseSensitive': False,
                    'alias': 'timestamp',
                    'doc_value': True,
                    'chn': False
                },
                'level': {
                    'type': 'text',
                    'token': [',', ' ', "'", '"', ';', '=', '(', ')', '[', ']', '{', '}', '?', '@', '&', '<', '>', '/', ':', '\n', '\t', '\r'],
                    'caseSensitive': False,
                    'alias': 'level',
                    'doc_value': True,
                    'chn': False
                },
                'message': {
                    'type': 'text',
                    'token': [',', ' ', "'", '"', ';', '=', '(', ')', '[', ']', '{', '}', '?', '@', '&', '<', '>', '/', ':', '\n', '\t', '\r'],
                    'caseSensitive': False,
                    'alias': 'message',
                    'doc_value': True,
                    'chn': False
                },
                'machine_type': {
                    'type': 'text',
                    'token': [',', ' ', "'", '"', ';', '=', '(', ')', '[', ']', '{', '}', '?', '@', '&', '<', '>', '/', ':', '\n', '\t', '\r'],
                    'caseSensitive': False,
                    'alias': 'machine_type',
                    'doc_value': True,
                    'chn': False
                },
                'machine_id': {
                    'type': 'text',
                    'token': [',', ' ', "'", '"', ';', '=', '(', ')', '[', ']', '{', '}', '?', '@', '&', '<', '>', '/', ':', '\n', '\t', '\r'],
                    'caseSensitive': False,
                    'alias': 'machine_id',
                    'doc_value': True,
                    'chn': False
                },
                'hostname': {
                    'type': 'text',
                    'token': [',', ' ', "'", '"', ';', '=', '(', ')', '[', ']', '{', '}', '?', '@', '&', '<', '>', '/', ':', '\n', '\t', '\r'],
                    'caseSensitive': False,
                    'alias': 'hostname',
                    'doc_value': True,
                    'chn': False
                },
                'ip_address': {
                    'type': 'text',
                    'token': [',', ' ', "'", '"', ';', '=', '(', ')', '[', ']', '{', '}', '?', '@', '&', '<', '>', '/', ':', '\n', '\t', '\r'],
                    'caseSensitive': False,
                    'alias': 'ip_address',
                    'doc_value': True,
                    'chn': False
                }
            }
        }
        index_config = IndexConfig()
        index_config.from_json(logstore_index)
        self.client.create_index(self.project_name, self.logstore_name, index_config)


# 示例使用
if __name__ == "__main__":
    # 配置参数 - 请替换为您的实际配置
    config = {
        "endpoint": "your-aliyun-endpoint",
        "access_key_id": "your-access-key-id",
        "access_key_secret": "your-access-key-secret",
        "project": "your-project",
        "logstore": "your-logstore",
        "machine_type": "robot-server",  # 机器类型
        "machine_id": "robot-001",       # 机器唯一标识(可选)
        "batch_size": 50,                # 批量大小
        "sync_interval": 30,             # 同步间隔(秒)
        "local_log_enabled": True,       # 启用本地日志
        "local_log_dir": "./app_logs",   # 本地日志目录
        "local_log_level": logging.DEBUG # 本地日志级别
    }
    
    # 创建日志收集器
    collector = AliyunLogCollector(**config)
    
    # 模拟添加日志
    for i in range(100):
        collector.add_log(
            level="INFO",
            message=f"Test log message {i}",
        )
        time.sleep(0.1)
    
    # 添加一条错误日志
    collector.add_log(
        level="ERROR",
        message="Something went wrong",
    )
    
    # 运行一段时间后停止(在实际应用中不需要手动停止)
    time.sleep(120)
    collector.stop()