coding = "utf-8"

from gevent import monkey
monkey.patch_all(subprocess=False)

from flask import Flask, jsonify, Response, request, session
from flask_cors import CORS
import cv2
import numpy as np
import threading
import time
import os
import logging
import datetime
from flask_socketio import SocketIO, emit
import schedule
import requests
import json
from upload_to_nas import DataUploader
import uuid
from upload_to_ks3 import RobotDataProcessor
from utils import setup_from_yaml,get_machine_info
from robot_data_uploader import config

log_machine_id = None

class CustomFormatter(logging.Formatter):
    def __init__(self, run_type, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.run_type = run_type
 
    def format(self, record):
        record.machine_id = globals().get("log_machine_id", "default_id")
        record.run_type = self.run_type
        return super().format(record)

class VideoStream:
    def __init__(self, stream_id, stream_name):
        self.stream_id = stream_id
        self.name = stream_name
        self.running = False
        self.frame_buffers = [None, None]  # 双缓冲
        self.buffer_index = 0
        self.lock = threading.Lock()
         
        

    def start(self):
        """启动视频流（仅标记为运行）"""
        if self.running:
            print(f"已经启动视频流")
            return True
        self.running = True
        return True
    
    def stop(self):
        """停止视频流"""
        self.running = False

    def update_frame(self, frame_data):
        """接收外部帧数据并更新当前帧"""
        if not self.running:
            return
        
        # # 解码图像
        # img = cv2.imdecode(np.frombuffer(frame_data, np.uint8), cv2.IMREAD_COLOR)
        # if img is None:
        #     return
        
        # # 压缩图像（可选）
        # img = cv2.resize(img, (640, 480))
        # encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 80]
        # _, jpeg = cv2.imencode('.jpg', img, encode_param)
        # compressed_frame = jpeg.tobytes()
        if frame_data is None:
            return
        
        compressed_frame = frame_data

        with self.lock:
            self.buffer_index = 1 - self.buffer_index
            self.frame_buffers[self.buffer_index] = compressed_frame

    def get_frame(self):
        if not self.running:
            return self.generate_blank_frame()
        with self.lock:
            return self.frame_buffers[self.buffer_index]
    
    @staticmethod 
    def generate_blank_frame():
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        _, jpeg = cv2.imencode('.jpg', blank)
        return jpeg.tobytes()

 
def get_machine_id():
    return "default_machine_id"


class FlaskServer:
    def __init__(self):
        # 初始化Flask应用
        config_dict = setup_from_yaml()
        if config_dict['device_server_type'] == 'release':
            self.web = config_dict['platform_server_ip_release']
            self.machine_id_path = config_dict['machine_id_path_release']
            self.machine_unique_code = config_dict['machine_code_path_release']
            config.SERVER_URL = config_dict['ks3_data_path_release']
            config.BUCKET_NAME = config_dict['ks3_data_bucket_release']
            config.UPLOAD_TARGET = config_dict['ks3_data_target_release']
        if config_dict['device_server_type'] == 'dev':
            self.web = config_dict['platform_server_ip_dev']
            self.machine_id_path = config_dict['machine_id_path_dev']
            self.machine_unique_code = config_dict['machine_code_path_dev']
            config.SERVER_URL = config_dict['ks3_data_path_dev']
            config.BUCKET_NAME = config_dict['ks3_data_bucket_dev']
            config.UPLOAD_TARGET = config_dict['ks3_data_target_dev']
        if config_dict['device_server_type'] == 'demo':
            self.web = config_dict['platform_server_ip_demo']
            self.machine_id_path = config_dict['machine_id_path_demo']
            self.machine_unique_code = config_dict['machine_code_path_demo']
            config.SERVER_URL = config_dict['ks3_data_path_demo']
            config.BUCKET_NAME = config_dict['ks3_data_bucket_demo']
            config.UPLOAD_TARGET = config_dict['ks3_data_target_demo']
        self.run_type = config_dict['device_server_type']
        self.port = config_dict['device_server_port']
        self.upload_type = config_dict['upload_type']
        self.upload_time = str(config_dict['upload_time'])
        self.robot_type = config_dict['robot_type']

        self.app = Flask(__name__)
        self.socketio = SocketIO(self.app, cors_allowed_origins="*")
        CORS(self.app)

        self.session = requests.Session() 
        self.token = None

        # 初始化日志
        self.init_logging()
        
        # 初始化实例变量
        self.robot_sid = None
        self.video_list = {}
        self.video_timestamp = time.time()
        self.video_streams = {}
        self.stream_status = {}
        self.frame_lock = threading.Lock()
        self.upload_lock = threading.Lock()  # 用于保护 self.upload_nas_flag 的访问
        self.init_streams_flag = False
        self.task_steps = {}
        self.machine_information = None
        self.machine_information_timestamp = None
        self.upload_thread = threading.Thread(target=self.time_job, daemon=True)
        self.upload_nas_flag = False
        self.upload_ks3_flag = False
        self.upload_id = '0'
        self._running = False
        self.ks3_processor = RobotDataProcessor(
            fold_path=config_dict['device_data_path'],
            server_url=config_dict['device_server_ip']
        )
        self.nas_processor = DataUploader()
        self.replay_data = None
        # 响应模板
        self.response_start_collection = {
            "timestamp": time.time(),
            "msg": None
        }
        self.response_finish_collection = {
            "timestamp": time.time(),
            "msg": None,
            "data": None
        }
        self.response_submit_collection = {
            "timestamp": time.time(),
            "msg": None
        }
        self.response_discard_collection = {
            "timestamp": time.time(),
            "msg": None
        }
        
        self.response_start_replay = {
            "timestamp": time.time(),
            "msg": None,
            "data": None

        }
        # 注册路由
        self.register_routes()


    # ---------------------------------生成设备ID---------------------------------------------
    def get_unique_code(self):
        # """获取 MAC 地址（格式：A1B2C3D4E5F6）"""
        # mac = uuid.getnode()
        # mac_hex = '{:012X}'.format(mac)  # 12 位大写十六进制
        # return mac_hex
        """生成随机唯一标识（不依赖硬件）"""
        return str(uuid.uuid4())
    
    def generate_machine_id(self):
        """生成机器 ID（MAC地址_aloha）"""
        mac = self.get_address()
        return f"{mac}_aloha"  # 格式：A1B2C3D4E5F6_aloha
    
    def save_machine_platform_id(self,machine_id,unique_code):
        """保存ID 到"""
        try:
            os.makedirs(os.path.dirname(self.machine_id_path), exist_ok=True)  # 确保目录存在
            with open(self.machine_id_path, "w") as f:
                f.write(machine_id)
            logging.info(f"机器 ID 已保存到 {self.machine_id_path}")
            os.makedirs(os.path.dirname(self.machine_unique_code), exist_ok=True)  # 确保目录存在
            with open(self.machine_unique_code, "w") as f:
                f.write(unique_code)
            logging.info(f"标识码 ID 已保存到 {self.machine_unique_code}")
            return True
        except Exception as e:
            logging.error(f"保存 ID 失败: {e}")
            return False
    
    def load_machine_id(self):
        """读取机器 ID"""
        global log_machine_id  # 声明使用全局变量
        if not os.path.exists(self.machine_id_path):
            logging.info("未找到机器 ID 文件，将生成新 ID")
            return None
        try:
            with open(self.machine_id_path, "r") as f:
                machine_id = f.read().strip()
                log_machine_id = machine_id
                logging.info(f"已加载机器 ID")
                return machine_id
        except Exception as e:
            logging.error(f"读取机器 ID 失败: {e}")
            return None
        
    def load_unique_code(self):
        """读取标识 ID"""
        if not os.path.exists(self.machine_unique_code):
            logging.info("未找到标识码 文件")
            return None
        try:
            with open(self.machine_unique_code, "r") as f:
                unique_code = f.read().strip()
                logging.info(f"已加载标识码 ID")
                return unique_code
        except Exception as e:
            logging.error(f"读取标识码 失败: {e}")
            return None

    
    # --------------------------------定时任务-------------------------------------------------
    def login(self):
        """发送登录请求"""
        logging.info("[API Request] login - 开始登录云平台")
        url = f"{self.web}/login"
         
        data = {
            "username": "eai_data_collect",
            "password": "eai_collect@2025"
        }
        
        try:
            response = self.session.post(url, json=data)
            logging.info(f"[API Response] login - 状态码: {response.status_code}")
            
            if response.status_code == 200:
                response_data = response.json()
                logging.info("[API Response] login - 登录成功")
                self.token = response_data["token"]
                return True
            else:
                logging.error(f"[API Response] login - 登录失败，状态码: {response.status_code}")
                return False
                
        except requests.exceptions.RequestException as e:
            logging.error(f"[API Error] login - 请求异常: {str(e)}")
            return False
        
    def make_request_with_token(self, path, data=None, method="POST"):
        """发送带有 token 和请求体的请求（支持 GET/POST/PUT）"""
        if not self.token:
            logging.warning("[API Warning] make_request_with_token - 未登录，无法发送请求")
            return None

        url = f"{self.web}/{path}"
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json"
        }

        try:
            logging.info(f"[API Request] make_request_with_token - 方法: {method}, 路径: {path}, 数据: {data}")
            
            # 根据 method 参数选择请求方式
            if method == "GET":
                response = self.session.get(url, headers=headers)
            elif method == "POST":
                response = self.session.post(url, headers=headers, json=data)
            elif method == "PUT":
                response = self.session.put(url, headers=headers, json=data)
            else:
                logging.error(f"[API Error] make_request_with_token - 不支持的请求方法: {method}")
                return None

            logging.info(f"[API Response] make_request_with_token - 状态码: {response.status_code}")
            
            if response.status_code == 200:
                response_data = response.json()
                logging.info(f"[API Response] make_request_with_token - 成功: {response_data}")
                return response_data
            else:
                logging.error(f"[API Response] make_request_with_token - 失败，状态码: {response.status_code}, 响应: {response.text}")
                return None
        except requests.exceptions.RequestException as e:
            logging.error(f"[API Error] make_request_with_token - 请求异常: {str(e)}")
            return None
 
    def local_to_nas(self, task_id_list=None):  # 添加参数
        logging.info(f"[Task] local_to_nas - 任务执行开始于: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        if self.login():
            self.nas_processor.upload(task_id_list)  # 假设upload方法也需要这个参数
            with self.upload_lock:
                self.upload_nas_flag = False
            logging.info("[Task] local_to_nas - 任务执行完成")
        else:
            with self.upload_lock:
                self.upload_nas_flag = False
            logging.error("[Task] local_to_nas - 任务执行失败，登录不成功")

    def local_to_ks3(self,task_id_list=None):
        logging.info(f"[Task] local_to_ks3 - 任务执行开始于: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        if self.login():    
            self.ks3_processor.encode_and_upload(self.token,task_id_list)
            with self.upload_lock:
                self.upload_ks3_flag = False
            logging.info("[Task] local_to_ks3 - 任务执行完成")
        else:
            with self.upload_lock:
                self.upload_ks3_flag = False
            logging.error("[Task] local_to_ks3 - 任务执行失败，登录不成功")

    def time_job(self):
        if self.upload_type == 'nas':
            schedule.every().day.at(self.upload_time).do(self.local_to_nas)
        else:
            schedule.every().day.at(self.upload_time).do(self.local_to_ks3)
        logging.info(f"[Task] time_job - 定时任务已启动，每天{self.upload_time}执行...")
        try:
            while True:
                schedule.run_pending()
                time.sleep(60)
        except KeyboardInterrupt:
            logging.info("[Task] time_job - 定时任务已停止")

    #---------------------------初始化---------------------------------------------------------   
    def init_logging(self):
        """初始化日志配置"""
        now = datetime.datetime.now()
        file_name = "/opt/wanx_studio/log/server/" + now.strftime("%Y.%m.%d.%H.%M") + ".log"
        log_dir = "/opt/wanx_studio/log/server"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        # logging.basicConfig(
        #     filename=file_name,
        #     level=logging.DEBUG,
        #     format='%(asctime)s - %(levelname)s - %(message)s',
        #     filemode="a",
        #     encoding='utf-8'  # Python 3.9+ 支持
        # )
        # # 添加控制台输出
        # console_handler = logging.StreamHandler()
        # console_handler.setLevel(logging.INFO)
        # formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        # console_handler.setFormatter(formatter)
        # logging.getLogger('').addHandler(console_handler)
         # 获取 root logger
        logger = logging.getLogger('')
        logger.setLevel(logging.DEBUG)  # 设置最低日志级别
    
        # 1. 文件日志（使用 CustomFormatter）
        file_handler = logging.FileHandler(file_name, mode="a", encoding="utf-8")
        file_formatter = CustomFormatter(
            run_type=self.run_type,
            fmt='%(asctime)s - %(levelname)s - %(machine_id)s - %(run_type)s - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
        # 2. 控制台日志（普通格式）
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

    def register_machine(self,unique_code):
        data = {
            'device_body':self.robot_type,
            'device_code':unique_code
        }
        response_data = self.make_request_with_token('eai/device/register', data, method="POST")
        logging.info(response_data)
        if response_data['code'] == 200:
            machine_id = response_data['data']['device_id']
            if self.save_machine_platform_id(machine_id,unique_code):
                return machine_id
        else:
            return None

    def init_or_get_machine_id(self):
        if self.login():
            """主逻辑：检查是否存在，不存在则生成并保存"""
            # 1. 尝试读取现有 ID
            machine_id = self.load_machine_id()
            if machine_id:
                return machine_id
            try:
                # 2.尝试读取标识文件
                unique_code = self.load_unique_code()
                if unique_code:
                    # 依据标识码获取机器编码
                    machine_id = self.register_machine(unique_code)
                    if machine_id:
                        return machine_id
                # 3. 都不存在则生成新 ID
                logging.info("未找到机器 ID和标识码，正在生成...")
                unique_code = self.get_unique_code()
                machine_id = self.register_machine(unique_code)
                if machine_id:
                    logging.info(f"生成的机器 ID")
                    return machine_id
                else:
                    logging.info(f"生成的机器 ID 失败！！！！")
                    return machine_id
            except Exception as e:
                logging.error(f"未知错误，无法生成机器 ID: {e}")
                return None
            
    def set_is_connect_false(self,d):
        if isinstance(d, dict):
            for key, value in d.items():
                if key == "is_connect":
                    d[key] = False
                else:
                    self.set_is_connect_false(value)
        elif isinstance(d, list):
            for item in d:
                self.set_is_connect_false(item)

    def update_machine_information_to_platform(self):
        try:
            device_id = self.init_or_get_machine_id()
            if self.machine_information:
                data = {
                        "device_id": device_id,
                        "device_code": self.load_unique_code(),
                    }
                data.update(self.machine_information)
                if abs(time.time() - self.machine_information_timestamp) > 80:
                    self.set_is_connect_false(data) 
                response_data = self.make_request_with_token('eai/device', data, method="PUT")
                logging.info(f"设备信息更新到平台反馈：{response_data}")
            else:
                logging.warning(f"设备未上报信息")
                if self.robot_type == "aloha":
                    time.sleep(10)
                    data = {
                            "device_id": device_id,
                            "device_code": self.load_unique_code(),
                        }
                    data.update(get_machine_info())
                    response_data = self.make_request_with_token('eai/device', data, method="PUT")
                    logging.info(f"设备信息更新到平台反馈：{response_data}")
        except Exception as e:
            logging.error(f"未知错误，无法更新机器信息: {e}")

    def _run_periodic_update(self):
        """定时执行更新任务的内部方法"""
        while self._running:
            self.update_machine_information_to_platform()
            time.sleep(60)  # 每分钟执行一次
 
    def start_periodic_update(self):
        """启动定时更新线程"""
        if not self._running:
            self._running = True
            thread = threading.Thread(target=self._run_periodic_update, daemon=True)
            thread.start()
            logging.info("设备信息定时更新线程已启动(每分钟)")
 
    def stop_periodic_update(self):
        """停止定时更新线程"""
        self._running = False
        logging.info("设备信息定时更新线程已停止")

    def register_routes(self):
        """注册所有路由"""
        # 系统信息
        self.app.add_url_rule('/api/info', 'system_info', self.system_info, methods=['GET'])
        self.app.add_url_rule('/api/device_information', 'device_information', self.device_information, methods=['GET'])
        
        # 视频流管理
        self.app.add_url_rule('/api/stream_info', 'get_streams', self.get_streams, methods=['GET'])
        self.app.add_url_rule('/api/start_stream', 'start_stream', self.start_stream, methods=['POST'])
        self.app.add_url_rule('/api/get_stream/<stream_id>', 'stream_video', self.stream_video, methods=['GET'])
        self.app.add_url_rule('/api/stop_stream/<stream_id>', 'stop_stream', self.stop_stream, methods=['POST'])
        
        # 采集控制
        self.app.add_url_rule('/api/start_collection', 'start_collection', self.start_collection, methods=['POST'])
        self.app.add_url_rule('/api/finish_collection', 'finish_collection', self.finish_collection, methods=['POST'])
        self.app.add_url_rule('/api/discard_collection', 'discard_collection', self.discard_collection, methods=['POST'])
        self.app.add_url_rule('/api/submit_collection', 'submit_collection', self.submit_collection, methods=['POST'])
        
        # 回放
        self.app.add_url_rule('/api/start_replay', 'start_replay', self.start_replay, methods=['POST'])
        
        # 手动上传
        self.app.add_url_rule('/api/manual_upload_nas', 'manual_upload_nas', self.manual_upload_nas, methods=['POST']) 
        self.app.add_url_rule('/api/manual_upload_ks3', 'manual_upload_ks3', self.manual_upload_ks3, methods=['GET']) 
        self.app.add_url_rule('/api/get_upload_mode', 'get_upload_mode', self.get_upload_mode, methods=['GET']) 
        self.app.add_url_rule('/api/manual_upload', 'manual_upload', self.manual_upload, methods=['POST']) 

        # nas上传反馈
        self.app.add_url_rule('/api/upload_start', 'upload_start', self.upload_start, methods=['POST'])
        self.app.add_url_rule('/api/upload_finish', 'upload_finish', self.upload_finish, methods=['POST'])
        self.app.add_url_rule('/api/upload_fail', 'upload_fail', self.upload_fail, methods=['POST'])
        self.app.add_url_rule('/api/upload_process', 'upload_process', self.upload_process, methods=['POST'])
       
        # ks3反馈接口
        self.app.add_url_rule('/api/upload_task_id', 'upload_task_id', self.upload_task_id, methods=['POST'])
        self.app.add_url_rule('/api/upload_start_ks3', 'upload_start_ks3', self.upload_start_ks3, methods=['POST'])
        self.app.add_url_rule('/api/upload_finish_ks3', 'upload_finish_ks3', self.upload_finish_ks3, methods=['POST'])
        
    
        # 机器人接口
        self.app.add_url_rule('/robot/update_stream/<stream_id>', 'update_frame', self.update_frame, methods=['POST'])
        self.app.add_url_rule('/robot/stream_info', 'robot_get_video_list', self.robot_get_video_list, methods=['POST'])
        self.app.add_url_rule('/robot/response', 'robot_response', self.robot_response, methods=['POST'])
        self.app.add_url_rule('/robot/get_task_steps', 'get_task_steps', self.get_task_steps, methods=['GET'])
        self.app.add_url_rule('/robot/update_machine_information', 'update_machine_information', self.update_machine_information, methods=['POST'])
        
        # WebSocket事件
        self.socketio.on_event('connect', self.handle_connect)
        self.socketio.on_event('HEARTBEAT', self.handle_heartbeat)
        self.socketio.on_event('disconnect', self.handle_disconnect)
    
    def send_message_to_robot(self, sid, message):
        """向特定机器人客户端发送消息"""
        logging.info(f"[WebSocket] send_message_to_robot - 发送消息到 {sid}: {message}")
        try:
            self.socketio.emit('robot_command', message, room=sid, namespace='/')
        except Exception as e:
            logging.error(f"[WebSocket] send_message_to_robot_error:{str(e)}")


    
    # ---------------------- WebSocket 事件处理 ----------------------
    def handle_connect(self):
        self.robot_sid = request.sid
        logging.info(f"[WebSocket] handle_connect - 客户端连接: {self.robot_sid}")
    
    def handle_heartbeat(self):
        """响应心跳包"""
        logging.debug("[WebSocket] handle_heartbeat - 收到心跳包")
        emit('HEARTBEAT_RESPONSE', {'server': 'alive'})
    
    def handle_disconnect(self):
        logging.info(f"[WebSocket] handle_disconnect - 客户端断开连接: {self.robot_sid}")
    
    # ---------------------- 路由处理方法 ----------------------
    def system_info(self):
        """获取系统信息"""
        logging.info("[API] system_info - 获取系统信息请求")
        try:
            active_count = sum(1 for s in self.stream_status.values() if s["active"])
            
            response_data = {
                "status": "running",
                "streams_active": active_count,
                "total_streams": len(self.stream_status),
                "timestamp": time.time(),
                "streams": self.stream_status
            }
            logging.info("[API] system_info - 返回系统信息")
            return jsonify(response_data), 200
        except Exception as e:
            logging.error(f"[API Error] system_info - 异常: {str(e)}")
            return jsonify({"error": str(e)}), 500
        
    def device_information(self):
        """获取系统信息"""
        logging.info("[API] device_information - 获取设备信息请求")
        try:
            response_data = {
                "code": 200,
                "data": {
                    "device_id":self.load_machine_id()
                },
                "msg": "success"
            }
            return jsonify(response_data), 200
        except Exception as e:
            logging.error(f"[API Error] device_information - 异常: {str(e)}")
            response_data = {
                "code": 500,
                "data": {},
                "msg": str(e)
            }
            return jsonify(response_data), 500
    
    def get_streams(self):
        try:
            logging.info("[API] get_streams - 获取视频流列表请求")
            response_data = {
                "code": 200,
                "data": self.video_list,
                "msg": "success"
            }
            logging.info("[API] get_streams - 返回视频流列表")
            return jsonify(response_data), 200
        except Exception as e:
            logging.error(f"[API Error] get_streams - 异常: {str(e)}")
            response_data = {
                "code": 500,
                "data": {},
                "msg": str(e)
            }
            return jsonify(response_data), 500
    
    def start_stream(self):
        try:
            logging.info("[API] start_stream - 启动视频流请求")
            data = request.get_json()
            logging.debug(f"[API] start_stream - 请求数据: {data}")
            
            stream_id = data.get('stream_id')
            if stream_id not in self.video_streams:
                logging.warning(f"[API] start_stream - 无效的视频流ID: {stream_id}")
                response_data = {
                    "code": 404,
                    "data": {},
                    "msg": "无效的视频流ID"
                }
                return jsonify(response_data), 200
                
            success = self.video_streams[stream_id].start()
            if success:
                self.stream_status[stream_id]["active"] = True
                logging.info(f"[API] start_stream - 成功启动视频流: {stream_id}")
                response_data = {
                    "code": 200,
                    "data": {},
                    "msg": "success"
                }
                return jsonify(response_data), 200
            else:
                logging.error(f"[API] start_stream - 启动视频流失败: {stream_id}")
                response_data = {
                    "code": 404,
                    "data": {},
                    "msg": "启动视频流失败"
                }
                return jsonify(response_data), 200
        except Exception as e:
            logging.error(f"[API Error] start_stream - 异常: {str(e)}")
            response_data = {
                "code": 500,
                "data": {},
                "msg": str(e)
            }
            return jsonify(response_data), 500
    
    def stream_video(self, stream_id):
        try:
            logging.info(f"[API] stream_video - 获取视频流请求: {stream_id}")
            
            try:
                stream_id = int(stream_id)
            except ValueError:
                logging.warning(f"[API] stream_video - 无效的流ID: {stream_id}")
                response_data = {
                    "code": 400,
                    "data": {},
                    "msg": "无效的流ID,必须为数字"
                }
                return jsonify(response_data), 200
            
            if stream_id not in self.video_streams:
                logging.warning(f"[API] stream_video - 视频流不存在: {stream_id}")
                response_data = {
                    "code": 404,
                    "data": {},
                    "msg": "视频流不存在"
                }
                return jsonify(response_data), 200
            
            if not self.video_streams[stream_id].running:
                logging.warning(f"[API] stream_video - 视频流未开启: {stream_id}")
                response_data = {
                    "code": 404,
                    "data": {},
                    "msg": "视频流未开启"
                }
                return jsonify(response_data), 200
            
            def generate():
                max_retries = 10
                retry_count = 0
                try:
                    while True:
                        frame = self.video_streams[stream_id].get_frame()
                        if frame:
                            retry_count = 0
                            yield (b'--frame\r\n'
                                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                        else:
                            retry_count += 1
                            if retry_count >= max_retries:
                                logging.warning(f"[API] stream_video - 超过最大重试次数: {stream_id}")
                                break
                        time.sleep(0.03)
                except GeneratorExit:
                    logging.info(f"[API] stream_video - 客户端断开连接: {stream_id}")
            
            logging.info(f"[API] stream_video - 开始传输视频流: {stream_id}")
            return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')
        except Exception as e:
            logging.error(f"[API Error] stream_video - 异常: {str(e)}")
            response_data = {
                "code": 500,
                "data": {},
                "msg": str(e)
            }
            return jsonify(response_data), 500
    
    def stop_stream(self, stream_id):
        try:
            logging.info(f"[API] stop_stream - 停止视频流请求: {stream_id}")
            
            try:
                stream_id = int(stream_id)
            except ValueError:
                logging.warning(f"[API] stop_stream - 无效的流ID: {stream_id}")
                response_data = {
                    "code": 400,
                    "data": {},
                    "msg": "无效的流ID,必须为数字"
                }
                return jsonify(response_data), 200
            
            if stream_id not in self.video_streams:
                logging.warning(f"[API] stop_stream - 视频流不存在: {stream_id}")
                response_data = {
                    "code": 404,
                    "data": {},
                    "msg": "视频流不存在"
                }
                return jsonify(response_data), 200
                
            self.video_streams[stream_id].stop()
            self.stream_status[stream_id]["active"] = False
            logging.info(f"[API] stop_stream - 成功停止视频流: {stream_id}")
            
            response_data = {
                "code": 200,
                "data": {},
                "msg": "stopped"
            }
            return jsonify(response_data), 200
        except Exception as e:
            logging.error(f"[API Error] stop_stream - 异常: {str(e)}")
            response_data = {
                "code": 500,
                "data": {},
                "msg": str(e)
            }
            return jsonify(response_data), 500
    
    def init_streams(self):
        """初始化视频流"""
        logging.info("[API] init_streams - 初始化视频流")
        if 'streams' in self.video_list:
            for stream in self.video_list['streams']:
                self.video_streams[stream['id']] = VideoStream(stream['id'], stream['name'])
                self.stream_status[stream['id']] = {
                    "name": str(stream['name']),
                    "active": False
                }
        logging.debug(f"[API] init_streams - 初始化完成: {self.video_list}")
    
    def start_collection(self):
        try:
            logging.info("[API] start_collection - 开始采集请求")
            data = request.get_json()
            self.replay_data = data
            logging.debug(f"[API] start_collection - 请求数据: {data}")
            if self.upload_id != str(0):
                response_data = {
                            "code": 404,
                            "data": {},
                            "msg": '数据上传中，请勿采集'
                        }
                return jsonify(response_data), 200
                
            data['machine_id'] = self.load_machine_id() or "default_machine_id"
            self.task_steps = data
            now_time = time.time()
            self.send_message_to_robot(self.robot_sid, message={'cmd': 'start_collection', 'msg': data})
            
            while True:
                if 0 < self.response_start_collection["timestamp"] - now_time < 15:
                    if self.response_start_collection['msg'] == "success":
                        logging.info("[API] start_collection - 采集开始成功")
                        response_data = {
                            "code": 200,
                            "data": {},
                            "msg": self.response_start_collection['msg']
                        }
                        return jsonify(response_data), 200
                    else:
                        logging.warning(f"[API] start_collection - 采集开始失败: {self.response_start_collection['msg']}")
                        response_data = {
                            "code": 404,
                            "data": {},
                            "msg": self.response_start_collection['msg']
                        }
                        return jsonify(response_data), 200
                else:
                    time.sleep(0.02)
                if time.time() - now_time > 15:
                    logging.warning("[API] start_collection - 机器人响应超时")
                    response_data = {
                        "code": 404,
                        "data": {},
                        "msg": "机器人响应超时"
                    }
                    return jsonify(response_data), 200
        except Exception as e:
            logging.error(f"[API Error] start_collection - 异常: {str(e)}")
            response_data = {
                "code": 500,
                "data": {},
                "msg": str(e)
            }
            return jsonify(response_data), 500
    
    def finish_collection(self):
        try:
            logging.info("[API] finish_collection - 完成采集请求")
            data = request.get_json()
            logging.debug(f"[API] finish_collection - 请求数据: {data}")
            now_time = time.time()
            self.send_message_to_robot(self.robot_sid, message={'cmd': 'finish_collection'})
            
            while True:
                if 0 < self.response_finish_collection["timestamp"] - now_time < 100:
                    if self.response_finish_collection['msg'] == "success":
                        logging.info("[API] finish_collection - 采集完成成功")
                        self.response_finish_collection['data']["device_id"] = self.load_machine_id()
                        response_data = {
                            "code": 200,
                            "data": self.response_finish_collection['data'],
                            "msg": self.response_finish_collection['msg']
                        }
                        return jsonify(response_data), 200
                    else:
                        logging.warning(f"[API] finish_collection - 采集完成失败: {self.response_finish_collection['msg']}")
                        response_data = {
                            "code": 404,
                            "data": {},
                            "msg": self.response_finish_collection['msg']
                        }
                        return jsonify(response_data), 200
                else:
                    time.sleep(0.02)
                if time.time() - now_time > 100:
                    logging.warning("[API] finish_collection - 机器人响应超时")
                    response_data = {
                        "code": 404,
                        "data": {},
                        "msg": "机器人响应超时"
                    }
                    return jsonify(response_data), 200
        except Exception as e:
            logging.error(f"[API Error] finish_collection - 异常: {str(e)}")
            response_data = {
                "code": 500,
                "data": {},
                "msg": str(e)
            }
            return jsonify(response_data), 500
        
    def discard_collection(self):
        try:
            logging.info("[API] discard_collection - 丢弃采集请求")
            data = request.get_json()
            logging.debug(f"[API] discard_collection - 请求数据: {data}")
            
            now_time = time.time()
            self.send_message_to_robot(self.robot_sid, message={'cmd': 'discard_collection'})
            
            while True:
                if 0 < self.response_discard_collection["timestamp"] - now_time < 8:
                    if self.response_discard_collection['msg'] == "success":
                        logging.info("[API] discard_collection - 采集丢弃成功")
                        response_data = {
                            "code": 200,
                            "data": {},
                            "msg": "success"
                        }
                        return jsonify(response_data), 200
                    else:
                        logging.warning(f"[API] discard_collection - 采集丢弃失败: {self.response_discard_collection['msg']}")
                        response_data = {
                            "code": 404,
                            "data": {},
                            "msg": self.response_discard_collection['msg']
                        }
                        return jsonify(response_data), 200
                else:
                    time.sleep(0.02)
                if time.time() - now_time > 8:
                    logging.warning("[API] discard_collection - 机器人响应超时")
                    response_data = {
                        "code": 404,
                        "data": {},
                        "msg": "机器人响应超时"
                    }
                    return jsonify(response_data), 200
        except Exception as e:
            logging.error(f"[API Error] discard_collection - 异常: {str(e)}")
            response_data = {
                "code": 500,
                "data": {},
                "msg": str(e)
            }
            return jsonify(response_data), 500
        
    def submit_collection(self):
        try:
            logging.info("[API] submit_collection - 提交采集请求")
            data = request.get_json()
            logging.debug(f"[API] submit_collection - 请求数据: {data}")
            
            now_time = time.time()
            self.send_message_to_robot(self.robot_sid, message={'cmd': 'submit_collection'})
            
            while True:
                if 0 < self.response_submit_collection["timestamp"] - now_time < 5:
                    if self.response_submit_collection['msg'] == "success":
                        logging.info("[API] submit_collection - 采集提交成功")
                        response_data = {
                            "code": 200,
                            "data": {},
                            "msg": "success"
                        }
                        return jsonify(response_data), 200
                    else:
                        logging.warning(f"[API] submit_collection - 采集提交失败: {self.response_submit_collection['msg']}")
                        response_data = {
                            "code": 404,
                            "data": {},
                            "msg": self.response_submit_collection['msg']
                        }
                        return jsonify(response_data), 200
                else:
                    time.sleep(0.02)
                if time.time() - now_time > 5:
                    logging.warning("[API] submit_collection - 机器人响应超时")
                    response_data = {
                        "code": 404,
                        "data": {},
                        "msg": "机器人响应超时"
                    }
                    return jsonify(response_data), 200
        except Exception as e:
            logging.error(f"[API Error] submit_collection - 异常: {str(e)}")
            response_data = {
                "code": 500,
                "data": {},
                "msg": str(e)
            }
            return jsonify(response_data), 500
        
    def start_replay(self):
        """处理开始回放的API请求"""
        try:
            logging.info("[API] start_replay - 开始回放请求")
            request_data = request.get_json()
            logging.debug(f"[API] start_replay - 请求数据: {request_data}")
            if not request_data:
                request_data = self.replay_data
            # 发送开始回放指令给机器人
            start_time = time.time()
            self.send_message_to_robot(
                self.robot_sid,
                message={'cmd': 'start_replay', 'msg': request_data}  # 指令改为 start_replay
            )
            # 等待机器人响应（最多15秒）
            while True:
                current_time = time.time()
                elapsed_time = current_time - start_time
                # 检查是否收到有效响应（15秒内）
                if 0 < self.response_start_replay.get("timestamp", 0) - start_time < 5:
                    if self.response_start_replay.get('msg') == "success":
                        logging.info("[API] start_replay - 回放启动成功")
                        response_data = {
                            "code": 200,
                            "data": self.response_start_replay.get('data'),
                            "msg": "回放启动成功"
                        }
                        return jsonify(response_data), 200
                    else:
                        error_msg = self.response_start_replay.get('msg', "未知错误")
                        logging.warning(f"[API] start_replay - 回放启动失败: {error_msg}")
                        response_data = {
                            "code": 400,
                            "data": {},
                            "msg": error_msg
                        }
                        return jsonify(response_data), 200
                else:
                    time.sleep(0.02)  # 避免CPU空转
                # 超时处理
                if elapsed_time > 5:
                    logging.warning("[API] start_replay - 机器人响应超时")
                    response_data = {
                        "code": 408,  # 408 表示请求超时
                        "data": {},
                        "msg": "机器人响应超时"
                    }
                    return jsonify(response_data), 200
        except Exception as e:
            logging.error(f"[API Error] start_replay - 异常: {str(e)}")
            response_data = {
                "code": 500,
                "data": {},
                "msg": f"服务器内部错误: {str(e)}"
            }
            return jsonify(response_data), 500
    
    def standby(self):
        if 'user_id' not in session:
            logging.warning("[API] standby - 未授权访问")
            return jsonify({"error": "Unauthorized"}), 401
        logging.info("[API] standby - 待命状态")
        pass
    
    def get_upload_mode(self):
        try:
            response_data = {
                "code": 200,
                "data": {
                    "mode": self.upload_type
                },
                "msg": "success"
            }
            logging.info(f"[API] get_upload_mode - 获取上传模式:{self.upload_type}")
            return jsonify(response_data), 200
        except Exception as e:
            logging.error(f"[API Error] get_upload_mode - 异常: {str(e)}")
            response_data = {
                "code": 500,
                "data": {},
                "msg": str(e)
            }
            return jsonify(response_data), 500
        
    def manual_upload(self):
        try:
            request_data = request.get_json()
            upload_type = request_data["upload_type"]
            task_id_list = request_data["task_id_list"]

            # 定义上传类型与内部方法的映射
            upload_methods = {
                "nas": self._upload_nas,  # 内部方法，接收 task_id_list
                "ks3": self._upload_ks3   # 内部方法，接收 task_id_list
            }

            if upload_type not in upload_methods:
                response_data = {
                    "code": 400,
                    "data": {},
                    "msg": f"Unsupported upload_type: {upload_type}. Must be one of {list(upload_methods.keys())}"
                }
                return jsonify(response_data), 400

            # 调用内部方法并传递 task_id_list
            return upload_methods[upload_type](task_id_list)

        except Exception as e:
            logging.error(f"[API Error] manual_upload - 异常: {str(e)}")
            response_data = {
                "code": 500,
                "data": {},
                "msg": str(e)
            }
            return jsonify(response_data), 500

    # 内部方法：接收 task_id_list 参数
    def _upload_nas(self, task_id_list):
        try:
            logging.info(f"[API] _upload_nas - 手动上传NAS请求, task_id_list: {task_id_list}")
            with self.upload_lock:
                if self.upload_nas_flag:
                    logging.warning("[API] _upload_nas - 数据上传中")
                    response_data = {
                        "code": 601,
                        "data": {},
                        "msg": '数据上传中'
                    }
                    return jsonify(response_data), 200
                else:
                    self.upload_nas_flag = True
                    if not self.nas_processor.nas_auth.get_auth_sid():
                        logging.error("[API] _upload_nas - 连接NAS异常")
                        response_data = {
                            "code": 601,
                            "data": {},
                            "msg": '连接nas异常'
                        }
                        self.upload_nas_flag = False
                        return jsonify(response_data), 200
                    if not self.login():
                        logging.error("[API] _upload_nas - 网络异常")
                        response_data = {
                            "code": 601,
                            "data": {},
                            "msg": '网络异常'
                        }
                        self.upload_nas_flag = False
                        return jsonify(response_data), 200
                    # 将 task_id_list 传递给线程
                    upload_manual_thread = threading.Thread(
                        target=self.local_to_nas,
                        args=(task_id_list,),  # 传递参数
                        daemon=True
                    )
                    upload_manual_thread.start()
                    logging.info("[API] _upload_nas - 启动上传线程")
                    response_data = {
                        "code": 200,
                        "data": {},
                        "msg": 'success'
                    }
                    return jsonify(response_data), 200
        except Exception as e:
            logging.error(f"[API Error] _upload_nas - 异常: {str(e)}")
            response_data = {
                "code": 500,
                "data": {},
                "msg": str(e)
            }
            return jsonify(response_data), 500

    # 内部方法：接收 task_id_list 参数
    def _upload_ks3(self, task_id_list):
        try:
            logging.info(f"[API] _upload_ks3 - 手动上传KS3请求, task_id_list: {task_id_list}")
            with self.upload_lock:
                if self.upload_ks3_flag:
                    logging.warning("[API] _upload_ks3 - 数据上传中")
                    response_data = {
                        "code": 601,
                        "data": {},
                        "msg": '数据上传中'
                    }
                    return jsonify(response_data), 200
                else:
                    self.upload_ks3_flag = True
                    if not self.login():
                        logging.error("[API] _upload_ks3 - 网络异常")
                        response_data = {
                            "code": 601,
                            "data": {},
                            "msg": '网络异常'
                        }
                        self.upload_ks3_flag = False
                        return jsonify(response_data), 200
                    # 将 task_id_list 传递给线程
                    upload_manual_thread = threading.Thread(
                        target=self.local_to_ks3,
                        args=(task_id_list,),  # 传递参数
                        daemon=True
                    )
                    upload_manual_thread.start()
                    logging.info("[API] _upload_ks3 - 启动上传线程")
                    response_data = {
                        "code": 200,
                        "data": {},
                        "msg": 'success'
                    }
                    return jsonify(response_data), 200
        except Exception as e:
            logging.error(f"[API Error] _upload_ks3 - 异常: {str(e)}")
            response_data = {
                "code": 500,
                "data": {},
                "msg": str(e)
            }
            return jsonify(response_data), 500

    # 保留原有路由绑定（兼容性）
    def manual_upload_nas(self):
        # 调用内部方法，传递空列表或默认值
        return self._upload_nas(task_id_list=[])

    def manual_upload_ks3(self):
        # 调用内部方法，传递空列表或默认值
        return self._upload_ks3(task_id_list=[])
          

    # ---------------------------------------upload----------------------------------------------
    def upload_start(self):
        try:
            logging.info("[API] upload_start - 上传开始通知")
            data = request.get_json()
            logging.debug(f"[API] upload_start - 请求数据: {data}")
            
            data['transfer_type'] = 'local_to_nas'
            self.make_request_with_token('eai/dts/upload/start', data, method="POST")
            logging.info("[API] upload_start - 上传开始通知处理完成")
            return jsonify({}), 200
        except Exception as e:
            logging.error(f"[API Error] upload_start - 异常: {str(e)}")
            return jsonify({'error': str(e)}), 500
        
    def upload_start_ks3(self):
        try:
            logging.info("[API] upload_start - 上传开始通知")
            data = request.get_json()
            print(data)
            logging.debug(f"[API] upload_start - 请求数据: {data}")
            
            self.make_request_with_token('eai/dts/upload/start', data, method="POST")
            logging.info("[API] upload_start - 上传开始通知处理完成")
            return jsonify({}), 200
        except Exception as e:
            logging.error(f"[API Error] upload_start - 异常: {str(e)}")
            return jsonify({'error': str(e)}), 500

    def upload_finish_ks3(self):
        try:
            logging.info("[API] upload_finish - 上传完成通知")
            data = request.get_json()
            logging.debug(f"[API] upload_finish - 请求数据: {data}")
            self.make_request_with_token('eai/dts/upload/complete', data, method="POST")
            logging.info("[API] upload_finish - 上传完成通知处理完成")
            return jsonify({}), 200
        except Exception as e:
            logging.error(f"[API Error] upload_finish - 异常: {str(e)}")
            return jsonify({'error': str(e)}), 500
        
    def upload_finish(self):
        try:
            logging.info("[API] upload_finish - 上传完成通知")
            data = request.get_json()
            logging.debug(f"[API] upload_finish - 请求数据: {data}")
            
            response_data = {
                "task_id": data["task_id"],                 
                "task_data_id": data["task_data_id"],           
                "transfer_type": "local_to_nas",     
                "status": "SUCCESS" 
            }
            self.make_request_with_token('eai/dts/upload/complete', response_data, method="POST")
            logging.info("[API] upload_finish - 上传完成通知处理完成")
            return jsonify({}), 200
        except Exception as e:
            logging.error(f"[API Error] upload_finish - 异常: {str(e)}")
            return jsonify({'error': str(e)}), 500
    
    def upload_fail(self):
        try:
            logging.info("[API] upload_fail - 上传失败通知")
            data = request.get_json()
            logging.debug(f"[API] upload_fail - 请求数据: {data}")
            
            if data.get("expand"):
                response_data = {
                    "task_id": data["task_id"],                 
                    "task_data_id": data["task_data_id"],           
                    "transfer_type": "local_to_nas",     
                    "status": "FAILED",
                    "expand": data['expand'] 
                }
            else:
                response_data = {
                    "task_id": data["task_id"],                 
                    "task_data_id": data["task_data_id"],           
                    "transfer_type": "local_to_nas",     
                    "status": "FAILED",
                    "expand": '{"nas_failed_msg":"网络通讯错误"}' 
                }
            self.make_request_with_token('eai/dts/upload/complete', response_data, method="POST")
            logging.info("[API] upload_fail - 上传失败通知处理完成")
            return jsonify({}), 200
        except Exception as e:
            logging.error(f"[API Error] upload_fail - 异常: {str(e)}")
            return jsonify({'error': str(e)}), 500

    def upload_process(self):
        try:
            logging.info("[API] upload_process - 上传进度通知")
            data = request.get_json()
            logging.debug(f"[API] upload_process - 请求数据: {data}")
            
            data['transfer_type'] = 'local_to_nas'
            self.make_request_with_token('eai/dts/upload/process', data, method="POST")
            logging.info("[API] upload_process - 上传进度通知处理完成")
            return jsonify({}), 200
        except Exception as e:
            logging.error(f"[API Error] upload_process - 异常: {str(e)}")
            return jsonify({'error': str(e)}), 500
        
    def upload_task_id(self):
        try:
            logging.info("[API] upload_task_id - 上传id通知")
            data = request.get_json()
            logging.debug(f"[API] upload_task_id - 请求数据: {data}")
            self.upload_id  = data['task_id']
            return jsonify({}), 200
        except Exception as e:
            logging.error(f"[API Error] upload_task_id - 异常: {str(e)}")
            return jsonify({'error': str(e)}), 500

    # ---------------------------------------robot------------------------------------------------
    def update_frame(self, stream_id):
        try:
            #logging.info(f"[API] update_frame - 更新帧请求: {stream_id}")
            
            try:
                stream_id = int(stream_id)
            except ValueError:
                #logging.warning(f"[API] update_frame - 无效的流ID: {stream_id}")
                return jsonify({"error": "无效的流ID,必须为数字"}), 400

            if stream_id not in self.video_streams:
                #logging.error(f"[API] update_frame - 无效的视频流ID: {stream_id}")
                return jsonify({"error": "无效的视频流ID"}), 401

            frame_data = request.get_data()
            if not frame_data:
                #logging.error("[API] update_frame - 未接收到帧数据")
                return jsonify({"error": "未接收到帧数据"}), 402
            
            # try:
            #     img = cv2.imdecode(np.frombuffer(frame_data, np.uint8), cv2.IMREAD_COLOR)
            #     if img is None:
            #         raise ValueError("Invalid JPEG data")
            # except Exception as e:
            #     #logging.error(f"[API] update_frame - 无效的帧数据: {str(e)}")
            #     return jsonify({"error": "无效的帧数据"}), 404

            self.video_streams[stream_id].update_frame(frame_data)
            #logging.info(f"[API] update_frame - 成功更新帧: {stream_id}")
            return jsonify({"msg": "帧已更新"}), 200
        except Exception as e:
            logging.error(f"[API Error] update_frame - 异常: {str(e)}")
            return jsonify({"error": f"服务器内部错误: {str(e)}"}), 500
    
    def robot_get_video_list(self):
        try:
            logging.info("[API] robot_get_video_list - 机器人获取视频列表请求")
            new_list = request.get_json()
            logging.debug(f"[API] robot_get_video_list - 请求数据: {new_list}")
            
            self.video_list = json.loads(new_list)
            logging.debug(f"[API] robot_get_video_list - 更新视频列表: {self.video_list}")
            
            self.video_timestamp = time.time()
            if not self.init_streams_flag:
                self.init_streams()
                self.init_streams_flag = True
            logging.info("[API] robot_get_video_list - 处理完成")
            return jsonify({}), 200
        except Exception as e:
            logging.error(f"[API Error] robot_get_video_list - 异常: {str(e)}")
            return jsonify({"error": f"服务器内部错误: {str(e)}"}), 500
    
    def robot_response(self):
        try:
            logging.info("[API] robot_response - 机器人响应")
            data = request.get_json()
            logging.debug(f"[API] robot_response - 响应数据: {data}")
            
            if data["cmd"] == "start_collection":
                self.response_start_collection = {
                    "timestamp": time.time(),
                    "msg": data["msg"]
                }
            elif data["cmd"] == "finish_collection":
                self.response_finish_collection = {
                    "timestamp": time.time(),
                    "msg": data["msg"],
                    "data": data["data"]
                }
            elif data["cmd"] == "discard_collection":
                self.response_discard_collection = {
                    "timestamp": time.time(),
                    "msg": data["msg"]
                }
            elif data["cmd"] == "submit_collection":
                self.response_submit_collection = {
                    "timestamp": time.time(),
                    "msg": data["msg"]
                }
            elif data["cmd"] == "start_replay":
                self.response_start_replay = {
                    "timestamp": time.time(),
                    "msg": data["msg"],
                    "data": data["data"]
                }
            logging.info("[API] robot_response - 响应处理完成")
            return jsonify({}), 200
        except Exception as e:
            logging.error(f"[API Error] robot_response - 异常: {str(e)}")
            return jsonify({"error": f"服务器内部错误: {str(e)}"}), 500
    
    def update_machine_information(self):
        try:
            logging.info("[API] update_machine_information - 机器人响应")
            data = request.get_json()
            logging.debug(f"[API] update_machine_information - 响应数据: {data}")
            self.machine_information = data
            self.machine_information_timestamp = time.time()
            return jsonify({}), 200
        except Exception as e:
            logging.error(f"[API Error] update_machine_information - 异常: {str(e)}")
            return jsonify({"error": f"服务器内部错误: {str(e)}"}), 500
    
    def get_task_steps(self):
        logging.info("[API] get_task_steps - 获取任务步骤请求")
        logging.debug(f"[API] get_task_steps - 返回数据: {self.task_steps}")
        return jsonify(self.task_steps), 200
    
    def run(self):
        logging.info("[Server] run - 启动服务器")
        self.upload_thread.start()
        self.start_periodic_update()
        self.socketio.run(self.app, host='0.0.0.0', port=self.port, debug=False)




if __name__ == '__main__':
    server = FlaskServer()
    server.run()
