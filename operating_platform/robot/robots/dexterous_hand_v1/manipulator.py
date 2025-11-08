import zmq  
import threading  
import time  
import json
import numpy as np  
import torch  
import cv2  
from operating_platform.robot.robots.configs import DexterousHandRobotConfig
from operating_platform.robot.robots.com_configs.cameras import CameraConfig, OpenCVCameraConfig ,IntelRealSenseCameraConfig
from operating_platform.robot.robots.camera import Camera
  
pose_ipc_address = "ipc:///tmp/dr-robot-dexterous-hand-pose"   
pose_context = zmq.Context()  
pose_socket = pose_context.socket(zmq.PAIR)  
pose_socket.connect(pose_ipc_address)  
pose_socket.setsockopt(zmq.RCVTIMEO, 300)  

image_ipc_address = "ipc:///tmp/dr-robot-dexterous-hand-image"   
image_context = zmq.Context()  
image_socket = image_context.socket(zmq.PAIR)  
image_socket.connect(image_ipc_address)  
image_socket.setsockopt(zmq.RCVTIMEO, 300)  
  
recv_wrist_data = {}  
recv_finger_data = {}  
recv_head_data = {}  
recv_images = {}
recv_full_skeleton = {}  
lock = threading.Lock()  
  
def pose_recv_server():
    while True:    
        try:   
            
            message_parts = pose_socket.recv_multipart()  
            
            if len(message_parts) < 1:    
                continue    
    
            event_id = message_parts[0].decode('utf-8')  
            buffer_bytes = message_parts[1]  
            # print(event_id)
            # print(len(buffer_bytes))
            # print(1)
            # 检查是否有第三个部分（metadata）  
            metadata = {}  
            if len(message_parts) >= 3:  
                try:  
                    metadata = json.loads(message_parts[2].decode('utf-8'))  
                except:  
                    metadata = {}  
    
            if 'wrist_left' in event_id:    
                pose_left_data = np.frombuffer(buffer_bytes, dtype=np.float32) 
                # print(pose_left_data)
                with lock:    
                    recv_wrist_data[event_id] = pose_left_data
 
            elif 'wrist_right' in event_id:    
                pose_right_data = np.frombuffer(buffer_bytes, dtype=np.float32)  
                with lock:    
                    recv_wrist_data[event_id] = pose_right_data   
    
            elif 'finger_left' in event_id:    
                finger_left_data = np.frombuffer(buffer_bytes, dtype=np.float32)    
                with lock:    
                    recv_finger_data[event_id] = finger_left_data  
            elif 'finger_right' in event_id:    
                finger_right_data = np.frombuffer(buffer_bytes, dtype=np.float32)  
                with lock:    
                    recv_finger_data[event_id] = finger_right_data   
    
            elif 'head_pose' in event_id:    
                head_data = np.frombuffer(buffer_bytes, dtype=np.float32)    
                with lock:    
                    recv_head_data[event_id] = head_data

            elif 'left_full_skeleton' in event_id:    
                left_full_skeleton = np.frombuffer(buffer_bytes, dtype=np.float32)    
                with lock:    
                    recv_full_skeleton[event_id] = left_full_skeleton
            
            elif 'right_full_skeleton' in event_id:    
                right_full_skeleton = np.frombuffer(buffer_bytes, dtype=np.float32)    
                with lock:    
                    recv_full_skeleton[event_id] = right_full_skeleton

        except zmq.Again:    
            continue    
        except Exception as e:    
            print("recv error:", e)    
            break

def image_recv_server():
    while True:    
        try:   
            message_parts = image_socket.recv_multipart()  
            if len(message_parts) < 1:    
                continue    
    
            event_id = message_parts[0].decode('utf-8')  
            buffer_bytes = message_parts[1] 
              
            metadata = {}  
            if len(message_parts) >= 3:  
                try:  
                    metadata = json.loads(message_parts[2].decode('utf-8'))  
                except:  
                    metadata = {}  
    

            if 'image' in event_id:
                # 解码图像
                if 'depth' not in event_id:
                    img_array = np.frombuffer(buffer_bytes, dtype=np.uint8)
                    frame = img_array.reshape((480, 640, 3))  # 已经是 RGB 格式 
                    if frame is not None:
                        with lock:
                            #print(f"Received event_id = {event_id}")
                            recv_images[event_id] = frame
                elif 'depth' in event_id: 
                    print(len(buffer_bytes))
                    depth_array = np.frombuffer(buffer_bytes, dtype=np.uint16)
                    depth_frame = depth_array.reshape((480, 640))  # 已经是 RGB 格式
                    if depth_frame is not None:
                        # 排除无效值（可选，根据传感器特性，例如0或65535可能是无效深度）
                        valid_mask = (depth_frame > 0) & (depth_frame < 65535)
                        if np.any(valid_mask):
                            # 仅对有效区域归一化，避免无效值干扰
                            valid_depth = depth_frame[valid_mask]
                            min_val = valid_depth.min()
                            max_val = valid_depth.max()
                        else:
                            # 若全无效，强制范围为0~65535
                            min_val, max_val = 0, 65535

                        depth_16bit = cv2.normalize(
                        depth_frame,
                        None,
                        alpha=0,
                        beta=65535,
                        norm_type=cv2.NORM_MINMAX,
                        dtype=cv2.CV_16U  # 转换为16位无符号整数
                        )
                        rgb_depth = cv2.cvtColor(depth_16bit, cv2.COLOR_GRAY2RGB)  # 扩展维度到(480,640,3)以实现保存
                        # 存储归一化后的图像用于显示
                        with lock:
                            # print(f"Received event_id = {event_id}")
                            recv_images[event_id] = rgb_depth

                           
        except zmq.Again:    
            continue    
        except Exception as e:    
            print("recv error:", e)    
            break



class OpenCVCamera:
    def __init__(self, config: OpenCVCameraConfig):
        self.config = config
        self.camera_index = config.camera_index
        self.port = None
    
        
        self.capture_width = config.width
        self.capture_height = config.height

        
        if config.rotation in [-90, 90]:
            self.width = config.height
            self.height = config.width
        else:
            self.width = config.width
            self.height = config.height

        self.fps = config.fps
        self.channels = config.channels
        self.color_mode = config.color_mode
        self.mock = config.mock

        self.camera = None
        self.is_connected = False
        self.thread = None
        self.stop_event = None
        self.color_image = None
        self.logs = {}

class IntelRealSenseCamera:  
    def __init__(self, config: IntelRealSenseCameraConfig):  
        self.config = config  
        self.name = config.name  
        self.serial_number = config.serial_number  
        self.port = None  
    
        self.capture_width = config.width  
        self.capture_height = config.height  
    
        if config.rotation in [-90, 90]:  
            self.width = config.height  
            self.height = config.width  
        else:  
            self.width = config.width  
            self.height = config.height  
  
        self.fps = config.fps  
        self.channels = config.channels  
        self.color_mode = config.color_mode  
        self.use_depth = config.use_depth  
        self.force_hardware_reset = config.force_hardware_reset  
        self.mock = config.mock  
  
        self.camera = None  
        self.pipeline = None    
        self.is_connected = False  
        self.thread = None  
        self.stop_event = None  
        self.color_image = None  
        self.depth_image = None    
        self.logs = {}

def make_cameras_from_configs(camera_configs: dict[str, CameraConfig]) -> list[Camera]:
    cameras = {}

    for key, cfg in camera_configs.items():
        if cfg.type == "opencv":
            cameras[key] = OpenCVCamera(cfg)
        elif cfg.type == "intelrealsense":  
            cameras[key] = IntelRealSenseCamera(cfg)
        else:
            raise ValueError(f"The camera type '{cfg.type}' is not valid.")
    return cameras

class DexterousHandManipulator:  
    def __init__(self, config: DexterousHandRobotConfig):  
        self.config = config  
        self.robot_type = self.config.type  
        
        self.use_videos = self.config.use_videos
        
        self.microphones = self.config.microphones
        
        self.cameras =  make_cameras_from_configs(self.config.cameras)
        
        self.follower_arms = {} 
        self.follower_arms["left_wrist"] = self.config.left_wrist_tracker.motors  
        self.follower_arms["right_wrist"] = self.config.right_wrist_tracker.motors  
        self.follower_arms["head"] = self.config.head_tracker.motors

        self.finger_sensors = {}  
        self.finger_sensors["left_finger"] = self.config.left_finger_sensors.motors  
        self.finger_sensors["right_finger"] = self.config.right_finger_sensors.motors 
        
        self.full_skeleton = {}  
        self.full_skeleton["left_skeleton"] = self.config.left_full_skeleton.motors  
        self.full_skeleton["right_skeleton"] = self.config.right_full_skeleton.motors

        pose_recv_thread = threading.Thread(target=pose_recv_server, daemon=True)  
        pose_recv_thread.start()   

        image_recv_thread = threading.Thread(target=image_recv_server, daemon=True)  
        image_recv_thread.start()  
        
        self.is_connected = False 
        self.logs = {}
        self.frame_counter = 0

    def get_motor_names(self, motor_groups: dict[str, dict]) -> list:   
        return [f"{group}_{motor}" for group, motors in motor_groups.items() for motor in motors]    
  
    @property    
    def camera_features(self) -> dict:   
        cam_ft = {}  
        for cam_key, cam in self.cameras.items():  
            key = f"observation.images.{cam_key}"  
            cam_ft[key] = {  
                "shape": (cam.height, cam.width, cam.channels),  
                "names": ["height", "width", "channels"],  
                "info": None,  
            }  
        return cam_ft 
  
    @property    
    def motor_features(self) -> dict:  
        all_motor_groups = {**self.follower_arms, **self.finger_sensors, **self.full_skeleton}  
        action_names = self.get_motor_names(all_motor_groups)  
        state_names = self.get_motor_names(all_motor_groups)  
        
        return {  
            "action": {    
                "dtype": "float32",     
                "shape": (len(action_names),),    
                "names": action_names,    
            },  
            "observation.state": {    
                "dtype": "float32",    
                "shape": (len(state_names),),    
                "names": state_names,    
            },
            "observation.state.full_skeleton": {  
                "dtype": "float32",  
                "shape": (150,),  
                "names": ["left_full_skeleton", "right_full_skeleton"],  
            },

            "observation.state.wrist": {  
                "dtype": "float32",  
                "shape": (14,),  
                "names": ["left_wrist_pose", "right_wrist_pose"],  
            },  
    
            "observation.state.finger": {  
                "dtype": "float32",   
                "shape": (30,),  
                "names": ["left_finger_joints", "right_finger_joints"],  
            },  

            "observation.state.head": {  
                "dtype": "float32",  
                "shape": (7,),  
                "names": ["head_pose"],  
            }
        } 
  
    @property  
    def features(self):  
        return {**self.motor_features, **self.camera_features}  
  
    def connect(self):  
        timeout = 10 
        start_time = time.perf_counter()  
  
        while True:  
            wrist_ready = all(key in recv_wrist_data for key in ['wrist_left', 'wrist_right'])  
            finger_ready = all(key in recv_finger_data for key in ['finger_left', 'finger_right'])  
            head_ready = 'head_pose' in recv_head_data  
            image_ready = 'image_top' in recv_images  
            print(wrist_ready,finger_ready,image_ready,head_ready)
            if wrist_ready and finger_ready and head_ready and image_ready:  
                break  
  
            if time.perf_counter() - start_time > timeout:  
                raise TimeoutError("设备连接超时")  
  
            time.sleep(0.1)  
  
        self.is_connected = True  
  
    
    def teleop_step(self, record_data=False):  
        if not self.is_connected:  
            raise Exception("设备未连接")  
    
        if not record_data:  
            return  
    
        with lock:  
            # 收集状态数据  
            state_data = [] 
            wrist_data = []
            finger_data = []
            head_data = []
            full_skeleton = []     

            if 'left_full_skeleton' in recv_full_skeleton:  
                state_data.extend(recv_full_skeleton['left_full_skeleton']) 
                full_skeleton.extend(recv_full_skeleton["left_full_skeleton"])  
            if 'right_full_skeleton' in recv_full_skeleton:    
                state_data.extend(recv_full_skeleton['right_full_skeleton'])
                full_skeleton.extend(recv_full_skeleton['right_full_skeleton'])  
            
            if 'wrist_left' in recv_wrist_data:  
                state_data.extend(recv_wrist_data['wrist_left'])  
                wrist_data.extend(recv_wrist_data["wrist_left"])  
            if 'wrist_right' in recv_wrist_data:    
                state_data.extend(recv_wrist_data['wrist_right'])
                wrist_data.extend(recv_wrist_data['wrist_right'])    
            
            if 'finger_left' in recv_finger_data:    
                state_data.extend(recv_finger_data['finger_left'])
                finger_data.extend(recv_finger_data["finger_left"])    
            if 'finger_right' in recv_finger_data:    
                state_data.extend(recv_finger_data['finger_right'])
                finger_data.extend(recv_finger_data["finger_right"])    
            
            if 'head_pose' in recv_head_data:    
                state_data.extend(recv_head_data['head_pose'])
                head_data.extend(recv_head_data["head_pose"])  

            full_skeleton_tensor=torch.from_numpy(np.array(full_skeleton,dtype=np.float32))
            wrist_tensor=torch.from_numpy(np.array(wrist_data, dtype=np.float32))
            finger_tensor=torch.from_numpy(np.array(finger_data, dtype=np.float32))
            head_tensor=torch.from_numpy(np.array(head_data, dtype=np.float32))
            state = torch.from_numpy(np.array(state_data, dtype=np.float32))  
            
            # 正确获取图像数据 - 遍历所有相机  
            images = {}  
            for name in self.cameras:  
                if name in recv_images:  
                    images[name] = torch.from_numpy(recv_images[name])  
                else:  
                    images[name] = torch.from_numpy(np.zeros((480, 640, 3), dtype=np.uint8))  
    
        obs_dict = {
            "observation.state.full_skeleton": full_skeleton_tensor,
            "observation.state.wrist": wrist_tensor,  
            "observation.state.finger": finger_tensor,   
            "observation.state.head": head_tensor,  
            "observation.state": state,  
        }  
        
        # 添加图像数据  
        for name in self.cameras:  
            obs_dict[f"observation.images.{name}"] = images[name]  
        
        action_dict = {  
            "action": state  
        }  
    
        return obs_dict, action_dict
  
    def disconnect(self):  
        self.is_connected = False
