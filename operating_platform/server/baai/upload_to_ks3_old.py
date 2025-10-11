import os
import json
import subprocess
from pathlib import Path
import requests
from collections import OrderedDict
import shutil
from typing import List, Dict, Union, Optional
from robot_data_uploader.collect_uploader import BaaiRobotDataUploader
from robot_data_uploader.config import UPLOAD_TARGET
import datetime
from utils import setup_from_yaml

class RobotDataProcessor:
    def __init__(self, fold_path: str = "/home/robot/dataset/", server_url: str = "http://localhost:8088"):
        """
        初始化机器人数据处理类
        
        Args:
            fold_path (str): 数据集基础路径
            server_url (str): 本地服务器URL
        """
        self.fold_path = fold_path
        self.server_url = server_url
        self.session = requests.Session()
        self.token = None
        
    def set_token(self, token: str):
        """设置KS3上传token"""
        self.token = token
        
    def local_server_request(self, api_url: str, data: Dict):
        """
        向本地服务器发送POST请求
        
        Args:
            api_url (str): API端点
            data (Dict): 要发送的数据
        """
        print(f"[INFO] 发送请求到服务器: {api_url}")
        print(f"[DEBUG] 请求数据: {data}")
        
        try:
            response = self.session.post(
                f"{self.server_url}/{api_url}",
                json=data
            )
            response.raise_for_status()
            print(f"[INFO] 服务器响应: {response.json()}")
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"[ERROR] 请求失败: {str(e)}")
            return None
    
    def local_server_task_id_request(self, task_id: Union[str, int]):
        """
        发送任务ID到服务器
        
        Args:
            task_id (Union[str, int]): 任务ID
        """
        data = {"task_id": str(task_id)}
        self.local_server_request('api/upload_task_id', data)
    def get_date_offset(self, days_offset):
        """获取指定偏移量的日期（格式：YYYYMMDD）"""
        target_date = datetime.datetime.now() - datetime.timedelta(days=days_offset)
        return target_date.strftime("%Y%m%d")

    def read_common_record_json(self, each_task_path: str) -> Dict:
        """
        读取common_record.json文件
        Args:
            each_task_path (str): 任务路径
            
        Returns:
            Dict: 包含task_id, task_name, machine_id的字典
        """
        each_common_record_path = os.path.join(each_task_path, 'meta', 'common_record.json')
        try:
            with open(each_common_record_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                print(f"[DEBUG] 读取common_record.json成功: {data}")
                return {
                    "task_id": data.get("task_id"),
                    "task_name": data.get("task_name"),
                    "machine_id": data.get("machine_id")
                }
        except Exception as e:
            print(f"[ERROR] 读取common_record.json失败: {str(e)}")
            return {}
    
    def read_common_record_json_status(self, each_task_path: str) -> int:
        """
        读取common_record.json中的上传状态
        Args:
            each_task_path (str): 任务路径
            
        Returns:
            int: 上传状态 (0: 未上传, 1: 上传成功, 2: 上传失败)
        """
        each_common_record_path = os.path.join(each_task_path, 'meta', 'common_record.json')
        try:
            with open(each_common_record_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                status = data.get('upload_status', 0)
                print(f"[DEBUG] 当前上传状态: {status}")
                return status
        except Exception as e:
            print(f"[ERROR] 读取上传状态失败: {str(e)}")
            return 0
    
    def read_opdata_path_json(self, each_task_path: str) -> List[int]:
        """
        读取op_dataid.jsonl文件
        
        Args:
            each_task_path (str): 任务路径
            
        Returns:
            List[int]: 数据ID列表
        """
        task_object_list = []
        each_opdata_path = os.path.join(each_task_path, "meta", "op_dataid.jsonl")
        
        try:
            with open(each_opdata_path, 'r', encoding='utf-8') as file:
                for line in file:
                    try:
                        json_object_data = json.loads(line.strip())
                        dataid = int(json_object_data['dataid'])
                        task_object_list.append(dataid)
                    except json.JSONDecodeError as e:
                        print(f"[WARNING] 解析JSON行失败: {line.strip()}, 错误: {e}")
            print(f"[DEBUG] 读取到 {len(task_object_list)} 个数据ID")
            return task_object_list
        except Exception as e:
            print(f"[ERROR] 读取op_dataid.jsonl失败: {str(e)}")
            return []
    
    def read_info_json(self, each_task_path: str) -> Optional[int]:
        """
        读取info.json文件获取FPS
        
        Args:
            each_task_path (str): 任务路径
            
        Returns:
            Optional[int]: FPS值
        """
        each_info_path = os.path.join(each_task_path, 'meta', 'info.json')
        try:
            with open(each_info_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                fps = data.get("fps")
                print(f"[DEBUG] 读取到FPS值: {fps}")
                return fps
        except Exception as e:
            print(f"[ERROR] 读取info.json失败: {str(e)}")
            return None
    
    def get_img_video_path(self, each_task_path: str, camera_images: str, camera_images_path: str) -> tuple:
        """
        获取图像和视频路径列表
        
        Args:
            each_task_path (str): 任务路径
            camera_images (str): 相机名称
            camera_images_path (str): 相机图像路径
            
        Returns:
            tuple: (img_path_list, video_path_list)
        """
        img_path_list = []
        video_path_list = []
        depth_video_path_list = []
        
        try:
            entries = os.listdir(camera_images_path)
            for episode_index in entries:
                img_path = os.path.join(camera_images_path, episode_index)
                video_name = episode_index + '.mp4'
                depth_video_name = episode_index + '.avi'
                video_path = os.path.join(each_task_path, 'videos', 'chunk-000',camera_images, video_name)
                depth_video_path = os.path.join(each_task_path, 'depth', 'chunk-000',camera_images, depth_video_name)
                img_path_list.append(img_path)
                video_path_list.append(video_path)
                depth_video_path_list.append(depth_video_path)
            
            if len(img_path_list) != len(video_path_list):
                print("[ERROR] 图像和视频路径数量不匹配")
                return [], []
                
            print(f"[DEBUG] 找到 {len(img_path_list)} 组图像和视频路径")
            return img_path_list, video_path_list,depth_video_path_list
        except Exception as e:
            print(f"[ERROR] 获取路径列表失败: {str(e)}")
            return [], []
    
    def encode_video_frames(self, imgs_dir: Union[Path, str], video_path: Union[Path, str], fps: int) -> bool:
        """
        编码普通视频帧
        
        Args:
            imgs_dir (Union[Path, str]): 图像目录
            video_path (Union[Path, str]): 输出视频路径
            fps (int): 帧率
            
        Returns:
            bool: 是否成功
        """
        print(f"[INFO] 开始编码普通视频: {video_path}")
        try:
            imgs_dir = Path(imgs_dir)
            video_path = Path(video_path)
            video_path.parent.mkdir(parents=True, exist_ok=True)

            ffmpeg_args = OrderedDict([
                ("-f", "image2"),
                ("-r", str(fps)),
                ("-i", str(imgs_dir / "frame_%06d.jpg")),
                ("-vcodec", "libx264"),
                ("-pix_fmt", "yuv420p"),
                ("-g", "5"),
                ("-crf", "18"),
                ("-loglevel", "error"),
            ])

            ffmpeg_cmd = ["ffmpeg"] + [item for pair in ffmpeg_args.items() for item in pair] + [str(video_path)]
            print(f"[DEBUG] 执行FFmpeg命令: {' '.join(ffmpeg_cmd)}")
            
            subprocess.run(ffmpeg_cmd, check=True, stdin=subprocess.DEVNULL)
            
            if not video_path.exists():
                raise OSError(f"视频文件未生成: {video_path}")
                
            print(f"[INFO] 普通视频编码成功: {video_path}")
            return True
        except Exception as e:
            print(f"[ERROR] 普通视频编码失败: {str(e)}")
            return False
        
    def encode_label_video_frames(self, imgs_dir: Union[Path, str], video_path: Union[Path, str], fps: int) -> bool:
        """
        编码普通视频帧
        
        Args:
            imgs_dir (Union[Path, str]): 图像目录
            video_path (Union[Path, str]): 输出视频路径
            fps (int): 帧率
            
        Returns:
            bool: 是否成功
        """
        print(f"[INFO] 开始编码普通视频: {video_path}")
        try:
            imgs_dir = Path(imgs_dir)
            video_path = Path(video_path)
            video_path.parent.mkdir(parents=True, exist_ok=True)
            #  ("-vcodec", "libx264"),

            ffmpeg_args = OrderedDict([
                ("-f", "image2"),
                ("-r", str(fps)),
                ("-i", str(imgs_dir / "frame_%06d.jpg")),
                ("-vcodec", "libx264"),
                ("-pix_fmt", "yuv420p"),
                ("-g", "20"),
                ("-crf", "23"),
                ("-loglevel", "error"),
            ])

            ffmpeg_cmd = ["ffmpeg"] + [item for pair in ffmpeg_args.items() for item in pair] + [str(video_path)]
            print(f"[DEBUG] 执行FFmpeg命令: {' '.join(ffmpeg_cmd)}")
            
            subprocess.run(ffmpeg_cmd, check=True, stdin=subprocess.DEVNULL)
            
            if not video_path.exists():
                raise OSError(f"视频文件未生成: {video_path}")
                
            print(f"[INFO] 普通视频编码成功: {video_path}")
            return True
        except Exception as e:
            print(f"[ERROR] 普通视频编码失败: {str(e)}")
            return False
    
    def encode_depth_video_frames(self, imgs_dir: Union[Path, str], video_path: Union[Path, str], fps: int) -> bool:
        """
        编码深度视频帧
        
        Args:
            imgs_dir (Union[Path, str]): 图像目录
            video_path (Union[Path, str]): 输出视频路径
            fps (int): 帧率
            
        Returns:
            bool: 是否成功
        """
        print(f"[INFO] 开始编码深度视频: {video_path}")
        try:
            imgs_dir = Path(imgs_dir)
            video_path = Path(video_path)
            video_path.parent.mkdir(parents=True, exist_ok=True)

            ffmpeg_args = [
                "ffmpeg",
                "-f", "image2",
                "-r", str(fps),
                "-i", str(imgs_dir / "frame_%06d.png"),
                "-vcodec", "ffv1",
                "-loglevel", "error",
                "-pix_fmt", "gray16le",
                "-y",
                str(video_path)
            ]
            
            print(f"[DEBUG] 执行FFmpeg命令: {' '.join(ffmpeg_args)}")
            subprocess.run(ffmpeg_args, check=True, stdin=subprocess.DEVNULL)
            
            if not video_path.exists():
                raise OSError(f"视频文件未生成: {video_path}")
                
            print(f"[INFO] 深度视频编码成功: {video_path}")
            return True
        except Exception as e:
            print(f"[ERROR] 深度视频编码失败: {str(e)}")
            return False
    
    def finish_upload_data(self, task_id: Union[str, int], task_data_id: List[int], status: str, expand: Optional[Dict] = None) -> Dict:
        """
        构建上传完成数据
        
        Args:
            task_id (Union[str, int]): 任务ID
            task_data_id (List[int]): 任务数据ID列表
            status (str): 状态 (SUCCESS/FAILED)
            expand (Optional[Dict]): 扩展信息
            
        Returns:
            Dict: 响应数据
        """
        response_data = {
            "task_id": int(task_id),
            "task_data_ids": task_data_id,
            "transfer_type": "local_to_ks3",
            "status": status,
            "expand": expand or '{}'
        }
        print(f"[DEBUG] 构建上传完成数据: {response_data}")
        return response_data
    
    def start_upload_data(self, task_id: Union[str, int], task_data_id: List[int], source_path: str, target_path: str) -> Dict:
        """
        构建开始上传数据
        
        Args:
            task_id (Union[str, int]): 任务ID
            task_data_id (List[int]): 任务数据ID列表
            source_path (str): 源路径
            target_path (str): 目标路径
            
        Returns:
            Dict: 响应数据
        """
        response_data = {
            "task_id": int(task_id),
            "task_data_ids": task_data_id,
            "source_path": source_path,
            "target_path": target_path,
            "transfer_type": "local_to_ks3"
        }
        print(f"[DEBUG] 构建开始上传数据: {response_data}")
        return response_data
    
    def upload_dir(self, directory: str, target_directory: str) -> bool:
        """
        上传目录到KS3
        
        Args:
            directory (str): 源目录
            target_directory (str): 目标目录
            
        Returns:
            bool: 是否成功
        """
        if not self.token:
            print("[ERROR] 未设置KS3上传token")
            return False
            
        print(f"[INFO] 开始上传目录: {directory} -> {target_directory}")
        try:
            
            uploader = BaaiRobotDataUploader(use_direct_auth=False)
            uploader.set_eai_token(eai_token=self.token)
            uploader.get_ks3_sts()
            uploader.set_max_worker(4)
            
            result = uploader.batch_upload(
                directory=directory,
                target_directory=target_directory,
                skip_exist=False,
                show_progress=False
            )
            
            print(f"[INFO] 目录上传完成: {directory}")
            return True
        except Exception as e:
            print(f"[ERROR] 目录上传失败: {str(e)}")
            return False
    
    @staticmethod
    def delete_directory(path: str):
        """
        删除目录
        
        Args:
            path (str): 要删除的目录路径
        """
        try:
            shutil.rmtree(path)
            print(f"[INFO] 成功删除目录: {path}")
        except OSError as e:
            print(f"[ERROR] 删除目录失败: {path}, 错误: {e.strerror}")
    
    @staticmethod
    def modify_json(path: str, upload_status: int):
        """
        修改JSON文件中的上传状态
        
        Args:
            path (str): JSON文件路径
            upload_status (int): 新的上传状态
        """
        try:
            with open(path, 'r', encoding='utf-8') as file:
                existing_data = json.load(file)
            
            existing_data['upload_status'] = upload_status
            
            with open(path, 'w', encoding='utf-8') as file:
                json.dump(existing_data, file, indent=4, ensure_ascii=False)
                
            print(f"[INFO] 成功修改JSON文件: {path}, 新状态: {upload_status}")
        except Exception as e:
            print(f"[ERROR] 修改JSON文件失败: {path}, 错误: {str(e)}")

    @staticmethod
    def modify_feature_dtypes(local_nas_info_path, target_fields, new_dtype):
        """
        修改 metadata 中 features 的 dtype（仅修改 target_fields 中的字段）
        
        Args:
            each_task_path (str): 任务路径（包含 'meta/info.json'）
            target_fields (list): 需要修改的字段名列表（如 ["timestamp", "frame_index"]）
            new_dtype (str): 新的 dtype（如 "float64"）
        
        Returns:
            bool: 修改成功返回 True，失败返回 False
        """
        
        try:
            # 1. 读取 JSON 文件
            with open(local_nas_info_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
            
            # 2. 检查并修改 dtype
            if "features" not in metadata:
                raise ValueError("Invalid metadata: missing 'features' key.")
            
            features = metadata["features"]
            modified_fields = []
            matches = []
            matches = [field for field in target_fields if "depth" in field]  # 子字符串匹配
            if matches:
                metadata["video_depth_path"] = "depth/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.avi"

            metadata["total_videos"] = int(metadata["total_episodes"]*len(target_fields))
            metadata["video_path"] = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
            metadata["image_path"] = None
            for field_name, field_info in features.items():
                if field_name in target_fields:
                    field_info["dtype"] = new_dtype
                    modified_fields.append(field_name)
            
            if not modified_fields:
                print(f"[WARNING] No target fields found in {local_nas_info_path}. Available fields: {list(features.keys())}")
                return False
            
            # 3. 写回 JSON 文件
            with open(local_nas_info_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=4, ensure_ascii=False)
            
            print(f"[INFO] Successfully modified dtype for {modified_fields} in {local_nas_info_path}")
            return True
        
        except FileNotFoundError:
            print(f"[ERROR] File not found: {local_nas_info_path}")
        except json.JSONDecodeError:
            print(f"[ERROR] Invalid JSON format in {local_nas_info_path}")
        except Exception as e:
            print(f"[ERROR] Failed to modify {local_nas_info_path}: {str(e)}")
        
        return False
    
    def process_date_data(self, date_data: str, task_id_list):
        """
        处理指定日期的数据
        
        Args:
            date_data (str): 日期字符串 (格式: YYYY-MM-DD)
        """
        print(f"\n[INFO] 开始处理日期数据: {date_data}")
        config_dict = setup_from_yaml()
        if config_dict["device_server_type"] == "release":
            local_name = 'user'
        else:
            local_name = 'dev'
        directory_path = os.path.join(self.fold_path, date_data, local_name)
        
        if not os.path.exists(directory_path):
            print(f"[WARNING] 数据路径不存在: {directory_path}")
            return
            
        entries = os.listdir(directory_path)
        subdirectories = [entry for entry in entries if os.path.isdir(os.path.join(directory_path, entry))]
        
        for task_data_name in subdirectories:
            print(f"\n[INFO] 处理任务: {task_data_name}")
            ffmpeg_encode_flag = True
            each_task_path = os.path.join(directory_path, task_data_name)
            
            # 检查是否有images目录
            entries_1 = os.listdir(each_task_path)
            subdirectories_1 = [entry for entry in entries_1 if os.path.isdir(os.path.join(each_task_path, entry))]
            task_info = self.read_common_record_json(each_task_path)
            if not task_info:
                continue
                
            task_id = task_info['task_id']
            if task_id_list:
                if int(task_id) not in task_id_list:
                    continue
            machine_id = task_info['machine_id']
            task_data_id = self.read_opdata_path_json(each_task_path)
            if not task_data_id:
                continue
            
            if 'images' in subdirectories_1:
                # 处理图像和视频
                try:
                    
                    fps = self.read_info_json(each_task_path)
                    if fps is None:
                        continue
                        
                    # 通知服务器开始处理任务
                    self.local_server_task_id_request(task_id)
                    
                    # 处理每个相机的图像
                    each_images_path = os.path.join(each_task_path, 'images')
                    entries_2 = os.listdir(each_images_path)
                    
                    for camera_images in entries_2:
                        camera_images_path = os.path.join(each_images_path, camera_images)
                        if os.path.isdir(camera_images_path):
                            img_list, video_list,depth_video_path_list = self.get_img_video_path(each_task_path, camera_images, camera_images_path)
                            
                            if 'depth' in camera_images:
                                # 处理深度图像
                                if img_list:
                                    for img_path, video_path in zip(img_list, depth_video_path_list):
                                        print(f"[INFO] 处理深度图像: {img_path} -> {video_path}")
                                        if not self.encode_depth_video_frames(img_path, video_path, fps):
                                            ffmpeg_encode_flag = False
                            else:
                                # 处理普通图像
                                if img_list:
                                    for img_path, video_path in zip(img_list, video_list):
                                        print(f"[INFO] 处理普通图像mp4: {img_path} -> {video_path}")
                                        if not self.encode_video_frames(img_path, video_path, fps):
                                            ffmpeg_encode_flag = False
                                            
                    each_info_path = os.path.join(each_task_path, 'meta', 'info.json')
                    self.modify_feature_dtypes(each_info_path,entries_2,'video')  
                except Exception as e:
                    print(f"[ERROR] 处理任务 {task_data_name} 失败: {str(e)}")
                    ffmpeg_encode_flag = False
            else:
                status = self.read_common_record_json_status(each_task_path)
                if status == 1:
                    # 没有images目录，检查上传状态
                    if date_data == self.get_date_offset(9):
                        # 如果是前天数据且已上传成功，则删除
                        # 这里假设date_data是前天日期，实际逻辑可能需要调整
                        self.delete_directory(each_task_path)
                    continue
                
            # 上传处理后的数据
            if ffmpeg_encode_flag:
                if 'images' in subdirectories_1:
                    self.delete_directory(os.path.join(each_task_path, 'images'))
                
                target_path = os.path.join(UPLOAD_TARGET,'collect', task_data_name, machine_id, date_data)
                target_path_sdk = os.path.join('collect', task_data_name, machine_id, date_data)
                
                # 通知服务器开始上传
                start_data = self.start_upload_data(task_id, task_data_id, each_task_path, target_path)
                self.local_server_request('api/upload_start_ks3', start_data)
                
                # 执行上传
                if self.upload_dir(each_task_path, target_path_sdk):
                    # 上传成功
                    finish_data = self.finish_upload_data(task_id, task_data_id, 'SUCCESS')
                    self.local_server_request('api/upload_finish_ks3', finish_data)
                    self.modify_json(os.path.join(each_task_path, 'meta', 'common_record.json'), 1)
                else:
                    # 上传失败
                    finish_data = self.finish_upload_data(
                        task_id, task_data_id, 'FAILED', 
                        '{"ks3_failed_msg": "网络通信错误"}'
                    )
                    self.local_server_request('api/upload_finish_ks3', finish_data)
                    self.modify_json(os.path.join(each_task_path, 'meta', 'common_record.json'), 2)
            
    
    def encode_and_upload(self, token: str, task_id_list):
        """
        主处理流程：编码视频并上传
        
        Args:
            token (str): KS3上传token
        """
        self.set_token(token)
        
        date_functions = []
        for i in range(10):
            date_data = self.get_date_offset(i)  # i=0（今天）到 i=6（6 天前）
            date_functions.append(("数据", date_data))  # 每个条目绑定不同的日期
 
        
        for date_name, date_func in date_functions:
            print(f"\n[INFO] 开始处理 {date_name}")
            date_data = date_func
            self.process_date_data(date_data, task_id_list)
        # 通知服务器任务处理完成
        self.local_server_task_id_request(0)
