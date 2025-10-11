# coding=utf-8

import datetime
import os
import time
import random
import json
import shutil
import re
from nas_sdk import NASAuthenticator
import subprocess
from typing import List, Dict, Union, Optional
import requests
from pathlib import Path
from collections import OrderedDict
from utils import setup_from_yaml


class DataUploader:
    def __init__(self):
        self.coding = "utf-8"
        self.nas_auth = NASAuthenticator()
        self.session = requests.Session()
        
        # 加载配置
        self.config_dict = setup_from_yaml()
        self.server_url = self.config_dict['device_server_ip']
        self.fold_path1 = self.config_dict['device_data_path']
        
        # 根据环境类型设置路径
        if self.config_dict['device_server_type'] == 'release':
            self.nas_path = self.config_dict['nas_cache_path_release']
            self.nas_data_path = self.config_dict['nas_data_path_release']
            self.local_name = 'user'
        if self.config_dict['device_server_type'] == 'dev':
            self.nas_path = self.config_dict['nas_cache_path_dev']
            self.nas_data_path = self.config_dict['nas_data_path_dev']
            self.local_name = 'dev'
        if self.config_dict['device_server_type'] == 'demo':
            self.nas_path = self.config_dict['nas_cache_path_demo']
            self.nas_data_path = self.config_dict['nas_data_path_demo']
            self.local_name = 'dev'

        self.robot_type = self.config_dict['robot_type']

    @staticmethod
    def get_date_offset(days_offset):
        """获取指定偏移量的日期（格式：YYYYMMDD）"""
        target_date = datetime.datetime.now() - datetime.timedelta(days=days_offset)
        return target_date.strftime("%Y%m%d")

    @staticmethod
    def copy_files(source_paths, destination_paths):
        """
        将源文件路径列表中的文件复制到目标文件路径列表中。
        
        :param source_paths: 源文件路径列表
        :param destination_paths: 目标文件路径列表
        :raises ValueError: 如果源文件路径和目标文件路径数量不匹配
        :raises FileNotFoundError: 如果源文件不存在
        """
        if len(source_paths) != len(destination_paths):
            raise ValueError("源文件路径和目标文件路径数量不匹配")

        for src, dest in zip(source_paths, destination_paths):
            try:
                # 确保目标目录存在
                os.makedirs(os.path.dirname(dest), exist_ok=True)
                shutil.copy2(src, dest)
                print(f"成功复制文件: {src} -> {dest}")
            except FileNotFoundError:
                print(f"源文件不存在: {src}")
            except Exception as e:
                print(f"复制文件时发生错误: {src} -> {dest}, 错误信息: {e}")

    @staticmethod
    def add_random_milliseconds():
        # 获取当前时间戳（毫秒）
        current_timestamp = int(time.time() * 1000)
        
        # 生成一个随机的毫秒数（例如，1到1000毫秒之间）
        random_milliseconds = random.randint(1, 1000)
        
        # 将毫秒转换为秒，因为 time.sleep() 使用秒
        wait_time = random_milliseconds / 1000.0
        
        # 等待指定的时间
        time.sleep(wait_time)
        # 计算新的时间戳
        new_timestamp = current_timestamp + random_milliseconds
        return new_timestamp

    @staticmethod
    def get_today_date():
        # 获取当前日期和时间
        today = datetime.datetime.now()
        
        # 格式化日期为字符串，格式为 "YYYY-MM-DD"
        date_string = today.strftime("%Y%m%d")
        return date_string

    @staticmethod
    def get_yesterday_date():
        """返回昨天的日期，格式：YYYYMMDD"""
        yesterday = datetime.datetime.now() - datetime.timedelta(days=1)
        return yesterday.strftime("%Y%m%d")

    @staticmethod
    def get_day_before_yesterday_date():
        """返回前天的日期，格式：YYYYMMDD"""
        day_before_yesterday = datetime.datetime.now() - datetime.timedelta(days=2)
        return day_before_yesterday.strftime("%Y%m%d")

    @staticmethod
    def get_today_time():
        # 获取当前日期和时间
        today = datetime.datetime.now()
        
        # 格式化日期为字符串，格式为 "YYYY-MM-DD"
        date_string = today.strftime("%Y%m%d%H%M%S")
        return date_string

    @staticmethod
    def extract_number(episode_name):
        # 使用正则表达式提取数字部分
        match = re.match(r'episode_(\d+)', episode_name)
        if match:
            return int(match.group(1))
        return 0  # 如果没有匹配，返回一个默认值

    @staticmethod
    def get_path_after_last_data(file_path, name):
        """
        获取路径中最后一个名为 'data' 的子目录之后的路径。
        
        :param file_path: 文件的完整路径
        :return: 最后一个 'data' 目录之后的路径
        """
        parts = file_path.split(os.sep)
        last_data_index = -1

        # 查找最后一个 'data' 目录的索引
        for i, part in enumerate(parts):
            if part == name:
                last_data_index = i

        # 如果找到了 'data' 目录，则返回其后的路径
        if last_data_index != -1 and last_data_index + 1 < len(parts):
            return os.path.join(*parts[last_data_index + 1:])
        else:
            return file_path  # 如果没有找到 'data' 目录，返回完整路径

    @staticmethod
    def delete_directory(directory_path):
        """
        删除指定目录及其所有内容。
        
        :param directory_path: 要删除的目录路径
        """
        try:
            # 检查目录是否存在
            if os.path.exists(directory_path):
                # 删除目录及其所有子目录和文件
                shutil.rmtree(directory_path)
                print(f"目录 '{directory_path}' 已成功删除。")
            else:
                print(f"目录 '{directory_path}' 不存在。")
        except Exception as e:
            print(f"删除目录时发生错误: {e}")

    @staticmethod
    def delete_file(file_list):
        for file_path in file_list:
            # 检查文件是否存在
            if os.path.exists(file_path):
                try:
                    # 删除文件
                    os.remove(file_path)
                    print(f"文件 {file_path} 已成功删除")
                except OSError as e:
                    print(f"删除文件时出错: {e}")
            else:
                print(f"文件 {file_path} 不存在")

    @staticmethod
    def increment_episode_number(filename, n=0):
        """
        增加文件名中数字部分的数值，并保持位数不变。
        
        :param filename: 原始文件名
        :param n: 要增加的数值，默认为 1
        :return: 修改后的文件名
        """
        # 使用正则表达式查找文件名中的数字部分
        match = re.search(r'(\d+)', filename)
        if n == 0:
           return filename
        if match:
            # 提取数字部分
            number_str = match.group(1)
            # 将数字部分转换为整数并增加 n
            new_number = int(number_str) + n
            # 计算数字部分的位数
            num_digits = len(number_str)
            # 格式化新的数字部分，保持位数不变
            new_number_str = f"{new_number:0{num_digits}d}"
            # 替换文件名中的数字部分
            new_filename = re.sub(r'\d+', new_number_str, filename, count=1)
            return new_filename
        else:
            # 如果没有找到数字部分，返回原始文件名
            return filename

    def get_nth_file_in_subdirectories(self, fold_path, m, n=0, task=0, middle_name=None, meta=0, nas_data_path=None):
        """
        获取指定目录或其子目录中的第 n 个文件。
        
        :param fold_path: 目录路径
        :param m: 要获取的文件索引（从 0 开始）
        :return: 找到的第 n 个文件的路径，如果没有找到则返回 None
        """
        # 检查目录是否存在
        if not os.path.isdir(fold_path):
            print(f"目录不存在: {fold_path}")
            return None

        # 获取目录中的所有条目
        pre_entries = os.listdir(fold_path)

        has_directory = False
        has_file = False
        file_list = []
        nas_file_list = []

        if meta:
            for subdir in pre_entries:
                # if subdir == "common_record.json":
                #     continue
                entry = os.path.join(fold_path, subdir)
                file_list.append(entry)
                nas_file = self.get_path_after_last_data(entry, task)
                nas_final_path = os.path.join(nas_data_path, middle_name, nas_file)
                nas_file_list.append(nas_final_path)
            return file_list, nas_file_list

        for entry in pre_entries:
            entry_path = os.path.join(fold_path, entry)
            if os.path.isdir(entry_path):
                has_directory = True
            elif os.path.isfile(entry_path):
                has_file = True
            break

        if has_file:
            entries = sorted(pre_entries, key=self.extract_number)
            file_path = os.path.join(fold_path, entries[m])
            nas_file_path = os.path.join(fold_path, self.increment_episode_number(entries[m], n))
            nas_file = self.get_path_after_last_data(nas_file_path, task)
            nas_final_path = os.path.join(nas_data_path, middle_name, nas_file)
            return file_path, nas_final_path
        elif has_directory:
            # 遍历子目录，查找第 n 个文件
            for subdir in pre_entries:
                pre_entry = os.listdir(os.path.join(fold_path, subdir))
                entry = sorted(pre_entry, key=self.extract_number)
                file_list.append(os.path.join(fold_path, subdir, entry[m]))
                nas_file_path = os.path.join(fold_path, subdir, self.increment_episode_number(entry[m], n))
                nas_file = self.get_path_after_last_data(nas_file_path, task)
                nas_final_path = os.path.join(nas_data_path, middle_name, nas_file)
                nas_file_list.append(nas_final_path)
            return file_list, nas_file_list

    @staticmethod
    def change_number(task_path, nas_meta_file_path, n, m, end_line):
        json_objects = []
        with open(task_path, 'r', encoding='utf-8') as file:
            # 跳过前 m 行
            for _ in range(m):
                next(file, None)  # 使用 next() 跳过 m 行

            # 处理剩余的行
            # 遍历剩余行，筛选指定范围
            for current_line, line in enumerate(file):
                # 检查是否在目标行范围内
                if current_line + m >= end_line:
                    break
                try:
                    # 去除行末的换行符，并解析为 JSON 对象
                    json_object = json.loads(line.strip())
                    json_object_copy = json_object.copy()
                    json_object_copy["episode_index"] = int(json_object_copy["episode_index"]) + n
                    json_objects.append(json_object_copy)
                except json.JSONDecodeError as e:
                    print(f"解析 JSON 失败，行内容: {line.strip()}, 错误信息: {e}")

        # 将 json_objects 中的对象追加到文件
        with open(nas_meta_file_path, 'a', encoding='utf-8') as file:
            for json_obj in json_objects:
                file.write(json.dumps(json_obj, ensure_ascii=False) + '\n')

    @staticmethod
    def change_number_info(task_path, nas_meta_file_path, frame_path, m, n):
        frames = 0
        with open(frame_path, 'r', encoding='utf-8') as file:
            # 跳过前 m 行
            for _ in range(m):
                next(file, None)  # 使用 next() 跳过 m 行

            # 处理剩余的行
            for current_line, line in enumerate(file):
                # 检查是否在目标行范围内
                if current_line + m >= n:
                    break
                try:
                    # 去除行末的换行符，并解析为 JSON 对象
                    json_object = json.loads(line.strip())
                    frames += json_object["length"]
                except json.JSONDecodeError as e:
                    print(f"解析 JSON 失败，行内容: {line.strip()}, 错误信息: {e}")

        with open(task_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            total_videos = data["total_videos"]

        with open(nas_meta_file_path, "r", encoding="utf-8") as f:
            data_nas = json.load(f)
            data_nas["total_episodes"] += (n - m)
            data_nas["total_frames"] += frames
            data_nas["total_videos"] += int((total_videos / n) * (n - m))
            # data_nas["video_path"] = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.avi"
            # data_nas["image_path"] = None
        # 将更新后的数据写回 nas_meta_file_path
        with open(nas_meta_file_path, "w", encoding="utf-8") as f:
            json.dump(data_nas, f, ensure_ascii=False, indent=4)

    @staticmethod
    def create_json(path, data):
        # 创建并写入 JSON 文件
        with open(path, 'w') as file:
            json.dump(data, file, indent=4)

    @staticmethod
    def modify_json(path, episodes_id):
        # 读取现有 JSON 文件
        try:
            with open(path, 'r', encoding='utf-8') as file:
                existing_data = json.load(file)
        except FileNotFoundError:
            existing_data = {}
        
        # 修改数据
        existing_data['last_upload_id'] = episodes_id
        existing_data['last_upload_time'] = DataUploader.get_today_time()
        
        # 写回文件
        with open(path, 'w', encoding='utf-8') as file:
            json.dump(existing_data, file, indent=4, ensure_ascii=False)

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

    @staticmethod
    def grant_recursive_rw_permission(directory, username, pd):
        """
        递归授予用户对目录的读写权限（直接使用 chmod + sudo）
        :param directory: 目标目录路径（如 '/path/to/dir'）
        :param username: 要授权的用户名
        """
        try:
            # 1. 递归修改所有者（确保用户有权操作文件）
            subprocess.run(
                f"echo {pd} | sudo -S chown -R {username}:{username} {directory}",
                shell=True,
                check=True
            )

            # 2. 递归授予读写权限（755：所有者rwx，其他用户r-x）
            subprocess.run(
                f"echo {pd} | sudo -S chmod -R 755 {directory}",
                shell=True,
                check=True
            )

            print(f"成功授权用户 '{username}' 对目录 '{directory}' 的读写权限！")
        except subprocess.CalledProcessError as e:
            print(f"命令执行失败: {e}")
        except Exception as e:
            print(f"发生错误: {e}")

    @staticmethod
    def read_info_json(each_task_path: str) -> Optional[int]:
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

    @staticmethod
    def encode_video_frames(imgs_dir: Union[Path, str], video_path: Union[Path, str], fps: int) -> bool:
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
            
            # 自动检测图片后缀
            supported_extensions = ['.jpg', '.jpeg', '.png']
            detected_ext = None
            for ext in supported_extensions:
                if list(imgs_dir.glob(f"*{ext}")):  # 检查是否存在该扩展名的文件
                    detected_ext = ext
                    break
            
            if not detected_ext:
                raise ValueError(f"在 {imgs_dir} 中未找到支持的图片文件（支持的后缀: {', '.join(supported_extensions)}）")
            
            print(f"[DEBUG] 检测到图片后缀: {detected_ext}")

            ffmpeg_args = OrderedDict([
                ("-f", "image2"),
                ("-r", str(fps)),
                ("-i", str(imgs_dir / f"frame_%06d{detected_ext}")),
                ("-vcodec", "libx264"),
                ("-pix_fmt", "yuv420p"),
                ("-g", "5"),
                ("-crf", "18"),
                ("-loglevel", "error"),
            ])

            ffmpeg_cmd = ["ffmpeg"] + [item for pair in ffmpeg_args.items() for item in pair] + [str(video_path)]
            print(f"[DEBUG] 执行FFmpeg命令: {' '.join(ffmpeg_cmd)}")
            
            subprocess.run(ffmpeg_cmd, check=True)
            
            if not video_path.exists():
                raise OSError(f"视频文件未生成: {video_path}")
                
            print(f"[INFO] 普通视频编码成功: {video_path}")
            return True
        except Exception as e:
            print(f"[ERROR] 普通视频编码失败: {str(e)}")
            return False


    @staticmethod
    def encode_label_video_frames(imgs_dir: Union[Path, str], video_path: Union[Path, str], fps: int) -> bool:
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
            
            # 自动检测图片后缀
            supported_extensions = ['.jpg', '.jpeg', '.png']
            detected_ext = None
            for ext in supported_extensions:
                if list(imgs_dir.glob(f"*{ext}")):  # 检查是否存在该扩展名的文件
                    detected_ext = ext
                    break
            
            if not detected_ext:
                raise ValueError(f"在 {imgs_dir} 中未找到支持的图片文件（支持的后缀: {', '.join(supported_extensions)}）")
            
            print(f"[DEBUG] 检测到图片后缀: {detected_ext}")

            ffmpeg_args = OrderedDict([
                ("-f", "image2"),
                ("-r", str(fps)),
                ("-i", str(imgs_dir / f"frame_%06d{detected_ext}")),
                ("-vcodec", "libx264"),
                ("-pix_fmt", "yuv420p"),
                ("-g", "5"),
                ("-crf", "18"),
                ("-loglevel", "error"),
            ])

            ffmpeg_cmd = ["ffmpeg"] + [item for pair in ffmpeg_args.items() for item in pair] + [str(video_path)]
            print(f"[DEBUG] 执行FFmpeg命令: {' '.join(ffmpeg_cmd)}")
            
            subprocess.run(ffmpeg_cmd, check=True)
            
            if not video_path.exists():
                raise OSError(f"视频文件未生成: {video_path}")
                
            print(f"[INFO] 普通视频编码成功: {video_path}")
            return True
        except Exception as e:
            print(f"[ERROR] 普通视频编码失败: {str(e)}")
            return False

    @staticmethod
    def encode_depth_video_frames(imgs_dir: Union[Path, str], video_path: Union[Path, str], fps: int) -> bool:
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
            subprocess.run(ffmpeg_args, check=True)
            
            if not video_path.exists():
                raise OSError(f"视频文件未生成: {video_path}")
                
            print(f"[INFO] 深度视频编码成功: {video_path}")
            return True
        except Exception as e:
            print(f"[ERROR] 深度视频编码失败: {str(e)}")
            return False

    @staticmethod
    def get_img_video_path(each_task_path: str, camera_images: str, camera_images_path: str) -> tuple:
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
                video_path = os.path.join(each_task_path, 'videos', 'chunk-000', camera_images, video_name)
                depth_video_path = os.path.join(each_task_path, 'depth', 'chunk-000', camera_images, depth_video_name)
                img_path_list.append(img_path)
                video_path_list.append(video_path)
                depth_video_path_list.append(depth_video_path)
            
            if len(img_path_list) != len(video_path_list):
                print("[ERROR] 图像和视频路径数量不匹配")
                return [], []
                
            print(f"[DEBUG] 找到 {len(img_path_list)} 组图像和视频路径")
            return img_path_list, video_path_list, depth_video_path_list
        except Exception as e:
            print(f"[ERROR] 获取路径列表失败: {str(e)}")
            return [], []

    def upload(self,task_id_list):
        if not self.nas_auth.get_auth_sid():
            print("[ERROR] 登录nas失败")
            return
        for i in range(10):
            date_data = self.get_date_offset(i)  # i=0 表示今天，i=1 表示昨天，...，i=6 表示 6 天前
            print(date_data)
            directory_path = os.path.join(self.fold_path1, date_data, self.local_name)
            
            if not os.path.exists(directory_path):
                print("数据路径不存在")
                continue 
            self.upload_day_task(directory_path,i,task_id_list)
        self.local_server_task_id_request(0)

    def upload_day_task(self,directory_path,day,task_id_list):
        entries = os.listdir(directory_path) # 各任务列表
        # 筛选出子目录
        subdirectories = [entry for entry in entries if os.path.isdir(os.path.join(directory_path, entry))] # 仅筛选目录
        try:
            for task_data_name in subdirectories: # 1813490901
                json_object_list = []
                each_task_path = os.path.join(directory_path, task_data_name)
                entries_task_detail = os.listdir(each_task_path) # 各任务子目录列表
                if not entries_task_detail:
                    continue
                if "meta" not in entries_task_detail: # 目录结构判断
                    for each_task_single_name in entries_task_detail:
                        each_task_single_path = os.path.join(each_task_path, each_task_single_name)
                        each_common_record_path = os.path.join(each_task_single_path, "meta", "common_record.json")
                        with open(each_common_record_path, "r", encoding="utf-8") as f:
                            data = json.load(f)
                            task_id = data["task_id"] # 云平台任务id
                            machine_id = data["machine_id"]
                            task_name = data["task_name"]
                            if task_id_list:
                                if int(task_id) not in task_id_list:
                                    print(f"task id {task_id} not in task_id_list{task_id_list}")
                                    break
                            if "last_upload_id" in data:
                                print(f"任务:{each_task_single_name} 已上传")
                                if day == 9:
                                    self.delete_directory(each_task_single_path)
                                continue
                        self.upload_single_task_2(each_task_single_path,task_id,machine_id,task_name,task_data_name)
                    continue
                each_common_record_path = os.path.join(each_task_path, "meta", "common_record.json")
                each_opdata_path = os.path.join(each_task_path, "meta", "op_dataid.jsonl")

                with open(each_opdata_path, 'r', encoding='utf-8') as file:
                    for line in file:
                        try:
                            # 去除行末的换行符，并解析为 JSON 对象
                            json_object_data = json.loads(line.strip())
                            json_object_list.append(json_object_data)
                            last_line_json = json_object_data  # 更新最后一行的 JSON 对象
                        except json.JSONDecodeError as e:
                            print(f"解析 JSON 失败，行内容: {line.strip()}, 错误信息: {e}")
                task_number = int(last_line_json["episode_index"]) + 1 # 需要上传的文件数量
                
                with open(each_common_record_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    task_id = data["task_id"] # 云平台任务id
                    if task_id_list:
                        if int(task_id) not in task_id_list:
                            print(f"task id {task_id} not in task_id_list{task_id_list}")
                            continue
                    machine_id = data["machine_id"]
                    task_name = data["task_name"]
                    if "last_upload_id" in data:
                        last_epid = data['last_upload_id']
                        if task_number <= last_epid: # 判断当前数据是否新增
                            print("没有需要上传的数据")
                            if day == 9:
                                self.delete_directory(each_task_path)
                            continue
                    else:
                        last_epid = 0
                self.upload_single_task(directory_path,each_task_path,each_common_record_path,json_object_list,task_id,machine_id,task_name,last_epid,task_number,task_data_name)
        except Exception as e:
            print(str(e))

    def upload_single_task(self,directory_path,each_task_path,each_common_record_path,json_object_list,task_id,machine_id,task_name,last_epid,task_number,task_data_name):
        try:
            fps = self.read_info_json(each_task_path)
            self.local_server_task_id_request(task_id)
            
            timestamp_nas = str(self.add_random_milliseconds()) # 等待毫秒，造成时间差
            
            task_nas_path = os.path.join(self.nas_path, task_id) # 缓存目录
            if not self.nas_auth.check_path_exists(task_nas_path):
                self.nas_auth.create_folder(task_nas_path)
            
            task_cache_name = machine_id + "***" + timestamp_nas # 创建缓存排队目录
            task_cache_nas_path = os.path.join(task_nas_path, task_cache_name)
            if not self.nas_auth.check_path_exists(task_cache_nas_path):
                self.nas_auth.create_folder(task_cache_nas_path)

            time.sleep(5) # 等待所有目录创造
            while True:
                try:
                    meta_file_list = []
                    entries_2 = self.nas_auth.get_directory_structure(task_nas_path)
                    subdirectories_2 = [entry.split("***") for entry in entries_2]
                    # 找到时间戳最小的设备编号
                    min_timestamp_device = min(subdirectories_2, key=lambda x: int(x[1]))[0]
                    print(f"时间戳最小的设备编号是: {min_timestamp_device}")
                    if min_timestamp_device == machine_id: # 轮到自己论次
                        middle_name = task_name + "_" + task_id
                        nas_target_path = os.path.join(self.nas_data_path, middle_name)
                        nas_meta_file_path = os.path.join(nas_target_path, "meta", "op_dataid.jsonl")
                        local_nas_meta_file_path = os.path.join(directory_path, "op_dataid.jsonl")
                        if self.nas_auth.check_file_exists(nas_meta_file_path):
                            self.nas_auth.download_file(nas_meta_file_path, local_nas_meta_file_path)
                        if os.path.exists(local_nas_meta_file_path): # 判断nas上是否有任务数据，有的话先合并json
                            with open(local_nas_meta_file_path, 'r', encoding='utf-8') as file:
                                for line in file:
                                    try:
                                        # 去除行末的换行符，并解析为 JSON 对象
                                        json_object = json.loads(line.strip())
                                        last_line_json = json_object  # 更新最后一行的 JSON 对象
                                    except json.JSONDecodeError as e:
                                        print(f"解析 JSON 失败，行内容: {line.strip()}, 错误信息: {e}")
                            last_episode_id = int(last_line_json["episode_index"]) + 1 #  nas上的数据id记录
                            last_episode_id = last_episode_id - last_epid

                            op_dataid_path = os.path.join(each_task_path, "meta", "op_dataid.jsonl")
                            self.change_number(op_dataid_path, local_nas_meta_file_path, last_episode_id, last_epid, task_number) # 增加数据id
                            meta_file_list.append(local_nas_meta_file_path)

                            local_nas_episodes_path = os.path.join(directory_path, "episodes.jsonl")
                            nas_episodes_file_path = os.path.join(self.nas_data_path, middle_name, "meta", "episodes.jsonl")
                            self.nas_auth.download_file(nas_episodes_file_path, local_nas_episodes_path)
                            episodes_path = os.path.join(each_task_path, "meta", "episodes.jsonl")
                            self.change_number(episodes_path, local_nas_episodes_path, last_episode_id, last_epid, task_number) # 增加数据id
                            meta_file_list.append(local_nas_episodes_path)

                            
                            local_nas_info_path = os.path.join(directory_path, "info.json") # nas上的info.json文件保存到本地的地址
                            nas_info_file_path = os.path.join(self.nas_data_path, middle_name, "meta", "info.json") # nas上的info.json地址
                            self.nas_auth.download_file(nas_info_file_path, local_nas_info_path) #下载nas文件到本地
                            info_path = os.path.join(each_task_path, "meta", "info.json") # 当前任务的info.json文件地址
                            self.change_number_info(info_path, local_nas_info_path, episodes_path, last_epid, task_number) # 修改info.json的统计数据
                            meta_file_list.append(local_nas_info_path) # 将下载到本地的info.json加入删除名单
                            
                            
                            local_episodes_stats_path = os.path.join(directory_path, "episodes_stats.jsonl")
                            nas_episodes_stats_file_path = os.path.join(self.nas_data_path, middle_name, "meta", "episodes_stats.jsonl")
                            self.nas_auth.download_file(nas_episodes_stats_file_path, local_episodes_stats_path)
                            episodes_stats_path = os.path.join(each_task_path, "meta", "episodes_stats.jsonl")           
                            self.change_number(episodes_stats_path, local_episodes_stats_path, last_episode_id, last_epid, task_number) # 增加数据id
                            meta_file_list.append(local_episodes_stats_path)
                            list1 = [local_nas_info_path, local_nas_meta_file_path, local_nas_episodes_path, local_episodes_stats_path]
                            list2 = [nas_info_file_path, nas_meta_file_path, nas_episodes_file_path, nas_episodes_stats_file_path]
                            for data_id in range(last_epid, task_number):
                                cloud_data_id = json_object_list[data_id]["dataid"]
                                self.organize_data(each_task_path,data_id,last_epid,last_episode_id,cloud_data_id,list1,list2,fps,task_data_name,middle_name,task_number,task_id,local_nas_info_path)

                            self.nas_auth.delete_folder(task_cache_nas_path)
                            self.delete_file(meta_file_list)  
                            self.modify_json(each_common_record_path, task_number)
                            break
                        else:
                            for data_id in range(task_number): # 遍历任务数量
                                cloud_data_id = json_object_list[data_id]["dataid"]
                                self.organize_data2(each_task_path,data_id,cloud_data_id,fps,task_data_name,middle_name,task_number,task_id)
                                
                            self.nas_auth.delete_folder(task_cache_nas_path)
                            self.modify_json(each_common_record_path,task_number)
                            break
                    else:
                        time.sleep(10)
                except Exception as e:
                    print(str(e))
                    self.nas_auth.delete_folder(task_cache_nas_path)
                    break
        except Exception as e:
            print(str(e))
 
    def organize_data(self,each_task_path,data_id,last_epid,last_episode_id,cloud_data_id,list1,list2,fps,task_data_name,middle_name,task_number,task_id,local_nas_info_path):
        local_file_list = []
        nas_file_list = []
        local_video_list = []
        nas_video_list = []
        ffmpeg_encode_flag = True
        entries_1 = os.listdir(each_task_path) 
        for task_part in entries_1:
            if task_part == "images" and data_id == last_epid:
                each_images_path = os.path.join(each_task_path, 'images')
                entries_2 = os.listdir(each_images_path)
                
                for camera_images in entries_2:
                    camera_images_path = os.path.join(each_images_path, camera_images)
                    if os.path.isdir(camera_images_path):
                        img_list, video_list, depth_video_path_list = self.get_img_video_path(each_task_path, camera_images, camera_images_path)
                        
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
                                        
                if ffmpeg_encode_flag:
                    self.delete_directory(os.path.join(each_task_path, 'images'))
                    self.modify_feature_dtypes(local_nas_info_path,entries_2,'video')
        entries_1 = os.listdir(each_task_path)                               
        for task_part in entries_1:
            if task_part == "meta":
                if data_id == task_number - 1:
                    local_file_list.extend(list1)
                    nas_file_list.extend(list2)
            elif task_part == "data":
                fold_path = os.path.join(each_task_path, task_part, "chunk-000")
                local_file, nas_file = self.get_nth_file_in_subdirectories(fold_path, data_id, last_episode_id, task_data_name, middle_name, 0, self.nas_data_path)
                if isinstance(local_file, str):
                    local_file_list.append(local_file)
                    nas_file_list.append(nas_file)
                elif isinstance(local_file, list):
                    local_file_list.extend(local_file)
                    nas_file_list.extend(nas_file)
            elif task_part == "videos":
                fold_path = os.path.join(each_task_path, task_part, "chunk-000")
                local_file, nas_file = self.get_nth_file_in_subdirectories(fold_path, data_id, last_episode_id, task_data_name, middle_name, 0, self.nas_data_path)
                if isinstance(local_file, str):
                    local_video_list.append(local_file)
                    nas_video_list.append(nas_file)
                elif isinstance(local_file, list):
                    local_video_list.extend(local_file)
                    nas_video_list.extend(nas_file)
            elif task_part == "label":
                fold_path = os.path.join(each_task_path, task_part, "chunk-000")
                local_file, nas_file = self.get_nth_file_in_subdirectories(fold_path, data_id, last_episode_id, task_data_name, middle_name, 0, self.nas_data_path)
                if isinstance(local_file, str):
                    local_video_list.append(local_file)
                    nas_video_list.append(nas_file)
                elif isinstance(local_file, list):
                    local_video_list.extend(local_file)
                    nas_video_list.extend(nas_file)
            elif task_part == "depth":
                fold_path = os.path.join(each_task_path, task_part, "chunk-000")
                local_file, nas_file = self.get_nth_file_in_subdirectories(fold_path, data_id, last_episode_id, task_data_name, middle_name, 0, self.nas_data_path)
                if isinstance(local_file, str):
                    local_video_list.append(local_file)
                    nas_video_list.append(nas_file)
                elif isinstance(local_file, list):
                    local_video_list.extend(local_file)
                    nas_video_list.extend(nas_file)
            elif task_part == "audio":
                fold_path = os.path.join(each_task_path, task_part, "chunk-000")
                local_file, nas_file = self.get_nth_file_in_subdirectories(fold_path, data_id, last_episode_id, task_data_name, middle_name, 0, self.nas_data_path)
                if isinstance(local_file, str):
                    local_video_list.append(local_file)
                    nas_video_list.append(nas_file)
                elif isinstance(local_file, list):
                    local_video_list.extend(local_file)
                    nas_video_list.extend(nas_file)
        local_file_list.extend(local_video_list)
        nas_file_list.extend(nas_video_list)
        task_msg = {
            "task_id": int(task_id),
            "task_data_id": int(cloud_data_id),
            "source_path": each_task_path,
            "target_path": str(nas_file_list)
        }
        self.nas_auth.upload_file(task_msg, local_file_list, nas_file_list)

    def organize_data2(self,each_task_path,data_id,cloud_data_id,fps,task_data_name,middle_name,task_number,task_id):
        ffmpeg_encode_flag = True
        info_encode_flag = False
        local_file_list = []
        nas_file_list = []
        local_video_list = []
        nas_video_list = []
        entries_1 = os.listdir(each_task_path) 
        for task_part in entries_1:
            if task_part == "images" and data_id == 0:
                each_images_path = os.path.join(each_task_path, 'images')
                entries_2 = os.listdir(each_images_path)
                
                for camera_images in entries_2:
                    camera_images_path = os.path.join(each_images_path, camera_images)
                    if os.path.isdir(camera_images_path):
                        img_list, video_list, depth_video_path_list = self.get_img_video_path(each_task_path, camera_images, camera_images_path)
                        
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
                                        
                if ffmpeg_encode_flag:
                    self.delete_directory(os.path.join(each_task_path, 'images'))
                    nas_info_path = os.path.join(each_task_path,"meta","info.json")
                    local_nas_info_path = os.path.join(each_task_path,"info.json")
                    shutil.copy(nas_info_path,local_nas_info_path)
                    self.modify_feature_dtypes(local_nas_info_path,entries_2,'video')
                    nas_nas_info_path = os.path.join(self.nas_data_path,middle_name, "meta", "info.json")
                    info_encode_flag = True

        entries_1 = os.listdir(each_task_path)           
        for task_part in entries_1:
            if task_part == "meta":
                if data_id == task_number - 1:
                    fold_path = os.path.join(each_task_path, task_part)
                    local_file, nas_file = self.get_nth_file_in_subdirectories(fold_path, data_id, 0, task_data_name, middle_name, 1,self.nas_data_path)
                    if isinstance(local_file, str):
                        local_file_list.append(local_file)
                        nas_file_list.append(nas_file)
                    elif isinstance(local_file, list):
                        local_file_list.extend(local_file)
                        nas_file_list.extend(nas_file)
                    if info_encode_flag:
                        if nas_nas_info_path in nas_file_list:
                            # 如果已经存在，找到所有匹配的索引并替换对应的 local 路径
                            for i in range(len(nas_file_list)):
                                if nas_file_list[i] == nas_nas_info_path:
                                    local_file_list[i] = local_nas_info_path
                                        
            elif task_part == "data":
                fold_path = os.path.join(each_task_path, task_part, "chunk-000")
                local_file, nas_file = self.get_nth_file_in_subdirectories(fold_path, data_id, 0, task_data_name, middle_name, 0,self.nas_data_path)
                if isinstance(local_file, str):
                    local_file_list.append(local_file)
                    nas_file_list.append(nas_file)
                elif isinstance(local_file, list):
                    local_file_list.extend(local_file)
                    nas_file_list.extend(nas_file)    
                    
            elif task_part == "videos":
                fold_path = os.path.join(each_task_path, task_part, "chunk-000")
                local_file, nas_file = self.get_nth_file_in_subdirectories(fold_path, data_id, 0, task_data_name, middle_name, 0,self.nas_data_path)
                if isinstance(local_file, str):
                    local_video_list.append(local_file)
                    nas_video_list.append(nas_file)
                elif isinstance(local_file, list):
                    local_video_list.extend(local_file)
                    nas_video_list.extend(nas_file)    
                    
            elif task_part == "label":
                fold_path = os.path.join(each_task_path, task_part, "chunk-000")
                local_file, nas_file = self.get_nth_file_in_subdirectories(fold_path, data_id, 0, task_data_name, middle_name, 0,self.nas_data_path)
                if isinstance(local_file, str):
                    local_video_list.append(local_file)
                    nas_video_list.append(nas_file)
                elif isinstance(local_file, list):
                    local_video_list.extend(local_file)
                    nas_video_list.extend(nas_file)
            elif task_part == "depth":
                fold_path = os.path.join(each_task_path, task_part, "chunk-000")
                local_file, nas_file = self.get_nth_file_in_subdirectories(fold_path, data_id, 0, task_data_name, middle_name, 0,self.nas_data_path)
                if isinstance(local_file, str):
                    local_video_list.append(local_file)
                    nas_video_list.append(nas_file)
                elif isinstance(local_file, list):
                    local_video_list.extend(local_file)
                    nas_video_list.extend(nas_file)    
            elif task_part == "audio":
                fold_path = os.path.join(each_task_path, task_part, "chunk-000")
                local_file, nas_file = self.get_nth_file_in_subdirectories(fold_path, data_id, 0, task_data_name, middle_name, 0,self.nas_data_path)
                if isinstance(local_file, str):
                    local_video_list.append(local_file)
                    nas_video_list.append(nas_file)
                elif isinstance(local_file, list):
                    local_video_list.extend(local_file)
                    nas_video_list.extend(nas_file)
        local_file_list.extend(local_video_list)
        nas_file_list.extend(nas_video_list)
        task_msg = {
            "task_id": int(task_id),
            "task_data_id": int(cloud_data_id),
            "source_path": each_task_path,
            "target_path":str(nas_file_list)
        }
        self.nas_auth.upload_file(task_msg,local_file_list,nas_file_list)
        if info_encode_flag:
            os.remove(local_nas_info_path)

    def organize_data3(self,each_task_path,data_id,cloud_data_id,fps,task_data_name,middle_name,task_id,each_common_record_path):
        ffmpeg_encode_flag = True
        local_file_list = []
        nas_file_list = []
        local_video_list = []
        nas_video_list = []
        entries_1 = os.listdir(each_task_path) 
        for task_part in entries_1:
            if task_part == "images":
                each_images_path = os.path.join(each_task_path, 'images')
                entries_2 = os.listdir(each_images_path)
                
                for camera_images in entries_2:
                    camera_images_path = os.path.join(each_images_path, camera_images)
                    if os.path.isdir(camera_images_path):
                        img_list, video_list, depth_video_path_list = self.get_img_video_path(each_task_path, camera_images, camera_images_path)
                        
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
                                        
                if ffmpeg_encode_flag:
                    self.delete_directory(os.path.join(each_task_path, 'images'))
                    local_nas_info_path = os.path.join(each_task_path,"meta","info.json")
                    self.modify_feature_dtypes(local_nas_info_path,entries_2,'video')
        
        if not ffmpeg_encode_flag:
            print("[ERROR] ffmpeg fail")
            return
        entries_1 = os.listdir(each_task_path)  
        nas_data_path = os.path.join(self.nas_data_path, 'single_collect_data')         
        for task_part in entries_1:
            if task_part == "meta":
                fold_path = os.path.join(each_task_path, task_part)
                
                local_file, nas_file = self.get_nth_file_in_subdirectories(fold_path, data_id, 0, task_data_name, middle_name, 1, nas_data_path)
                if isinstance(local_file, str):
                    local_file_list.append(local_file)
                    nas_file_list.append(nas_file)
                elif isinstance(local_file, list):
                    local_file_list.extend(local_file)
                    nas_file_list.extend(nas_file)
                                        
            elif task_part == "data":
                fold_path = os.path.join(each_task_path, task_part, "chunk-000")
                local_file, nas_file = self.get_nth_file_in_subdirectories(fold_path, data_id, 0, task_data_name, middle_name, 0, nas_data_path)
                if isinstance(local_file, str):
                    local_file_list.append(local_file)
                    nas_file_list.append(nas_file)
                elif isinstance(local_file, list):
                    local_file_list.extend(local_file)
                    nas_file_list.extend(nas_file)    
                    
            elif task_part == "videos":
                fold_path = os.path.join(each_task_path, task_part, "chunk-000")
                local_file, nas_file = self.get_nth_file_in_subdirectories(fold_path, data_id, 0, task_data_name, middle_name, 0, nas_data_path)
                if isinstance(local_file, str):
                    local_video_list.append(local_file)
                    nas_video_list.append(nas_file)
                elif isinstance(local_file, list):
                    local_video_list.extend(local_file)
                    nas_video_list.extend(nas_file)   
                     
                    
            elif task_part == "label":
                fold_path = os.path.join(each_task_path, task_part, "chunk-000")
                local_file, nas_file = self.get_nth_file_in_subdirectories(fold_path, data_id, 0, task_data_name, middle_name, 0, nas_data_path)
                if isinstance(local_file, str):
                    local_video_list.append(local_file)
                    nas_video_list.append(nas_file)
                elif isinstance(local_file, list):
                    local_video_list.extend(local_file)
                    nas_video_list.extend(nas_file)   

            elif task_part == "depth":
                fold_path = os.path.join(each_task_path, task_part, "chunk-000")
                local_file, nas_file = self.get_nth_file_in_subdirectories(fold_path, data_id, 0, task_data_name, middle_name, 0, nas_data_path)
                if isinstance(local_file, str):
                    local_video_list.append(local_file)
                    nas_video_list.append(nas_file)
                elif isinstance(local_file, list):
                    local_video_list.extend(local_file)
                    nas_video_list.extend(nas_file)   

            elif task_part == "audio":
                fold_path = os.path.join(each_task_path, task_part, "chunk-000")
                local_file, nas_file = self.get_nth_file_in_subdirectories(fold_path, data_id, 0, task_data_name, middle_name, 0, nas_data_path)
                if isinstance(local_file, str):
                    local_video_list.append(local_file)
                    nas_video_list.append(nas_file)
                elif isinstance(local_file, list):
                    local_video_list.extend(local_file)
                    nas_video_list.extend(nas_file)
        local_file_list.extend(local_video_list)
        nas_file_list.extend(nas_video_list)
        task_msg = {
            "task_id": int(task_id),
            "task_data_id": int(cloud_data_id),
            "source_path": each_task_path,
            "target_path":str(nas_file_list)
        }
        self.nas_auth.upload_file(task_msg,local_file_list,nas_file_list)
        self.modify_json(each_common_record_path, 1)

    def upload_single_task_2(self,each_task_single_path,task_id,machine_id,task_name,task_data_name):
        try:
            json_object_list = []
            each_opdata_path = os.path.join(each_task_single_path, "meta", "op_dataid.jsonl")
            each_common_record_path = os.path.join(each_task_single_path, "meta", "common_record.json")
            middle_name = task_name + "_" + task_id
            fps = self.read_info_json(each_task_single_path)
            self.local_server_task_id_request(task_id)
            data_id = 0
            with open(each_opdata_path, 'r', encoding='utf-8') as file:
                for line in file:
                    try:
                        # 去除行末的换行符，并解析为 JSON 对象
                        json_object_data = json.loads(line.strip())
                        json_object_list.append(json_object_data)
                        last_line_json = json_object_data  # 更新最后一行的 JSON 对象
                    except json.JSONDecodeError as e:
                        print(f"解析 JSON 失败，行内容: {line.strip()}, 错误信息: {e}")
            cloud_data_id = last_line_json["dataid"]
            self.organize_data3(each_task_single_path,data_id,cloud_data_id,fps,task_data_name,middle_name,task_id,each_common_record_path)
        except Exception as e:
            print(str(e))

