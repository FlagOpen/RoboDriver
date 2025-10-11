import os
import datetime
import json
import shutil

import numpy as np
import pandas as pd
from pathlib import Path
import cv2


def calculate_thresholds_fps(meta_dir, episodes_stats, info_json, episode_id, threshold_percentage=0.1):
    """计算动作突变阈值和FPS"""
    # 读取JSON文件（逐行查找指定episode_id）
    target_json = None
    with open(meta_dir / episodes_stats, 'r') as file:
        for line in file:
            try:
                json_object = json.loads(line.strip())
                if json_object["episode_index"] == episode_id:
                    target_json = json_object
                    break
            except json.JSONDecodeError as e:
                print(f"解析 JSON 失败，行内容: {line.strip()}, 错误信息: {e}")
                continue

    if target_json is None:
        raise ValueError(f"未找到 episode_id = {episode_id} 的数据！")

    # 提取 max 和 min，并计算范围
    max_vals = np.array(target_json['stats']['action']['max'], dtype=np.float32)
    min_vals = np.array(target_json['stats']['action']['min'], dtype=np.float32)
    
    # 检查维度一致性
    if len(max_vals) != len(min_vals):
        raise ValueError(f"max 和 min 的维度不一致！max: {len(max_vals)}, min: {len(min_vals)}")

    # 计算阈值
    action_range = max_vals - min_vals
    joint_change_thresholds = np.abs(action_range) * threshold_percentage
    joint_change_thresholds = np.where(
        joint_change_thresholds == 0,
        1e-9,
        joint_change_thresholds
    )

    # 读取FPS
    fps = None
    try:
        with open(meta_dir / info_json, "r", encoding="utf-8") as f:
            data = json.load(f)
            fps = data.get("fps")
            print(f"[DEBUG] 读取到FPS值: {fps}")
    except Exception as e:
        print(f"[ERROR] 读取info.json失败: {str(e)}")
    
    return joint_change_thresholds, fps


def validate_timestamps(df, fps):
    """校验时间戳是否符合固定帧率"""
    timestamps = df['timestamp'].values.astype(np.float64)
    
    # 检查单调性
    if not np.all(np.diff(timestamps) >= 0):
        return False, '检测到动作时间戳不单调递增'
    
    # 检查固定帧率
    expected_interval = 1.0 / fps
    actual_intervals = np.diff(timestamps)
    tolerance = 0.01  # 1% 容差
    relative_error = np.abs(actual_intervals - expected_interval) / expected_interval
    
    if np.any(relative_error > tolerance):
        bad_idx = np.where(relative_error > tolerance)[0][0]
        error_msg = (
            f"帧 {bad_idx} 的时间戳间隔不符合预期帧率 {fps} FPS。\n"
            f"预期间隔: {expected_interval:.6f} 秒，\n"
            f"实际间隔: {actual_intervals[bad_idx]:.6f} 秒"
        )
        return False, error_msg
    
    return True, None


def validate_action_data(df, joint_change_thresholds,cut_list=None):
    """校验动作数据质量"""
    if 'action' not in df.columns:
        return False, "Parquet文件中缺少'action'列"
    
    action_data = np.stack(df['action'].values)
    print(f"动作数据形状: {action_data.shape}")

    # 检查全零帧
    zero_frames = np.where(np.all(action_data == 0, axis=1))[0]
    if len(zero_frames) > 0:
        error_msg = f"检测到 {len(zero_frames)} 帧全零数据（无效动作），位于索引: {list(zero_frames)}"
        return False, error_msg

    # 检查相邻帧突变
    diff = np.abs(np.abs(action_data[1:]) - np.abs(action_data[:-1]))  # 计算帧间绝对差值
    # 只有当阈值 >= 0.1 时才检查突变
    threshold_mask = joint_change_thresholds >= 0.5  # 形状：(n_joints,)
    violations = (diff > joint_change_thresholds) & threshold_mask  # 仅当阈值 >= 0.1 时才可能触发违规
    n_violations_per_frame = np.sum(violations, axis=1)
    
    violation_indices = np.where(n_violations_per_frame > 0)[0]  # 所有违规帧的索引（0-based）
    if len(violation_indices) > 0:
        error_msg_details = "\n=== 突变检测结果 ===\n"
        error_msg_details += f"共检测到 {len(violation_indices)} 处突变帧：\n"
        
        for i, violation_frame_0based in enumerate(violation_indices):
            problematic_frame_idx = violation_frame_0based + 1  # 转换为1-based帧号
            exceeding_dims = np.where(violations[violation_frame_0based])[0]  # 突变的关节索引
            
            # 获取当前突变帧和前后帧的数据（如果存在）
            curr_frame_idx = violation_frame_0based  # 当前突变帧（0-based）
            prev_frame_idx = curr_frame_idx - 1      # 前一帧（0-based）
            next_frame_idx = curr_frame_idx + 1      # 后一帧（0-based）
            
            # 收集需要输出的帧索引（确保不越界）
            frame_indices = []
            if prev_frame_idx >= 0:
                frame_indices.append(prev_frame_idx)
            frame_indices.append(curr_frame_idx)
            if next_frame_idx < len(action_data):
                frame_indices.append(next_frame_idx)
            
            # 记录当前突变的信息（仅输出涉及突变关节的值）
            error_msg_details += f"\n突变 #{i+1}: 第 {problematic_frame_idx} 帧，涉及关节索引: {exceeding_dims}\n"
            error_msg_details += "相关帧数据（仅突变关节）:\n"
            
            for idx in frame_indices:
                frame_num = idx + 1  # 转换为1-based帧号
                # 仅提取突变关节索引处的值
                relevant_values = action_data[idx, exceeding_dims]
                error_msg_details += f"  第 {frame_num} 帧: {relevant_values}\n"
        
        print(error_msg_details)

    if np.any(n_violations_per_frame > 0):
        first_violation_frame = np.where(n_violations_per_frame > 0)[0][0] + 1
        problematic_frame_idx = first_violation_frame
        exceeding_dims = np.where(violations[problematic_frame_idx - 1])[0]
        error_msg = f"检测到 {problematic_frame_idx} 帧发生突变（无效动作），位于索引: {exceeding_dims}"
        return False, error_msg
    
    action_data_subset_list = []
    most_common_ratio_list = []
    trigger_cut = None
    if cut_list:
        for cut in cut_list:
             # 检查 cut 是否越界
            if cut[0] >= action_data.shape[1] or cut[1] > action_data.shape[1] or cut[0] < 0 or cut[1] <= cut[0]:
                print(f"Warning: 跳过越界的 cut 区间: {cut}")
                continue  # 跳过无效的 cut
            action_data_subset = action_data[:, cut[0]:cut[1]]  
            action_data_subset_list.append(action_data_subset)
    if action_data_subset_list:
        for action_data_sub in action_data_subset_list:
            # 检查重复动作
            unique_actions, counts = np.unique(action_data_sub, axis=0, return_counts=True)
            max_count = max(counts)
            most_common_ratio = max_count / len(action_data)
            most_common_ratio_list.append(most_common_ratio)
        max_ratio_idx = most_common_ratio_list.index(max(most_common_ratio_list))
        most_common_ratio = most_common_ratio_list[max_ratio_idx]
        trigger_cut = cut_list[max_ratio_idx]  # 记录对应的 cut[0], cut[1]
    else:
        unique_actions, counts = np.unique(action_data, axis=0, return_counts=True)
        max_count = max(counts)
        most_common_ratio = max_count / len(action_data)
    if most_common_ratio > 0.9:
        error_msg = f"检测到{most_common_ratio:.3%}帧{trigger_cut}动作重复"
        return False, error_msg
    msg = f"检测到{most_common_ratio:.3%}帧动作重复"
    return True, msg


def compare_images(img1, img2):
    """比较两张图像的相似度"""
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    hist1 = cv2.calcHist([gray1], [0], None, [256], [0, 256])
    hist2 = cv2.calcHist([gray2], [0], None, [256], [0, 256])
    
    similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    return similarity


def validate_image_data(img_files, camera_name, image_sample_interval=30, image_change_threshold=0.98):
    """校验图像数据"""
    if not img_files:
        return False, f"检测到{camera_name}没有图像数据"

    # 抽样检查
    sample_indices = list(range(0, len(img_files), image_sample_interval))
    if len(img_files) - 1 not in sample_indices:
        sample_indices.append(len(img_files) - 1)

    similar_pairs = []
    for i in range(len(sample_indices) - 1):
        idx1 = sample_indices[i]
        idx2 = sample_indices[i + 1]
        img1 = cv2.imread(str(img_files[idx1]))
        img2 = cv2.imread(str(img_files[idx2]))

        if img1 is None or img2 is None:
            return False, f"无法读取{camera_name}图像数据"
    
        similarity = compare_images(img1, img2)
        if similarity > image_change_threshold:
            similar_pairs.append((idx1, idx2, similarity))

    # 分析结果
    if len(similar_pairs) > len(sample_indices) * 0.5:
        first_img = cv2.imread(str(img_files[0]))
        all_same = True
        for img_file in img_files[1:]:
            img = cv2.imread(str(img_file))
            if not np.array_equal(first_img, img):
                all_same = False
                break
        if all_same:
            return False, f"检测到{camera_name}部分图像数据完全相同"
    
    # 检查首尾图像
    first_img = cv2.imread(str(img_files[0]))
    last_img = cv2.imread(str(img_files[-1]))
    end_similarity = compare_images(first_img, last_img)
    
    if end_similarity >= 1:
        return False, f"检测到{camera_name}首尾图像完全相同"
    
    if end_similarity > image_change_threshold:
        print(
            f"注意: 首尾图像非常相似 (相似度={end_similarity:.4f})。 "
            "请确认会话是否捕获了预期的动作。"
        )
    
    return True, None


def validate_frame_count(df, img_files):
    """校验动作数据和图像帧数是否一致"""
    if len(df) != len(img_files):
        return False, (
            f"帧数不匹配: 动作数据 {len(df)} 帧 vs "
            f"图像数据 {len(img_files)} 帧"
        )
    return True, None


def validate_session(_dir, session_id, 
                    episodes_stats="episodes_stats.jsonl", 
                    info_json="info.json",
                    image_sample_interval=30,
                    image_change_threshold=0.98,
                    threshold_percentage=0.5,
                    cut_list= None): # cut_list举例,用来切片action信息进行比对  [(0,75),(75,150),(150,201)]
    """验证单个会话的数据，返回结构化验证结果"""
    print(f"正在验证会话: {session_id}")
    
    # 初始化返回结构
    verification_result = {
        "camera_frame_rate": "pass",
        "camera_frame_rate_comment": "",
        "action_frame_rate": "pass",
        "action_frame_rate_comment": "",
        "file_integrity": "pass",
        "file_integrity_comment": ""
    }
    
    data_dir = Path(os.path.join(_dir,"data"))
    images_dir = Path(os.path.join(_dir,"images"))
    meta_dir = Path(os.path.join(_dir,"meta"))
    
    # 解析episode_id
    try:
        episode_id = int(session_id.split("_")[1])
    except (IndexError, ValueError):
        verification_result["file_integrity"] = "no pass"
        verification_result["file_integrity_comment"] = f"无效的会话ID格式: {session_id}"
        return {"verification": verification_result}
 
    # 计算阈值和FPS
    try:
        joint_change_thresholds, fps = calculate_thresholds_fps(
            meta_dir, episodes_stats, info_json, episode_id, threshold_percentage
        )
        if fps is None:
            fps = 30  # 默认值
    except Exception as e:
        verification_result["file_integrity"] = "no pass"
        verification_result["file_integrity_comment"] = str(e)
        return {"verification": verification_result}
 
    # 1. 加载数据
    parquet_path = data_dir / "chunk-000" / f"{session_id}.parquet"
    img_session_dir = images_dir
    
    if not parquet_path.exists():
        verification_result["file_integrity"] = "no pass"
        verification_result["file_integrity_comment"] = f"检测不到Parquet文件: {parquet_path}"
        return {"verification": verification_result}
    
    if not img_session_dir.exists():
        verification_result["file_integrity"] = "no pass"
        verification_result["file_integrity_comment"] = f"检测不到图像目录: {img_session_dir}"
        return {"verification": verification_result}
    
    try:
        df = pd.read_parquet(parquet_path)
    except Exception as e:
        verification_result["file_integrity"] = "no pass"
        verification_result["file_integrity_comment"] = f"读取Parquet文件失败: {str(e)}"
        return {"verification": verification_result}
 
    # 验证动作时间戳（帧率）
    valid, msg = validate_timestamps(df, fps)
    if not valid:
        verification_result["file_integrity"] = "no pass"
        verification_result["file_integrity_comment"] = msg
    
    
    # 验证动作数据质量
    valid, msg = validate_action_data(df, joint_change_thresholds,cut_list)
    if not valid:
        verification_result["action_frame_rate"] = "no pass"
        verification_result["action_frame_rate_comment"] = msg
       
    else:
        verification_result["action_frame_rate_comment"] = msg
    # 检查相机目录
    camera_dirs = [d for d in img_session_dir.glob("*") if d.is_dir()]
    if not camera_dirs:
        verification_result["camera_frame_rate"] = "no pass"
        verification_result["camera_frame_rate_comment"] = f"未找到相机目录: {img_session_dir}"
        return {"verification": verification_result}
    
    # 验证每个相机的数据
    for camera_dir in camera_dirs:
        camera_name = os.path.basename(camera_dir)
        camera_dir = camera_dir / session_id
        img_files = sorted(camera_dir.glob("frame_*.png"),
                         key=lambda x: int(x.stem.split("_")[-1]))
        if not img_files:
            img_files = sorted(camera_dir.glob("frame_*.jpg"),
                         key=lambda x: int(x.stem.split("_")[-1]))
        
        # 验证帧数一致性
        valid, msg = validate_frame_count(df, img_files)
        if not valid:
            verification_result["file_integrity"] = "no pass"
            verification_result["file_integrity_comment"] = msg
        
        # 验证图像数据
        valid, msg = validate_image_data(
            img_files, camera_name, image_sample_interval, image_change_threshold
        )
        if not valid:
            verification_result["camera_frame_rate"] = "no pass"
            verification_result["camera_frame_rate_comment"] = msg
    
    print(f"✅ 会话 {session_id} 验证完成")
    return {"verification": verification_result}

def get_today_date():
    # 获取当前日期和时间
    today = datetime.datetime.now()
    
    # 格式化日期为字符串，格式为 "YYYY-MM-DD"
    date_string = today.strftime("%Y%m%d")
    return date_string

 
def get_directory_size(directory):
    """递归计算文件夹的总大小（字节）"""
    total_size = 0
    for dirpath, _, filenames in os.walk(directory):
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            # 忽略无效链接（可选）
            if not os.path.exists(file_path):
                continue
            total_size += os.path.getsize(file_path)
    return total_size

def file_size(path, n):
    has_directory = False
    has_file = False
    file_size_bytes = 0

    pre_entries = os.listdir(path)
    if not pre_entries:  # 空目录情况
        return 0

    # 检查路径下是否有文件或目录
    for entry in pre_entries:
        entry_path = os.path.join(path, entry)
        if os.path.isdir(entry_path):
            has_directory = True
        elif os.path.isfile(entry_path):
            has_file = True
        break  # 只需要检查第一个条目

    if has_file:
        for file_name in pre_entries:
            base, ext = os.path.splitext(file_name)
            if not ext:
                continue
            if "_" not in base:
                continue
            prefix, old_num = base.rsplit("_", 1)
            num_digits = len(old_num)
            new_num = str(n).zfill(num_digits)
            new_file_name = f"{prefix}_{new_num}{ext}"
            file_path = os.path.join(path, new_file_name)
            if os.path.isfile(file_path):
                file_size_bytes += os.path.getsize(file_path)
                break
        return file_size_bytes

    elif has_directory:
        for subdir in pre_entries:
            subdir_path = os.path.join(path, subdir)
            if not os.path.isdir(subdir_path):
                continue
            # 先检查子目录中的文件
            found = False
            for file_name in os.listdir(subdir_path):
                base, ext = os.path.splitext(file_name)
                if "_" not in base:
                    continue
                prefix, old_num = base.rsplit("_", 1)
                num_digits = len(old_num)
                new_num = str(n).zfill(num_digits)
                new_file_name = f"{prefix}_{new_num}{ext}"
                file_path = os.path.join(subdir_path, new_file_name)
                if os.path.isfile(file_path):
                    file_size_bytes += os.path.getsize(file_path)
                    found = True
                    break
                # 如果没找到文件，递归检查子目录
                if not found:
                    file_size_bytes += get_directory_size(file_path)
                break
                
        return file_size_bytes

    return 0

def get_data_size(fold_path, data): # 文件大小单位(MB)
    try:
        size_bytes = 0

        # directory_path = os.path.join(fold_path, get_today_date())
        # print(directory_path)
        # #directory_path = os.path.join(fold_path1, "2025701")
        # if not os.path.exists(directory_path):
        #     return 400

        # task_path = os.path.join(directory_path,f"{str(data['task_name'])}_{str(data['task_id'])}")

        task_path=fold_path
        opdata_path = os.path.join(task_path,"meta","op_dataid.jsonl")
        with open(opdata_path, 'r', encoding='utf-8') as file:
            for line in file:
                try:
                    # 去除行末的换行符，并解析为 JSON 对象
                    json_object_data = json.loads(line.strip())
                    if json_object_data["dataid"] == str(data["task_data_id"]):
                        episode_index = json_object_data["episode_index"]
                        break
                except json.JSONDecodeError as e:
                    print(f"解析 JSON 失败，行内容: {line.strip()}, 错误信息: {e}")
        
        entries_1 = os.listdir(task_path) 
        for entry in entries_1:
            data_path = os.path.join(task_path,entry,"chunk-000")
            if entry == "meta":
                continue
            if entry == "videos":
                if "images" in entries_1:
                    continue
                data_path = os.path.join(task_path,entry,"chunk-000")
                size_bytes += file_size(data_path,episode_index)
            if entry == "images":
                data_path = os.path.join(task_path,entry)
                print(data_path)
                size_bytes += file_size(data_path,episode_index)
            if entry == "data":
                data_path = os.path.join(task_path,entry,"chunk-000")
                size_bytes += file_size(data_path,episode_index)
            if entry == "audio":
                data_path = os.path.join(task_path,entry,"chunk-000")
                size_bytes += file_size(data_path,episode_index)
        size_mb = round(size_bytes / (1024 * 1024), 2)
        return size_mb


    except Exception as e:
        print(str(e))
        return 500
    




def get_data_duration(fold_path,data):  # 文件时长单位(s)
    try:
        # directory_path = os.path.join(fold_path, get_today_date())
        # print(directory_path)
        # #directory_path = os.path.join(fold_path1, "2025701")
        # if not os.path.exists(directory_path):
        #     return 30
        
        # task_path = os.path.join(directory_path,f"{str(data['task_name'])}_{str(data['task_id'])}")
        task_path = fold_path
        info_path = os.path.join(task_path,"meta","info.json")
        opdata_path = os.path.join(task_path,"meta","op_dataid.jsonl")
        episodes_path = os.path.join(task_path,"meta","episodes.jsonl")
        with open(info_path,"r",encoding="utf-8") as f:
            info_data = json.load(f)
            fps = info_data["fps"] # 
        with open(opdata_path, 'r', encoding='utf-8') as file:
            for line in file:
                try:
                    # 去除行末的换行符，并解析为 JSON 对象
                    json_object_data = json.loads(line.strip())
                    if json_object_data["dataid"] == str(data["task_data_id"]):
                        episode_index = json_object_data["episode_index"]
                        break
                except json.JSONDecodeError as e:
                    print(f"解析 JSON 失败，行内容: {line.strip()}, 错误信息: {e}")
        with open(episodes_path, 'r', encoding='utf-8') as file:
            for line in file:
                try:
                    # 去除行末的换行符，并解析为 JSON 对象
                    json_object_data = json.loads(line.strip())
                    if json_object_data["episode_index"] == episode_index:
                        length= json_object_data["length"]
                        break
                except json.JSONDecodeError as e:
                    print(f"解析 JSON 失败，行内容: {line.strip()}, 错误信息: {e}")
        duration = round(length/fps,2)
        return duration        
    except Exception as e:
        print(str(e))
        return 30

def update_dataid_json(path, episode_index, data):
    opdata_path = os.path.join(path, "meta", "op_dataid.jsonl")

    append_data = {
        "episode_index": episode_index,
        "dataid": str(data["task_data_id"]),
        "machine_id":str(data["machine_id"]),
    }
    
    # 以追加模式打开文件（如果不存在则创建）
    with open(opdata_path, 'a', encoding='utf-8') as f:
        # 写入一行 JSON 数据（每行一个 JSON 对象）
        f.write(json.dumps(append_data, ensure_ascii=False) + '\n')

def find_epindex_from_dataid_json(path: str, task_data_id: str) -> int:
    """
    根据 task_data_id 从 op_dataid.jsonl 文件中查询对应的 episode_index
    
    Args:
        path: 数据根目录路径（包含 meta 子目录）
        task_data_id: 需要查询的任务数据ID
    
    Returns:
        int: 对应的 episode_index 值
        
    Raises:
        FileNotFoundError: 当 op_dataid.jsonl 文件不存在时
        ValueError: 当指定 task_data_id 未找到时
    """
    opdata_path = os.path.join(path, "meta", "op_dataid.jsonl")
    
    if not os.path.exists(opdata_path):
        raise FileNotFoundError(f"元数据文件不存在: {opdata_path}")
    
    # 规范化 task_data_id 类型（确保字符串比较）
    target_id = str(task_data_id).strip()
    
    with open(opdata_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                record = json.loads(line.strip())
                # 严格匹配 dataid 字段（考虑大小写和空格）
                if str(record.get("dataid", "")).strip() == target_id:
                    return int(record["episode_index"])
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                # 跳过无效行但记录警告（实际项目中可添加日志）
                continue
    
    raise ValueError(f"未找到 task_data_id={task_data_id} 对应的 episode_index")

def delete_dataid_json(path, episode_index, data):
    opdata_path = os.path.join(path, "meta", "op_dataid.jsonl")
    
    # 构建要删除的匹配条件
    target_episode = episode_index
    target_dataid = str(data["task_data_id"])
    
    # 如果文件不存在，直接返回（无内容可删除）
    if not os.path.exists(opdata_path):
        return
    
    # 临时存储过滤后的数据
    filtered_data = []
    
    # 读取并过滤文件内容
    with open(opdata_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                entry = json.loads(line.strip())
                # 定义匹配条件（同时匹配episode_index和dataid）
                if (entry.get("episode_index") == target_episode and 
                    entry.get("dataid") == target_dataid):
                    continue  # 跳过匹配的条目（即删除）
                filtered_data.append(entry)
            except json.JSONDecodeError:
                continue  # 跳过无效JSON行
    
    # 覆盖写回文件（不含匹配条目）
    with open(opdata_path, 'w', encoding='utf-8') as f:
        for entry in filtered_data:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

def update_common_record_json(path, data):
    opdata_path = os.path.join(path, "meta", "common_record.json")
    os.makedirs(os.path.dirname(opdata_path), exist_ok=True)
    if os.path.isfile(opdata_path):
        with open(opdata_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if "task_id" in data:
                return
    overwrite_data = {
        "task_id": str(data["task_id"]),
        "task_name": str(data["task_name"]),
        "machine_id": str(data["machine_id"]),
    }
    
    # 以追加模式打开文件（如果不存在则创建）
    with open(opdata_path, 'w', encoding='utf-8') as f:
        # 写入一行 JSON 数据（每行一个 JSON 对象）
        f.write(json.dumps(overwrite_data, ensure_ascii=False) + '\n')

def check_disk_space(min_gb=1): 
    # 获取根目录（/）的磁盘使用情况（Docker 默认挂载点）
    total, used, free = shutil.disk_usage("/") 
    # 转换为 GB
    free_gb = free // (2**30)  # 1 GB = 2^30 bytes 
    if free_gb < min_gb:
        print(f"警告：剩余存储空间不足 {min_gb}GB（当前剩余 {free_gb}GB）")
        return False
    else:
        print(f"存储空间充足（剩余 {free_gb}GB）")
        return True

# if __name__ == '__main__':
#     fold_path = '/home/liuyou/Documents'
#     data = {
#         "task_id": "187",
#         "task_name": "刀具安全取放",
#         "task_data_id": "2043",
#         "collector_id":"001",
#         "task_steps": [
#             {
#                 "doruation": "10",
#                 "instruction": "put"
#             },
#             {
#                 "doruation": "2",
#                 "instruction": "close"
#             },
#             {
#                 "doruation": "5",
#                 "instruction": "clean"
#             }
#         ]
#     } # 之后作为参数传递
#     print(data_size(fold_path,data))
#     print(data_duration(fold_path,data))
        
if __name__ == '__main__':
    _dir = "/home/liuyou/Documents/DoRobot/dataset/20250918/dev/倒水_111_277/倒水_111_277_6274"
    session_id = "episode_000000"
    print(validate_session(_dir, session_id,  episodes_stats="episodes_stats.jsonl", 
                    info_json="info.json",
                    image_sample_interval=30,
                    image_change_threshold=0.98,
                    threshold_percentage=0.5,
                    cut_list= None))
    
    









