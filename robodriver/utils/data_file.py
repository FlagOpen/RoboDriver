import datetime
import json
import os
import shutil
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from robodriver.utils.qc_tools import VideoCorruptionChecker, check_action_state_naming_compliance


def calculate_thresholds_fps(
    meta_dir, episodes_stats, info_json, episode_id, threshold_percentage=0.1
):
    """计算动作突变阈值和FPS"""
    # 读取JSON文件（逐行查找指定episode_id）
    target_json = None
    with open(meta_dir / episodes_stats, "r") as file:
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
    max_vals = np.array(target_json["stats"]["action"]["max"], dtype=np.float32)
    min_vals = np.array(target_json["stats"]["action"]["min"], dtype=np.float32)

    # 检查维度一致性
    if len(max_vals) != len(min_vals):
        raise ValueError(
            f"max 和 min 的维度不一致！max: {len(max_vals)}, min: {len(min_vals)}"
        )

    # 计算阈值
    action_range = max_vals - min_vals
    joint_change_thresholds = np.abs(action_range) * threshold_percentage
    joint_change_thresholds = np.where(
        joint_change_thresholds == 0, 1e-9, joint_change_thresholds
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
    timestamps = df["timestamp"].values.astype(np.float64)

    # 检查单调性
    if not np.all(np.diff(timestamps) >= 0):
        return False, "检测到动作时间戳不单调递增"

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


def validate_action_data(df, joint_change_thresholds, cut_list=None):
    """校验动作数据质量"""
    if "action" not in df.columns:
        return False, "Parquet文件中缺少'action'列"
    
    # 定义需要屏蔽的关节索引（夹爪数据，可根据需求修改）
    skip_joint_indices = []

    action_data = np.stack(df["action"].values)
    print(f"动作数据形状: {action_data.shape}")

    # 检查全零帧
    zero_frames = np.where(np.all(action_data == 0, axis=1))[0]
    if len(zero_frames) > 0:
        error_msg = f"检测到 {len(zero_frames)} 帧全零数据（无效动作），位于索引: {list(zero_frames)}"
        return False, error_msg

    # 检查相邻帧突变
    diff = np.abs(
        np.abs(action_data[1:]) - np.abs(action_data[:-1])
    )  # 计算帧间绝对差值
    # 只有当阈值 >= 0.1 时才检查突变
    threshold_mask = joint_change_thresholds >= 0.5  # 形状：(n_joints,)
    violations = (
        diff > joint_change_thresholds
    ) & threshold_mask  # 仅当阈值 >= 0.1 时才可能触发违规
    n_violations_per_frame = np.sum(violations, axis=1)

    violation_indices = np.where(n_violations_per_frame > 0)[
        0
    ]  # 所有违规帧的索引（0-based）
    if len(violation_indices) > 0:
        error_msg_details = "\n=== 突变检测结果 ===\n"
        error_msg_details += f"共检测到 {len(violation_indices)} 处突变帧：\n"

        for i, violation_frame_0based in enumerate(violation_indices):
            problematic_frame_idx = violation_frame_0based + 1  # 转换为1-based帧号
            exceeding_dims = np.where(violations[violation_frame_0based])[
                0
            ]  # 突变的关节索引

            # 获取当前突变帧和前后帧的数据（如果存在）
            curr_frame_idx = violation_frame_0based  # 当前突变帧（0-based）
            prev_frame_idx = curr_frame_idx - 1  # 前一帧（0-based）
            next_frame_idx = curr_frame_idx + 1  # 后一帧（0-based）

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
        # 获取该帧所有违规关节索引
        all_exceeding_dims = np.where(violations[first_violation_frame - 1])[0]
        
        # 方法：将数组转为列表，筛选不在skip_joint_indices中的索引
        valid_exceeding_dims = [dim for dim in all_exceeding_dims if dim not in skip_joint_indices]
        
        # 只有当存在有效违规索引（非屏蔽索引）时，才返回错误
        if len(valid_exceeding_dims) > 0:
            error_msg = f"检测到第 {first_violation_frame} 帧发生突变（无效动作），涉及关节索引: {valid_exceeding_dims}"
            return False, error_msg

    action_data_subset_list = []
    most_common_ratio_list = []
    trigger_cut = None
    if cut_list:
        for cut in cut_list:
            # 检查 cut 是否越界
            if (
                cut[0] >= action_data.shape[1]
                or cut[1] > action_data.shape[1]
                or cut[0] < 0
                or cut[1] <= cut[0]
            ):
                print(f"Warning: 跳过越界的 cut 区间: {cut}")
                continue  # 跳过无效的 cut
            action_data_subset = action_data[:, cut[0] : cut[1]]
            action_data_subset_list.append(action_data_subset)
    if action_data_subset_list:
        for action_data_sub in action_data_subset_list:
            # 检查重复动作
            unique_actions, counts = np.unique(
                action_data_sub, axis=0, return_counts=True
            )
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


def validate_media_data(
    media_files, media_type, camera_name, image_sample_interval=30, image_change_threshold=0.98
):
    """校验图像数据"""
    if not media_files:
        return False, f"检测到{camera_name}没有数据"
    # 图像数据校验（原有逻辑）
    if media_type == "image":
        return validate_image_data(media_files, camera_name, image_sample_interval, image_change_threshold)
    
     # 视频数据校验（新增逻辑，兼容单视频/多视频文件）
    elif media_type == "video":
        checker = VideoCorruptionChecker(image_sample_interval, image_change_threshold)
        for video_file in media_files:
            # 执行视频校验，返回布尔值（是否损坏）
            is_corrupted = checker.check_video_corruption(video_file)
            if is_corrupted:
                return False, f"相机{camera_name}的视频文件{video_file.name}异常"
        return True, None
    else:
        return False, f"不支持的媒体类型: {media_type}"


def validate_image_data(img_files, camera_name, image_sample_interval=30, image_change_threshold=0.98):
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
        # 抽样验证，避免遍历所有图像耗时过长
        check_indices = list(range(0, len(img_files), max(1, len(img_files)//100)))
        for idx in check_indices[1:]:
            img = cv2.imread(str(img_files[idx]))
            if img is None or not np.array_equal(first_img, img):
                all_same = False
                break
        if all_same:
            return False, f"检测到{camera_name}的图像数据完全相同"

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


def validate_frame_count(df, media_files, media_type):
    """校验动作数据和图像/视频帧数是否一致
    新增逻辑：针对视频，每个视频的帧数都必须和df的帧数完全匹配
    """
    action_frame_count = len(df)  # 预期帧数（每个视频都需等于该值）

    # 图像帧数：保持原有逻辑（总文件数和df帧数一致）
    if media_type == "image":
        media_frame_count = len(media_files)
        if action_frame_count != media_frame_count:
            return False, (
                f"帧数不匹配: 动作数据 {action_frame_count} 帧 vs "
                f"图像总文件数 {media_frame_count} 帧"
            )
    # 视频帧数：逐个校验每个视频的帧数是否等于df帧数
    elif media_type == "video":
        for video_file in media_files:
            cap = cv2.VideoCapture(str(video_file))
            # 校验视频是否能正常打开
            if not cap.isOpened():
                cap.release()  # 确保释放资源
                return False, f"无法获取视频文件{video_file.name}的帧数（文件无法打开）"
            # 获取当前视频的帧数
            video_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()  # 及时释放视频流资源
            
            # 逐个校验：当前视频帧数是否与动作数据帧数一致
            if video_frame_count != action_frame_count:
                return False, (
                    f"视频{video_file.name}帧数不匹配: "
                    f"预期（动作数据）{action_frame_count} 帧 vs 实际 {video_frame_count} 帧"
                )
    # 不支持的媒体类型
    else:
        return False, f"不支持的媒体类型: {media_type}"

    # 所有校验通过
    return True, None

def validate_session(
    _dir, 
    session_id,
    episodes_stats="episodes_stats.jsonl",
    info_json="info.json",
    image_sample_interval=30,
    image_change_threshold=0.98,  
    threshold_percentage=0.5,
    cut_list=None, # cut_list举例,用来切片action信息进行比对  [(0,75),(75,150),(150,201)]
):  
    """验证单个会话的数据，返回结构化验证结果

    Args:
        _dir (str/Path): 会话根目录
        session_id (str): 会话ID（格式：episode_000000）
        episodes_stats (str, optional):  episode统计文件名称，默认"episodes_stats.jsonl"
        info_json (str, optional): 会话信息JSON文件名称，默认"info.json"
        image_sample_interval (int, optional): 图像抽样校验间隔，默认30（每30帧抽样1张）
        image_change_threshold (float, optional): 图像相似度判定阈值，取值范围[0,1]。
            值越大，允许图像相似度越高（校验越宽松）；值越小，要求图像差异越明显（校验越严格）。
            默认值0.98，适用于动作幅度较小、图像小幅变化的场景，避免误判。
        threshold_percentage (float, optional): 动作关节突变判定百分比阈值，取值范围[0,1]。
            基于动作关节的最大最小差值（动作范围）计算突变阈值（突变阈值=动作范围×该百分比）。
            值越大，允许的帧间动作突变幅度越大（校验越宽松）；值越小，突变容忍度越低（校验越严格）。
            默认值0.5（50%），平衡严格性与容错性，避免正常动作波动误判为无效突变。
        cut_list (list[tuple], optional): action数据切片区间列表，用于分段校验动作重复率，默认None

    Returns:
        dict: 结构化验证结果，包含各类校验项的通过状态及备注
    """
    print(f"正在验证会话: {session_id}")

    # 初始化返回结构
    verification_result = {
        "camera_frame_rate": "pass",
        "camera_frame_rate_comment": "",
        "action_frame_rate": "pass",
        "action_frame_rate_comment": "",
        "file_integrity": "pass",
        "file_integrity_comment": "",
    }

    data_dir = Path(os.path.join(_dir, "data"))
    images_dir = Path(os.path.join(_dir, "images"))
    meta_dir = Path(os.path.join(_dir, "meta"))
    videos_dir = Path(os.path.join(_dir, "videos","chunk-000"))

    # 解析episode_id
    try:
        episode_id = int(session_id.split("_")[1])
    except (IndexError, ValueError):
        verification_result["file_integrity"] = "no pass"
        verification_result["file_integrity_comment"] = (
            f"无效的会话ID格式: {session_id}"
        )
        return {"verification": verification_result}
    # print(check_action_state_naming_compliance(meta_dir / info_json))
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
        verification_result["file_integrity_comment"] = (
            f"检测不到Parquet文件: {parquet_path}"
        )
        return {"verification": verification_result}

    try:
        df = pd.read_parquet(parquet_path)
    except Exception as e:
        verification_result["file_integrity"] = "no pass"
        verification_result["file_integrity_comment"] = f"读取Parquet文件失败: {str(e)}"
        return {"verification": verification_result}
    
    # 2. 确定媒体目录和类型（核心：优先videos，后images）
    media_dir = None
    media_type = None
    if videos_dir.exists():
        media_dir = videos_dir
        media_type = "video"
        print(f"[INFO] 检测到videos目录，优先使用该目录校验: {videos_dir}")
    elif images_dir.exists():
        media_dir = images_dir
        media_type = "image"
        print(f"[INFO] 未检测到videos目录，使用images目录校验: {images_dir}")
    else:
        verification_result["camera_frame_rate"] = "no pass"
        verification_result["file_integrity"] = "no pass"
        verification_result["file_integrity_comment"] = "未检测到videos/images目录"
        return {"verification": verification_result}

    # 验证动作时间戳（帧率）
    valid, msg = validate_timestamps(df, fps)
    if not valid:
        verification_result["file_integrity"] = "no pass"
        verification_result["file_integrity_comment"] = msg

    # 验证动作数据质量
    valid, msg = validate_action_data(df, joint_change_thresholds, cut_list)
    if not valid:
        verification_result["action_frame_rate"] = "no pass"
        verification_result["action_frame_rate_comment"] = msg

    else:
        verification_result["action_frame_rate_comment"] = msg
    # 检查相机目录
    camera_dirs = [d for d in media_dir.glob("*") if d.is_dir()]
    if not camera_dirs:
        verification_result["camera_frame_rate"] = "no pass"
        verification_result["camera_frame_rate_comment"] = (
            f"未找到相机目录: {media_dir}"
        )
        return {"verification": verification_result}

    # 验证每个相机的数据
    for camera_dir in camera_dirs:
        camera_name = os.path.basename(camera_dir)
        camera_session_dir = camera_dir / session_id
        # 查找媒体文件
        media_files = []
        if media_type == "video":
            # 支持常见视频格式：mp4, avi, mov, mkv
            video_extensions = [".mp4", ".avi", ".mov", ".mkv"]
            for ext in video_extensions:
                video_files = sorted(
                    camera_dir.glob(f"*{ext}"),
                    key=lambda x: int(x.stem.split("_")[-1]) if "_" in x.stem else 0
                )
                media_files.extend(video_files)
        elif media_type == "image":
            # 原有图像查找逻辑：优先png，后jpg
            media_files = sorted(
                camera_session_dir.glob("frame_*.png"),
                key=lambda x: int(x.stem.split("_")[-1])
            )
            if not media_files:
                media_files = sorted(
                    camera_session_dir.glob("frame_*.jpg"),
                    key=lambda x: int(x.stem.split("_")[-1])
                )

        # 验证帧数一致性
        valid, msg = validate_frame_count(df, media_files, media_type)
        if not valid:
            verification_result["file_integrity"] = "no pass"
            verification_result["file_integrity_comment"] = msg

        # 验证图像数据
        valid, msg = validate_media_data(
            media_files, media_type, camera_name, image_sample_interval, image_change_threshold
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

def has_valid_image_files(dir_path):
    """
    递归遍历多层级目录，判断是否存在有效图片文件
    :param dir_path: 目标目录路径
    :return: bool - 存在有效图片返回True，否则返回False
    """
    # 定义常见图片后缀名（可根据实际需求补充）
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp')
    
    # 使用os.walk递归遍历所有子目录和文件
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            # 判断文件后缀是否为图片格式（忽略大小写）
            if file.lower().endswith(image_extensions):
                # 找到任意一个有效图片文件，立即返回True
                return True
    # 遍历完所有目录和文件，未找到有效图片，返回False
    return False


def get_data_size(fold_path, data):  # 文件大小单位(MB)
    try:
        size_bytes = 0

        task_path = fold_path
        opdata_path = os.path.join(task_path, "meta", "op_dataid.jsonl")
        with open(opdata_path, "r", encoding="utf-8") as file:
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
            data_path = os.path.join(task_path, entry, "chunk-000")
            if entry == "meta":
                continue
            if entry == "videos":
                if "images" in entries_1:
                    images_dir = os.path.join(task_path, "images")
                    if has_valid_image_files(images_dir):
                        # 仅当images目录有有效图片时，才跳过videos统计
                        continue
                data_path = os.path.join(task_path, entry, "chunk-000")
                size_bytes += file_size(data_path, episode_index)
            if entry == "images":
                data_path = os.path.join(task_path, entry)
                size_bytes += file_size(data_path, episode_index)
            if entry == "data":
                data_path = os.path.join(task_path, entry, "chunk-000")
                size_bytes += file_size(data_path, episode_index)
            if entry == "audio":
                data_path = os.path.join(task_path, entry, "chunk-000")
                size_bytes += file_size(data_path, episode_index)
        size_mb = round(size_bytes / (1024 * 1024), 2)
        return size_mb

    except Exception as e:
        print(str(e))
        return 500


def get_data_duration(fold_path, data):  # 文件时长单位(s)
    try:
        task_path = fold_path
        info_path = os.path.join(task_path, "meta", "info.json")
        opdata_path = os.path.join(task_path, "meta", "op_dataid.jsonl")
        episodes_path = os.path.join(task_path, "meta", "episodes.jsonl")
        with open(info_path, "r", encoding="utf-8") as f:
            info_data = json.load(f)
            fps = info_data["fps"]  #
        with open(opdata_path, "r", encoding="utf-8") as file:
            for line in file:
                try:
                    # 去除行末的换行符，并解析为 JSON 对象
                    json_object_data = json.loads(line.strip())
                    if json_object_data["dataid"] == str(data["task_data_id"]):
                        episode_index = json_object_data["episode_index"]
                        break
                except json.JSONDecodeError as e:
                    print(f"解析 JSON 失败，行内容: {line.strip()}, 错误信息: {e}")
        with open(episodes_path, "r", encoding="utf-8") as file:
            for line in file:
                try:
                    # 去除行末的换行符，并解析为 JSON 对象
                    json_object_data = json.loads(line.strip())
                    if json_object_data["episode_index"] == episode_index:
                        length = json_object_data["length"]
                        break
                except json.JSONDecodeError as e:
                    print(f"解析 JSON 失败，行内容: {line.strip()}, 错误信息: {e}")
        duration = round(length / fps, 2)
        return duration
    except Exception as e:
        print(str(e))
        return 30


def update_dataid_json(path, episode_index, data):
    opdata_path = os.path.join(path, "meta", "op_dataid.jsonl")

    append_data = {
        "episode_index": episode_index,
        "dataid": str(data["task_data_id"]),
        "machine_id": str(data["machine_id"]),
    }

    # 以追加模式打开文件（如果不存在则创建）
    with open(opdata_path, "a", encoding="utf-8") as f:
        # 写入一行 JSON 数据（每行一个 JSON 对象）
        f.write(json.dumps(append_data, ensure_ascii=False) + "\n")


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

    with open(opdata_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                record = json.loads(line.strip())
                # 严格匹配 dataid 字段（考虑大小写和空格）
                if str(record.get("dataid", "")).strip() == target_id:
                    return int(record["episode_index"])
            except (json.JSONDecodeError, KeyError, ValueError):
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
    with open(opdata_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                entry = json.loads(line.strip())
                # 定义匹配条件（同时匹配episode_index和dataid）
                if (
                    entry.get("episode_index") == target_episode
                    and entry.get("dataid") == target_dataid
                ):
                    continue  # 跳过匹配的条目（即删除）
                filtered_data.append(entry)
            except json.JSONDecodeError:
                continue  # 跳过无效JSON行

    # 覆盖写回文件（不含匹配条目）
    with open(opdata_path, "w", encoding="utf-8") as f:
        for entry in filtered_data:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def update_common_record_json(path, data):
    opdata_path = os.path.join(path, "meta", "common_record.json")
    os.makedirs(os.path.dirname(opdata_path), exist_ok=True)
    if os.path.isfile(opdata_path):
        with open(opdata_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if "task_id" in data:
                return
    overwrite_data = {
        "task_id": str(data["task_id"]),
        "task_name": str(data["task_name"]),
        "machine_id": str(data["machine_id"]),
    }
    # 假设data变量已提前定义（包含所需的所有键值对）
    # overwrite_data = {
    #     "task_id": str(data["task_id"]),
    #     "task_name": str(data["task_name"]),
    #     "machine_id": str(data["machine_id"]),
    #     "scene_type": str(data["scene_type"]),       # 场景标签
    #     "task_description": str(data["task_description"]),  # 任务描述
    #     "tele_type": str(data["tele_type"]),         # 遥操作类型
    #     "task_instruction": str(data["task_instruction"])   # 任务步骤
    # }

    # 以追加模式打开文件（如果不存在则创建）
    with open(opdata_path, "w", encoding="utf-8") as f:
        # 写入一行 JSON 数据（每行一个 JSON 对象）
        f.write(json.dumps(overwrite_data, ensure_ascii=False) + "\n")


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

if __name__ == "__main__":
    _dir = "/home/rm/DoRobot/dataset/20251213/dev/place the okra into the bowl_place the okra into the pink bowl_502/place_the_okra_into_the_bowl_502_63101_old"
    session_id = "episode_000000"
    print(
        validate_session(
            _dir,
            session_id,
            episodes_stats="episodes_stats.jsonl",
            info_json="info.json",
            image_sample_interval=30,
            image_change_threshold=0.98,
            threshold_percentage=0.5,
            cut_list=None,
        )
    )
