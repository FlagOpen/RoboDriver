from pathlib import Path
import cv2
import numpy as np
import av
import json
import re


class VideoCorruptionChecker:
    """视频文件损坏检测类（整合帧相似度检查，返回布尔值结果）"""

    def __init__(self, image_sample_interval=30, image_change_threshold=0.98):
        """
        初始化检测类
        :param image_sample_interval: 帧抽样间隔
        :param image_change_threshold: 帧相似度阈值（超过该值判定为异常）
        """
        self.result = {
            "is_corrupted": False,
            "errors": [],
            "warnings": [],
            "file_path": "",
        }
        self.sample_interval = image_sample_interval
        self.change_threshold = image_change_threshold

    def _compare_images(self, img1, img2):
        """内部方法：比较两张图像的相似度"""
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        hist1 = cv2.calcHist([gray1], [0], None, [256], [0, 256])
        hist2 = cv2.calcHist([gray2], [0], None, [256], [0, 256])

        similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
        return similarity

    def _check_video_frame_similarity(self, video_path):
        """内部方法：校验视频帧相似度（异常则标记为损坏）"""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            self.result["is_corrupted"] = True
            self.result["errors"].append(f"无法打开视频文件以检查帧相似度")
            return

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # 生成抽样索引
        sample_indices = list(range(0, frame_count, self.sample_interval))
        if frame_count - 1 not in sample_indices:
            sample_indices.append(frame_count - 1)

        # 提取抽样帧
        sample_frames = []
        for idx in sample_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                sample_frames.append(frame)
        cap.release()

        if len(sample_frames) < 2:
            return  # 帧数不足，不判定为损坏

        # 检查抽样帧相似度是否过高
        similar_pairs = 0
        for i in range(len(sample_frames) - 1):
            img1 = sample_frames[i]
            img2 = sample_frames[i + 1]
            similarity = self._compare_images(img1, img2)
            if similarity > self.change_threshold:
                similar_pairs += 1

        # 超过50%相似对则判定为异常
        if similar_pairs > len(sample_frames) * 0.5:
            # 进一步检查是否所有帧完全相同
            first_frame = sample_frames[0]
            all_same = True
            for frame in sample_frames[1:]:
                if not np.array_equal(first_frame, frame):
                    all_same = False
                    break
            if all_same:
                self.result["is_corrupted"] = True
                self.result["errors"].append("视频所有抽样帧完全相同，判定为损坏")
                return

        # 检查首尾帧是否完全相同
        first_frame = sample_frames[0]
        last_frame = sample_frames[-1]
        end_similarity = self._compare_images(first_frame, last_frame)
        if end_similarity >= 1.0:
            self.result["is_corrupted"] = True
            self.result["errors"].append("视频首尾帧完全相同，判定为损坏")

    def check_video_corruption(self, video_path: str | Path) -> bool:
        """
        检查视频文件是否损坏（整合文件完整性+帧相似度检查）
        Args:
            video_path: 视频文件路径（支持字符串或Path对象）
        Returns:
            bool: True=视频损坏/异常，False=视频正常
        """
        # 重置结果字典，避免多次调用时数据残留
        self._reset_result()
        # 统一转换为Path对象
        video_path_obj = Path(video_path)
        self.result["file_path"] = str(video_path_obj)

        # 1. 检查文件是否存在
        if not video_path_obj.exists():
            self.result["is_corrupted"] = True
            self.result["errors"].append(f"文件不存在: {video_path_obj}")
            return self.result["is_corrupted"]

        # 2. 检查文件是否为空
        if video_path_obj.stat().st_size == 0:
            self.result["is_corrupted"] = True
            self.result["errors"].append("文件大小为0（空文件）")
            return self.result["is_corrupted"]

        container = None
        try:
            # 3. 尝试打开视频容器
            container = av.open(str(video_path_obj))
        except av.error.InvalidDataError as e:
            self.result["is_corrupted"] = True
            self.result["errors"].append(f"无效数据格式: {str(e)}")
            return self.result["is_corrupted"]
        except av.error.FileNotFoundError as e:
            self.result["is_corrupted"] = True
            self.result["errors"].append(f"文件未找到: {str(e)}")
            return self.result["is_corrupted"]
        except Exception as e:
            self.result["is_corrupted"] = True
            self.result["errors"].append(f"打开视频失败: {type(e).__name__}: {str(e)}")
            return self.result["is_corrupted"]

        # 4. 检查视频流是否存在
        video_stream = None
        for stream in container.streams.video:
            video_stream = stream
            break

        if video_stream is None:
            self.result["is_corrupted"] = True
            self.result["errors"].append("容器中未找到视频流")
            container.close()
            return self.result["is_corrupted"]

        # 5. 尝试读取前5帧，检查完整性
        frame_count = 0
        read_errors = 0

        try:
            for _frame in container.decode(video_stream):
                frame_count += 1
                # 只检查前5帧，提高检测效率
                if frame_count >= 5:
                    break
        except av.error.InvalidDataError as e:
            read_errors += 1
            self.result["errors"].append(f"读取帧 {frame_count} 失败: {str(e)}")
        except Exception as e:
            read_errors += 1
            self.result["errors"].append(f"读取帧 {frame_count} 失败: {type(e).__name__}: {str(e)}")

        # 根据读帧错误标记文件损坏
        if read_errors > 0:
            self.result["is_corrupted"] = True

        # 检查是否读取到有效帧
        if frame_count == 0:
            self.result["is_corrupted"] = True
            self.result["errors"].append("无法从视频中读取任何帧")

        # 6. 检查视频元数据（时长）- 仅记录警告，不判定损坏
        try:
            duration = container.duration
            if duration is None or duration <= 0:
                self.result["warnings"].append("视频时长无效或不可用")
        except Exception as e:
            self.result["warnings"].append(f"获取时长失败: {str(e)}")

        # 7. 检查视频元数据（帧率）- 仅记录警告，不判定损坏
        try:
            fps = video_stream.average_rate
            if fps is None or float(fps) <= 0:
                self.result["warnings"].append(f"视频帧率无效: {fps}")
        except Exception as e:
            self.result["warnings"].append(f"获取帧率失败: {str(e)}")

        # 关闭视频容器，释放资源
        container.close()

        # 8. 整合：视频帧相似度检查（核心新增逻辑）
        self._check_video_frame_similarity(video_path_obj)

        # 返回布尔值：是否损坏（True=损坏，False=正常）
        return self.result["is_corrupted"]

    def _reset_result(self):
        """重置结果字典，用于多次调用时清空历史数据"""
        self.result = {
            "is_corrupted": False,
            "errors": [],
            "warnings": [],
            "file_path": "",
        }


"""
视频帧跳变检测类

检测视频中是否存在由于关键帧遗漏、画面过度等导致的帧跳变

使用方法:
    1. 导入类并实例化
    2. 调用检测方法
    3. 获取检测结果

示例:
    from qc_drop_frame import VideoFrameDropDetector

    # 实例化检测器
    detector = VideoFrameDropDetector(phash_dist_threshold=15)

    # 检测视频
    result = detector.check_frame_drops("/path/to/video.mp4")

    # 打印结果
    detector.print_result(result)
"""
class VideoFrameDropDetector:
    """视频帧跳变检测类"""

    def __init__(self, phash_dist_threshold: float = 15):
        """
        初始化帧跳变检测器

        Args:
            phash_dist_threshold: pHash汉明距离阈值，超过此值表示可能存在帧跳变
        """
        self.phash_dist_threshold = phash_dist_threshold

    def compute_phash(self, image: np.ndarray, hash_size: int = 8) -> str:
        """
        计算图像的感知哈希 (pHash)

        Args:
            image: 输入图像 (BGR格式)
            hash_size: 哈希大小 (默认 8x8)

        Returns:
            哈希字符串
        """
        # 转换为灰度图
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 调整大小
        resized = cv2.resize(gray, (hash_size, hash_size))

        # 计算DCT
        dct = cv2.dct(np.float32(resized))

        # 取左上角8x8区域的平均值
        dct_left = dct[:8, :8]
        avg = np.mean(dct_left)

        # 生成二进制哈希
        return "".join(["1" if x > avg else "0" for x in dct_left.flatten()])

    def hamming_distance(self, hash1: str, hash2: str) -> int:
        """计算两个哈希的汉明距离"""
        return sum(c1 != c2 for c1, c2 in zip(hash1, hash2))

    def check_frame_drops(self, video_path: str | Path) -> dict:
        """
        检测视频中的帧跳变（使用pHash计算相邻帧的相似度变化），保持原有算法逻辑不变

        Args:
            video_path: 视频文件路径（字符串或Path对象）

        Returns:
            检查结果字典
        """
        # 统一路径格式
        video_path = Path(video_path) if isinstance(video_path, str) else video_path
        result = {
            "total_frames": 0,
            "drop_frame_locations": [],  # [(frame_idx, distance), ...]
            "max_distance": 0,
            "avg_distance": 0,
            "drop_frame_count": 0,
            "has_drop_frames": False,
            "score": 1.0,  # 评分 0-1，1 为最好
            "errors": [],
            "warnings": [],
            "file_path": str(video_path),
        }

        # 检查文件是否存在
        if not video_path.exists():
            error_msg = f"File does not exist: {video_path}"
            result["errors"].append(error_msg)
            print(f"❌ {error_msg}")
            return result

        # 尝试打开视频
        try:
            container = av.open(str(video_path))
        except Exception as e:
            error_msg = f"Failed to open video: {type(e).__name__}: {str(e)}"
            result["errors"].append(error_msg)
            print(f"❌ {error_msg}")
            return result

        # 获取视频流
        video_stream = None
        for stream in container.streams.video:
            video_stream = stream
            break

        if video_stream is None:
            error_msg = "No video stream found"
            result["errors"].append(error_msg)
            print(f"❌ {error_msg}")
            container.close()
            return result

        print(f"Processing video: {video_path}")
        print(f"Drop frame detection threshold: {self.phash_dist_threshold}")

        prev_hash = None
        distances = []
        frame_idx = 0

        try:
            for frame in container.decode(video_stream):
                frame_idx += 1

                # 转换为 numpy 数组并转换为 BGR
                image = frame.to_ndarray(format="bgr24")

                # 计算 pHash
                current_hash = self.compute_phash(image)

                if prev_hash is None:
                    prev_hash = current_hash
                    continue

                # 计算汉明距离
                distance = self.hamming_distance(prev_hash, current_hash)
                distances.append(distance)

                if distance > self.phash_dist_threshold:
                    # 可能存在帧跳变
                    result["drop_frame_locations"].append((frame_idx, distance))
                    print(f"⚠️ Large frame difference at frame {frame_idx}: distance={distance}")

                # 更新最大距离
                result["max_distance"] = max(result["max_distance"], distance)
                prev_hash = current_hash

                # 进度输出
                if frame_idx % 100 == 0:
                    print(f"Processed {frame_idx} frames...")

        except Exception as e:
            error_msg = f"Error processing frames: {type(e).__name__}: {str(e)}"
            result["errors"].append(error_msg)
            print(f"❌ {error_msg}")

        # 填充结果基本信息
        result["total_frames"] = frame_idx
        result["drop_frame_count"] = len(result["drop_frame_locations"])

        # 计算平均距离、评分等
        if distances:
            result["avg_distance"] = np.mean(distances)

            # 计算评分：基于drop frames 的比例
            proportion = result["drop_frame_count"] / len(distances)
            result["score"] = max(0, 1.0 - proportion)

            if result["drop_frame_count"] > 0:
                result["has_drop_frames"] = True
                print(f"❌ Detected {result['drop_frame_count']} potential frame drops")
            else:
                print("✅ No frame drops detected")

        container.close()
        return result

    @staticmethod
    def print_result(result: dict) -> None:
        """
        格式化打印检测结果

        Args:
            result: check_frame_drops方法返回的结果字典
        """
        print("\n" + "=" * 80)
        print("FRAME DROP CHECK RESULT")
        print("=" * 80)
        print(f"File: {result['file_path']}")
        print(f"Total Frames: {result['total_frames']}")
        print(f"Potential Frame Drops: {result['drop_frame_count']}")
        print(f"Max Frame Distance: {result['max_distance']}")
        print(f"Average Frame Distance: {result['avg_distance']:.2f}")
        print(f"Score (0-1): {result['score']:.3f}")
        status = "✅ OK" if not result["has_drop_frames"] else "❌ PROBLEM DETECTED"
        print(f"Status: {status}")

        # 打印帧跳变位置
        if result["drop_frame_locations"]:
            print("\nPotential Frame Drop Locations (first 10):")
            for idx, (frame_idx, distance) in enumerate(result["drop_frame_locations"][:10], 1):
                print(f"  {idx}. Frame {frame_idx}: distance={distance}")

            if len(result["drop_frame_locations"]) > 10:
                remaining = len(result["drop_frame_locations"]) - 10
                print(f"  ... and {remaining} more")

        # 打印错误信息
        if result["errors"]:
            print(f"\nErrors ({len(result['errors'])}):")
            for error in result["errors"]:
                print(f"  ❌ {error}")

        # 打印警告信息
        if result["warnings"]:
            print(f"\nWarnings ({len(result['warnings'])}):")
            for warning in result["warnings"]:
                print(f"  ⚠️ {warning}")
        print("=" * 80)

"""
视频连续静止帧检测类

检测视频中是否存在过多连续的几乎完全相同的帧（表示视频可能卡住或损坏）

使用方法:
    1. 导入类并实例化
    2. 调用检测方法
    3. 获取/打印检测结果

示例:
    from qc_consecutive_static_frames import VideoConsecutiveStaticDetector

    # 实例化检测器
    detector = VideoConsecutiveStaticDetector(phash_dist_threshold=5, static_frames_threshold=10)

    # 检测视频
    result = detector.check_consecutive_static_frames("/path/to/video.mp4")

    # 格式化打印结果
    detector.print_result(result)
"""

class VideoConsecutiveStaticDetector:
    """视频连续静止帧检测类"""

    def __init__(self, phash_dist_threshold: float = 5, static_frames_threshold: int = 10):
        """
        初始化连续静止帧检测器

        Args:
            phash_dist_threshold: pHash距离阈值 (0-64, 越小表示帧越相似)
            static_frames_threshold: 被视为"连续静止"的帧数阈值
        """
        self.phash_dist_threshold = phash_dist_threshold
        self.static_frames_threshold = static_frames_threshold

    def compute_phash(self, image: np.ndarray, hash_size: int = 8) -> str:
        """
        计算图像的感知哈希 (pHash)

        Args:
            image: 输入图像 (BGR格式)
            hash_size: 哈希大小 (默认 8x8)

        Returns:
            哈希字符串
        """
        # 转换为灰度图
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 调整大小
        resized = cv2.resize(gray, (hash_size, hash_size))

        # 计算DCT
        dct = cv2.dct(np.float32(resized))

        # 取左上角8x8区域的平均值
        dct_left = dct[:8, :8]
        avg = np.mean(dct_left)

        # 生成二进制哈希
        phash = "".join(["1" if x > avg else "0" for x in dct_left.flatten()])

        return phash

    def hamming_distance(self, hash1: str, hash2: str) -> int:
        """计算两个哈希的汉明距离"""
        return sum(c1 != c2 for c1, c2 in zip(hash1, hash2))

    def check_consecutive_static_frames(self, video_path: str | Path) -> dict:
        """
        检测视频中的连续静止帧，保持原有算法逻辑不变

        Args:
            video_path: 视频文件路径（字符串或Path对象）

        Returns:
            检查结果字典
        """
        # 统一路径格式
        video_path = Path(video_path) if isinstance(video_path, str) else video_path
        result = {
            "total_frames": 0,
            "static_frame_groups": [],  # [(start_idx, end_idx, count), ...]
            "max_static_frames": 0,
            "has_excessive_static": False,
            "score": 1.0,  # 评分 0-1，1 为最好
            "errors": [],
            "warnings": [],
            "file_path": str(video_path),
        }

        # 检查文件是否存在
        if not video_path.exists():
            error_msg = f"File does not exist: {video_path}"
            result["errors"].append(error_msg)
            print(f"❌ {error_msg}")
            return result

        # 尝试打开视频
        try:
            container = av.open(str(video_path))
        except Exception as e:
            error_msg = f"Failed to open video: {type(e).__name__}: {str(e)}"
            result["errors"].append(error_msg)
            print(f"❌ {error_msg}")
            return result

        # 获取视频流
        video_stream = None
        for stream in container.streams.video:
            video_stream = stream
            break

        if video_stream is None:
            error_msg = "No video stream found"
            result["errors"].append(error_msg)
            print(f"❌ {error_msg}")
            container.close()
            return result

        # 输出初始化信息
        print(f"Processing video: {video_path}")
        print(f"pHash threshold: {self.phash_dist_threshold}, Static threshold: {self.static_frames_threshold} frames")

        prev_hash = None
        static_group_start = None
        static_count = 0
        frame_idx = 0

        try:
            for frame in container.decode(video_stream):
                frame_idx += 1

                # 转换为 numpy 数组并转换为 BGR
                image = frame.to_ndarray(format="bgr24")

                # 计算 pHash
                current_hash = self.compute_phash(image)

                if prev_hash is None:
                    prev_hash = current_hash
                    static_count = 1
                    static_group_start = frame_idx
                    continue

                # 计算汉明距离
                distance = self.hamming_distance(prev_hash, current_hash)

                if distance <= self.phash_dist_threshold:
                    # 相似帧，累加静止计数
                    static_count += 1
                else:
                    # 不相似帧，结束之前的静止组
                    if static_count >= self.static_frames_threshold:
                        group = (static_group_start, frame_idx - 1, static_count)
                        result["static_frame_groups"].append(group)
                        warning_msg = (
                            f"⚠️ Found consecutive static frames: "
                            f"frames {static_group_start}-{frame_idx-1} ({static_count} frames)"
                        )
                        print(warning_msg)

                    # 更新最大静止帧数
                    if static_count > result["max_static_frames"]:
                        result["max_static_frames"] = static_count

                    # 开始新的静止组
                    prev_hash = current_hash
                    static_count = 1
                    static_group_start = frame_idx

                # 进度输出（每100帧）
                if frame_idx % 100 == 0:
                    print(f"Processed {frame_idx} frames...")

        except Exception as e:
            error_msg = f"Error processing frames: {type(e).__name__}: {str(e)}"
            result["errors"].append(error_msg)
            print(f"❌ {error_msg}")

        # 处理最后一组静止帧
        if static_count >= self.static_frames_threshold:
            group = (static_group_start, frame_idx, static_count)
            result["static_frame_groups"].append(group)
            warning_msg = (
                f"⚠️ Found consecutive static frames at end: "
                f"frames {static_group_start}-{frame_idx} ({static_count} frames)"
            )
            print(warning_msg)

        # 最终更新最大静止帧数
        if static_count > result["max_static_frames"]:
            result["max_static_frames"] = static_count

        # 填充结果基本信息
        result["total_frames"] = frame_idx

        # 计算评分和判断是否存在过多静止帧
        if result["static_frame_groups"]:
            result["has_excessive_static"] = True
            # 评分基于最长的静止帧组占总帧数的比例
            proportion = result["max_static_frames"] / frame_idx if frame_idx > 0 else 1.0
            result["score"] = max(0, 1.0 - proportion)
            print("❌ Video has excessive static frames")
        else:
            result["score"] = 1.0
            print("✅ No excessive consecutive static frames detected")

        container.close()
        return result

    @staticmethod
    def print_result(result: dict) -> None:
        """
        格式化打印检测结果

        Args:
            result: check_consecutive_static_frames方法返回的结果字典
        """
        print("\n" + "=" * 80)
        print("CONSECUTIVE STATIC FRAMES CHECK RESULT")
        print("=" * 80)
        print(f"File: {result['file_path']}")
        print(f"Total Frames: {result['total_frames']}")
        print(f"Max Consecutive Static Frames: {result['max_static_frames']}")
        print(f"Score (0-1): {result['score']:.3f}")
        status = "✅ OK" if not result["has_excessive_static"] else "❌ PROBLEM DETECTED"
        print(f"Status: {status}")

        # 打印静止帧组信息
        if result["static_frame_groups"]:
            print(f"\nStatic Frame Groups ({len(result['static_frame_groups'])}):")
            for start, end, count in result["static_frame_groups"]:
                print(f"  Frames {start}-{end}: {count} consecutive static frames")

        # 打印错误信息
        if result["errors"]:
            print(f"\nErrors ({len(result['errors'])}):")
            for error in result["errors"]:
                print(f"  ❌ {error}")

        # 打印警告信息
        if result["warnings"]:
            print(f"\nWarnings ({len(result['warnings'])}):")
            for warning in result["warnings"]:
                print(f"  ⚠️ {warning}")
        print("=" * 80)


def check_action_state_naming_compliance(json_file_path):
    """
    校验info.json中action和observation.state的names命名是否符合指定规范
    参数:
        json_file_path: info.json文件的路径
    返回:
        校验结果字典，包含合规状态和违规信息
    """
    # 定义命名规范的正则表达式模式（覆盖所有要求的命名格式）
    # 模式说明：
    # {dir} 对应 right/left（机械臂方向）
    # 匹配所有指定的命名格式：关节角、手部关节角、夹爪开合度、末端位置、末端姿态
    naming_patterns = [
        r"^(?P<dir>right|left)_arm_joint_(?P<num>\d+)_rad$",          # {dir}_arm_joint_{num}_rad
        r"^(?P<dir>right|left)_hand_joint_(?P<num>\d+)_rad$",         # {dir}_hand_joint_{num}_rad
        r"^(?P<dir>right|left)_gripper_open_scale$",                  # {dir}_gripper_open_scale
        r"^(?P<dir>right|left)_eef_pos_(?P<axis>x|y|z)$",             # {dir}_eef_pos_{axis} (x/y/z)
        r"^(?P<dir>right|left)_eef_rot_(?P<axis>x|y|z)$"              # {dir}_eef_rot_{axis} (x/y/z)
    ]
    # 编译所有正则模式
    compiled_patterns = [re.compile(pattern) for pattern in naming_patterns]

    # 初始化校验结果
    result = {
        "is_compliant": True,
        "violations": []
    }

    try:
        # 读取并解析JSON文件
        with open(json_file_path, 'r', encoding='utf-8') as f:
            info_data = json.load(f)

        # 提取features中的action和observation.state字段
        features = info_data.get("features", {})
        target_fields = {
            "action": features.get("action", {}).get("names", []),
            "observation.state": features.get("observation.state", {}).get("names", [])
        }

        # 遍历每个目标字段（action/observation.state）进行校验
        for field_name, name_list in target_fields.items():
            if not isinstance(name_list, list):
                violation_msg = f"字段{field_name}的names不是列表类型，校验失败"
                result["violations"].append(violation_msg)
                result["is_compliant"] = False
                continue

            # 遍历names列表中的每个名称进行匹配
            for idx, name in enumerate(name_list):
                is_match = False
                # 检查是否匹配任一命名模式
                for pattern in compiled_patterns:
                    if pattern.match(name):
                        is_match = True
                        break
                # 若不匹配，记录违规信息
                if not is_match:
                    violation_msg = f"字段{field_name}的names列表中，索引{idx}的名称「{name}」不符合命名规范"
                    result["violations"].append(violation_msg)
                    result["is_compliant"] = False

    except FileNotFoundError:
        result["is_compliant"] = False
        result["violations"].append(f"未找到JSON文件：{json_file_path}")
    except json.JSONDecodeError:
        result["is_compliant"] = False
        result["violations"].append(f"JSON文件格式错误，无法解析：{json_file_path}")
    except Exception as e:
        result["is_compliant"] = False
        result["violations"].append(f"校验过程中发生异常：{str(e)}")

    return result

# 测试调用示例（将路径替换为你的info.json实际路径）
if __name__ == "__main__":
    json_path = "info.json"  # 你的info.json文件路径
    check_result = check_action_state_naming_compliance(json_path)
    if check_result["is_compliant"]:
        print("✅ action和observation.state的names命名全部符合规范")
    else:
        print("❌ 存在命名违规情况：")
        for violation in check_result["violations"]:
            print(f"  - {violation}")