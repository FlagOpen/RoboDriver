import queue
import sys
import threading
import time  # 添加 time 模块
from pathlib import Path
from typing import Dict, List, Optional

import sounddevice as sd
import soundfile as sf


class AsyncAudioWriter:
    """异步多设备音频录制器

    示例:
        writer = AsyncAudioWriter(
            microphones={
                "left": 0,   # 设备ID 0
                "right": 1   # 设备ID 1
            },
            savepath={
                "left": Path("./cache/audio/left.wav"),   # 使用Path对象
                "right": Path("./cache/audio/right.wav")  # 使用Path对象
            }
        )
        writer.start()
        # ... 录制中 ...
        writer.stop()
        writer.wait_until_done()  # 确保所有数据写入完成
    """

    def __init__(
        self,
        microphones: Dict[str, int],
        savepath: Dict[str, Path],  # 修改为Path对象字典
        samplerate: Optional[int] = None,
        channels: int = 1,
        subtype: Optional[str] = "PCM_16",
    ):
        """
        初始化音频录制器

        Args:
            microphones: 设备ID映射字典 (设备名称 -> 设备ID)
            savepath: 保存路径映射字典 (设备名称 -> Path对象)
            samplerate: 采样率 (None表示使用设备默认值)
            channels: 声道数 (默认1)
            subtype: 音频格式 (如 "PCM_24", None表示使用soundfile默认)
        """
        # 验证每个microphones键都能在savepath中找到对应的键（通过后缀匹配）
        for mic_key in microphones.keys():
            # 查找包含mic_key的savepath键
            matching_keys = [k for k in savepath.keys() if k.endswith(mic_key)]

            if not matching_keys:
                raise ValueError(
                    f"savepath中找不到与microphones键 '{mic_key}' 对应的键\n"
                    f"microphones keys: {list(microphones.keys())}\n"
                    f"savepath keys: {list(savepath.keys())}"
                )

        self.microphones = microphones
        self.savepath = savepath  # 现在是一个字典 {设备名: Path对象}
        self.samplerate = samplerate
        self.channels = channels
        self.subtype = subtype

        # 内部状态
        self._tasks: List[Dict] = []  # 存储每个任务的资源
        self._is_started = False
        self._stop_event = threading.Event()
        self._stop_timeout = 5.0  # 停止超时时间(秒)

    def start(self):
        """启动所有录音设备"""
        if self._is_started:
            raise RuntimeError("录音已启动，请先调用stop()")

        # 确保状态完全重置
        self._stop_event.clear()

        try:
            for name, device_id in self.microphones.items():
                matching_keys = [k for k in self.savepath.keys() if k.endswith(name)]

                if not matching_keys:
                    raise KeyError(f"在savepath中找不到与'{name}'匹配的键")

                # 选择最长的匹配（最具体的路径）
                best_match = max(matching_keys, key=len)
                file_path = self.savepath[best_match]

                # 确保父目录存在
                file_path.parent.mkdir(exist_ok=True, parents=True)

                # 检查文件是否已存在
                if file_path.exists():
                    # 改为删除现有文件而不是抛出异常
                    print(f"警告: 文件 {file_path} 已存在，将被覆盖")
                    file_path.unlink()

                print(f"开始录制 {name} 到 {file_path} (设备ID: {device_id})")

                # 查询设备信息
                try:
                    device_info = sd.query_devices(device_id, "input")
                except Exception as e:
                    raise RuntimeError(f"设备 {device_id} 查询失败: {str(e)}") from e

                # 确定采样率
                actual_samplerate = self.samplerate or int(
                    device_info["default_samplerate"]
                )

                # 创建资源
                q = queue.Queue(maxsize=100)  # 限制队列大小防止内存溢出
                try:
                    file = sf.SoundFile(
                        file_path,
                        mode="x",
                        samplerate=actual_samplerate,
                        channels=self.channels,
                        subtype=self.subtype,
                    )
                except Exception as e:
                    raise RuntimeError(f"无法创建文件 {file_path}: {str(e)}") from e

                # 创建流 (使用绑定回调避免lambda问题)
                stream = sd.InputStream(
                    samplerate=actual_samplerate,
                    device=device_id,
                    channels=self.channels,
                    callback=self._make_callback(q),
                )

                # 创建写入线程
                thread = threading.Thread(
                    target=self._writer_thread, args=(q, file, file_path), daemon=True
                )

                # 保存任务资源
                task = {
                    "filename": file_path,
                    "device_id": device_id,
                    "samplerate": actual_samplerate,
                    "queue": q,
                    "file": file,
                    "stream": stream,
                    "thread": thread,
                }
                self._tasks.append(task)

            # 启动所有资源
            for task in self._tasks:
                task["stream"].start()
                task["thread"].start()

            self._is_started = True
            print(f"已启动 {len(self._tasks)} 个录音设备")

        except Exception:
            self._cleanup_tasks()
            raise

    def _make_callback(self, q: queue.Queue):
        """创建线程安全的音频回调函数"""

        def callback(indata, frames, time, status):
            if status:
                print(f"音频状态警告: {status}", file=sys.stderr)
            try:
                q.put_nowait(indata.copy())
            except queue.Full:
                print(f"警告: 队列溢出，丢弃 {frames} 帧", file=sys.stderr)

        return callback

    def _writer_thread(self, q: queue.Queue, file: sf.SoundFile, filename: Path):
        """文件写入线程"""
        try:
            while not self._stop_event.is_set():
                try:
                    # 使用超时避免永久阻塞
                    data = q.get(timeout=0.5)
                    file.write(data)
                except queue.Empty:
                    continue
                except Exception as e:
                    print(f"写入文件 {filename} 时出错: {str(e)}", file=sys.stderr)
                    break
        finally:
            # 确保文件关闭
            try:
                if not file.closed:
                    file.close()
                    print(f"已关闭音频文件: {filename}")
            except Exception as e:
                print(f"关闭文件 {filename} 失败: {str(e)}", file=sys.stderr)

    def stop(self):
        """停止所有录音设备（非阻塞）"""
        if not self._is_started:
            return

        print("停止录音...")
        self._stop_event.set()

        # 停止音频流
        for task in self._tasks:
            try:
                if task["stream"].active:  # 检查流是否活跃
                    task["stream"].stop()
                    print(f"已停止设备 {task['device_id']} 的音频流")
            except Exception as e:
                print(f"停止设备 {task['device_id']} 失败: {str(e)}", file=sys.stderr)

        self._is_started = False

    def wait_until_done(self, timeout: Optional[float] = None):
        """等待所有写入操作完成

        Args:
            timeout: 等待超时时间(秒)，None表示无限等待
        """
        if not self._tasks:
            return

        print("等待写入完成...")
        actual_timeout = timeout if timeout is not None else self._stop_timeout
        start_time = time.time()

        for task in self._tasks:
            # 计算剩余等待时间
            elapsed = time.time() - start_time
            remaining = max(0, actual_timeout - elapsed)

            if remaining <= 0:
                print(
                    f"警告: 等待超时，强制终止任务: {task['filename']}", file=sys.stderr
                )
                break

            # 等待线程结束
            task["thread"].join(timeout=remaining)
            if task["thread"].is_alive():
                print(f"警告: 文件 {task['filename']} 写入超时", file=sys.stderr)

        # 清理资源
        self._cleanup_tasks()
        print("所有录音任务已完成")

    def _cleanup_tasks(self):
        """清理所有任务资源"""
        for task in self._tasks:
            # 清空队列
            while not task["queue"].empty():
                try:
                    task["queue"].get_nowait()
                except queue.Empty:
                    break

            # 关闭文件流
            try:
                if "file" in task and not task["file"].closed:
                    task["file"].close()
            except:
                pass

            # 关键修复: 显式关闭PortAudio流
            try:
                if "stream" in task and task["stream"]:
                    if task["stream"].active:
                        task["stream"].stop()
                    task["stream"].close()  # 释放底层资源
                    print(f"已关闭设备 {task['device_id']} 的音频流")
            except Exception as e:
                print(f"关闭音频流失败: {str(e)}", file=sys.stderr)

        # 重置状态变量
        self._tasks = []
        # 不要清除_stop_event，因为start方法会重置它
        self._is_started = False

    def __del__(self):
        """析构函数 - 确保资源被释放"""
        if self._is_started:
            print(
                "警告: AsyncAudioWriter被销毁时仍在运行! 调用stop()和wait_until_done()确保安全停止",
                file=sys.stderr,
            )
            self.stop()
            self.wait_until_done()
