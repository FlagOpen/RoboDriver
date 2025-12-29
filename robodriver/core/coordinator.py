import asyncio
import queue
import threading
import time
from datetime import datetime
from typing import Dict, Optional

import aiohttp
import cv2
import logging_mp
import socketio
from lerobot.teleoperators import Teleoperator

from robodriver.core.recorder import Record, RecordConfig
from robodriver.core.replayer import DatasetReplayConfig, ReplayConfig, replay
from robodriver.dataset.dorobot_dataset import *
from robodriver.dataset.visual.visual_dataset import visualize_dataset
from robodriver.robots.daemon import Daemon
from robodriver.utils.constants import (
    DEFAULT_FPS,
    DOROBOT_DATASET,
    RERUN_WEB_PORT,
    RERUN_WS_PORT,
)
from robodriver.utils.data_file import check_disk_space, find_epindex_from_dataid_json
from robodriver.utils.utils import cameras_to_stream_json, get_current_git_branch

logger = logging_mp.get_logger(__name__)


class Coordinator:
    def __init__(
        self,
        daemon: Daemon,
        teleop: Optional[Teleoperator],
        server_url="http://localhost:8088",
    ):
        self.server_url = server_url
        # 异步客户端
        self.sio = socketio.AsyncClient()
        self.session = aiohttp.ClientSession(
            connector=aiohttp.TCPConnector(limit=10, limit_per_host=10)
        )

        self.daemon = daemon
        self.teleop = teleop

        self.running = False
        self.last_heartbeat_time = 0
        self.heartbeat_interval = 2
        self.recording = False
        self.replaying = False
        self.saveing = False

        self.cameras = {"image_top": 1, "image_right": 2}

        # 注册异步回调
        self.sio.on("HEARTBEAT_RESPONSE", self.__on_heartbeat_response_handle)
        self.sio.on("connect", self.__on_connect_handle)
        self.sio.on("disconnect", self.__on_disconnect_handle)
        self.sio.on("robot_command", self.__on_robot_command_handle)

        self.record = None

    ####################### Client Start/Stop ############################
    async def start(self):
        """启动客户端"""
        self.running = True
        await self.sio.connect(self.server_url)
        # 用 asyncio 任务发心跳
        asyncio.create_task(self.send_heartbeat_loop())

    async def stop(self):
        self.running = False
        await self.sio.disconnect()
        await self.session.close()
        logger.info("异步客户端已停止")

    ####################### Client Handle ############################
    async def __on_heartbeat_response_handle(self, data):
        """心跳响应回调"""
        logger.info("收到心跳响应:", data)

    async def __on_connect_handle(self):
        """连接成功回调"""
        logger.info("成功连接到服务器")

    async def __on_disconnect_handle(self):
        """断开连接回调"""
        logger.info("与服务器断开连接")

    async def __on_robot_command_handle(self, data):
        """收到机器人命令回调"""
        logger.info("收到服务器命令:", data)
        global task_id
        global task_name
        global task_data_id
        global repo_id
        # 根据命令类型进行响应
        if data.get("cmd") == "video_list":
            logger.info("处理更新视频流命令...")
            response_data = cameras_to_stream_json(self.cameras)
            # 发送响应
            try:
                response = self.session.post(
                    f"{self.server_url}/robot/stream_info",
                    json=response_data,
                )
                logger.info(f"已发送响应 [{data.get('cmd')}]: {response_data}")
            except Exception as e:
                logger.error(f"发送响应失败 [{data.get('cmd')}]: {e}")

        elif data.get("cmd") == "start_collection":
            logger.info("处理开始采集命令...")
            msg = data.get("msg")

            if not check_disk_space(min_gb=2):
                logger.warning("存储空间不足,小于2GB,取消采集！")
                await self.send_response("start_collection", "存储空间不足,小于2GB")
                return

            if self.replaying == True:
                logger.warning("Replay is running, cannot start collection.")
                await self.send_response("start_collection", "fail")
                return

            if self.recording == True:
                self.record.stop()
                self.record.discard()
                self.recording = False

            self.recording = True

            task_id = msg.get("task_id")
            task_name = msg.get("task_name")
            task_data_id = msg.get("task_data_id")
            countdown_seconds = msg.get("countdown_seconds", 3)
            task_dir = f"{task_name}_{task_id}"
            repo_id = f"{task_name}_{task_id}_{task_data_id}"

            date_str = datetime.now().strftime("%Y%m%d")

            # 构建目标目录路径
            dataset_path = DOROBOT_DATASET

            git_branch_name = get_current_git_branch()
            if "release" in git_branch_name:
                target_dir = dataset_path / date_str / "user" / task_dir / repo_id
            elif "dev" in git_branch_name:
                target_dir = dataset_path / date_str / "dev" / task_dir / repo_id
            else:
                target_dir = dataset_path / date_str / "dev" / task_dir / repo_id

            # 判断是否存在对应文件夹以决定是否启用恢复模式
            resume = False

            # 检查数据集目录是否存在
            if not dataset_path.exists():
                logger.info(
                    f"Dataset directory '{dataset_path}' does not exist. Cannot resume."
                )
            else:
                # 检查目标文件夹是否存在且为目录
                if target_dir.exists() and target_dir.is_dir():
                    # resume = True
                    # logging.info(f"Found existing directory for repo_id '{repo_id}'. Resuming operation.")

                    logger.info(
                        f"Found existing directory for repo_id '{repo_id}'. Delete directory."
                    )
                    shutil.rmtree(target_dir)
                    time.sleep(0.5)  # make sure delete success.
                else:
                    logger.info(
                        f"No directory found for repo_id '{repo_id}'. Starting fresh."
                    )

            # resume 变量现在可用于后续逻辑
            logger.info(f"Resume mode: {'Enabled' if resume else 'Disabled'}")

            record_cfg = RecordConfig(
                fps=DEFAULT_FPS,
                single_task=task_name,
                repo_id=repo_id,
                video=self.daemon.robot.use_videos,
                resume=resume,
                root=target_dir,
            )
            self.record = Record(
                fps=DEFAULT_FPS,
                robot=self.daemon.robot,
                daemon=self.daemon,
                teleop=self.teleop,
                record_cfg=record_cfg,
                record_cmd=msg,
            )
            # 发送响应
            await self.send_response("start_collection", "success")
            # 开始采集倒计时
            logger.info(f"开始采集倒计时{countdown_seconds}s...")
            time.sleep(countdown_seconds)

            # 开始采集
            self.record.start()

        elif data.get("cmd") == "finish_collection":
            logger.info("处理完成采集命令...")
            if self.replaying == True:
                logger.warning("Replay is running, cannot finish collection.")
                await self.send_response("finish_collection", "fail")
                return

            if not self.saveing and self.record.save_data is None:
                # 如果不在保存状态，立即停止记录并保存
                self.saveing = True
                self.record.stop()
                self.record.save()
                self.recording = False
                self.saveing = False

            # 如果正在保存，循环等待直到 self.record.save_data 有数据
            while self.saveing:
                time.sleep(0.1)  # 避免CPU过载，适当延迟
            # 此时无论 saveing 状态如何，self.record.save_data 已有有效数据
            response_data = {
                "msg": "success",
                "data": self.record.save_data,
            }
            # 发送响应
            await self.send_response(
                "finish_collection", response_data["msg"], response_data
            )

        elif data.get("cmd") == "discard_collection":
            # 模拟处理丢弃采集
            logger.info("处理丢弃采集命令...")

            if self.replaying == True:
                logger.warning("Replay is running, cannot discard collection.")
                await self.send_response("discard_collection", "fail")
                return

            self.record.stop()
            self.record.discard()
            self.recording = False

            # 发送响应
            await self.send_response("discard_collection", "success")

        elif data.get("cmd") == "submit_collection":
            # 模拟处理提交采集
            logger.info("处理提交采集命令...")
            time.sleep(0.01)  # 模拟处理时间

            if self.replaying == True:
                logger.warning("Replay is running, cannot submit collection.")
                await self.send_response("submit_collection", "fail")
                return
            # 发送响应
            await self.send_response("submit_collection", "success")

        elif data.get("cmd") == "start_replay":
            logger.info("处理开始回放命令...")
            msg = data.get("msg")
            if self.recording == True:
                logger.warning("Recording is running, cannot start replay.")
                await self.send_response("start_replay", "fail")
                return
            if self.replaying == True:
                logger.warning("Replay is already running.")
                await self.send_response("start_replay", "fail")
                return
            self.replaying = True

            task_id = msg.get("task_id")
            task_name = msg.get("task_name")
            task_data_id = msg.get("task_data_id")
            task_dir = f"{task_name}_{task_id}"
            repo_id = f"{task_name}_{task_id}_{task_data_id}"

            date_str = datetime.now().strftime("%Y%m%d")

            # 构建目标目录路径
            dataset_path = DOROBOT_DATASET
            git_branch_name = get_current_git_branch()
            if "release" in git_branch_name:
                target_dir = dataset_path / date_str / "user" / task_dir / repo_id
            elif "dev" in git_branch_name:
                target_dir = dataset_path / date_str / "dev" / task_dir / repo_id
            else:
                target_dir = dataset_path / date_str / "dev" / task_dir / repo_id

            ep_index = find_epindex_from_dataid_json(target_dir, task_data_id)

            dataset = DoRobotDataset(repo_id, root=target_dir)

            logger.info(
                f"开始回放数据集: {repo_id}, 目标目录: {target_dir}, 任务数据ID: {task_data_id}, 回放索引: {ep_index}"
            )

            replay_dataset_cfg = DatasetReplayConfig(
                repo_id, ep_index, target_dir, fps=DEFAULT_FPS
            )
            replay_cfg = ReplayConfig(self.daemon.robot, replay_dataset_cfg)

            # 用于线程间通信的异常队列
            error_queue = queue.Queue()
            # 用于通知replay线程停止的事件
            stop_event = threading.Event()

            def visual_worker():
                """visual工作线程函数"""
                try:
                    # 主线程执行可视化（阻塞直到窗口关闭或超时）
                    visualize_dataset(
                        dataset,
                        mode="distant",
                        episode_index=ep_index,
                        web_port=RERUN_WEB_PORT,
                        ws_port=RERUN_WS_PORT,
                        stop_event=stop_event,  # 需要replay函数支持stop_event参数
                        open_browser=False,
                    )
                except Exception as e:
                    error_queue.put(e)

            # 创建并启动replay线程
            visual_thread = threading.Thread(
                target=visual_worker,
                name="VisualThread",
                daemon=True,  # 设置为守护线程，主程序退出时自动终止
            )
            visual_thread.start()

            # 发送响应
            response_data = {
                "data": {
                    "url": f"http://localhost:{RERUN_WEB_PORT}/?url=ws://localhost:{RERUN_WS_PORT}",
                },
            }
            await self.send_response("start_replay", "success", response_data)

            try:
                replay(replay_cfg)

            finally:
                # 无论可视化是否正常结束，都通知replay线程停止
                stop_event.set()
                # 等待replay线程安全退出（设置合理超时）
                visual_thread.join(timeout=5.0)

                # 检查线程是否已退出
                if visual_thread.is_alive():
                    logger.warning("Warning: Visual thread did not exit cleanly")

                # 处理子线程异常
                try:
                    error = error_queue.get_nowait()
                    raise RuntimeError(
                        f"Visual failed in thread: {str(error)}"
                    ) from error
                except queue.Empty:
                    pass
            self.replaying = False

            logger.info("=" * 20 + "Replay Complete Success!" + "=" * 20)

    ####################### Client Send to Server ############################
    async def send_heartbeat_loop(self):
        """定期发送心跳"""
        while self.running:
            current_time = time.time()
            if current_time - self.last_heartbeat_time >= self.heartbeat_interval:
                try:
                    await self.sio.emit("HEARTBEAT")
                    self.last_heartbeat_time = current_time
                except Exception as e:
                    logger.error(f"发送心跳失败: {e}")
            time.sleep(1)
            await self.sio.wait()

    # 发送回复请求
    async def send_response(self, cmd, msg, data=None):
        payload = {"cmd": cmd, "msg": msg}
        if data:
            payload.update(data)
        try:
            async with self.session.post(
                f"{self.server_url}/robot/response",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=2),
            ) as resp:
                logger.info(f"已发送响应 [{cmd}]: {payload}")
        except Exception as e:
            logger.error(f"发送响应失败 [{cmd}]: {e}")

    ####################### Robot API ############################
    def stream_info(self, info: Dict[str, int]):
        self.cameras = info.copy()
        logger.info(f"更新摄像头信息: {self.cameras}")

    def stream_info_add(self, camera_name: str, camera_id: int):
        """添加或更新单个摄像头的流信息
        
        Args:
            camera_name: 摄像头名字
            camera_id: 摄像头编号
        """
        if not hasattr(self, 'cameras'):
            self.cameras = {}
        
        self.cameras[camera_name] = camera_id
        logger.info(f"添加摄像头 {camera_name} 编号: {camera_id}")
        
        # 可选：返回更新后的总流数
        return sum(self.cameras.values())

    async def update_stream_info_to_server(self):
        stream_info_data = cameras_to_stream_json(self.cameras)
        logger.info(f"stream_info_data: {stream_info_data}")
        try:
            # 2. 异步post加await，确保请求发送
            async with self.session.post(
                f"{self.server_url}/robot/stream_info",
                json=stream_info_data,
                timeout=aiohttp.ClientTimeout(total=2),
            ) as response:
                if response.status == 200:
                    logger.info("摄像头流信息已同步到服务器")
                else:
                    logger.warning(f"同步流信息失败: {response.status}")
        except Exception as e:
            logger.error(f"同步流信息异常: {e}")

    async def update_stream_async(self, name, frame):
        _, jpeg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        url = f"{self.server_url}/robot/update_stream/{self.cameras[name]}"
        try:
            # 超时给短一点，丢几帧对视频流影响不大
            async with self.session.post(
                url, data=jpeg.tobytes(), timeout=aiohttp.ClientTimeout(total=0.2)
            ) as resp:
                if resp.status != 200:
                    txt = await resp.text()
                    logger.error(f"Server error {resp.status}: {txt}")
        except asyncio.TimeoutError:
            logger.warning("update_stream timeout")
        except Exception as e:
            logger.error("update_stream exception:", e)
