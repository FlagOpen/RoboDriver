import asyncio
import json
import queue
import shutil
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import aiohttp
import cv2
import logging_mp
from lerobot.teleoperators import Teleoperator

from robodriver.core.recorder import Record, RecordConfig
from robodriver.core.replayer import DatasetReplayConfig, ReplayConfig, replay
from robodriver.dataset.dorobot_dataset import DoRobotDataset
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

logger = logging_mp.getLogger(__name__)


class Coordinator:
    def __init__(
        self,
        daemon: Daemon,
        teleop: Optional[Teleoperator],
        server_url: str = "http://localhost:8088",
    ):
        self.server_url = server_url
        self.ws_url = f"{server_url.replace('http', 'ws')}/ws"  # 假设WebSocket端点是/ws

        # HTTP 会话 (用于非WebSocket请求)
        self.session = aiohttp.ClientSession(
            connector=aiohttp.TCPConnector(limit=10, limit_per_host=10)
        )

        # WebSocket 相关
        self.websocket: Optional[aiohttp.ClientWebSocketResponse] = None
        self.ws_lock = asyncio.Lock()
        self.ws_connected = False
        self.ws_reconnect_delay = 2.0  # 重连延迟(秒)

        self.daemon = daemon
        self.teleop = teleop

        self.running = False
        self.last_heartbeat_time = 0
        self.heartbeat_interval = 2
        self.recording = False
        self.replaying = False
        self.saveing = False

        self.cameras = {"image_top": 1, "image_right": 2}

        # 消息路由表 (替代socketio的事件注册)
        self.message_handlers = {
            "HEARTBEAT_RESPONSE": self._handle_heartbeat_response,
            "robot_command": self._handle_robot_command,
        }

        self.record = None

    ####################### Client Start/Stop ############################
    async def start(self):
        """启动客户端"""
        self.running = True
        # 启动WebSocket连接任务
        asyncio.create_task(self._websocket_connection_loop())
        # 启动心跳任务
        asyncio.create_task(self._send_heartbeat_loop())

    async def stop(self):
        """停止客户端"""
        self.running = False
        # 关闭WebSocket连接
        async with self.ws_lock:
            if self.websocket and not self.websocket.closed:
                await self.websocket.close()
                self.ws_connected = False

        # 关闭HTTP会话
        await self.session.close()
        logger.info("Coordinator已停止")

    ####################### WebSocket Connection Management ############################
    async def _websocket_connection_loop(self):
        """管理WebSocket连接的主循环"""
        while self.running:
            try:
                async with self.ws_lock:
                    if not self.websocket or self.websocket.closed:
                        logger.info(f"正在连接WebSocket: {self.ws_url}")
                        self.websocket = await self.session.ws_connect(
                            self.ws_url,
                            heartbeat=30.0,  # aiohttp内置的心跳
                            timeout=10.0,
                        )
                        self.ws_connected = True
                        logger.info("WebSocket连接成功")
                        # 触发连接事件
                        await self._on_connect()

                # 启动消息接收循环
                await self._receive_messages()

            except aiohttp.WSServerHandshakeError as e:
                logger.error(f"WebSocket握手失败: {e}")
            except aiohttp.ClientConnectionError as e:
                logger.error(f"WebSocket连接失败: {e}")
            except Exception as e:
                logger.exception(f"WebSocket连接异常: {e}")

            # 重连前清理
            async with self.ws_lock:
                if self.websocket and not self.websocket.closed:
                    await self.websocket.close()
                self.ws_connected = False

            # 触发断开连接事件
            await self._on_disconnect()

            # 重连延迟
            if self.running:
                logger.info(f"{self.ws_reconnect_delay}秒后尝试重连...")
                await asyncio.sleep(self.ws_reconnect_delay)

    async def _receive_messages(self):
        """接收并处理WebSocket消息"""
        if not self.websocket:
            return

        async for msg in self.websocket:
            if msg.type == aiohttp.WSMsgType.TEXT:
                try:
                    data = json.loads(msg.data)
                    event = data.get("event")
                    payload = data.get("data", {})

                    if handler := self.message_handlers.get(event):
                        await handler(payload)
                    else:
                        logger.warning(f"未处理的事件类型: {event}")
                except json.JSONDecodeError:
                    logger.error(f"无效的JSON消息: {msg.data}")
                except Exception as e:
                    logger.exception(f"处理消息时出错: {e}")

            elif msg.type == aiohttp.WSMsgType.CLOSED:
                logger.info("WebSocket连接已关闭")
                break
            elif msg.type == aiohttp.WSMsgType.ERROR:
                logger.error(f"WebSocket错误: {self.websocket.exception()}")
                break

    async def _on_connect(self):
        """WebSocket连接成功回调"""
        logger.info("成功连接到服务器")
        # 可选: 发送初始状态
        await self._update_stream_info_to_server()

    async def _on_disconnect(self):
        """WebSocket断开连接回调"""
        logger.info("与服务器断开连接")

    ####################### Message Handlers ############################
    async def _handle_heartbeat_response(self, data: dict):
        """心跳响应回调"""
        logger.debug(f"收到心跳响应: {data}")

    async def _handle_robot_command(self, data: dict):
        """收到机器人命令回调"""
        logger.info(f"收到服务器命令: {data}")
        cmd = data.get("cmd")
        msg = data.get("msg", {})

        try:
            if cmd == "video_list":
                await self._handle_video_list_command(msg)
            elif cmd == "start_collection":
                await self._handle_start_collection_command(msg)
            elif cmd == "finish_collection":
                await self._handle_finish_collection_command(msg)
            elif cmd == "discard_collection":
                await self._handle_discard_collection_command(msg)
            elif cmd == "submit_collection":
                await self._handle_submit_collection_command(msg)
            elif cmd == "start_replay":
                await self._handle_start_replay_command(msg)
            else:
                logger.warning(f"未知命令: {cmd}")
                await self.send_response(cmd, "unknown_command")

        except Exception as e:
            logger.exception(f"处理命令 '{cmd}' 时出错")
            await self.send_response(cmd, "internal_error", {"error": str(e)})

    # 以下是各个命令的具体处理方法 (重构自原__on_robot_command_handle)
    async def _handle_video_list_command(self, msg: dict):
        logger.info("处理更新视频流命令...")
        response_data = cameras_to_stream_json(self.cameras)
        try:
            async with self.session.post(
                f"{self.server_url}/robot/stream_info",
                json=response_data,
                timeout=aiohttp.ClientTimeout(total=2),
            ) as response:
                if response.status == 200:
                    logger.info(f"已发送流信息响应: {response_data}")
                else:
                    logger.warning(f"发送流信息失败: HTTP {response.status}")
        except Exception as e:
            logger.error(f"发送流信息响应失败: {e}")

    async def _handle_start_collection_command(self, msg: dict):
        logger.info("处理开始采集命令...")
        global task_id, task_name, task_data_id, repo_id

        if not check_disk_space(min_gb=2):
            logger.warning("存储空间不足,小于2GB,取消采集！")
            await self.send_response("start_collection", "storage_insufficient")
            return

        if self.replaying:
            logger.warning("Replay is running, cannot start collection.")
            await self.send_response("start_collection", "replay_in_progress")
            return

        # 停止任何现有录制
        if self.recording and self.record:
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
        dataset_path = Path(DOROBOT_DATASET)

        # 确定存储路径 (基于git分支)
        git_branch_name = get_current_git_branch()
        env = "user" if "release" in git_branch_name else "dev"
        target_dir = dataset_path / date_str / env / task_dir / repo_id

        # 清理已有目录
        if target_dir.exists():
            logger.info(f"删除已存在的目录: {target_dir}")
            shutil.rmtree(target_dir)
            await asyncio.sleep(0.5)  # 确保删除完成

        # 配置录制
        record_cfg = RecordConfig(
            fps=DEFAULT_FPS,
            single_task=task_name,
            repo_id=repo_id,
            video=self.daemon.robot.use_videos,
            resume=False,
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

        # 发送成功响应
        await self.send_response("start_collection", "success")

        # 倒计时
        logger.info(f"开始采集倒计时{countdown_seconds}s...")
        await asyncio.sleep(countdown_seconds)

        # 启动录制
        logger.info("开始采集数据...")
        self.record.start()

    async def _handle_finish_collection_command(self, msg: dict):
        logger.info("处理完成采集命令...")

        if self.replaying:
            logger.warning("Replay is running, cannot finish collection.")
            await self.send_response("finish_collection", "replay_in_progress")
            return

        # 确保录制已开始
        if not self.recording or not self.record:
            logger.warning("没有活动的录制会话")
            await self.send_response("finish_collection", "no_active_recording")
            return

        # 停止录制
        self.saveing = True
        try:
            self.record.stop()
            # 保存数据 (同步操作，可能需要较长时间)
            save_data = await asyncio.to_thread(self.record.save)
            self.recording = False

            # 准备响应
            response_data = {
                "msg": "success",
                "data": save_data,
            }
            await self.send_response("finish_collection", "success", response_data)
        finally:
            self.saveing = False

    async def _handle_discard_collection_command(self, msg: dict):
        logger.info("处理丢弃采集命令...")

        if self.replaying:
            logger.warning("Replay is running, cannot discard collection.")
            await self.send_response("discard_collection", "replay_in_progress")
            return

        if not self.recording or not self.record:
            logger.warning("没有活动的录制会话可丢弃")
            await self.send_response("discard_collection", "no_active_recording")
            return

        # 停止并丢弃
        self.record.stop()
        self.record.discard()
        self.recording = False

        await self.send_response("discard_collection", "success")

    async def _handle_submit_collection_command(self, msg: dict):
        logger.info("处理提交采集命令...")

        if self.replaying:
            logger.warning("Replay is running, cannot submit collection.")
            await self.send_response("submit_collection", "replay_in_progress")
            return

        # 这里应该有实际的提交逻辑
        await asyncio.sleep(0.01)  # 模拟处理时间
        await self.send_response("submit_collection", "success")

    async def _handle_start_replay_command(self, msg: dict):
        logger.info("处理开始回放命令...")
        global task_id, task_name, task_data_id, repo_id

        if self.recording:
            logger.warning("Recording is running, cannot start replay.")
            await self.send_response("start_replay", "recording_in_progress")
            return

        if self.replaying:
            logger.warning("Replay is already running.")
            await self.send_response("start_replay", "replay_already_running")
            return

        self.replaying = True

        task_id = msg.get("task_id")
        task_name = msg.get("task_name")
        task_data_id = msg.get("task_data_id")
        task_dir = f"{task_name}_{task_id}"
        repo_id = f"{task_name}_{task_id}_{task_data_id}"

        date_str = datetime.now().strftime("%Y%m%d")
        dataset_path = Path(DOROBOT_DATASET)

        # 确定存储路径
        git_branch_name = get_current_git_branch()
        env = "user" if "release" in git_branch_name else "dev"
        target_dir = dataset_path / date_str / env / task_dir / repo_id

        # 查找episode索引
        ep_index = find_epindex_from_dataid_json(target_dir, task_data_id)
        if ep_index is None:
            logger.error(f"找不到任务数据ID {task_data_id} 对应的episode")
            await self.send_response("start_replay", "episode_not_found")
            self.replaying = False
            return

        # 加载数据集
        try:
            dataset = DoRobotDataset(repo_id, root=target_dir)
        except Exception as e:
            logger.exception(f"加载数据集失败: {e}")
            await self.send_response(
                "start_replay", "dataset_load_failed", {"error": str(e)}
            )
            self.replaying = False
            return

        logger.info(
            f"开始回放数据集: {repo_id}, 目标目录: {target_dir}, "
            f"任务数据ID: {task_data_id}, 回放索引: {ep_index}"
        )

        # 配置回放
        replay_dataset_cfg = DatasetReplayConfig(
            repo_id, ep_index, target_dir, fps=DEFAULT_FPS
        )
        replay_cfg = ReplayConfig(self.daemon.robot, replay_dataset_cfg)

        # 创建通信原语
        error_queue = queue.Queue()
        stop_event = threading.Event()

        # 启动可视化线程
        def visual_worker():
            try:
                visualize_dataset(
                    dataset,
                    mode="distant",
                    episode_index=ep_index,
                    web_port=RERUN_WEB_PORT,
                    ws_port=RERUN_WS_PORT,
                    stop_event=stop_event,
                )
            except Exception as e:
                error_queue.put(e)

        visual_thread = threading.Thread(
            target=visual_worker,
            name="VisualThread",
            daemon=True,
        )
        visual_thread.start()

        # 发送回放URL响应
        response_data = {
            "data": {
                "url": f"http://localhost:{RERUN_WEB_PORT}/?url=ws://localhost:{RERUN_WS_PORT}",
            },
        }
        await self.send_response("start_replay", "success", response_data)

        try:
            # 执行回放 (同步操作)
            await asyncio.to_thread(replay, replay_cfg)
        finally:
            # 清理
            stop_event.set()
            visual_thread.join(timeout=5.0)

            if visual_thread.is_alive():
                logger.warning("可视化线程未在5秒内退出")

            # 检查可视化错误
            if not error_queue.empty():
                error = error_queue.get()
                logger.error(f"可视化线程错误: {error}")

            self.replaying = False

        logger.info("=" * 20 + "Replay Complete Success!" + "=" * 20)

    ####################### Client Communication ############################
    async def _send_heartbeat_loop(self):
        """定期发送心跳"""
        while self.running:
            current_time = time.time()
            if current_time - self.last_heartbeat_time >= self.heartbeat_interval:
                try:
                    await self._send_heartbeat()
                    self.last_heartbeat_time = current_time
                except Exception as e:
                    logger.error(f"发送心跳失败: {e}")
            await asyncio.sleep(0.5)  # 高频检查

    async def _send_heartbeat(self):
        """发送心跳消息"""
        async with self.ws_lock:
            if self.ws_connected and self.websocket and not self.websocket.closed:
                await self.websocket.send_json({"event": "HEARTBEAT", "data": {}})
                logger.debug("已发送心跳")

    async def send_response(self, cmd: str, msg: str, data: Optional[dict] = None):
        """发送HTTP响应到服务器"""
        payload = {"cmd": cmd, "msg": msg}
        if data:
            payload.update(data)

        try:
            async with self.session.post(
                f"{self.server_url}/robot/response",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=2),
            ) as resp:
                if resp.status == 200:
                    logger.debug(f"已发送响应 [{cmd}]: {msg}")
                else:
                    logger.warning(f"响应失败 [{cmd}]: HTTP {resp.status}")
        except Exception as e:
            logger.error(f"发送响应失败 [{cmd}]: {e}")

    async def _update_stream_info_to_server(self):
        """更新流信息到服务器"""
        stream_info_data = cameras_to_stream_json(self.cameras)
        logger.debug(f"更新流信息: {stream_info_data}")
        try:
            async with self.session.post(
                f"{self.server_url}/robot/stream_info",
                json=stream_info_data,
                timeout=aiohttp.ClientTimeout(total=2),
            ) as response:
                if response.status == 200:
                    logger.debug("摄像头流信息已同步到服务器")
                else:
                    logger.warning(f"同步流信息失败: {response.status}")
        except Exception as e:
            logger.error(f"同步流信息异常: {e}")

    ####################### Stream Management ############################
    def stream_info(self, info: Dict[str, int]):
        """更新本地摄像头信息"""
        self.cameras = info.copy()
        logger.info(f"更新摄像头信息: {self.cameras}")
        # 异步更新服务器
        asyncio.create_task(self._update_stream_info_to_server())

    async def update_stream_async(self, name: str, frame):
        """异步更新视频流"""
        if name not in self.cameras:
            logger.warning(f"未知的摄像头流: {name}")
            return

        try:
            _, jpeg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            url = f"{self.server_url}/robot/update_stream/{self.cameras[name]}"

            # 使用短超时，丢帧可接受
            async with self.session.post(
                url, data=jpeg.tobytes(), timeout=aiohttp.ClientTimeout(total=0.2)
            ) as resp:
                if resp.status != 200:
                    txt = await resp.text()
                    logger.error(f"流更新失败 {name}: {resp.status} - {txt}")
        except asyncio.TimeoutError:
            logger.debug(f"流 {name} 更新超时 (可接受)")
        except Exception as e:
            logger.error(f"流 {name} 更新异常: {e}")
