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
from robodriver.core.inferencer import InferenceConfig, Inferencer
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

logger = logging_mp.getLogger(__name__)


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
        self.inferring = False
        self.saveing = False

        self.cameras = {"image_top": 1, "image_right": 2}

        # 注册异步回调
        self.sio.on("HEARTBEAT_RESPONSE", self.__on_heartbeat_response_handle)
        self.sio.on("connect", self.__on_connect_handle)
        self.sio.on("disconnect", self.__on_disconnect_handle)
        self.sio.on("robot_command", self.__on_robot_command_handle)

        self.record = None
        self.inferencer = None

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
            target_dir = dataset_path / date_str / "dev" / task_dir / repo_id
            # if "release" in git_branch_name or "main" in git_branch_name:
            #     target_dir = dataset_path / date_str / "user" / task_dir / repo_id
            # elif "dev" in git_branch_name:
            #     target_dir = dataset_path / date_str / "dev" / task_dir / repo_id
            # else:
            #     target_dir = dataset_path / date_str / "dev" / task_dir / repo_id

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
            target_dir = dataset_path / date_str / "dev" / task_dir / repo_id
            # if "release" in git_branch_name or "main" in git_branch_name:
            #     target_dir = dataset_path / date_str / "user" / task_dir / repo_id
            # elif "dev" in git_branch_name:
            #     target_dir = dataset_path / date_str / "dev" / task_dir / repo_id
            # else:
            #     target_dir = dataset_path / date_str / "dev" / task_dir / repo_id

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
                        mode="local",
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
                    "url": f"http://127.0.0.1:{RERUN_WEB_PORT}/?url=rerun%2Bhttp%3A%2F%2F127.0.0.1%3A{RERUN_WS_PORT}%2Fproxy",
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

        elif data.get("cmd") == "start_inference":
            logger.info("处理开始推理命令...")
            msg = data.get("msg")
            
            # 检查是否有其他任务在运行
            if self.recording:
                logger.warning("Recording is running, cannot start inference.")
                await self.send_response("start_inference", "Recording is running, cannot start inference")
                return
            if self.replaying:
                logger.warning("Replay is running, cannot start inference.")
                await self.send_response("start_inference", "Replay is running, cannot start inference")
                return
            if self.inferring:
                logger.warning("Inference is already running.")
                await self.send_response("start_inference", "Inference is already running")
                return
            
            try:
                # 解析配置
                fps = msg.get("fps", 30)
                prompt = msg.get("prompt", "default_task")
                print(prompt)

                # 支持从 data_channel 提取完整 URL 信息
                data_channel = msg.get("data_channel", {})
                # policy_host = "localhost"
                # policy_host = "192.168.19.2"
                policy_host = "106.63.14.88"
                # policy_port = 8087
                policy_port = 8003
                # policy_path = "/inference"
                policy_path = ""

                # if data_channel.get("url"):
                #     # 解析 ws://host:port/path 格式
                #     import re
                #     url_match = re.match(r"ws://([^:/]+):(\d+)(/\S*)?", data_channel["url"])
                #     if url_match:
                #         policy_host = url_match.group(1)
                #         policy_port = int(url_match.group(2))
                #         if url_match.group(3):
                #             policy_path = url_match.group(3)
                #     else:
                #         logger.warning(f"Could not parse data_channel URL: {data_channel['url']}")

                # 策略类型：支持 "flagscale"（默认）、"openpi"、"bc"
                # "bc" 使用 bc_robodriver 风格的推理路径（直接观察提取 + msgpack WebSocket）
                policy_type = msg.get("policy_type", "bc")

                logger.info(
                    f"Starting inference: policy_type={policy_type}, "
                    f"host={policy_host}, port={policy_port}, path={policy_path}, fps={fps}"
                )

                # 创建 inferencer 配置
                infer_cfg = InferenceConfig(
                    policy_host=policy_host,
                    policy_port=policy_port,
                    policy_path=policy_path,
                    prompt=prompt,
                    fps=fps,
                    policy_type=policy_type,
                )

                # 创建 inferencer
                self.inferencer = Inferencer(
                    robot=self.daemon.robot,
                    daemon=self.daemon,
                    teleop=self.teleop,
                    infer_cfg=infer_cfg,
                )

                # 连接数据通道
                self.inferencer.connect()

                # 启动推理
                self.inferring = True
                self.inferencer.start()

                logger.info("Inference started successfully")
                await self.send_response("start_inference", "success")

            except Exception as e:
                self.inferring = False
                logger.error(f"Failed to start inference: {e}")
                await self.send_response("start_inference", str(e))

        elif data.get("cmd") == "reset_pose":
            logger.info("处理重置位姿命令...")
            
            # 检查是否有其他任务在运行
            if self.recording:
                logger.warning("Recording is running, cannot reset pose.")
                await self.send_response("reset_pose", "Recording is running, cannot reset pose")
                return
            if self.replaying:
                logger.warning("Replay is running, cannot reset pose.")
                await self.send_response("reset_pose", "Replay is running, cannot reset pose")
                return
            if self.inferring:
                logger.warning("Inference is running, cannot reset pose.")
                await self.send_response("reset_pose", "Inference is running, cannot reset pose")
                return
            
            try:
                robot = self.daemon.robot
                action_features = robot.action_features
                
                # 读取当前 observation 获取关节位置
                observation = self.daemon.get_observation()
                if observation is None:
                    observation = robot.get_observation()
                
                # 构建起始位置和目标位置
                start_pose = {}
                target_pose = {}
                
                for key in action_features.keys():
                    # 将 leader_ 替换为 follower_ 去 observation 中查找当前值
                    obs_key = key.replace("leader_", "follower_")
                    if obs_key in observation:
                        start_pose[key] = float(observation[obs_key])
                    else:
                        # 兜底：如果找不到对应字段，使用 0
                        logger.warning(f"Could not find {obs_key} in observation, using 0 as start value")
                        start_pose[key] = 0.0
                    
                    # 目标值：gripper 保持当前值不调整，其他关节复位到 0
                    if "gripper" in key:
                        target_pose[key] = 100
                    else:
                        target_pose[key] = 0.0  # 关节归零
                
                # 规划匀速轨迹（1.5秒）
                num_steps = int(1.5 * DEFAULT_FPS)  # 约 45 步 @ 30fps
                logger.info(f"Planning reset trajectory: {num_steps} steps, duration: 1.5s")
                logger.info(f"Start pose: {start_pose}")
                logger.info(f"Target pose: {target_pose}")
                
                # 逐步发送插值动作
                for i in range(1, num_steps + 1):
                    if not self.running:
                        logger.warning("Client disconnected, aborting reset trajectory")
                        break
                    
                    t = i / num_steps  # 插值系数 0 -> 1
                    action = {}
                    for key in action_features.keys():
                        # 线性插值: start + (target - start) * t
                        action[key] = start_pose[key] + (target_pose[key] - start_pose[key]) * t
                    
                    robot.send_action(action)
                    time.sleep(1.0 / DEFAULT_FPS)
                
                logger.info("Reset pose completed successfully")
                await self.send_response("reset_pose", "success")
                
            except Exception as e:
                logger.error(f"Failed to reset pose: {e}")
                await self.send_response("reset_pose", str(e))

        elif data.get("cmd") == "stop_inference":
            logger.info("处理停止推理命令...")
            
            if not self.inferring:
                logger.warning("Inference is not running")
                await self.send_response("stop_inference", "success")
                return
            
            try:
                if self.inferencer is not None:
                    self.inferencer.stop()
                self.inferring = False
                logger.info("Inference stopped successfully")
                await self.send_response("stop_inference", "success")
                
            except Exception as e:
                logger.error(f"Failed to stop inference: {e}")
                self.inferring = False
                await self.send_response("stop_inference", str(e))

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
