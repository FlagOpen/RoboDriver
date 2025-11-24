import threading
import time
from typing import Any

import logging_mp
import numpy as np
import torch
from lerobot.cameras import make_cameras_from_configs
from lerobot.robots.robot import Robot
from lerobot.utils.errors import DeviceNotConnectedError

from .config import SO101AIODoraRobotConfig
from .recv_zmq import (
    recv_image_server,
    recv_images,
    recv_images_status,
    recv_joint,
    recv_joint_server,
    recv_joint_status,
)
from .status import SO101AIODoraRobotStatus

logger = logging_mp.get_logger(__name__)
CONNECT_TIMEOUT_FRAME = 10


class SO101AIODoraRobot(Robot):
    def __init__(self, config: SO101AIODoraRobotConfig):
        self.config = config
        self.status = SO101AIODoraRobotStatus()
        self.robot_type = self.config.type

        self.use_videos = self.config.use_videos

        self.microphones = self.config.microphones

        self.leader_arms = {}
        self.leader_arms["main_leader"] = self.config.leader_arms["main"]

        self.follower_arms = {}
        self.follower_arms["main_follower"] = self.config.follower_arms["main"]

        self.cameras = make_cameras_from_configs(self.config.cameras)

        self.connect_excluded_cameras = ["image_pika_pose"]

        self.recv_image_thread = threading.Thread(target=recv_image_server, daemon=True)
        self.recv_image_thread.start()

        self.recv_joint_thread = threading.Thread(target=recv_joint_server, daemon=True)
        self.recv_joint_thread.start()

        self.is_connected = False
        self.logs = {}

    def get_motor_names(self, arms: dict[str, dict]) -> list:
        return [f"{arm}_{motor}" for arm, bus in arms.items() for motor in bus.motors]

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
    def microphone_features(self) -> dict:
        mic_ft = {}
        for mic_key, mic in self.microphones.items():
            key = f"observation.audio.{mic_key}"
            mic_ft[key] = {
                "shape": (1,),
                "names": ["channels"],
                "info": None,
            }
        return mic_ft

    @property
    def motor_features(self) -> dict:
        action_names = self.get_motor_names(self.leader_arms)
        state_names = self.get_motor_names(self.follower_arms)
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
        }

    def connect(self):
        timeout = 50  # 统一的超时时间（秒）
        start_time = time.perf_counter()

        # 定义所有需要等待的条件及其错误信息
        conditions = [
            (
                lambda: all(
                    name in recv_images
                    for name in self.cameras
                    if name not in self.connect_excluded_cameras
                ),
                lambda: [name for name in self.cameras if name not in recv_images],
                "等待摄像头图像超时",
            ),
            (
                lambda: all(
                    any(name in key for key in recv_joint) for name in self.leader_arms
                ),
                lambda: [
                    name
                    for name in self.leader_arms
                    if not any(name in key for key in recv_joint)
                ],
                "等待主臂关节角度超时",
            ),
            (
                lambda: all(
                    any(name in key for key in recv_joint)
                    for name in self.follower_arms
                ),
                lambda: [
                    name
                    for name in self.follower_arms
                    if not any(name in key for key in recv_joint)
                ],
                "等待从臂关节角度超时",
            ),
        ]

        # 跟踪每个条件是否已完成
        completed = [False] * len(conditions)

        while True:
            # 检查每个未完成的条件
            for i in range(len(conditions)):
                if not completed[i]:
                    condition_func = conditions[i][0]
                    if condition_func():
                        completed[i] = True

            # 如果所有条件都已完成，退出循环
            if all(completed):
                break

            # 检查是否超时
            if time.perf_counter() - start_time > timeout:
                failed_messages = []
                for i in range(len(completed)):
                    if not completed[i]:
                        condition_func, get_missing, base_msg = conditions[i]
                        missing = get_missing()

                        # 重新检查条件是否满足（可能刚好在最后一次检查后满足）
                        if condition_func():
                            completed[i] = True
                            continue

                        # 如果没有 missing，也视为满足
                        if not missing:
                            completed[i] = True
                            continue

                        # 计算已接收的项
                        if i == 0:
                            received = [
                                name for name in self.cameras if name not in missing
                            ]
                        else:
                            received = [
                                name
                                for name in self.follower_arms
                                if name not in missing
                            ]

                        # 构造错误信息
                        msg = f"{base_msg}: 未收到 [{', '.join(missing)}]; 已收到 [{', '.join(received)}]"
                        failed_messages.append(msg)

                # 如果所有条件都已完成，break
                if not failed_messages:
                    break

                # 抛出超时异常
                raise TimeoutError(
                    f"连接超时，未满足的条件: {'; '.join(failed_messages)}"
                )

            # 减少 CPU 占用
            time.sleep(0.01)

        # ===== 新增成功打印逻辑 =====
        success_messages = []
        # 摄像头连接状态
        if conditions[0][0]():
            cam_received = [
                name
                for name in self.cameras
                if name in recv_images and name not in self.connect_excluded_cameras
            ]
            success_messages.append(f"摄像头: {', '.join(cam_received)}")

        # 主臂数据状态
        arm_data_types = [
            "主臂关节角度",
        ]
        for i, data_type in enumerate(arm_data_types, 1):
            if conditions[i][0]():
                arm_received = [
                    name
                    for name in self.leader_arms
                    if any(name in key for key in (recv_joint,)[i - 1])
                ]
                success_messages.append(f"{data_type}: {', '.join(arm_received)}")

        # 从臂数据状态
        arm_data_types = [
            "从臂关节角度",
        ]
        for i, data_type in enumerate(arm_data_types, 1):
            if conditions[i][0]():
                arm_received = [
                    name
                    for name in self.follower_arms
                    if any(name in key for key in (recv_joint,)[i - 1])
                ]
                success_messages.append(f"{data_type}: {', '.join(arm_received)}")

        # 打印成功连接信息
        # print("\n[连接成功] 所有设备已就绪:")
        # for msg in success_messages:
        #     print(f"  - {msg}")
        # print(f"  总耗时: {time.perf_counter() - start_time:.2f}秒\n")
        log_message = "\n[连接成功] 所有设备已就绪:\n"
        log_message += "\n".join(f"  - {msg}" for msg in success_messages)
        log_message += f"\n  总耗时: {time.perf_counter() - start_time:.2f}秒\n"
        logger.info(log_message)
        # ===========================

        for i in range(self.status.specifications.camera.number):
            self.status.specifications.camera.information[i].is_connect = True
        for i in range(self.status.specifications.arm.number):
            self.status.specifications.arm.information[i].is_connect = True

        self.is_connected = True

    @property
    def features(self):
        return {**self.motor_features, **self.camera_features}

    @property
    def has_camera(self):
        return len(self.cameras) > 0

    @property
    def num_cameras(self):
        return len(self.cameras)

    # def calibrate(self) -> None:
    #     if self.calibration:
    #         # self.calibration is not empty here
    #         user_input = input(
    #             f"Press ENTER to use provided calibration file associated with the id {self.id}, or type 'c' and press ENTER to run calibration: "
    #         )
    #         if user_input.strip().lower() != "c":
    #             logger.info(f"Writing calibration file associated with the id {self.id} to the motors")
    #             self.bus.write_calibration(self.calibration)
    #             return

    #     logger.info(f"\nRunning calibration of {self}")
    #     self.bus.disable_torque()
    #     for motor in self.bus.motors:
    #         self.bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)

    #     input(f"Move {self} to the middle of its range of motion and press ENTER....")
    #     homing_offsets = self.bus.set_half_turn_homings()

    #     print(
    #         "Move all joints sequentially through their entire ranges "
    #         "of motion.\nRecording positions. Press ENTER to stop..."
    #     )
    #     range_mins, range_maxes = self.bus.record_ranges_of_motion()

    #     self.calibration = {}
    #     for motor, m in self.bus.motors.items():
    #         self.calibration[motor] = MotorCalibration(
    #             id=m.id,
    #             drive_mode=0,
    #             homing_offset=homing_offsets[motor],
    #             range_min=range_mins[motor],
    #             range_max=range_maxes[motor],
    #         )

    #     self.bus.write_calibration(self.calibration)
    #     self._save_calibration()
    #     print("Calibration saved to", self.calibration_fpath)

    def teleop_step(
        self,
        record_data=False,
    ) -> None | tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        if not self.is_connected:
            raise DeviceNotConnectedError(
                "Aloha is not connected. You need to run `robot.connect()`."
            )

        for key in recv_images_status:
            recv_images_status[key] = max(0, recv_images_status[key] - 1)

        for key in recv_joint_status:
            recv_joint_status[key] = max(0, recv_joint_status[key] - 1)

        if not record_data:
            return

        follower_joint = {}
        for name in self.follower_arms:
            for match_name in recv_joint:
                if name in match_name:
                    now = time.perf_counter()

                    byte_array = np.zeros(6, dtype=np.float32)
                    pose_read = recv_joint[match_name]

                    byte_array[:6] = pose_read[:]
                    byte_array = np.round(byte_array, 3)

                    follower_joint[name] = torch.from_numpy(byte_array)

                    self.logs[f"read_follower_{name}_joint_dt_s"] = (
                        time.perf_counter() - now
                    )

        leader_joint = {}
        for name in self.leader_arms:
            for match_name in recv_joint:
                if name in match_name:
                    now = time.perf_counter()

                    byte_array = np.zeros(6, dtype=np.float32)
                    pose_read = recv_joint[match_name]

                    byte_array[:6] = pose_read[:]
                    byte_array = np.round(byte_array, 3)

                    leader_joint[name] = torch.from_numpy(byte_array)

                    self.logs[f"read_leader_{name}_joint_dt_s"] = (
                        time.perf_counter() - now
                    )

        # 记录当前关节角度
        state = []
        for name in self.follower_arms:
            if name in follower_joint:
                state.append(follower_joint[name])
        state = torch.cat(state)

        # 将关节目标位置添加到 action 列表中
        action = []
        for name in self.leader_arms:
            if name in leader_joint:
                action.append(leader_joint[name])
        action = torch.cat(action)

        # Capture images from cameras
        images = {}
        for name in self.cameras:
            now = time.perf_counter()
            images[name] = recv_images[name]
            images[name] = torch.from_numpy(images[name])
            self.logs[f"read_camera_{name}_dt_s"] = time.perf_counter() - now

        # Populate output dictionnaries and format to pytorch
        obs_dict, action_dict = {}, {}
        obs_dict["observation.state"] = state
        action_dict["action"] = action
        for name in self.cameras:
            obs_dict[f"observation.images.{name}"] = images[name]

        # print("end teleoperate record")
        return obs_dict, action_dict

    def send_action(self, action: dict[str, Any]):
        """The provided action is expected to be a vector."""
        if not self.is_connected:
            raise DeviceNotConnectedError(
                "KochRobot is not connected. You need to run `robot.connect()`."
            )

        for name in self.leader_arms:
            goal_joint = [
                val for key, val in action.items() if name in key and "joint" in key
            ]
            # goal_gripper = [ val for key, val in action.items() if name in key and "gripper" in key]

            # goal_joint = action[(arm_index*arm_action_dim+from_idx):(arm_index*arm_action_dim+to_idx)]
            # goal_gripper = action[arm_index*arm_action_dim + 12]
            # arm_index += 1
            goal_joint_numpy = np.array(
                [t.item() for t in goal_joint], dtype=np.float32
            )
            # goal_gripper_numpy = np.array([t.item() for t in goal_gripper], dtype=np.float32)
            # position = np.concatenate([goal_joint_numpy, goal_gripper_numpy], axis=0)

            so101_zmq_send(f"action_joint_{name}", goal_joint_numpy, wait_time_s=0.01)

    def update_status(self) -> str:
        for i in range(self.status.specifications.camera.number):
            match_name = self.status.specifications.camera.information[i].name
            for name in recv_images_status:
                if match_name in name:
                    self.status.specifications.camera.information[i].is_connect = (
                        True if recv_images_status[name] > 0 else False
                    )

        for i in range(self.status.specifications.arm.number):
            match_name = self.status.specifications.arm.information[i].name
            for name in recv_joint_status:
                if match_name in name:
                    self.status.specifications.arm.information[i].is_connect = (
                        True if recv_joint_status[name] > 0 else False
                    )

        return self.status.to_json()

    def disconnect(self):
        if not self.is_connected:
            raise DeviceNotConnectedError(
                "Aloha is not connected. You need to run `robot.connect()` before disconnecting."
            )

        self.is_connected = False
        global running_recv_image_server
        global running_recv_joint_server
        running_recv_image_server = False
        running_recv_joint_server = False

        self.recv_image_thread.join()
        self.recv_joint_thread.join()

    def __del__(self):
        if getattr(self, "is_connected", False):
            self.disconnect()
