import time
import torch
import logging_mp
import numpy as np

from typing import Any

from lerobot.utils.errors import DeviceNotConnectedError, DeviceAlreadyConnectedError
from lerobot.teleoperators.teleoperator import Teleoperator

from .config import SO101LeaderDoraTeleoperatorConfig
from .status import  SO101LeaderDoraTeleoperatorStatus
from .node import  SO101LeaderDoraTeleoperatorNode


logger = logging_mp.getLogger(__name__)


class SO101LeaderDoraTeleoperator(Teleoperator):
    config_class = SO101LeaderDoraTeleoperatorConfig
    name = "so101_leader_dora"

    def __init__(self, config: SO101LeaderDoraTeleoperatorConfig):
        super().__init__(config)
        self.config = config
        self.teleoperator_type = self.config.type

        # self.leader_arms = {}
        # self.leader_arms['main_leader'] = self.config.leader_arms["main"]
        self.motors = config.motors # 大道至简。所有的action，不管是左右手臂，还是头部等等，能动的都算做motor。

        self.status = SO101LeaderDoraTeleoperatorStatus()
        self.teleoperator_dora_node = SO101LeaderDoraTeleoperatorNode()
        self.teleoperator_dora_node.start()

        self.connected = False
        self.logs = {}

    @property
    def action_features(self) -> dict[str, type]:
        return {f"{motor}.pos": float for motor in self.motors}
    
    @property
    def feedback_features(self) -> dict[str, type]:
        return {}
    
    @property
    def is_connected(self) -> bool:
        return self.connected
    
    def connect(self):
        timeout = 20
        start_time = time.perf_counter()

        if self.connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        # 定义所有需要等待的条件及其错误信息
        conditions = [
            (
                lambda: all(
                    any(name in key for key in self.teleoperator_dora_node.recv_joint)
                    for name in self.motors
                ),
                lambda: [name for name in self.motors if not any(name in key for key in self.teleoperator_dora_node.recv_joint)],
                "等待主臂关节角度超时"
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

                        # # 计算已接收的项
                        # if i == 0:
                        #     received = [name for name in self.cameras if name not in missing]
                        # else:
                        #     received = [name for name in self.follower_arms if name not in missing]
                        received = [name for name in self.motors if name not in missing]

                        # 构造错误信息
                        msg = f"{base_msg}: 未收到 [{', '.join(missing)}]; 已收到 [{', '.join(received)}]"
                        failed_messages.append(msg)

                # 如果所有条件都已完成，break
                if not failed_messages:
                    break

                # 抛出超时异常
                raise TimeoutError(f"连接超时，未满足的条件: {'; '.join(failed_messages)}")

            # 减少 CPU 占用
            time.sleep(0.01)

        # ===== 新增成功打印逻辑 =====
        success_messages = []
        # 摄像头连接状态
        # if conditions[0][0]():
        #     cam_received = [name for name in self.cameras 
        #                 if name in self.teleoperator_dora_node.recv_images and name not in self.connect_excluded_cameras]
        #     success_messages.append(f"摄像头: {', '.join(cam_received)}")

        # 主臂数据状态
        arm_data_types = ["主臂关节角度",]
        for i, data_type in enumerate(arm_data_types, 0):
            if conditions[i][0]():
                arm_received = [name for name in self.motors 
                            if any(name in key for key in (self.teleoperator_dora_node.recv_joint,)[i-1])]
                success_messages.append(f"{data_type}: {', '.join(arm_received)}")
        
        # # 从臂数据状态
        # arm_data_types = ["从臂关节角度",]
        # for i, data_type in enumerate(arm_data_types, 1):
        #     if conditions[i][0]():
        #         arm_received = [name for name in self.follower_arms 
        #                     if any(name in key for key in (self.teleoperator_dora_node.recv_joint,)[i-1])]
        #         success_messages.append(f"{data_type}: {', '.join(arm_received)}")
        
        log_message = "\n[连接成功] 所有设备已就绪:\n"
        log_message += "\n".join(f"  - {msg}" for msg in success_messages)
        log_message += f"\n  总耗时: {time.perf_counter() - start_time:.2f} 秒\n"
        logger.info(log_message)
        # ===========================


        for i in range(self.status.specifications.arm.number):
            self.status.specifications.arm.information[i].is_connect = True

        self.connected = True
    
    # def teleop_step(
    #     self, record_data=False, 
    # ) -> None | tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:

    #     if not self.connected:
    #         raise DeviceNotConnectedError(
    #             f"{self} is not connected. You need to run `robot.connect()`."
    #         )
        
    #     for key in self.teleoperator_dora_node.recv_joint_status:
    #         self.teleoperator_dora_node.recv_joint_status[key] = max(0, self.teleoperator_dora_node.recv_joint_status[key] - 1)

    #     if not record_data:
    #         return
                    
    #     leader_joint = {}
    #     for name in self.leader_arms:
    #         for match_name in self.teleoperator_dora_node.recv_joint:
    #             if name in match_name:
    #                 now = time.perf_counter()

    #                 byte_array = np.zeros(6, dtype=np.float32)
    #                 pose_read = self.teleoperator_dora_node.recv_joint[match_name]

    #                 byte_array[:6] = pose_read[:]
    #                 byte_array = np.round(byte_array, 3)
                    
    #                 leader_joint[name] = torch.from_numpy(byte_array)

    #                 self.logs[f"read_leader_{name}_joint_dt_s"] = time.perf_counter() - now

    #     #记录当前关节角度
    #     state = []
    #     for name in self.follower_arms:
    #         if name in follower_joint:
    #             state.append(follower_joint[name])
    #     state = torch.cat(state)

    #     #将关节目标位置添加到 action 列表中
    #     action = []
    #     for name in self.leader_arms:
    #         if name in leader_joint:
    #             action.append(leader_joint[name])
    #     action = torch.cat(action)

    #     # Capture images from cameras
    #     images = {}
    #     for name in self.cameras:
    #         now = time.perf_counter()
    #         images[name] = self.teleoperator_dora_node.recv_images[name]
    #         images[name] = torch.from_numpy(images[name])
    #         self.logs[f"read_camera_{name}_dt_s"] = time.perf_counter() - now

    #     # Populate output dictionnaries and format to pytorch
    #     obs_dict, action_dict = {}, {}
    #     obs_dict["observation.state"] = state
    #     action_dict["action"] = action
    #     for name in self.cameras:
    #         obs_dict[f"observation.images.{name}"] = images[name]

    #     # print("end teleoperate record")
    #     return obs_dict, action_dict

    @property
    def is_calibrated(self) -> bool:
        """Whether the teleoperator is currently calibrated or not. Should be always `True` if not applicable"""
        return True

    def calibrate(self) -> None:
        """
        Calibrate the teleoperator if applicable. If not, this should be a no-op.

        This method should collect any necessary data (e.g., motor offsets) and update the
        :pyattr:`calibration` dictionary accordingly.
        """
        pass

    def configure(self) -> None:
        """
        Apply any one-time or runtime configuration to the teleoperator.
        This may include setting motor parameters, control modes, or initial state.
        """
        pass
    
    def get_action(self) -> dict[str, float]:
        if not self.connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        for key in self.teleoperator_dora_node.recv_joint_status:
            self.teleoperator_dora_node.recv_joint_status[key] = max(0, self.teleoperator_dora_node.recv_joint_status[key] - 1)

        # Read arm position
        start = time.perf_counter()
        action = {
            f"{motor}.pos": val
            for motor in self.motors
                for name, val in self.teleoperator_dora_node.recv_joint.items() 
                    if motor in name
        }
        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read state: {dt_ms:.1f} ms")

        return action

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        if not self.connected:
            raise DeviceNotConnectedError(
                f"{self} is not connected. You need to run `robot.connect()`."
            )
        
        logger.critical(f"{self}: send_feedback() not implemented.")
        raise NotImplementedError

    def update_status(self) -> str:
        for i in range(self.status.specifications.arm.number):
            match_name = self.status.specifications.arm.information[i].name
            for name in self.teleoperator_dora_node.recv_joint_status:
                if match_name in name:
                    self.status.specifications.arm.information[i].is_connect = True if self.teleoperator_dora_node.recv_joint_status[name]>0 else False

        return self.status.to_json()

    def disconnect(self):
        if not self.connected:
            raise DeviceNotConnectedError(
                "Teleoperator is not connected. You need to run `teleoperator.connect()` before disconnecting."
            )

        self.teleoperator_dora_node.running = False
        self.teleoperator_dora_node.thread.join()

        self.connected = False
        logger.info(f"{self} is not connected.")

    def __del__(self):
        if getattr(self, "connected", False):
            self.disconnect()
