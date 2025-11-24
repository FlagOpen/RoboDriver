"""
设备信息(机器臂、相机）定时推送脚本 (machine)
========================================

一、功能描述
--------
1. 支持设备基本信息配置（可选填）
2. 动态更新设备连接状态
3. 将设备信息格式化为JSON
4. 每分钟自动采集设备信息（如机械臂、相机等）并推送至服务端

二、适用场景
--------
- 机械臂、相机状态监控
- 设备心跳上报

三、配置要求
--------
1. 开发人员需根据实际设备修改入参  `new_machine` 配置项
2. 如无某些设备（如机械臂），对应字段可留空或删除

四、睿尔曼机器设备信息示例
--------
{
    "device_name": "睿尔曼双臂轮式机器人V1.1",
    "device_body": "睿尔曼",
    "specifications": {
        "end_type": "",
        "fps": 20,
        "camera": {
            "number": 3,
            "information": [
                {
                    "name": "cam_high",
                    "chinese_name": "头部摄像头",
                    "type": "Intel RealSense D435",
                    "width": 640,
                    "height": 480,
                    "is_connect": true
                },
                {
                    "name": "cam_left_wrist",
                    "chinese_name": "左腕摄像头",
                    "type": "Intel RealSense D435",
                    "width": 640,
                    "height": 480,
                    "is_connect": true
                },
                {
                    "name": "cam_right_wrist",
                    "chinese_name": "右腕摄像头",
                    "type": "Intel RealSense D435",
                    "width": 640,
                    "height": 480,
                    "is_connect": true
                }
            ]
        },
        "piper": {
            "number": 2,
            "information": [
                {
                    "name": "piper_left",
                    "type": "RM65-B",
                    "start_pose": [
                        -90.0,
                        90.0,
                        90.0,
                        -90.0,
                        0.0,
                        0.0,
                        0.0
                    ],
                    "joint_p_limit": [
                        169.0,
                        102.0,
                        169.0,
                        52.0,
                        169.0,
                        117.0,
                        169.0
                    ],
                    "joint_n_limit": [
                        -169.0,
                        -102.0,
                        -169.0,
                        -167.0,
                        -169.0,
                        -87.0,
                        -169.0
                    ],
                    "is_connect": true
                },
                {
                    "name": "piper_right",
                    "type": "RM65-B",
                    "start_pose": [
                        -90.0,
                        90.0,
                        90.0,
                        -90.0,
                        0.0,
                        0.0,
                        0.0
                    ],
                    "joint_p_limit": [
                        169.0,
                        102.0,
                        169.0,
                        52.0,
                        169.0,
                        117.0,
                        169.0
                    ],
                    "joint_n_limit": [
                        -169.0,
                        -102.0,
                        -169.0,
                        -167.0,
                        -169.0,
                        -87.0,
                        -169.0
                    ],
                    "is_connect": true
                }
            ]
        }
    }
}
"""

import json
import threading
import time
from dataclasses import asdict, dataclass, field
from typing import List, Optional

import requests


@dataclass
class CameraInfo:
    name: str = ""
    chinese_name: str = ""
    type: str = ""
    width: int = 0
    height: int = 0
    is_connect: bool = False


@dataclass
class CameraConfig:
    number: int = 0
    information: List[CameraInfo] = field(default_factory=list)

    def __post_init__(self):
        if not self.information:  # 如果 information 为空，则 number 设为 0
            self.number = 0
        else:
            self.number = len(self.information)


@dataclass
class PiperInfo:
    name: str = ""
    type: str = ""
    start_pose: List[float] = field(default_factory=list)
    joint_p_limit: List[float] = field(default_factory=list)
    joint_n_limit: List[float] = field(default_factory=list)
    is_connect: bool = False


@dataclass
class PiperConfig:
    number: int = 0
    information: List[PiperInfo] = field(default_factory=list)

    def __post_init__(self):
        if not self.information:  # 如果 information 为空，则 number 设为 0
            self.number = 0
        else:
            self.number = len(self.information)


@dataclass
class Specifications:
    end_type: str = ""
    fps: int = 20
    camera: Optional[CameraConfig] = None
    piper: Optional[PiperConfig] = None


@dataclass
class MachineInformation:
    device_name: str = ""
    device_body: str = ""
    specifications: Specifications = field(default_factory=Specifications)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: dict) -> "MachineInformation":
        # 处理 specifications 字段
        specs_data = data.get("specifications", {})

        # 处理 camera 字段（可能是 dict 或 list）
        if "camera" in specs_data:
            if specs_data["camera"] is None:
                pass  # 保持 None
            elif isinstance(specs_data["camera"], dict):
                if "information" in specs_data["camera"]:
                    info_list = specs_data["camera"]["information"]
                    if isinstance(info_list, list):
                        specs_data["camera"]["information"] = [
                            CameraInfo(**cam) for cam in info_list
                        ]
                specs_data["camera"] = CameraConfig(**specs_data["camera"])
            elif isinstance(specs_data["camera"], list):
                specs_data["camera"] = CameraConfig(
                    information=[CameraInfo(**cam) for cam in specs_data["camera"]]
                )

        # 处理 piper 字段（可能是 dict 或 list）
        if "piper" in specs_data:
            if specs_data["piper"] is None:
                pass  # 保持 None
            elif isinstance(specs_data["piper"], dict):
                if "information" in specs_data["piper"]:
                    info_list = specs_data["piper"]["information"]
                    if isinstance(info_list, list):
                        specs_data["piper"]["information"] = [
                            PiperInfo(**pipe) for pipe in info_list
                        ]
                specs_data["piper"] = PiperConfig(**specs_data["piper"])
            elif isinstance(specs_data["piper"], list):
                specs_data["piper"] = PiperConfig(
                    information=[PiperInfo(**pipe) for pipe in specs_data["piper"]]
                )

        # 更新 specifications
        if specs_data:
            data["specifications"] = Specifications(**specs_data)

        return cls(**data)


class MachineInformationPost:
    def __init__(self):
        self._machine_info_dict = None  # 内部存储字典格式
        self._lock = threading.Lock()  # 线程锁
        self._running = False  # 控制线程启停

    def init_machine_information(self, new_machine: MachineInformation):
        """初始化设备信息（存储为字典格式，确保与示例结构一致）"""
        # 深拷贝原始数据，避免修改传入对象
        machine_dict = new_machine.to_dict()

        # 处理 camera 字段：如果传入的是列表，转换为字典格式
        if "camera" in machine_dict["specifications"]:
            camera = machine_dict["specifications"]["camera"]
            if isinstance(camera, list):  # 您的传参是列表
                machine_dict["specifications"]["camera"] = {
                    "number": len(camera),
                    "information": camera,
                }
            elif isinstance(camera, dict):  # 如果已经是字典，确保有 number 字段
                if "information" in camera and "number" not in camera:
                    camera["number"] = len(camera["information"])

        # 处理 piper 字段：同上
        if "piper" in machine_dict["specifications"]:
            piper = machine_dict["specifications"]["piper"]
            if isinstance(piper, list):  # 您的传参是列表
                machine_dict["specifications"]["piper"] = {
                    "number": len(piper),
                    "information": piper,
                }
            elif isinstance(piper, dict):  # 如果已经是字典，确保有 number 字段
                if "information" in piper and "number" not in piper:
                    piper["number"] = len(piper["information"])

        self._machine_info_dict = machine_dict
        print("初始化后的数据结构:")
        print(json.dumps(self._machine_info_dict, indent=2, ensure_ascii=False))

    def update_connection_status(
        self, device_type=None, device_name=None, new_status=None, set_all=None
    ):
        """更新设备连接状态（线程安全）"""
        with self._lock:
            if not self._machine_info_dict:
                raise RuntimeError("请先调用 init_machine_information 初始化数据")

            # 批量设置所有设备
            if set_all is not None and device_type is not None:
                if device_type not in ["camera", "piper"]:
                    raise ValueError("设备类型必须是 'camera' 或 'piper'")
                devices = (
                    self._machine_info_dict["specifications"]
                    .get(device_type, {})
                    .get("information", [])
                )
                for device in devices:
                    device["is_connect"] = bool(new_status)
                return

            # 单个设备设置
            elif device_type and device_name is not None and new_status is not None:
                if device_type not in ["camera", "piper"]:
                    raise ValueError("设备类型必须是 'camera' 或 'piper'")
                devices = (
                    self._machine_info_dict["specifications"]
                    .get(device_type, {})
                    .get("information", [])
                )
                for device in devices:
                    if device["name"] == device_name:
                        device["is_connect"] = bool(new_status)
                        return
                raise ValueError(f"未找到名称 '{device_name}' 的 {device_type} 设备")

            else:
                raise ValueError(
                    "参数组合错误：请提供 (device_type+device_name+new_status) 或 (device_type+set_all+new_status)"
                )

    def send_machine_info_periodically(self, interval_seconds=60):
        """定时发送设备信息（线程安全）"""
        url = "http://localhost:8088/robot/update_machine_information"
        headers = {"Content-Type": "application/json"}
        self._running = True

        while self._running:
            try:
                with self._lock:
                    if not self._machine_info_dict:
                        continue
                    json_data = json.dumps(self._machine_info_dict, ensure_ascii=False)

                response = requests.post(
                    url, data=json_data, headers=headers, timeout=10
                )

                if response.status_code == 200:
                    print(f"[{time.ctime()}] 信息发送成功: {response.text}")
                else:
                    print(f"[{time.ctime()}] 发送失败，状态码: {response.status_code}")

            except requests.exceptions.RequestException as e:
                print(f"[{time.ctime()}] 请求异常: {str(e)}")

            time.sleep(interval_seconds)

    def start(self, new_machine: MachineInformation):
        """启动后台线程"""
        self.init_machine_information(new_machine)
        self._thread = threading.Thread(
            target=self.send_machine_info_periodically, daemon=True
        )
        self._thread.start()

    def stop(self):
        """停止后台线程"""
        self._running = False
        if hasattr(self, "_thread"):
            self._thread.join()


# 示例用法
if __name__ == "__main__":
    new_machine = MachineInformation(
        device_name="星海图R1-Lite",
        device_body="星海图",
        specifications=Specifications(
            end_type="二指夹爪",
            fps=30,
            camera=[
                CameraInfo(
                    name="top_left",
                    chinese_name="头部左摄像头",
                    type="纯双目视觉相机",
                    width=1280,
                    height=720,
                    is_connect=False,
                ),
                CameraInfo(
                    name="top_right",
                    chinese_name="头部右摄像头",
                    type="纯双目视觉相机",
                    width=1280,
                    height=720,
                    is_connect=False,
                ),
                CameraInfo(
                    name="wrist_left",
                    chinese_name="腕部左摄像头",
                    type="单目深度相机",
                    width=640,
                    height=360,
                    is_connect=False,
                ),
                CameraInfo(
                    name="wrist_right",
                    chinese_name="腕部右摄像头",
                    type="单目深度相机",
                    width=640,
                    height=360,
                    is_connect=False,
                ),
            ],
            piper=[
                PiperInfo(
                    name="piper_left",
                    type="Galaxea A1X + Galaxea G1 - 7DOF",
                    start_pose=[],
                    joint_p_limit=[165.0, 180.0, 0.0, 90.0, 90.0, 165.0],
                    joint_n_limit=[-165.0, 0.0, -190.0, -90.0, -90.0, -165.0],
                    is_connect=False,
                ),
                PiperInfo(
                    name="piper_right",
                    type="Galaxea A1X + Galaxea G1 - 7DOF",
                    start_pose=[],
                    joint_p_limit=[165.0, 180.0, 0.0, 90.0, 90.0, 165.0],
                    joint_n_limit=[-165.0, 0.0, -190.0, -90.0, -90.0, -165.0],
                    is_connect=False,
                ),
            ],
        ),
    )

    # 启动后台任务
    machine_post = MachineInformationPost()
    machine_post.start(new_machine)

    # 更新状态（非阻塞）
    machine_post.update_connection_status(
        device_type="camera", set_all=True, new_status=True
    )
    machine_post.update_connection_status(
        device_type="piper", set_all=True, new_status=True
    )
