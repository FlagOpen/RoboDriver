from dataclasses import dataclass

from operating_platform.robot.robots.statuses import (
    RobotStatus,
    CameraStatus,
    ArmStatus,
    CameraInfo,
    ArmInfo,
)


@RobotStatus.register_subclass("galaxea")
@dataclass
class GALAXEARobotStatus(RobotStatus):
    device_name: str = "星海图R1-Lite"
    device_body: str = "星海图"

    def __post_init__(self):
        self.specifications.end_type = "二指夹爪"
        self.specifications.fps = 30
        self.specifications.camera = CameraStatus(
            information=[
                CameraInfo(
                    name="image_top_left",
                    chinese_name="头部左摄像头",
                    type="纯双目视觉相机",
                    width=1280,
                    height=720,
                    is_connect=False
                ),
                CameraInfo(
                    name="image_top_right",
                    chinese_name="头部右摄像头",
                    type="纯双目视觉相机",
                    width=1280,
                    height=720,
                    is_connect=False
                ),
                CameraInfo(
                    name="image_wrist_left",
                    chinese_name="腕部左摄像头",
                    type="单目深度相机",
                    width=640,
                    height=360,
                    is_connect=False
                ),
                CameraInfo(
                    name="image_wrist_right",
                    chinese_name="腕部右摄像头",
                    type="单目深度相机",
                    width=640,
                    height=360,
                    is_connect=False
                )
            ]
        )

        self.specifications.arm = ArmStatus(
            information=[
                ArmInfo(
                    name="piper_left",
                    type="Galaxea A1X + Galaxea G1 - 7DOF",
                    start_pose=[],
                    joint_p_limit=[165.0,180.0,0.0,90.0,90.0,165.0],
                    joint_n_limit=[-165.0,0.0,-190.0,-90.0,-90.0,-165.0],
                    is_connect=False
                    ),
                ArmInfo(
                    name="piper_right",
                    type="Galaxea A1X + Galaxea G1 - 7DOF",
                    start_pose=[],
                    joint_p_limit=[165.0,180.0,0.0,90.0,90.0,165.0],
                    joint_n_limit=[-165.0,0.0,-190.0,-90.0,-90.0,-165.0],
                    is_connect=False
                ),
            ]
        )