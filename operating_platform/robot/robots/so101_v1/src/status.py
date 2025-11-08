from dataclasses import dataclass

from operating_platform.robot.robots.statuses import (
    RobotStatus,
    CameraStatus,
    ArmStatus,
    CameraInfo,
    ArmInfo,
)


@RobotStatus.register_subclass("so101")
@dataclass
class SO101RobotStatus(RobotStatus):
    device_name: str = "SO101_V1"
    device_body: str = "SO101"

    def __post_init__(self):
        self.specifications.end_type = "二指夹爪"
        self.specifications.fps = 30
        self.specifications.camera = CameraStatus(
            information=[
                CameraInfo(
                    name="image_top",
                    chinese_name="头部摄像头",
                    type="单目视觉相机",
                    width=640,
                    height=480,
                    is_connect=False
                ),
                CameraInfo(
                    name="image_wrist",
                    chinese_name="腕部摄像头",
                    type="单目视觉相机",
                    width=640,
                    height=480,
                    is_connect=False
                ),
            ]
        )

        self.specifications.arm = ArmStatus(
            information=[
                ArmInfo(
                    name="leader",
                    type="so101 主臂 5DOF",
                    start_pose=[],
                    joint_p_limit=[90.0, 90.0, 90.0, 90.0, 90.0],
                    joint_n_limit=[-90.0, -90.0, -90.0, -90.0, -90.0],
                    is_connect=False
                ),
                ArmInfo(
                    name="follower",
                    type="so101 从臂 5DOF",
                    start_pose=[],
                    joint_p_limit=[90.0, 90.0, 90.0, 90.0, 90.0],
                    joint_n_limit=[-90.0, -90.0, -90.0, -90.0, -90.0],
                    is_connect=False
                ),
            ]
        )
