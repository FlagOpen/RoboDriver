from dataclasses import dataclass, field

from operating_platform.robot.robots.com_configs.cameras import (
    CameraConfig,
    OpenCVCameraConfig,
)

from operating_platform.robot.robots.com_configs.motors import (
    RosMotorsBusConfig,
    MotorsBusConfig,
)

from operating_platform.robot.robots.configs import RobotConfig


@RobotConfig.register_subclass("galaxea")
@dataclass
class GALAXEARobotConfig(RobotConfig):
    leader_arms: dict[str, MotorsBusConfig] = field(
        default_factory=lambda: {
            "main": RosMotorsBusConfig(
                topic="[/motion_target/target_joint_state_arm_left,/motion_target/target_joint_state_arm_right,/tabletop/hdas/feedback_arm_left,/tabletop/hdas/feedback_arm_right,]",
                motors={
                    # name: (index, model)
                    "left_arm_joint_1_rad": [1, "galaxea_motor"],
                    "left_arm_joint_2_rad": [2, "galaxea_motor"],
                    "left_arm_joint_3_rad": [3, "galaxea_motor"],
                    "left_arm_joint_4_rad": [4, "galaxea_motor"],
                    "left_arm_joint_5_rad": [5, "galaxea_motor"],
                    "left_arm_joint_6_rad": [6, "galaxea_motor"],
                    "left_gripper_degree_mm": [7, "galaxea_motor"],
                    "right_arm_joint_1_rad": [8, "galaxea_motor"],
                    "right_arm_joint_2_rad": [9, "galaxea_motor"],
                    "right_arm_joint_3_rad": [10, "galaxea_motor"],
                    "right_arm_joint_4_rad": [11, "galaxea_motor"],
                    "right_arm_joint_5_rad": [12, "galaxea_motor"],
                    "right_arm_joint_6_rad": [13, "galaxea_motor"],
                    "right_gripper_degree_mm": [14, "galaxea_motor"],
                },
            ),
        }
    )

    follower_arms: dict[str, MotorsBusConfig] = field(
        default_factory=lambda: {
            "main": RosMotorsBusConfig(
                topic="[/hdas/feedback_arm_left,/hdas/feedback_arm_right]",
                motors={
                    # name: (index, model)
                    "left_arm_joint_1_rad": [1, "galaxea_motor"],
                    "left_arm_joint_2_rad": [2, "galaxea_motor"],
                    "left_arm_joint_3_rad": [3, "galaxea_motor"],
                    "left_arm_joint_4_rad": [4, "galaxea_motor"],
                    "left_arm_joint_5_rad": [5, "galaxea_motor"],
                    "left_arm_joint_6_rad": [6, "galaxea_motor"],
                    "left_gripper_degree_mm": [15, "galaxea_motor"],
                    "right_arm_joint_1_rad": [16, "galaxea_motor"],
                    "right_arm_joint_2_rad": [17, "galaxea_motor"],
                    "right_arm_joint_3_rad": [18, "galaxea_motor"],
                    "right_arm_joint_4_rad": [19, "galaxea_motor"],
                    "right_arm_joint_5_rad": [20, "galaxea_motor"],
                    "right_arm_joint_6_rad": [21, "galaxea_motor"],
                    "right_gripper_degree_mm": [30, "galaxea_motor"],
                },
            ),
        }
    )

    cameras: dict[str, CameraConfig] = field(
        default_factory=lambda: {
            "image_top_left": OpenCVCameraConfig(
                camera_index=0,
                fps=30,
                width=1280,
                height=720,
                channels = 3,
                info = {
                    "video.fps": 30.0,
                    "video.height": 720,
                    "video.width": 1280,
                    "video.channels": 3,
                    "video.codec": "libx264",
                    "video.pix_fmt": "yuv420p",
                    "video.is_depth_map": False,
                    "has_audio": False
                }
            ),
            "image_top_right": OpenCVCameraConfig(
                camera_index=1,
                fps=30,
                width=1280,
                height=720,
                channels = 3,
                info = {
                    "video.fps": 30.0,
                    "video.height": 720,
                    "video.width": 1280,
                    "video.channels": 3,
                    "video.codec": "libx264",
                    "video.pix_fmt": "yuv420p",
                    "video.is_depth_map": False,
                    "has_audio": False
                }
                
            ),
            "image_wrist_left": OpenCVCameraConfig(
                camera_index=2,
                fps=30,
                width=640,
                height=360,
                channels = 3,
                info = {
                    "video.fps": 30.0,
                    "video.height": 720,
                    "video.width": 1280,
                    "video.channels": 3,
                    "video.codec": "libx264",
                    "video.pix_fmt": "yuv420p",
                    "video.is_depth_map": False,
                    "has_audio": False
                }
            ),
            "image_wrist_right": OpenCVCameraConfig(
                camera_index=3,
                fps=30,
                width=640,
                height=360,
                channels = 3,
                info = {
                    "video.fps": 30.0,
                    "video.height": 360,
                    "video.width": 640,
                    "video.channels": 3,
                    "video.codec": "libx264",
                    "video.pix_fmt": "yuv420p",
                    "video.is_depth_map": False,
                    "has_audio": False
                }
            ),
            "image_depth_wrist_left": OpenCVCameraConfig(
                camera_index=4,
                fps=30,
                width=640,
                height=360,
                channels=1,
                info = {
                    "video.fps": 30.0,
                    "video.height": 360,
                    "video.width": 640,
                    "video.channels": 1,
                    "video.codec": "ffv1",
                    "video.pix_fmt": "gray16le",
                    "video.is_depth_map": True,
                    "has_audio": False
                }
            ),
            "image_depth_wrist_right": OpenCVCameraConfig(
                camera_index=5,
                fps=30,
                width=640,
                height=360,
                channels=1,
                info = {
                    "video.fps": 30.0,
                    "video.height": 360,
                    "video.width": 640,
                    "video.channels": 1,
                    "video.codec": "ffv1",
                    "video.pix_fmt": "gray16le",
                    "video.is_depth_map": True,
                    "has_audio": False
                }
            ),
        }
    )

    use_videos: bool = False

    microphones: dict[str, int] = field(
        default_factory=lambda: {
            # "audio_right": 2,
            # "audio_left": 4,
        }
    )
    