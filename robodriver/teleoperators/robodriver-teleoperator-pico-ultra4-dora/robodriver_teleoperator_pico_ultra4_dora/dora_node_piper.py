"""
Pico Ultra4 遥操 Piper 的 Dora 节点

将 PiperTeleopController 适配为 Dora 节点，
在 tick 事件驱动下执行一次 IK + 控制循环，
同时在 grip 激活时以 DoRobotDataset（LeRobot 兼容）格式保存数据。
"""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "xcb")
import time
from pathlib import Path
import numpy as np
import pyarrow as pa
import cv2
from dora import Node

from xrobotoolkit_teleop.hardware.piper_teleop_controller import (
    PiperTeleopController,
    DEFAULT_PIPER_MANIPULATOR_CONFIG,
)
from xrobotoolkit_teleop.utils.path_utils import ASSET_PATH

from robodriver.dataset.dorobot_dataset import DoRobotDataset, DoRobotDatasetMetadata

URDF_PATH = os.getenv("URDF_PATH", os.path.join(ASSET_PATH, "piper/piper.urdf"))
CAN_PORT = os.getenv("CAN_BUS", "can0")
SCALE_FACTOR = float(os.getenv("SCALE_FACTOR", "1.5"))
CONTROL_RATE_HZ = int(os.getenv("CONTROL_RATE_HZ", "50"))
RECORD_DIR = os.getenv("RECORD_DIR", os.path.expanduser("~/recordings/pico_piper"))
RECORD_FPS = int(os.getenv("RECORD_FPS", "30"))
REPO_ID = os.getenv("REPO_ID", "pico_piper")
TASK = os.getenv("TASK", "teleoperation")
USE_VIDEOS = os.getenv("USE_VIDEOS", "false").lower() == "true"

# Piper 7 维状态：joint1-6 + gripper
STATE_DIM = 7
IMAGE_HEIGHT = 480
IMAGE_WIDTH = 640

FEATURES = {
    "observation.state": {
        "dtype": "float32",
        "shape": (STATE_DIM,),
        "names": ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "gripper"],
    },
    "observation.images.camera_top": {
        "dtype": "video" if USE_VIDEOS else "image",
        "shape": (IMAGE_HEIGHT, IMAGE_WIDTH, 3),
        "names": ["height", "width", "channel"],
    },
    "observation.images.camera_wrist": {
        "dtype": "video" if USE_VIDEOS else "image",
        "shape": (IMAGE_HEIGHT, IMAGE_WIDTH, 3),
        "names": ["height", "width", "channel"],
    },
    "action": {
        "dtype": "float32",
        "shape": (STATE_DIM,),
        "names": ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "gripper"],
    },
}


def make_dataset() -> DoRobotDataset:
    """创建或续接 DoRobotDataset，绕过 robot.microphones 依赖。"""
    obj = DoRobotDataset.__new__(DoRobotDataset)
    meta = DoRobotDatasetMetadata.__new__(DoRobotDatasetMetadata)
    meta.repo_id = REPO_ID
    meta.root = Path(RECORD_DIR)

    info_path = meta.root / "meta" / "info.json"
    if info_path.exists():
        # 续接已有数据集
        meta.load_metadata()
    else:
        # 全新数据集
        meta = DoRobotDatasetMetadata.create(
            repo_id=REPO_ID,
            fps=RECORD_FPS,
            root=RECORD_DIR,
            robot_type="piper",
            features=FEATURES,
            use_videos=USE_VIDEOS,
            use_audios=False,
        )

    obj.meta = meta
    obj.repo_id = obj.meta.repo_id
    obj.root = obj.meta.root
    obj.revision = None
    obj.tolerance_s = 1e-4
    obj.image_writer = None
    obj.audio_writer = None
    obj.episode_buffer = obj.create_episode_buffer()
    obj.episodes = None
    obj.hf_dataset = obj.create_hf_dataset()
    obj.image_transforms = None
    obj.delta_timestamps = None
    obj.delta_indices = None
    obj.episode_data_index = None
    obj.video_backend = "pyav"
    return obj


def decode_image(data: pa.Array, metadata: dict) -> np.ndarray | None:
    encoding = metadata.get("encoding", "bgr8")
    width = metadata.get("width", IMAGE_WIDTH)
    height = metadata.get("height", IMAGE_HEIGHT)
    buf = data.to_numpy(zero_copy_only=False).astype(np.uint8)
    if encoding == "bgr8":
        img = buf.reshape((height, width, 3))
    elif encoding in ("jpeg", "jpg", "png", "bmp", "webp"):
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if img is None:
            return None
    else:
        return None
    # DoRobotDataset 期望 RGB
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def main():
    import threading

    node = Node()

    dataset_ref = [None]
    controller_ref = [None]
    robot_ready = threading.Event()

    def setup():
        dataset_ref[0] = make_dataset()
        ctrl = PiperTeleopController(
            robot_urdf_path=URDF_PATH,
            manipulator_config=DEFAULT_PIPER_MANIPULATOR_CONFIG,
            can_port=CAN_PORT,
            scale_factor=SCALE_FACTOR,
            control_rate_hz=CONTROL_RATE_HZ,
            enable_log_data=False,
            enable_camera=False,
            visualize_placo=False,
        )
        ctrl._robot_setup()
        controller_ref[0] = ctrl
        robot_ready.set()

    threading.Thread(target=setup, daemon=True).start()

    latest_images: dict[str, np.ndarray] = {}
    was_active = False

    for event in node:
        if event["type"] != "INPUT":
            continue

        eid = event["id"]

        # 缓存相机图像
        if eid in ("camera_top_image", "camera_wrist_image"):
            meta = event["metadata"]
            val = event["value"]
            cam_key = "camera_top" if eid == "camera_top_image" else "camera_wrist"
            if cam_key == "camera_top":
                try:
                    buf_dbg = val.to_numpy(zero_copy_only=False).astype(np.uint8)
                    print(f"[DBG] {cam_key}: len={len(buf_dbg)}, first4={buf_dbg[:4].tolist()}")
                except Exception as e:
                    print(f"[DBG] {cam_key} to_numpy error: {e}")
            img = decode_image(val, meta)
            if img is not None:
                latest_images[cam_key] = img
            else:
                print(f"[DBG] {cam_key} decode failed: encoding={meta.get('encoding')}, len={len(val)}")
            continue

        if eid != "tick":
            continue

        # robot 未就绪时只显示图像，跳过控制
        if not robot_ready.is_set():
            if latest_images:
                frames_to_show = []
                for cam_key in ("camera_top", "camera_wrist"):
                    img = latest_images.get(cam_key)
                    if img is not None:
                        bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                        cv2.putText(bgr, f"{cam_key.upper()} | INIT", (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
                        frames_to_show.append(bgr)
                if frames_to_show:
                    try:
                        combined = np.hstack(frames_to_show) if len(frames_to_show) > 1 else frames_to_show[0]
                        cv2.imshow("Pico Teleop", combined)
                        cv2.waitKey(1)
                    except Exception as e:
                        print(f"[imshow error] {e}")
            continue

        # IK + 控制
        controller = controller_ref[0]
        dataset = dataset_ref[0]
        controller._update_robot_state()
        controller._update_gripper_target()
        controller._update_ik()
        controller._send_command()

        q = controller.piper.get_joint_positions()
        gripper = controller.piper.get_gripper_position()
        state = np.append(q, gripper).astype(np.float32)

        is_active = controller.active.get("right_arm", False)

        # grip 激活时录制（有哪路图像用哪路，缺失的用上一帧或零填充）
        if is_active and latest_images:
            top = latest_images.get(
                "camera_top", np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.uint8)
            )
            wrist = latest_images.get(
                "camera_wrist", np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.uint8)
            )
            frame = {
                "observation.state": state,
                "observation.images.camera_top": top,
                "observation.images.camera_wrist": wrist,
                "action": state,
                "task": TASK,
            }
            dataset.add_frame(frame)

        # grip 松开时保存 episode
        if was_active and not is_active:
            n_frames = dataset.episode_buffer["size"]
            if n_frames > 0:
                ep_idx = dataset.save_episode()
                print(f"[Recorder] Episode {ep_idx} saved ({n_frames} frames)")
                dataset.episode_buffer = dataset.create_episode_buffer()
            else:
                print("[Recorder] No frames recorded, discarding.")

        was_active = is_active

        # 实时显示相机画面
        if latest_images:
            frames_to_show = []
            for cam_key in ("camera_top", "camera_wrist"):
                img = latest_images.get(cam_key)
                if img is not None:
                    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    label = f"{'TOP' if cam_key == 'camera_top' else 'WRIST'} | {'REC' if is_active else 'IDLE'}"
                    cv2.putText(bgr, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                                (0, 0, 255) if is_active else (0, 255, 0), 2)
                    frames_to_show.append(bgr)
            if frames_to_show:
                try:
                    combined = np.hstack(frames_to_show) if len(frames_to_show) > 1 else frames_to_show[0]
                    cv2.imshow("Pico Teleop", combined)
                    key = cv2.waitKey(1)
                    if key == ord("q"):
                        break
                except Exception as e:
                    print(f"[imshow error] {e}")

        # 发布关节状态
        metadata = event["metadata"]
        metadata["timestamp"] = time.time_ns()
        node.send_output(
            "follower_jointstate",
            pa.array(state),
            metadata,
        )

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
