"""
Pico Ultra4 双臂遥操 Piper 的 Dora 节点

两个独立的 PiperTeleopController（右臂 can0 + 左臂 can1），
在 tick 事件驱动下同时执行 IK + 控制，
分别输出 right_follower_jointstate 和 left_follower_jointstate。
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
    DEFAULT_PIPER_LEFT_MANIPULATOR_CONFIG,
)
from xrobotoolkit_teleop.utils.path_utils import ASSET_PATH

from robodriver.dataset.dorobot_dataset import DoRobotDataset, DoRobotDatasetMetadata

URDF_PATH = os.getenv("URDF_PATH", os.path.join(ASSET_PATH, "piper/piper.urdf"))
RIGHT_CAN_PORT = os.getenv("RIGHT_CAN_BUS", os.getenv("CAN_BUS", "can0"))
LEFT_CAN_PORT = os.getenv("LEFT_CAN_BUS", "can1")
SCALE_FACTOR = float(os.getenv("SCALE_FACTOR", "1.5"))
CONTROL_RATE_HZ = int(os.getenv("CONTROL_RATE_HZ", "50"))
RECORD_DIR = os.getenv("RECORD_DIR", os.path.expanduser("~/recordings/pico_piper"))
RECORD_FPS = int(os.getenv("RECORD_FPS", "30"))
REPO_ID = os.getenv("REPO_ID", "pico_piper")
TASK = os.getenv("TASK", "teleoperation")
USE_VIDEOS = os.getenv("USE_VIDEOS", "false").lower() == "true"

# 每臂 7 维：joint1-6 + gripper，双臂共 14 维
STATE_DIM = 7
IMAGE_HEIGHT = 480
IMAGE_WIDTH = 640

FEATURES = {
    "observation.state": {
        "dtype": "float32",
        "shape": (STATE_DIM * 2,),
        "names": [
            "right_joint1", "right_joint2", "right_joint3",
            "right_joint4", "right_joint5", "right_joint6", "right_gripper",
            "left_joint1", "left_joint2", "left_joint3",
            "left_joint4", "left_joint5", "left_joint6", "left_gripper",
        ],
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
        "shape": (STATE_DIM * 2,),
        "names": [
            "right_joint1", "right_joint2", "right_joint3",
            "right_joint4", "right_joint5", "right_joint6", "right_gripper",
            "left_joint1", "left_joint2", "left_joint3",
            "left_joint4", "left_joint5", "left_joint6", "left_gripper",
        ],
    },
}


def make_dataset() -> DoRobotDataset:
    """创建或续接 DoRobotDataset。"""
    obj = DoRobotDataset.__new__(DoRobotDataset)
    meta = DoRobotDatasetMetadata.__new__(DoRobotDatasetMetadata)
    meta.repo_id = REPO_ID
    meta.root = Path(RECORD_DIR)

    info_path = meta.root / "meta" / "info.json"
    if info_path.exists():
        meta.load_metadata()
    else:
        meta = DoRobotDatasetMetadata.create(
            repo_id=REPO_ID,
            fps=RECORD_FPS,
            root=RECORD_DIR,
            robot_type="piper_dual",
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
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def get_arm_state(ctrl: PiperTeleopController) -> np.ndarray:
    """读取单臂 7D 状态（joint1-6 + gripper）。"""
    q = ctrl.piper.get_joint_positions()
    gripper = ctrl.piper.get_gripper_position()
    return np.append(q, gripper).astype(np.float32)


def main():
    import threading

    node = Node()

    dataset_ref = [None]
    right_ctrl_ref = [None]
    left_ctrl_ref = [None]
    robot_ready = threading.Event()

    def setup():
        try:
            dataset_ref[0] = make_dataset()
            print(f"[SETUP] Dataset ready: {RECORD_DIR}")

            print(f"[SETUP] Init RIGHT arm on {RIGHT_CAN_PORT} ...")
            right_ctrl = PiperTeleopController(
                robot_urdf_path=URDF_PATH,
                manipulator_config=DEFAULT_PIPER_MANIPULATOR_CONFIG,
                can_port=RIGHT_CAN_PORT,
                scale_factor=SCALE_FACTOR,
                control_rate_hz=CONTROL_RATE_HZ,
                enable_log_data=False,
                enable_camera=False,
                visualize_placo=False,
            )
            right_ctrl._robot_setup()
            right_ctrl_ref[0] = right_ctrl
            print(f"[SETUP] RIGHT arm OK, active keys: {list(right_ctrl.active.keys())}")

            print(f"[SETUP] Init LEFT arm on {LEFT_CAN_PORT} ...")
            left_ctrl = PiperTeleopController(
                robot_urdf_path=URDF_PATH,
                manipulator_config=DEFAULT_PIPER_LEFT_MANIPULATOR_CONFIG,
                can_port=LEFT_CAN_PORT,
                scale_factor=SCALE_FACTOR,
                control_rate_hz=CONTROL_RATE_HZ,
                enable_log_data=False,
                enable_camera=False,
                visualize_placo=False,
            )
            left_ctrl._robot_setup()
            left_ctrl_ref[0] = left_ctrl
            print(f"[SETUP] LEFT arm OK, active keys: {list(left_ctrl.active.keys())}")

            print(f"[SETUP] right xr_client id={id(right_ctrl.xr_client)}, left xr_client id={id(left_ctrl.xr_client)}")
            print(f"[SETUP] XrClient._initialized={right_ctrl.xr_client.__class__._initialized}")

            robot_ready.set()
            print("[SETUP] Both arms ready, robot_ready set.")
        except Exception as e:
            import traceback
            print(f"[ERROR] Robot setup failed: {e}")
            traceback.print_exc()

    threading.Thread(target=setup, daemon=True).start()

    latest_images: dict[str, np.ndarray] = {}
    was_active = False
    _dbg_tick = 0

    for event in node:
        if event["type"] != "INPUT":
            continue

        eid = event["id"]

        # 缓存相机图像
        if eid in ("camera_top_image", "camera_wrist_image"):
            cam_key = "camera_top" if eid == "camera_top_image" else "camera_wrist"
            img = decode_image(event["value"], event["metadata"])
            if img is not None:
                latest_images[cam_key] = img
            continue

        if eid != "tick":
            continue

        # robot 未就绪时只显示图像
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

        right_ctrl = right_ctrl_ref[0]
        left_ctrl = left_ctrl_ref[0]
        dataset = dataset_ref[0]

        # 右臂控制
        try:
            right_ctrl._update_robot_state()
            right_ctrl._update_gripper_target()
            right_ctrl._update_ik()
            right_ctrl._send_command()
        except Exception as e:
            print(f"[ERROR] RIGHT arm control failed: {e}")

        # 左臂控制
        try:
            left_ctrl._update_robot_state()
            left_ctrl._update_gripper_target()
            left_ctrl._update_ik()
            left_ctrl._send_command()
        except Exception as e:
            print(f"[ERROR] LEFT arm control failed: {e}")

        # 读取状态
        right_state = get_arm_state(right_ctrl)
        left_state = get_arm_state(left_ctrl)
        combined_state = np.concatenate([right_state, left_state])

        right_active = right_ctrl.active.get("right_arm", False)
        left_active = left_ctrl.active.get("left_arm", False)
        is_active = right_active or left_active

        # DEBUG 每50帧打印一次
        _dbg_tick += 1
        if _dbg_tick % 50 == 0:
            grip_r = right_ctrl.xr_client.get_key_value_by_name("right_grip")
            grip_l = left_ctrl.xr_client.get_key_value_by_name("left_grip")
            print(f"[DBG] tick={_dbg_tick} right_grip={grip_r:.3f} left_grip={grip_l:.3f} "
                  f"right_active={right_active} left_active={left_active}")
            print(f"[DBG] right_active_keys={dict(right_ctrl.active)} left_active_keys={dict(left_ctrl.active)}")
            print(f"[DBG] right_state={np.round(right_state, 3)} left_state={np.round(left_state, 3)}")

        # 录制
        if is_active and latest_images:
            top = latest_images.get("camera_top", np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.uint8))
            wrist = latest_images.get("camera_wrist", np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.uint8))
            frame = {
                "observation.state": combined_state,
                "observation.images.camera_top": top,
                "observation.images.camera_wrist": wrist,
                "action": combined_state,
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

        # 实时显示
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
        node.send_output("right_follower_jointstate", pa.array(right_state), metadata)
        node.send_output("left_follower_jointstate", pa.array(left_state), metadata)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
