#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Visualize data of **all** frames of any episode of a Lightwheel dataset.

Note: The last frame of the episode doesn't always correspond to a final state.
That's because our datasets are composed of transition from state to state up to
the antepenultimate state associated to the ultimate action to arrive in the final state.
However, there might not be a transition from a final state to another state.

Note: This script aims to visualize the data used to train the neural networks.
~What you see is what you get~. When visualizing image modality, it is often expected to observe
lossy compression artifacts since these images have been decoded from compressed mp4 videos to
save disk space. The compression factor applied has been tuned to not affect success rate.

Examples:

- Visualize data stored on a local machine:
```
local$ python -m robodriver.scripts.visual_lightwheel \
    --repo-id lightwheel/example \
    --episode-index 0
```

- Visualize data stored on a distant machine with a local viewer:
```
distant$ python -m robodriver.scripts.visual_lightwheel \
    --repo-id lightwheel/example \
    --episode-index 0 \
    --save 1 \
    --output-dir path/to/directory

local$ scp distant:path/to/directory/lightwheel_example_episode_0.rrd .
local$ rerun lightwheel_example_episode_0.rrd
```

- Visualize data stored on a distant machine through streaming:
(You need to forward the websocket port to the distant machine, with
`ssh -L 9087:localhost:9087 username@remote-host`)
```
distant$ python -m robodriver.scripts.visual_lightwheel \
    --repo-id lightwheel/example \
    --episode-index 0 \
    --mode distant \
    --ws-port 9087

local$ rerun ws://localhost:9087
```

"""

import argparse
import gc
import json
import logging_mp
import time
import timeit
from collections.abc import Iterator
from pathlib import Path
from threading import Event

import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import rerun as rr
import torch
import torch.utils.data
import torchvision
import tqdm


logging_mp.basic_config(level=logging_mp.INFO)
logger = logging_mp.get_logger(__name__)


def decode_video_frames_torchvision(
    video_path: str,
    timestamps: list[float],
    tolerance_s: float,
    backend: str = "pyav",
    log_loaded_timestamps: bool = False,
) -> torch.Tensor:
    """Loads frames associated to the requested timestamps of a video

    The backend can be either "pyav" (default) or "video_reader".
    "video_reader" requires installing torchvision from source, see:
    https://github.com/pytorch/vision/blob/main/torchvision/csrc/io/decoder/gpu/README.rst
    (note that you need to compile against ffmpeg<4.3)

    While both use cpu, "video_reader" is supposedly faster than "pyav" but requires additional setup.
    For more info on video decoding, see `benchmark/video/README.md`

    See torchvision doc for more info on these two backends:
    https://pytorch.org/vision/0.18/index.html?highlight=backend#torchvision.set_video_backend

    Note: Video benefits from inter-frame compression. Instead of storing every frame individually,
    the encoder stores a reference frame (or a key frame) and subsequent frames as differences relative to
    that key frame. As a consequence, to access a requested frame, we need to load the preceding key frame,
    and all subsequent frames until reaching the requested frame. The number of key frames in a video
    can be adjusted during encoding to take into account decoding time and video size in bytes.
    """
    video_path = str(video_path)

    # set backend
    keyframes_only = False
    torchvision.set_video_backend(backend)
    if backend == "pyav":
        keyframes_only = True  # pyav doesnt support accuracte seek

    # set a video stream reader
    reader = torchvision.io.VideoReader(video_path, "video")

    # set the first and last requested timestamps
    first_ts = min(timestamps)
    last_ts = max(timestamps)

    # access closest key frame of the first requested frame
    reader.seek(first_ts, keyframes_only=keyframes_only)

    # load all frames until last requested frame
    loaded_frames = []
    loaded_ts = []
    for frame in reader:
        current_ts = frame["pts"]
        if log_loaded_timestamps:
            logger.info(f"frame loaded at timestamp={current_ts:.4f}")
        loaded_frames.append(frame["data"])
        loaded_ts.append(current_ts)
        if current_ts >= last_ts:
            break

    if backend == "pyav":
        reader.container.close()

    reader = None

    query_ts = torch.tensor(timestamps)
    loaded_ts = torch.tensor(loaded_ts)

    # compute distances between each query timestamp and timestamps of all loaded frames
    dist = torch.cdist(query_ts[:, None], loaded_ts[:, None], p=1)
    min_, argmin_ = dist.min(1)

    is_within_tol = min_ < tolerance_s
    assert is_within_tol.all(), (
        f"One or several query timestamps unexpectedly violate the tolerance ({min_[~is_within_tol]} > {tolerance_s=})."
        "It means that the closest frame that can be loaded from the video is too far away in time."
        "This might be due to synchronization issues with timestamps during data collection."
        "To be safe, we advise to ignore this item during training."
        f"\nqueried timestamps: {query_ts}"
        f"\nloaded timestamps: {loaded_ts}"
        f"\nvideo: {video_path}"
        f"\nbackend: {backend}"
    )

    # get closest frames to the query timestamps
    closest_frames = torch.stack([loaded_frames[idx] for idx in argmin_])
    closest_ts = loaded_ts[argmin_]

    if log_loaded_timestamps:
        logger.info(f"{closest_ts=}")

    # convert to the pytorch format which is float32 in [0,1] range (and channel first)
    closest_frames = closest_frames.type(torch.float32) / 255

    assert len(timestamps) == len(closest_frames)
    return closest_frames


LIGHTWHEEL_DATASET_VERSION = "v3.0"


def to_hwc_uint8_numpy(chw_float32_torch: torch.Tensor) -> np.ndarray:
    assert chw_float32_torch.dtype == torch.float32
    assert chw_float32_torch.ndim == 3
    c, h, w = chw_float32_torch.shape
    assert (
        c < h and c < w
    ), f"expect channel first images, but instead {chw_float32_torch.shape}"
    hwc_uint8_numpy = (
        (chw_float32_torch * 255).type(torch.uint8).permute(1, 2, 0).numpy()
    )
    return hwc_uint8_numpy


class LightwheelMetadata:
    """Metadata class for Lightwheel datasets."""

    def __init__(
        self,
        repo_id: str,
        root: str | Path | None = None,
        force_cache_sync: bool = False,
    ):
        self.repo_id = repo_id
        self.root = Path(root) if root is not None else Path.home() / ".cache" / "huggingface" / "lightwheel" / repo_id

        try:
            if force_cache_sync:
                raise FileNotFoundError
            self.load_metadata()
        except (FileNotFoundError, NotADirectoryError):
            self.root.mkdir(exist_ok=True, parents=True)
            # Try to load from info.json if available
            info_path = self.root / "meta" / "info.json"
            if info_path.exists():
                with open(info_path, "r") as f:
                    self.info = json.load(f)
            else:
                raise FileNotFoundError(
                    f"Could not find info.json for {repo_id} at {self.root}"
                )
            self.load_metadata_from_info()

    def load_metadata(self):
        """Load metadata from local files."""
        info_path = self.root / "meta" / "info.json"
        if info_path.exists():
            with open(info_path, "r") as f:
                self.info = json.load(f)
            self.load_metadata_from_info()
        else:
            raise FileNotFoundError(f"Could not find info.json at {info_path}")

    def load_metadata_from_info(self):
        """Parse info dict to extract metadata."""
        self.codebase_version = self.info.get("codebase_version")
        self.robot_type = self.info.get("robot_type")
        self.total_episodes = self.info.get("total_episodes", 0)
        self.total_frames = self.info.get("total_frames", 0)
        self.total_tasks = self.info.get("total_tasks", 0)
        self.chunks_size = self.info.get("chunks_size", 1000)
        self.fps = self.info.get("fps", 30.0)
        self.splits = self.info.get("splits", {"train": "0:1"})
        self.data_path = self.info.get(
            "data_path", "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
        )
        self.video_path = self.info.get(
            "video_path", "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
        )
        self.features = self.info.get("features", {})

    def get_data_file_path(self, ep_index: int) -> Path:
        """Get the parquet file path for an episode."""
        chunk_index = ep_index // self.chunks_size
        file_index = ep_index % self.chunks_size
        fpath = self.data_path.format(chunk_index=chunk_index, file_index=file_index)
        return Path(fpath)

    def get_video_file_path(self, ep_index: int, vid_key: str) -> Path:
        """Get the video file path for an episode and video key."""
        chunk_index = ep_index // self.chunks_size
        file_index = ep_index % self.chunks_size
        fpath = self.video_path.format(
            video_key=vid_key, chunk_index=chunk_index, file_index=file_index
        )
        return Path(fpath)

    @property
    def video_keys(self) -> list[str]:
        """Keys to access video modalities."""
        return [key for key, ft in self.features.items() if ft.get("dtype") == "video"]

    @property
    def image_keys(self) -> list[str]:
        """Keys to access image modalities."""
        return [key for key, ft in self.features.items() if ft.get("dtype") == "image"]

    @property
    def camera_keys(self) -> list[str]:
        """Keys to access visual modalities (regardless of storage method)."""
        return [
            key
            for key, ft in self.features.items()
            if ft.get("dtype") in ["video", "image"]
        ]

    @property
    def state_keys(self) -> list[str]:
        """Keys to access state modalities."""
        return [
            key
            for key, ft in self.features.items()
            if key.startswith("observation.state")
        ]

    @property
    def hand_position_keys(self) -> list[str]:
        """Keys for hand position data."""
        return [
            "observation.state.hand_left_world",
            "observation.state.hand_right_world",
        ]

    @property
    def hand_rotation_keys(self) -> list[str]:
        """Keys for hand rotation data (quaternions)."""
        return [
            "observation.state.hand_left_world_rotation",
            "observation.state.hand_right_world_rotation",
        ]

    @property
    def camera_position_keys(self) -> list[str]:
        """Keys for camera position data."""
        return [
            "observation.state.head_left_camera_position",
            "observation.state.head_right_camera_position",
        ]

    @property
    def camera_rotation_keys(self) -> list[str]:
        """Keys for camera rotation data (quaternions)."""
        return [
            "observation.state.head_left_camera_rotation",
            "observation.state.head_right_camera_rotation",
        ]

    @property
    def head_pose_position_keys(self) -> list[str]:
        """Keys for head pose position data."""
        return ["observation.state.head_pose_world"]

    @property
    def head_pose_rotation_keys(self) -> list[str]:
        """Keys for head pose rotation data (quaternions)."""
        return ["observation.state.head_pose_world_rotation"]

    def get_feature_shape(self, key: str) -> list:
        """Get the shape of a feature."""
        if key in self.features:
            return self.features[key].get("shape", [])
        return []

    def __repr__(self):
        feature_keys = list(self.features)
        return (
            f"{self.__class__.__name__}({{\n"
            f"    Repository ID: '{self.repo_id}',\n"
            f"    Total episodes: '{self.total_episodes}',\n"
            f"    Total frames: '{self.total_frames}',\n"
            f"    Features: '{feature_keys}',\n"
            "})',\n"
        )


class EpisodeSampler(torch.utils.data.Sampler):
    def __init__(self, dataset: "LightwheelDataset", episode_index: int):
        from_idx = dataset.episode_data_index["from"][episode_index].item()
        to_idx = dataset.episode_data_index["to"][episode_index].item()
        self.frame_ids = range(from_idx, to_idx)

    def __iter__(self) -> Iterator:
        return iter(self.frame_ids)

    def __len__(self) -> int:
        return len(self.frame_ids)


class LightwheelDataset(torch.utils.data.Dataset):
    """Dataset class for Lightwheel data format."""

    def __init__(
        self,
        repo_id: str,
        root: str | Path | None = None,
        episodes: list[int] | None = None,
        tolerance_s: float = 1e-4,
        force_cache_sync: bool = False,
        video_backend: str | None = None,
        resize_resolution: tuple[int, int] | None = None,  # (width, height)
    ):
        super().__init__()
        self.repo_id = repo_id
        self.root = Path(root) if root is not None else Path.home() / ".cache" / "huggingface" / "lightwheel" / repo_id
        self.tolerance_s = tolerance_s
        self.episodes = episodes
        self.video_backend = video_backend if video_backend else "pyav"
        self.resize_resolution = resize_resolution  # (width, height) or None for original
        self.parquet_table = None
        self.data = None
        self._data_loaded = False

        # Load metadata
        self.meta = LightwheelMetadata(
            repo_id, self.root, force_cache_sync=force_cache_sync
        )

        # Only load data for the episodes we need (lazy loading)
        self._load_data_for_episodes(episodes)
        self._build_episode_data_index()

    def _load_data_for_episodes(self, episodes: list[int] | None = None):
        """Load data only for specific episodes (lazy loading)."""
        if episodes is None:
            # Default: only load episode 0
            episodes = [0]
            self.episodes = episodes

        all_tables = []
        for ep_idx in episodes:
            parquet_path = self.root / self.meta.get_data_file_path(ep_idx)
            if not parquet_path.exists():
                raise FileNotFoundError(f"Parquet file not found: {parquet_path}")

            table = pq.read_table(parquet_path)
            all_tables.append(table)

        if len(all_tables) == 0:
            raise ValueError(f"No parquet files found for episodes {episodes}")

        # Concatenate tables
        self.parquet_table = pa.concat_tables(all_tables)
        self.num_frames = len(self.parquet_table)

        # Convert to column dict, handling None values robustly
        self.data = {}
        for col_name in self.parquet_table.column_names:
            col = self.parquet_table.column(col_name)
            pylist = col.to_pylist()
            
            # Replace None values at the top level
            pylist = [x if x is not None else 0 for x in pylist]
            
            try:
                # Try to stack as numpy array
                col_data = np.array(pylist)
                self.data[col_name] = col_data
            except Exception:
                # If stacking fails, keep as list
                self.data[col_name] = pylist

        self._data_loaded = True

    def _build_episode_data_index(self):
        """Build episode data index for fast lookup."""
        episode_indices = self.data.get("episode_index", None)
        if episode_indices is None:
            raise ValueError("No episode_index found in data")

        # Find from/to indices for each episode
        self.episode_data_index = {"from": [], "to": []}
        unique_episodes = np.unique(episode_indices)

        for ep_idx in unique_episodes:
            indices = np.where(episode_indices == ep_idx)[0]
            if len(indices) > 0:
                self.episode_data_index["from"].append(indices[0])
                self.episode_data_index["to"].append(indices[-1] + 1)
            else:
                self.episode_data_index["from"].append(0)
                self.episode_data_index["to"].append(0)

        # Convert to tensors for compatibility
        self.episode_data_index = {
            k: torch.tensor(v, dtype=torch.int64)
            for k, v in self.episode_data_index.items()
        }

    def __len__(self):
        return self.num_frames

    def __getitem__(self, idx: int) -> dict:
        """Get a single frame sample."""
        item = {}

        # Extract all features for this frame
        for key in self.data:
            val = self.data[key][idx]
            # Skip None values to avoid collate errors
            if val is None:
                continue
            # Convert to torch tensor
            if isinstance(val, np.ndarray):
                item[key] = torch.from_numpy(val)
            elif isinstance(val, (int, float)):
                item[key] = torch.tensor(val)
            else:
                # Try to convert
                try:
                    item[key] = torch.tensor(val)
                except Exception:
                    # Skip non-tensorizable values
                    pass

        # Handle video frames
        for vid_key in self.meta.video_keys:
            ep_idx = int(item.get("episode_index", 0))
            timestamp = item.get("timestamp", 0.0)

            # Get the actual episode index for file path
            file_ep_idx = ep_idx if self.episodes is None else self.episodes[ep_idx]

            video_path = self.root / self.meta.get_video_file_path(
                file_ep_idx, vid_key
            )

            # Determine output resolution
            if self.resize_resolution is not None:
                out_h, out_w = self.resize_resolution[1], self.resize_resolution[0]
            else:
                out_h, out_w = 2464, 3248  # Original resolution

            # Always initialize video frame tensor to avoid None values
            item[vid_key] = torch.zeros(3, out_h, out_w, dtype=torch.float32)

            if video_path.exists():
                try:
                    frames = decode_video_frames_torchvision(
                        video_path,
                        [float(timestamp)],
                        self.tolerance_s,
                        self.video_backend,
                    )
                    if frames.numel() > 0:
                        frame = frames.squeeze(0)
                        # Resize if needed
                        if self.resize_resolution is not None:
                            frame = torch.nn.functional.interpolate(
                                frame.unsqueeze(0),
                                size=(out_h, out_w),
                                mode='bilinear',
                                align_corners=False,
                            ).squeeze(0)
                        item[vid_key] = frame
                except Exception as e:
                    logger.warning(f"Failed to decode video frame {vid_key}: {e}")

        # Final safety check: ensure no None values in the returned item
        # This is critical for DataLoader collation
        item = {k: v for k, v in item.items() if v is not None}
        
        return item


def quaternion_to_rotation_matrix(qw, qx, qy, qz):
    """Convert quaternion to 3x3 rotation matrix."""
    # Normalize quaternion
    norm = np.sqrt(qw**2 + qx**2 + qy**2 + qz**2)
    if norm < 1e-8:
        return np.eye(3)
    qw, qx, qy, qz = qw / norm, qx / norm, qy / norm, qz / norm

    # Compute rotation matrix
    r00 = 1 - 2 * (qy**2 + qz**2)
    r01 = 2 * (qx * qy - qz * qw)
    r02 = 2 * (qx * qz + qy * qw)
    r10 = 2 * (qx * qy + qz * qw)
    r11 = 1 - 2 * (qx**2 + qz**2)
    r12 = 2 * (qy * qz - qx * qw)
    r20 = 2 * (qx * qz - qy * qw)
    r21 = 2 * (qy * qz + qx * qw)
    r22 = 1 - 2 * (qx**2 + qy**2)

    return np.array([[r00, r01, r02], [r10, r11, r12], [r20, r21, r22]])


def visualize_lightwheel_dataset(
    dataset: LightwheelDataset,
    episode_index: int,
    batch_size: int = 32,
    num_workers: int = 4,
    mode: str = "local",
    web_port: int = 9195,
    ws_port: int = 9285,
    save: bool = False,
    output_dir: Path | None = None,
    run_duration: float = 0.0,
    stop_event: Event | None = None,
    open_browser: bool = True,
) -> Path | None:
    """Visualize a Lightwheel dataset episode using Rerun."""
    if save:
        assert (
            output_dir is not None
        ), "Set an output directory where to write .rrd files with `--output-dir path/to/directory`."

    repo_id = dataset.repo_id

    logger.info("Loading dataloader")
    episode_sampler = EpisodeSampler(dataset, episode_index)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=num_workers,
        batch_size=batch_size,
        sampler=episode_sampler,
    )

    logger.info("Starting Rerun")

    if mode not in ["local", "distant"]:
        raise ValueError(mode)

    spawn_local_viewer = mode == "local" and not save
    rr.init(f"{repo_id}/episode_{episode_index}", spawn=spawn_local_viewer)

    try:
        if mode == "distant":
            server_uri = rr.serve_grpc(grpc_port=ws_port)
            rr.serve_web_viewer(
                open_browser=open_browser, connect_to=server_uri, web_port=web_port
            )
        logger.info("Logging to Rerun")

        # Performance timing
        total_frames = 0
        visualization_start = timeit.default_timer()
        load_times = []
        
        dataloader_iter = iter(dataloader)
        for batch_idx in tqdm.tqdm(range(len(dataloader)), total=len(dataloader)):
            # Time how long it takes to load each batch from dataloader
            batch_load_start = timeit.default_timer()
            batch = next(dataloader_iter)
            batch_load_time = timeit.default_timer() - batch_load_start
            load_times.append(batch_load_time)
            
            logger.info(f"Batch {batch_idx} load time: {batch_load_time:.2f}s")
            
            # iterate over the batch
            for i in range(len(batch.get("index", []))):
                frame_start = timeit.default_timer()
                
                frame_idx = int(batch.get("frame_index", [torch.tensor(0)])[i].item())
                timestamp = float(batch.get("timestamp", [torch.tensor(0.0)])[i].item())
                
                # Set time using new rerun API (rr.set_time)
                rr.set_time("frame_index", sequence=frame_idx)
                rr.set_time("timestamp", timestamp=timestamp)

                # Display camera images
                for key in dataset.meta.camera_keys:
                    if key not in batch:
                        continue
                    if "depth" in key:
                        continue
                    if key in batch and batch[key][i].numel() > 0:
                        rr.log(key, rr.Image(to_hwc_uint8_numpy(batch[key][i])))
                
                frame_time = timeit.default_timer() - frame_start
                if total_frames % 3 == 0:  # Log every 30 frames
                    elapsed = timeit.default_timer() - visualization_start
                    logger.info(f"Frame {total_frames}: {frame_time*1000:.1f}ms (elapsed: {elapsed:.1f}s)")

                # Display hand position data (3D points)
                for hand_key in dataset.meta.hand_position_keys:
                    if hand_key not in batch:
                        continue
                    try:
                        hand_data = batch[hand_key][i]
                        if hand_data.numel() > 0:
                            # Shape: [21, 3]
                            hand_points = hand_data.numpy()
                            rr.log(f"state/{hand_key}", rr.Points3D(hand_points))
                    except Exception as e:
                        logger.warning(f"Failed to visualize {hand_key}: {e}")

                # Display hand rotation data (quaternions as 3D arrows/transforms)
                for hand_key in dataset.meta.hand_rotation_keys:
                    if hand_key not in batch:
                        continue
                    try:
                        hand_data = batch[hand_key][i]
                        if hand_data.numel() > 0:
                            # Shape: [21, 4] - quaternions
                            quaternions = hand_data.numpy()
                            # Log each joint's orientation as arrows
                            for j, q in enumerate(quaternions):
                                qw, qx, qy, qz = q
                                rr.log(
                                    f"state/{hand_key}/joint_{j}",
                                    rr.Transform3D(
                                        rotation=rr.Quaternion(xyzw=[qx, qy, qz, qw])
                                    ),
                                )
                    except Exception as e:
                        logger.warning(f"Failed to visualize {hand_key}: {e}")

                # Display camera position
                for cam_pos_key in dataset.meta.camera_position_keys:
                    if cam_pos_key not in batch:
                        continue
                    try:
                        cam_pos = batch[cam_pos_key][i].numpy()
                        rr.log(f"state/{cam_pos_key}", rr.Points3D(cam_pos))
                    except Exception as e:
                        logger.warning(f"Failed to visualize {cam_pos_key}: {e}")

                # Display camera rotation
                for cam_rot_key in dataset.meta.camera_rotation_keys:
                    if cam_rot_key not in batch:
                        continue
                    try:
                        cam_rot = batch[cam_rot_key][i].numpy()
                        qw, qx, qy, qz = cam_rot
                        rr.log(
                            f"state/{cam_rot_key}",
                            rr.Transform3D(
                                rotation=rr.Quaternion(xyzw=[qx, qy, qz, qw])
                            ),
                        )
                    except Exception as e:
                        logger.warning(f"Failed to visualize {cam_rot_key}: {e}")

                # Display head pose position
                for head_pos_key in dataset.meta.head_pose_position_keys:
                    if head_pos_key not in batch:
                        continue
                    try:
                        head_pos = batch[head_pos_key][i].numpy()
                        rr.log(f"state/{head_pos_key}", rr.Points3D(head_pos))
                    except Exception as e:
                        logger.warning(f"Failed to visualize {head_pos_key}: {e}")

                # Display head pose rotation
                for head_rot_key in dataset.meta.head_pose_rotation_keys:
                    if head_rot_key not in batch:
                        continue
                    try:
                        head_rot = batch[head_rot_key][i].numpy()
                        qw, qx, qy, qz = head_rot
                        rr.log(
                            f"state/{head_rot_key}",
                            rr.Transform3D(
                                rotation=rr.Quaternion(xyzw=[qx, qy, qz, qw])
                            ),
                        )
                    except Exception as e:
                        logger.warning(f"Failed to visualize {head_rot_key}: {e}")

                # Display action dimensions if present
                if "action" in batch:
                    for dim_idx, val in enumerate(batch["action"][i]):
                        rr.log(f"action/{dim_idx}", rr.Scalars(val.item()))

                # Display observation.state dimensions
                if "observation.state" in batch:
                    for dim_idx, val in enumerate(batch["observation.state"][i]):
                        rr.log(f"state/{dim_idx}", rr.Scalars(val.item()))

                # Display subtask index
                if "subtask_index" in batch:
                    rr.log("subtask_index", rr.Scalars(batch["subtask_index"][i].item()))

                if "next.done" in batch:
                    rr.log("next.done", rr.Scalars(batch["next.done"][i].item()))

                if "next.reward" in batch:
                    rr.log("next.reward", rr.Scalars(batch["next.reward"][i].item()))

                if "next.success" in batch:
                    rr.log("next.success", rr.Scalars(batch["next.success"][i].item()))
                
                total_frames += 1

        # Print final timing summary
        total_elapsed = timeit.default_timer() - visualization_start
        logger.info(f"Visualization complete: {total_frames} frames processed in {total_elapsed:.1f}s")
        if total_frames > 0 and total_elapsed > 0:
            avg_fps = total_frames / total_elapsed
            logger.info(f"Average processing speed: {avg_fps:.1f} fps")
        if load_times:
            avg_load = sum(load_times) / len(load_times)
            max_load = max(load_times)
            logger.info(f"Batch load stats: avg={avg_load:.2f}s, max={max_load:.2f}s, total_load={sum(load_times):.2f}s")

        if mode == "local" and save:
            # save .rrd locally
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            repo_id_str = repo_id.replace("/", "_")
            rrd_path = output_dir / f"{repo_id_str}_episode_{episode_index}.rrd"
            rr.save(rrd_path)
            return rrd_path

        if not save or mode == "distant":  # Viewing mode active
            if run_duration > 0:
                logger.info(
                    f"Visualization complete. Auto-exiting in {run_duration} seconds..."
                )
                time.sleep(run_duration)
            else:
                logger.info("Visualization complete. Press Ctrl-C to exit.")
                try:
                    if stop_event is None:
                        print("stop_event should been set")
                        raise ValueError(stop_event)
                    while stop_event.is_set() == False:
                        time.sleep(0.05)
                except KeyboardInterrupt:
                    print("\nCtrl-C received. Exiting.")

    finally:
        rr.disconnect()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--repo-id",
        type=str,
        required=False,
        help="Name of hugging face repository containing a Lightwheel dataset.",
    )
    parser.add_argument(
        "--episode-index",
        type=int,
        required=True,
        help="Episode to visualize.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Root directory for the dataset stored locally (e.g. `--root data`). By default, the dataset will be loaded from hugging face cache folder, or downloaded from the hub if available.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory path to write a .rrd file when `--save 1` is set.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size loaded by DataLoader.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of processes of Dataloader for loading the data.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="local",
        help=(
            "Mode of viewing between 'local' or 'distant'. "
            "'local' requires data to be on a local machine. It spawns a viewer to visualize the data locally. "
            "'distant' creates a server on the distant machine where the data is stored. "
            "Visualize the data by connecting to the server with `rerun ws://localhost:PORT` on the local machine."
        ),
    )
    parser.add_argument(
        "--web-port",
        type=int,
        default=9195,
        help="Web port for rerun.io when `--mode distant` is set.",
    )
    parser.add_argument(
        "--ws-port",
        type=int,
        default=9285,
        help="Web socket port for rerun.io when `--mode distant` is set.",
    )
    parser.add_argument(
        "--save",
        type=int,
        default=0,
        help=(
            "Save a .rrd file in the directory provided by `--output-dir`. "
            "It also deactivates the spawning of a viewer. "
            "Visualize the data by running `rerun path/to/file.rrd` on your local machine."
        ),
    )

    parser.add_argument(
        "--tolerance-s",
        type=float,
        default=1e-4,
        help=(
            "Tolerance in seconds used to ensure data timestamps respect the dataset fps value"
            "This is argument passed to the constructor of LightwheelDataset and maps to its tolerance_s constructor argument"
            "If not given, defaults to 1e-4."
        ),
    )

    parser.add_argument(
        "--resize-resolution",
        type=str,
        default=None,
        help=(
            "Resize camera images to this resolution for faster loading. "
            "Format: WIDTHxHEIGHT (e.g., 640x480). "
            "If not specified, original resolution (3248x2464) is used."
        ),
    )

    args = parser.parse_args()
    kwargs = vars(args)
    repo_id = kwargs.pop("repo_id")
    root = kwargs.pop("root")
    tolerance_s = kwargs.pop("tolerance_s")
    episode_index = kwargs.pop("episode_index")
    resize_resolution = kwargs.pop("resize_resolution", None)

    # Parse resize resolution
    resize_res = None
    if resize_resolution is not None:
        try:
            w, h = resize_resolution.split("x")
            resize_res = (int(w), int(h))
            logger.info(f"Using resized resolution: {resize_res}")
        except (ValueError, AttributeError):
            logger.warning(f"Invalid resize resolution format: {resize_resolution}, using original resolution")

    # Remove episode_index from kwargs since it's passed separately
    kwargs.pop("episode_index", None)

    logger.info("Loading dataset")

    stop_event = Event()
    # Only load the specific episode requested for faster startup
    dataset = LightwheelDataset(
        repo_id,
        root=root,
        tolerance_s=tolerance_s,
        episodes=[episode_index],
        resize_resolution=resize_res,
    )

    visualize_lightwheel_dataset(dataset, episode_index=episode_index, **kwargs, stop_event=stop_event)


if __name__ == "__main__":
    main()