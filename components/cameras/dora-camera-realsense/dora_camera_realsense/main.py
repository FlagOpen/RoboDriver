"""TODO: Add docstring."""

import os
import time

import cv2
import numpy as np
import pyarrow as pa
import pyrealsense2 as rs
from dora import Node

RUNNER_CI = True if os.getenv("CI") == "true" else False

def main():
    """TODO: Add docstring."""
    flip = os.getenv("FLIP", "")
    device_serial = os.getenv("DEVICE_SERIAL", "")
    image_height = int(os.getenv("IMAGE_HEIGHT", "480"))
    image_width = int(os.getenv("IMAGE_WIDTH", "640"))
    encoding = os.getenv("ENCODING", "jpeg")  # 改为 jpeg 以降低带宽

    pipeline = rs.pipeline()

    config = rs.config()
    # 如果指定了 device_serial，尝试使用它，否则使用任何可用设备
    if device_serial:
        try:
            config.enable_device(device_serial)
            print(f"Attempting to connect to device {device_serial}")
        except Exception as e:
            print(f"Warning: Could not enable specific device {device_serial}: {e}")
            print("Connecting to any available RealSense device...")

    config.enable_stream(rs.stream.color, image_width, image_height, rs.format.rgb8, 30)
    # 注释掉深度流配置 - 只输出彩色图
    # config.enable_stream(rs.stream.depth, image_width, image_height, rs.format.z16, 30)

    # 不需要对齐，因为只有彩色流
    # align_to = rs.stream.color
    # align = rs.align(align_to)

    try:
        profile = pipeline.start(config)
    except RuntimeError as e:
        print(f"Error starting pipeline: {e}")
        raise

    rgb_profile = profile.get_stream(rs.stream.color)
    # 不需要获取深度配置
    # depth_profile = profile.get_stream(rs.stream.depth)
    # _depth_intr = depth_profile.as_video_stream_profile().get_intrinsics()
    rgb_intr = rgb_profile.as_video_stream_profile().get_intrinsics()

    # Warm up the camera - wait for a few frames to stabilize
    print("Warming up RealSense camera...")
    for _ in range(30):
        try:
            pipeline.wait_for_frames(timeout_ms=1000)
        except RuntimeError:
            pass
    print("RealSense camera ready")

    node = Node()

    start_time = time.time()

    pa.array([])  # initialize pyarrow array

    for event in node:
        # Run this example in the CI for 10 seconds only.
        if RUNNER_CI and time.time() - start_time > 10:
            break

        event_type = event["type"]

        if event_type == "INPUT":
            event_id = event["id"]

            if event_id == "tick":
                try:
                    frames = pipeline.wait_for_frames(timeout_ms=10000)
                except RuntimeError as e:
                    print(f"Warning: Failed to get frames: {e}")
                    continue
                # 不需要对齐，直接获取彩色帧
                # aligned_frames = align.process(frames)
                # aligned_depth_frame = aligned_frames.get_depth_frame()
                color_frame = frames.get_color_frame()
                if not color_frame:
                    continue

                # 只处理彩色图像
                # depth_image = np.asanyarray(aligned_depth_frame.get_data())
                # scaled_depth_image = depth_image
                frame = np.asanyarray(color_frame.get_data())

                ## Change rgb to bgr

                if flip == "VERTICAL":
                    frame = cv2.flip(frame, 0)
                elif flip == "HORIZONTAL":
                    frame = cv2.flip(frame, 1)
                elif flip == "BOTH":
                    frame = cv2.flip(frame, -1)

                metadata = event["metadata"]
                metadata["encoding"] = encoding
                metadata["width"] = int(frame.shape[1])
                metadata["height"] = int(frame.shape[0])

                # Get the right encoding
                if encoding == "bgr8":
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                elif encoding in ["jpeg", "jpg", "jpe", "bmp", "webp", "png"]:
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # imdecode expects BGR
                    ret, frame = cv2.imencode("." + encoding, frame)
                    if not ret:
                        print("Error encoding image...")
                        continue

                storage = pa.array(frame.ravel())
        
                metadata["resolution"] = [int(rgb_intr.ppx), int(rgb_intr.ppy)]
                metadata["focal_length"] = [int(rgb_intr.fx), int(rgb_intr.fy)]
                # metadata["principal_point"] = [int(rgb_intr.ppx), int(rgb_intr.ppy)]
                metadata["timestamp"] = time.time_ns()
                node.send_output("image", storage, metadata)
#                 metadata["encoding"] = "mono16"
#                 scaled_depth_image[scaled_depth_image > 5000] = 0
#                 node.send_output(
#                     "image_depth",
#                     pa.array(scaled_depth_image.ravel()),
#                     metadata,
#                 )
        elif event_type == "ERROR":
            raise RuntimeError(event["error"])
        
        if event_type == "STOP":
            break


if __name__ == "__main__":
    main()