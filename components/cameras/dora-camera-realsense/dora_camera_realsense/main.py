"""TODO: Add docstring."""

import os
import time

import cv2
import numpy as np
import pyarrow as pa
import pyrealsense2 as rs
from dora import Node

RUNNER_CI = True if os.getenv("CI") == "true" else False


def env_flag(name, default=False):
    """Read a boolean environment variable."""
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def main():
    """TODO: Add docstring."""
    flip = os.getenv("FLIP", "")
    device_serial = os.getenv("DEVICE_SERIAL", "")
    image_height = int(os.getenv("IMAGE_HEIGHT", "480"))
    image_width = int(os.getenv("IMAGE_WIDTH", "640"))
    encoding = os.getenv("ENCODING", "rgb8")
    enable_depth = env_flag("ENABLE_DEPTH")
    ctx = rs.context()
    devices = ctx.query_devices()
    if devices.size() == 0:
        raise ConnectionError("No realsense camera connected.")

    # Serial list
    serials = [device.get_info(rs.camera_info.serial_number) for device in devices]
    if device_serial and (device_serial not in serials):
        raise ConnectionError(
            f"Device with serial {device_serial} not found within: {serials}.",
        )

    pipeline = rs.pipeline()

    config = rs.config()
    config.enable_device(device_serial)
    config.enable_stream(rs.stream.color, image_width, image_height, rs.format.rgb8, 30)
    if enable_depth:
        config.enable_stream(
            rs.stream.depth,
            image_width,
            image_height,
            rs.format.z16,
            30,
        )

    try:
        profile = pipeline.start(config)
        rgb_profile = profile.get_stream(rs.stream.color)
        rgb_intr = rgb_profile.as_video_stream_profile().get_intrinsics()
        align = rs.align(rs.stream.color) if enable_depth else None
        node = Node()

        start_time = time.time()

        pa.array([])  # initialize pyarrow array

        for event in node:
            # Run this example in the CI for 10 seconds only.
            if RUNNER_CI and time.time() - start_time > 10:
                break

            event_type = event["type"]

            if event_type == "INPUT" and event["id"] == "tick":
                frames = pipeline.wait_for_frames()
                if enable_depth:
                    frames = align.process(frames)

                color_frame = frames.get_color_frame()
                if not color_frame:
                    continue

                frame = np.asanyarray(color_frame.get_data())

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
                    ret, frame = cv2.imencode("." + encoding, frame)
                    if not ret:
                        print("Error encoding image...")
                        continue

                storage = pa.array(frame.ravel())
        
                metadata["resolution"] = [int(rgb_intr.ppx), int(rgb_intr.ppy)]
                metadata["focal_length"] = [int(rgb_intr.fx), int(rgb_intr.fy)]
                metadata["timestamp"] = time.time_ns()
                node.send_output("image", storage, metadata)

                if enable_depth:
                    depth_frame = frames.get_depth_frame()
                    if depth_frame:
                        depth_image = np.asanyarray(depth_frame.get_data())
                        depth_image[depth_image > 5000] = 0
                        depth_metadata = metadata.copy()
                        depth_metadata["encoding"] = "mono16"
                        node.send_output(
                            "image_depth",
                            pa.array(depth_image.ravel()),
                            depth_metadata,
                        )
            elif event_type == "ERROR":
                raise RuntimeError(event["error"])
            elif event_type == "STOP":
                break
    finally:
        pipeline.stop()


if __name__ == "__main__":
    main()
