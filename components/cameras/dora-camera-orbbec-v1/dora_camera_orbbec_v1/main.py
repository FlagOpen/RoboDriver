"""Orbbec Gemini 335 color camera node — V4L2 backend (OpenCV).

Uses the kernel uvcvideo driver via /dev/videoN instead of pyorbbecsdk/libusb,
because libob_usb.so fails to parse UVC payload headers on this device,
producing all-constant (green) frames regardless of scene content.
"""

import os
import cv2
import numpy as np
import pyarrow as pa
from dora import Node

V4L2_DEVICE = os.getenv("V4L2_DEVICE", "/dev/video16")
IMAGE_WIDTH = int(os.getenv("IMAGE_WIDTH", "640"))
IMAGE_HEIGHT = int(os.getenv("IMAGE_HEIGHT", "480"))


def main():
    node = Node()

    cap = cv2.VideoCapture(V4L2_DEVICE, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, IMAGE_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, IMAGE_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, 30)
    # 请求 MJPG，带宽低且解码质量好
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

    if not cap.isOpened():
        print(f"[CAM_TOP] ERROR: cannot open {V4L2_DEVICE}", flush=True)
        return

    print(f"[CAM_TOP] opened {V4L2_DEVICE} "
          f"{int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
          f"{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))} "
          f"@ {cap.get(cv2.CAP_PROP_FPS):.0f}fps", flush=True)

    for event in node:
        if event["type"] == "INPUT" and event["id"] == "tick":
            ret, bgr_image = cap.read()
            if not ret or bgr_image is None:
                print("[CAM_TOP] frame read failed", flush=True)
                continue

            mean_val = bgr_image.mean()
            print(f"[CAM_TOP] mean={mean_val:.1f} std={bgr_image.std():.1f} size={bgr_image.shape}", flush=True)

            if mean_val < 5:
                continue

            ret2, jpeg_buf = cv2.imencode(".jpg", bgr_image)
            if not ret2:
                continue

            jpeg_array = np.ascontiguousarray(jpeg_buf).ravel()
            node.send_output("image", pa.array(jpeg_array), {
                "encoding": "jpeg",
                "width": IMAGE_WIDTH,
                "height": IMAGE_HEIGHT,
            })

    cap.release()


if __name__ == "__main__":
    main()
