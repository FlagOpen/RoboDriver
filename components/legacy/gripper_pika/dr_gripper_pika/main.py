"""TODO: Add docstring."""

import os
import time

import pyarrow as pa
from dora import Node
from pika import sense

RUNNER_CI = True if os.getenv("CI") == "true" else False
USB_PORT = os.getenv("USB_PORT", "/dev/ttyUSB0")


def main():
    start_time = time.time()
    node = Node()
    my_sense = sense(USB_PORT)

    if not my_sense.connect():
        print("连接 Pika Sense 设备失败，请检查设备连接和串口路径")
        return

    print("成功连接到 Pika Sense 设备")

    for event in node:
        # Run this example in the CI for 10 seconds only.
        if RUNNER_CI and time.time() - start_time > 10:
            break

        event_type = event["type"]

        if event_type == "INPUT":
            event_id = event["id"]

            if event_id == "tick":
                encoder_data = my_sense.get_encoder_data()

                gripper_value = my_sense.get_gripper_distance()

                # gripper_value = np.array(gripper_value, dtype=np.float32)

                node.send_output(
                    "distance", pa.array([gripper_value], type=pa.float32())
                )
        elif event_type == "STOP":
            break

        elif event_type == "ERROR":
            raise RuntimeError(event["error"])


if __name__ == "__main__":
    main()
