import json

import zmq
from dora import Node

# IPC Address
ipc_address = "ipc:///tmp/dr-component-pika-gripper"

context = zmq.Context()
socket = context.socket(zmq.PAIR)
socket.connect(ipc_address)
socket.setsockopt(zmq.SNDHWM, 2000)
socket.setsockopt(zmq.SNDBUF, 2**25)


if __name__ == "__main__":
    node = Node()

    for event in node:
        if event["type"] == "INPUT":
            event_id = event["id"]
            data_bytes = event["value"].to_numpy().tobytes()
            meta_bytes = json.dumps(event["metadata"]).encode("utf-8")

            try:
                socket.send_multipart(
                    [
                        event_id.encode("utf-8"),
                        data_bytes,
                        meta_bytes,
                    ],
                    flags=zmq.NOBLOCK,
                )
            except zmq.Again:
                pass
        elif event["type"] == "STOP":
            break

    # Close server
    running_server = False

    # Close zmq
    socket.close()
    context.term()
