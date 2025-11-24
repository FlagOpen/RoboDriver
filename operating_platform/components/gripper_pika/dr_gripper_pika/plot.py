"""TODO: Add docstring."""

import os

from dora import Node

RUNNER_CI = True if os.getenv("CI") == "true" else False


def main():
    node = Node()

    for event in node:
        event_type = event["type"]

        if event_type == "INPUT":
            event_id = event["id"]

            if "gripper" in event_id:
                value = event["value"]
                print(f"gripper_distance:{value}")
        elif event_type == "ERROR":
            raise RuntimeError(event["error"])


if __name__ == "__main__":
    main()
