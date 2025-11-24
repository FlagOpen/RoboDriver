import os

import requests
from dora import Node

stream_id_1 = os.getenv("STREAM_ID_1", "default_stream")
stream_id_2 = os.getenv("STREAM_ID_2", "default_stream")
server_url = os.getenv("SERVER_URL", "http://localhost:8080")
session = requests.Session()


node = Node()


def main():
    for event in node:
        event_type = event["type"]

        if event_type == "INPUT":
            input_id = event["id"]
            value = event["value"]

            if input_id == "robot_image-1":
                frame_data = value.to_numpy().tobytes()

                # Build URL
                url = f"{server_url}/api/update_stream/{stream_id_1}"

                # Send POST request
                try:
                    response = session.post(url, data=frame_data)
                    if response.status_code != 200:
                        print(
                            f"Server returned error: {response.status_code}, {response.text}"
                        )
                except requests.exceptions.RequestException as e:
                    print(f"Request failed: {e}")

            if input_id == "robot_image-2":
                frame_data = value.to_numpy().tobytes()

                # Build URL
                url = f"{server_url}/api/update_stream/{stream_id_2}"

                # Send POST request
                try:
                    response = session.post(url, data=frame_data)
                    if response.status_code != 200:
                        print(
                            f"Server returned error: {response.status_code}, {response.text}"
                        )
                except requests.exceptions.RequestException as e:
                    print(f"Request failed: {e}")


if __name__ == "__main__":
    main()
