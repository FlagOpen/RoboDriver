"""example of using dora-vive node"""

import logging

from dora import Node

logger = logging.getLogger(__name__)


def main() -> None:
    try:
        node = Node()
        for event in node:
            if event["type"] == "INPUT":
                struct = event["value"][0]
                serial_number = struct["serial_number"]
                if event["id"] == "imu":
                    acc = struct["acc"].as_py()
                    gyro = struct["gyro"].as_py()
                    mag = struct["mag"].as_py()
                    logger.info(
                        f"Receive msg: id=imu, serial_number={serial_number}, acc={acc}, gyro={gyro}, mag={mag}"
                    )

                if event["id"] == "pose":
                    position = struct["position"].as_py()
                    rotation = struct["rotation"].as_py()
                    logger.info(
                        f"Receive msg: id=pose, serial_number={serial_number}, position={position}, rotation={rotation} "
                    )
            elif event["type"] == "STOP":
                break
            elif event["type"] == "INPUT_CLOSED":
                break

    except KeyboardInterrupt:
        logger.info("\nExiting dora_vive_example...")
    except Exception as e:
        logger.info(f"error: {e}")
        raise e

    logger.info("dora_vive_example Exit")


if __name__ == "__main__":
    main()
