"""
Vive Tracker IMU and Pose Data Interface

This script interfaces with Vive trackers using pysurvive to collect IMU and pose data,
and sends the data through Dora using PyArrow schemas. It employs two threads:
1. A thread to receive data from Vive trackers.
2. A thread to send data via Dora.
"""

import logging
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

import pyarrow as pa
import pysurvive
from dora import Node

print("Current working directory:", Path.cwd())
print("Python path:", sys.path)


from dr_tracker_vive.pa_schema import pa_imu_schema as imu_schema
from dr_tracker_vive.pa_schema import pa_pose_schema as pose_schema

# Initialize logging
logger = logging.getLogger(__name__)
node = Node()
dora_stop_event = threading.Event()
survive_close_event = threading.Event()


@dataclass
class IMUData:
    """
    Thread-safe container for IMU data from Vive trackers.

    Attributes:
        _lock: Thread lock for atomic operations.
        _has_data: Flag indicating if data is available.
        acc: Accelerometer data (x, y, z).
        gyro: Gyroscope data (x, y, z).
        mag: Magnetometer data (x, y, z).
        serial_number: Serial number of the device.
    """

    _lock: threading.Lock = field(default_factory=threading.Lock)
    _has_data: bool = False
    acc: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    gyro: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    mag: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    serial_number: str = ""

    def update_data(
        self,
        serial_number: str,
        acc: List[float],
        gyro: List[float],
        mag: List[float],
    ) -> None:
        """Update IMU data in a thread-safe manner."""
        with self._lock:
            self.serial_number = serial_number
            self.acc = acc
            self.gyro = gyro
            self.mag = mag
            self._has_data = True

    def read_data(self) -> Tuple[bool, str, List[float], List[float], List[float]]:
        """Read IMU data in a thread-safe manner."""
        with self._lock:
            return (
                self._has_data,
                self.serial_number,
                self.acc.copy(),
                self.gyro.copy(),
                self.mag.copy(),
            )


@dataclass
class PoseData:
    """
    Thread-safe container for pose data from Vive trackers.

    Attributes:
        _lock: Thread lock for atomic operations.
        _has_data: Flag indicating if data is available.
        position: Position data (x, y, z).
        rotation: Rotation data as a quaternion (w, x, y, z).
        serial_number: Serial number of the device.
    """

    _lock: threading.Lock = field(default_factory=threading.Lock)
    _has_data: bool = False
    position: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    rotation: List[float] = field(default_factory=lambda: [1.0, 0.0, 0.0, 0.0])
    serial_number: str = ""

    def update_data(
        self,
        serial_number: str,
        position: List[float],
        rotation: List[float],
    ) -> None:
        """Update pose data in a thread-safe manner."""
        with self._lock:
            self.position = position
            self.rotation = rotation
            self.serial_number = serial_number
            self._has_data = True

    def read_data(self) -> Tuple[bool, str, List[float], List[float]]:
        """Read pose data in a thread-safe manner."""
        with self._lock:
            return (
                self._has_data,
                self.serial_number,
                self.position.copy(),
                self.rotation.copy(),
            )


def make_imu_func(imu_data: IMUData):
    """Generate an IMU callback function for pysurvive."""

    def imu_func(ctx, _mode, accelgyro: List[float], _timecode, _dev_id) -> None:
        try:
            if len(accelgyro) < 9:
                logger.error("Invalid IMU data length: %d", len(accelgyro))
                return

            acc = accelgyro[:3]
            gyro = accelgyro[3:6]
            mag = accelgyro[6:]
            serial_number = ctx.contents.serial_number.decode("utf-8")
            imu_data.update_data(serial_number, acc, gyro, mag)
        except Exception as e:
            logger.exception("Error processing IMU data: %s", e)

    return imu_func


def make_pose_func(pose_data: PoseData):
    """Generate a pose callback function for pysurvive."""

    def pose_func(ctx, _timecode, pose: List[float]) -> None:
        try:
            if len(pose) < 7:
                logger.error("Invalid pose data length: %d", len(pose))
                return

            position = pose[:3]
            rotation = pose[3:]
            serial_number = ctx.contents.serial_number.decode("utf-8")
            pose_data.update_data(serial_number, position, rotation)
        except Exception as e:
            logger.exception("Error processing pose data: %s", e)

    return pose_func


def receive_data_from_survive(imu_data: IMUData, pose_data: PoseData) -> None:
    """Receive IMU and pose data from Vive trackers via pysurvive."""
    logger.info("Starting Survive thread.")
    ctx = pysurvive.init(sys.argv)
    if ctx is None:
        logger.error("Vive device not connected.")
        survive_close_event.set()
        return

    try:
        pysurvive.install_imu_fn(ctx, make_imu_func(imu_data))
        pysurvive.install_pose_fn(ctx, make_pose_func(pose_data))

        while not dora_stop_event.is_set():
            if pysurvive.survive_poll(ctx) != 0:
                logger.error("Error polling from pysurvive.")
                survive_close_event.set()
                break
            time.sleep(0.001)
    except Exception as e:
        logger.exception("Survive error: %s", e)
        survive_close_event.set()
    finally:
        logger.info("Closing pysurvive.")
        survive_close_event.set()
        # pysurvive.free(ctx)  # Uncomment if pysurvive provides a cleanup function


def send_data_through_dora(imu_data: IMUData, pose_data: PoseData) -> None:
    """Send IMU and pose data via Dora outputs."""
    logger.info("Starting Dora thread.")
    try:
        for event in node:
            if dora_stop_event.is_set() or survive_close_event.is_set():
                logger.info("Dora loop received stop signal.")
                break

            if event["type"] == "INPUT" and event["id"] == "tick":
                has_imu, sn_imu, acc, gyro, mag = imu_data.read_data()
                has_pose, sn_pose, pos, rot = pose_data.read_data()

                if has_imu:
                    try:
                        imu_batch = pa.record_batch(
                            {
                                "serial_number": [sn_imu],
                                "acc": [acc],
                                "gyro": [gyro],
                                "mag": [mag],
                            },
                            schema=imu_schema,
                        )
                        node.send_output("imu", imu_batch)
                    except Exception as e:
                        logger.exception("Failed to send IMU data: %s", e)

                if has_pose:
                    try:
                        pose_batch = pa.record_batch(
                            {
                                "serial_number": [sn_pose],
                                "position": [pos],
                                "rotation": [rot],
                            },
                            schema=pose_schema,
                        )
                        node.send_output("pose", pose_batch)
                    except Exception as e:
                        logger.exception("Failed to send pose data: %s", e)

                time.sleep(0.01)

            elif event["type"] == "STOP":
                dora_stop_event.set()
                logger.info("\nExiting dora_vive_tracker...")
                break

    except Exception as e:
        logger.exception("Dora error: %s", e)
        dora_stop_event.set()
    finally:
        logger.info("Exiting Dora thread.")
        dora_stop_event.set()


def main() -> None:
    """Main function to initialize and manage threads."""
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    imu_data = IMUData()
    pose_data = PoseData()

    # Start threads
    survive_thread = threading.Thread(
        target=receive_data_from_survive,
        args=(imu_data, pose_data),
        daemon=True,
    )
    dora_thread = threading.Thread(
        target=send_data_through_dora,
        args=(imu_data, pose_data),
        daemon=True,
    )

    survive_thread.start()
    dora_thread.start()

    logger.info("All threads started.")

    try:
        # Wait for threads to finish
        dora_thread.join()
        survive_thread.join()
    except KeyboardInterrupt:
        logger.info("Received KeyboardInterrupt. Shutting down...")
        dora_stop_event.set()
        survive_close_event.set()
    finally:
        logger.info("Exiting gracefully.")


if __name__ == "__main__":
    main()
