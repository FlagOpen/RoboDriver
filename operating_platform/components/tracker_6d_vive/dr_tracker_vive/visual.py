import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import pyarrow as pa
from dora import Node
from scipy.spatial.transform import Rotation as R

# Environment configuration
GET_DEVICE_FROM = os.getenv(
    "GET_DEVICE_FROM", "None"
)  # Source of device filtering (e.g., by serial number)
DEVICE_SN = os.getenv("DEVICE_SN")  # Target device serial number
DISPLAY = os.getenv("DR_VIVE_DISPLAY", "False")  # Whether to enable 3D visualization

logger = logging.getLogger(__name__)


class TransformVisualizer:
    """
    A 3D visualizer for real-time 6DoF pose tracking and trajectory display.
    Displays coordinate frames and movement path in a 3D plot.
    """

    def __init__(self, axis_length=0.1, space_size=2.0):
        self.display_enabled = DISPLAY.lower() == "true"

        # Initialize all internal states regardless of display status
        self.current_transform = None
        self.trajectory_points = []
        self.fig = None
        self.ax = None

        if not self.display_enabled:
            return

        # Only initialize visualization components if enabled
        self.axis_length = axis_length
        self.space_size = space_size

        self.fig = plt.figure(figsize=(10, 8))
        self.fig.canvas.manager.window.title(f"6D Pose Visualizer: {DEVICE_SN}")
        self.ax = self.fig.add_subplot(111, projection="3d")
        self.ax.view_init(azim=-160)
        self._setup_plot()

    def _setup_plot(self):
        """Set up the 3D plotting environment with labels, limits, and base frame."""
        if not self.display_enabled:
            return

        identity_transform = np.eye(4)
        self._plot_transform(identity_transform, axis_length=2.0, label="Base Frame")

        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_zlabel("Z")
        self.ax.set_title("6D Pose Visualization")
        self.ax.set_box_aspect([1, 1, 1])

        limit_range = [-self.space_size, self.space_size]
        self.ax.set_xlim(limit_range)
        self.ax.set_ylim(limit_range)
        self.ax.set_zlim(limit_range)

        self.ax.legend()
        plt.tight_layout()
        plt.ion()  # Enable interactive mode
        plt.show()

    def _plot_transform(self, transform_matrix, axis_length=None, label=None):
        """
        Draw a 3D coordinate frame defined by a 4x4 transformation matrix.

        Args:
            transform_matrix (np.ndarray): 4x4 homogeneous transformation matrix.
            axis_length (float, optional): Length of each axis arrow.
            label (str, optional): Label to place at the origin.
        """
        if not self.display_enabled:
            return

        axis_length = axis_length or self.axis_length
        origin = transform_matrix[:3, 3]

        x_axis_end = origin + transform_matrix[:3, 0] * axis_length
        y_axis_end = origin + transform_matrix[:3, 1] * axis_length
        z_axis_end = origin + transform_matrix[:3, 2] * axis_length

        # Plot X, Y, Z axes
        self.ax.plot(
            [origin[0], x_axis_end[0]],
            [origin[1], x_axis_end[1]],
            [origin[2], x_axis_end[2]],
            c="r",
            linewidth=2,
            label="_X" if label else "",
        )
        self.ax.plot(
            [origin[0], y_axis_end[0]],
            [origin[1], y_axis_end[1]],
            [origin[2], y_axis_end[2]],
            c="g",
            linewidth=2,
            label="_Y" if label else "",
        )
        self.ax.plot(
            [origin[0], z_axis_end[0]],
            [origin[1], z_axis_end[1]],
            [origin[2], z_axis_end[2]],
            c="b",
            linewidth=2,
            label="_Z" if label else "",
        )

        if label:
            self.ax.text(
                *origin, label, color="k", fontsize=8, ha="center", va="center"
            )

    def _plot_trajectory(self, color="gray", label="Trajectory"):
        """Plot the recorded trajectory of the tracked device."""
        if not self.display_enabled or not self.trajectory_points:
            return

        points = np.array(self.trajectory_points)
        self.ax.plot(
            points[:, 0],
            points[:, 1],
            points[:, 2],
            color=color,
            label=label,
            linewidth=1.5,
        )

    def update_visualization(self):
        """Redraw the entire visualization with current pose and trajectory."""
        if not self.display_enabled:
            return

        self.ax.cla()  # Clear current axes
        self._setup_plot()  # Re-initialize plot settings

        # Draw current pose if available
        if self.current_transform is not None:
            self._plot_transform(self.current_transform, label="Current Pose")

        # Draw trajectory
        self._plot_trajectory()

        # Refresh canvas
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def update_transform(self, world_transform):
        """
        Update the visualizer with a new 6D pose (position + orientation).

        Args:
            world_transform (np.ndarray): 4x4 transformation matrix in world coordinates.
        """
        # Use absolute world transform directly (can be changed to relative if needed)
        relative_transform = world_transform

        # Store current transform
        self.current_transform = relative_transform.copy()

        # Record position for trajectory
        self.trajectory_points.append(relative_transform[:3, 3].tolist())

        # Trigger visualization update
        self.update_visualization()


def main():
    """Main loop: subscribe to pose data and visualize in real time."""
    node = Node()
    visualizer = TransformVisualizer(space_size=2.0)

    # Determine device filtering strategy
    if GET_DEVICE_FROM == "SN":
        if DEVICE_SN is None:
            raise ValueError(
                "DEVICE_SN environment variable must be set when GET_DEVICE_FROM=SN"
            )
        target_serial_number = DEVICE_SN
    else:
        target_serial_number = None

    try:
        for event in node:
            if event["type"] == "INPUT" and event["id"] == "pose":
                data = event["value"][0]
                serial_number = data.get("serial_number").as_py()

                # Filter by serial number if required
                if (
                    target_serial_number is not None
                    and serial_number != target_serial_number
                ):
                    continue

                # Extract position and rotation
                position_list = data["position"].as_py()
                rotation_list = data[
                    "rotation"
                ].as_py()  # Assuming [w, x, y, z] or [x, y, z, w]

                # print(f"position_list:{position_list}")

                # Handle quaternion format: assuming input is [w, x, y, z], convert to [x, y, z, w]
                position = np.array(position_list, dtype=np.float32)
                quat_xyzw = np.array(
                    [
                        rotation_list[1],
                        rotation_list[2],
                        rotation_list[3],
                        rotation_list[0],
                    ],
                    dtype=np.float32,
                )

                # print(f"position:{position}")

                # Create rotation object and get rotation matrix
                rotation_obj = R.from_quat(quat_xyzw)
                rotation_matrix = rotation_obj.as_matrix()

                # Construct 4x4 homogeneous transformation matrix
                T_world = np.eye(4)
                T_world[:3, :3] = rotation_matrix
                T_world[:3, 3] = position

                # Update visualizer
                visualizer.update_transform(T_world)

                # Forward flattened transform via Dora output
                node.send_output(
                    "tansform", pa.array(T_world.flatten(), type=pa.float32())
                )
                node.send_output("pose", pa.array(position_list, type=pa.float32()))
                node.send_output(
                    "rotation-quat", pa.array(rotation_list, type=pa.float32())
                )

            elif event["type"] == "STOP":
                logger.info("Received STOP event. Shutting down...")
                break

    except KeyboardInterrupt:
        logger.info("User interrupted. Exiting...")
    except Exception as e:
        logger.exception("Unexpected error occurred in Dora node: %s", e)
    finally:
        if visualizer.fig is not None:
            plt.close(visualizer.fig)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
