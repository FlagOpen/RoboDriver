import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

PLUGIN_DIR = (
    Path(__file__).resolve().parents[1]
    / "robodriver"
    / "robots"
    / "robodriver-robot-so101-aio-dora"
)
sys.path.insert(0, str(PLUGIN_DIR))
try:
    from robodriver_robot_so101_aio_dora.config import SO101AIODoraRobotConfig
    from robodriver_robot_so101_aio_dora.robot import SO101AIODoraRobot
finally:
    sys.path.pop(0)


def _robot(require_leader):
    robot = object.__new__(SO101AIODoraRobot)
    robot.config = SimpleNamespace(require_leader=require_leader)
    robot.connected = False
    robot.cameras = {}
    robot.connect_excluded_cameras = []
    robot.leader_motors = {"shoulder_pan": None}
    robot.follower_motors = {"shoulder_pan": None}
    robot.robot_dora_node = SimpleNamespace(
        recv_images={},
        recv_joint_leader={},
        recv_joint_follower={"follower_joint_shoulder_pan": 1.0},
        stop=lambda: None,
    )
    robot.status = SimpleNamespace(
        specifications=SimpleNamespace(
            camera=SimpleNamespace(number=0, information=[]),
            arm=SimpleNamespace(number=0, information=[]),
        )
    )
    return robot


class SO101AIODoraInferenceTest(unittest.TestCase):
    def test_leader_is_required_by_default(self):
        self.assertTrue(SO101AIODoraRobotConfig().require_leader)

    def test_inference_connect_does_not_require_leader(self):
        robot = _robot(require_leader=False)

        robot.connect()

        self.assertTrue(robot.is_connected)

    def test_default_connect_still_requires_leader(self):
        robot = _robot(require_leader=True)

        with (
            patch(
                "robodriver_robot_so101_aio_dora.robot.time.perf_counter",
                side_effect=[0.0, 0.0, 21.0],
            ),
            patch("robodriver_robot_so101_aio_dora.robot.time.sleep"),
        ):
            with self.assertRaisesRegex(TimeoutError, "主臂"):
                robot.connect()

    def test_aio_observation_matches_policy_state_and_camera_contract(self):
        from robodriver.core.policies import LeRobotPolicy

        obs = {
            "observation.state": np.zeros(6, dtype=np.float32),
            "observation.images.image_top": np.zeros((480, 640, 3), dtype=np.uint8),
            "observation.images.image_wrist": np.zeros((480, 640, 3), dtype=np.uint8),
            "observation.images.image_top_dep": np.zeros((480, 640, 3), dtype=np.uint8),
            "prompt": "AIO compatibility test",
        }
        policy = LeRobotPolicy()

        request = policy._convert_observation(obs)

        self.assertEqual(request["state"].shape, (6,))
        self.assertEqual(
            request["images"]["observation.images.image_top"].shape, (480, 640, 3)
        )
        self.assertIn("observation.images.image_top_dep", request["images"])


if __name__ == "__main__":
    unittest.main()
