import asyncio
import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import lerobot.processor
from lerobot.robots.config import RobotConfig

from robodriver.core.coordinator import Coordinator, _build_reset_target_pose

ROBOT_ROOT = Path(__file__).resolve().parents[1] / "robodriver" / "robots"


def _load_config(module_name, plugin_name, module_path):
    path = ROBOT_ROOT / plugin_name / module_path / "config.py"
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    with patch.object(
        RobotConfig, "register_subclass", return_value=lambda config_cls: config_cls
    ):
        spec.loader.exec_module(module)
    return module


SO101FollowerDoraRobotConfig = _load_config(
    "_test_so101_follower_config",
    "robodriver-robot-so101-follower-dora",
    "robodriver_robot_so101_follower_dora",
).SO101FollowerDoraRobotConfig
SO101AIODoraRobotConfig = _load_config(
    "_test_so101_aio_dora_config",
    "robodriver-robot-so101-aio-dora",
    "robodriver_robot_so101_aio_dora",
).SO101AIODoraRobotConfig
So101AioRos2RobotConfig = _load_config(
    "_test_so101_aio_ros2_config",
    "robodriver-robot-so101-aio-ros2",
    "robodriver_robot_so101_aio_ros2",
).So101AioRos2RobotConfig
GALAXEALITEFollowerRos2RobotConfig = _load_config(
    "_test_galaxealite_config",
    "robodriver-robot-galaxealite-follower-ros2",
    "robodriver_robot_galaxealite_follower_ros2",
).GALAXEALITEFollowerRos2RobotConfig

SO101_RESET_VALUES = [2.154, -103.385, 89.802, 74.857, -2.242, 0.784]


class ResetTargetPoseTest(unittest.TestCase):
    def test_unconfigured_robot_keeps_legacy_targets(self):
        target = _build_reset_target_pose(["joint.pos", "gripper.pos"], {})

        self.assertEqual(target, {"joint.pos": 0.0, "gripper.pos": 100.0})

    def test_gripper_override_replaces_configured_value(self):
        configured = {"joint.pos": 12.5, "gripper.pos": 0.784}

        target = _build_reset_target_pose(configured, configured, 100)

        self.assertEqual(target, {"joint.pos": 12.5, "gripper.pos": 100.0})

    def test_so101_configs_match_their_action_keys(self):
        configs = [
            (SO101FollowerDoraRobotConfig(), False),
            (SO101AIODoraRobotConfig(), True),
            (So101AioRos2RobotConfig(), True),
        ]
        for config, aio in configs:
            with self.subTest(config=type(config).__name__):
                motors = config.leader_motors if aio else config.motors
                prefix = "leader_" if aio else ""
                keys = [f"{prefix}{motor}.pos" for motor in motors]
                target = _build_reset_target_pose(keys, config.reset_pose)

                self.assertEqual(list(target), keys)
                self.assertEqual(list(target.values()), SO101_RESET_VALUES)

    def test_galaxea_config_matches_all_17_action_keys(self):
        config = GALAXEALITEFollowerRos2RobotConfig()
        keys = [
            f"follower_{joint}.pos"
            for component in config.follower_motors.values()
            for joint in component
        ]

        target = _build_reset_target_pose(keys, config.reset_pose)

        self.assertEqual(len(target), 17)
        self.assertEqual(target, config.reset_pose)

    def test_reset_command_sends_configured_target(self):
        configured = {
            "shoulder_pan.pos": 2.154,
            "gripper.pos": 0.784,
        }
        robot = SimpleNamespace(
            action_features={key: float for key in configured},
            config=SimpleNamespace(reset_pose=configured),
            get_observation=Mock(
                return_value={"shoulder_pan.pos": 0.0, "gripper.pos": 50.0}
            ),
            send_action=Mock(),
        )
        daemon = SimpleNamespace(robot=robot, get_observation=Mock(return_value=None))
        coordinator = object.__new__(Coordinator)
        coordinator.daemon = daemon
        coordinator.running = True
        coordinator.recording = False
        coordinator.replaying = False
        coordinator.inferring = False
        coordinator.send_response = AsyncMock()

        with (
            patch("robodriver.core.coordinator.DEFAULT_FPS", 2),
            patch("robodriver.core.coordinator.time.sleep"),
        ):
            asyncio.run(
                coordinator._Coordinator__on_robot_command_handle(
                    {"cmd": "reset_pose", "msg": {}}
                )
            )

        self.assertEqual(robot.send_action.call_count, 3)
        actual = robot.send_action.call_args.args[0]
        self.assertEqual(list(actual), list(configured))
        for key, value in configured.items():
            self.assertAlmostEqual(actual[key], value)
        coordinator.send_response.assert_awaited_once_with("reset_pose", "success")


if __name__ == "__main__":
    unittest.main()
