import unittest
from unittest.mock import patch

from robodriver.core.disturbance import ActionDisturbance, DisturbanceConfig


class ActionDisturbanceTest(unittest.TestCase):
    def test_manual_trigger_overrides_action_during_duration(self):
        disturbance = ActionDisturbance(
            DisturbanceConfig(
                enabled=True,
                duration_s=0.5,
                action_overrides={"gripper": 0.07},
            )
        )
        disturbance.trigger()

        with patch("robodriver.core.disturbance.time.monotonic", return_value=10.0):
            action = disturbance.apply({"joint": 1.0, "gripper": 0.0})
            self.assertTrue(disturbance.is_active)

        self.assertEqual(action, {"joint": 1.0, "gripper": 0.07})

    def test_inactive_disturbance_returns_original_action(self):
        action = {"gripper": 0.0}
        disturbance = ActionDisturbance(DisturbanceConfig(enabled=False))

        self.assertIs(disturbance.apply(action), action)

    @patch("robodriver.core.disturbance.random.uniform", return_value=2.0)
    def test_random_trigger_overrides_action(self, _uniform):
        disturbance = ActionDisturbance(
            DisturbanceConfig(
                enabled=True,
                manual_enabled=False,
                random_enabled=True,
                random_min_interval_s=1.0,
                random_max_interval_s=3.0,
                action_overrides={"gripper": 0.07},
            )
        )
        with patch("robodriver.core.disturbance.time.monotonic", return_value=10.0):
            disturbance.start()
        with patch("robodriver.core.disturbance.time.monotonic", return_value=12.0):
            action = disturbance.apply({"gripper": 0.0})

        self.assertEqual(action["gripper"], 0.07)

    def test_invalid_random_interval_is_rejected(self):
        with self.assertRaises(ValueError):
            DisturbanceConfig(
                random_min_interval_s=5.0,
                random_max_interval_s=1.0,
            )


if __name__ == "__main__":
    unittest.main()
