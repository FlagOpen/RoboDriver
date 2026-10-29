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


class _FakeEcodes:
    EV_KEY = 1


class _FakeInputDevice:
    """Stand-in for evdev.InputDevice driven by a class-level registry.

    registry maps a device path to either a list of supported key codes or an
    OSError to raise on open.
    """

    registry: dict = {}

    def __init__(self, path):
        self.path = path
        self.closed = False
        behaviour = self.registry[path]
        if isinstance(behaviour, OSError):
            raise behaviour
        self._keys = behaviour

    def capabilities(self):
        return {_FakeEcodes.EV_KEY: self._keys}

    def close(self):
        self.closed = True


class EvdevDeviceSelectionTest(unittest.TestCase):
    KEY_S_CODE = 31

    def setUp(self):
        _FakeInputDevice.registry = {}
        self.disturbance = ActionDisturbance(DisturbanceConfig(enabled=True))

    def _open_devices(self, paths, config=None):
        disturbance = self.disturbance
        if config is not None:
            disturbance = ActionDisturbance(config)
        return disturbance._open_evdev_devices(
            _FakeInputDevice, _FakeEcodes, lambda: paths, self.KEY_S_CODE
        )

    def test_auto_detect_keeps_only_devices_that_emit_manual_key(self):
        _FakeInputDevice.registry = {
            "/dev/input/event0": [self.KEY_S_CODE],
            "/dev/input/event1": [30],  # some other key
        }

        devices = self._open_devices(["/dev/input/event0", "/dev/input/event1"])

        self.assertEqual([d.path for d in devices], ["/dev/input/event0"])
        self.assertFalse(devices[0].closed)

    def test_auto_detect_closes_rejected_devices(self):
        _FakeInputDevice.registry = {"/dev/input/event1": [30]}

        opened = []

        class TrackingDevice(_FakeInputDevice):
            def __init__(self, path):
                super().__init__(path)
                opened.append(self)

        self.disturbance._open_evdev_devices(
            TrackingDevice, _FakeEcodes, lambda: ["/dev/input/event1"], self.KEY_S_CODE
        )

        self.assertTrue(opened[0].closed)

    def test_explicit_manual_device_is_trusted_without_capability_check(self):
        _FakeInputDevice.registry = {"/dev/input/event9": []}

        devices = self._open_devices(
            [],
            config=DisturbanceConfig(enabled=True, manual_device="/dev/input/event9"),
        )

        self.assertEqual([d.path for d in devices], ["/dev/input/event9"])

    def test_explicit_manual_device_open_failure_returns_empty(self):
        _FakeInputDevice.registry = {"/dev/input/event9": OSError("Permission denied")}

        devices = self._open_devices(
            [],
            config=DisturbanceConfig(enabled=True, manual_device="/dev/input/event9"),
        )

        self.assertEqual(devices, [])

    def test_inaccessible_auto_detected_device_is_skipped(self):
        _FakeInputDevice.registry = {
            "/dev/input/event0": OSError("Permission denied"),
            "/dev/input/event1": [self.KEY_S_CODE],
        }

        devices = self._open_devices(["/dev/input/event0", "/dev/input/event1"])

        self.assertEqual([d.path for d in devices], ["/dev/input/event1"])

    def test_falls_back_to_discovery_when_list_devices_fails(self):
        _FakeInputDevice.registry = {"/dev/input/event2": [self.KEY_S_CODE]}

        def failing_list_devices():
            raise RuntimeError("no /dev/input")

        with patch.object(
            self.disturbance,
            "_discover_evdev_device_paths",
            return_value=["/dev/input/event2"],
        ):
            devices = self.disturbance._open_evdev_devices(
                _FakeInputDevice, _FakeEcodes, failing_list_devices, self.KEY_S_CODE
            )

        self.assertEqual([d.path for d in devices], ["/dev/input/event2"])

    def test_discover_paths_dedupes_across_patterns(self):
        with (
            patch("robodriver.core.disturbance.os.path.isdir", return_value=True),
            patch(
                "robodriver.core.disturbance.glob.glob",
                side_effect=lambda pattern: (
                    ["/dev/input/event0"]
                    if pattern.endswith("event*")
                    else (
                        ["/dev/input/event0", "/dev/input/mouse0"]
                        if pattern.endswith("*")
                        else []
                    )
                ),
            ),
        ):
            paths = self.disturbance._discover_evdev_device_paths()

        self.assertEqual(paths, ["/dev/input/event0", "/dev/input/mouse0"])

    def test_unmapped_manual_key_disables_evdev_listener(self):
        disturbance = ActionDisturbance(DisturbanceConfig(enabled=True, manual_key="?"))

        self.assertFalse(disturbance._start_evdev_listener())
        self.assertEqual(disturbance._evdev_threads, [])

    def test_stop_closes_evdev_devices(self):
        device = _FakeInputDevice.__new__(_FakeInputDevice)
        device.path = "/dev/input/event0"
        device.closed = False
        self.disturbance._evdev_devices.append(device)

        self.disturbance.stop()

        self.assertTrue(device.closed)


if __name__ == "__main__":
    unittest.main()
