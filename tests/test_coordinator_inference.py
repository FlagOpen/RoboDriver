import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from robodriver.core.coordinator import Coordinator, _parse_inference_config
from robodriver.core.policies.bc_policy import BCPolicy


def _inference_message(policy_type="lerobot"):
    return {
        "cmd": "start_inference",
        "msg": {
            "device_body": "SO101",
            "chunk_size": 50,
            "fps": 30,
            "prompt": "pick up the block",
            "policy_type": policy_type,
            "data_channel": {
                "type": "websocket",
                "url": "wss://inference.example/svc/3",
                "headers": {"Authorization": "Bearer test-jwt"},
            },
        },
    }


class InferenceConfigTest(unittest.TestCase):
    def test_parses_lerobot_wss_config(self):
        config = _parse_inference_config(_inference_message())

        self.assertEqual(config.policy_type, "lerobot")
        self.assertEqual(config.policy_host, "wss://inference.example/svc/3")
        self.assertEqual(config.policy_port, 443)
        self.assertEqual(config.policy_path, "/svc/3")
        self.assertEqual(config.prompt, "pick up the block")
        self.assertEqual(config.fps, 30)
        self.assertEqual(
            config.headers, {"Authorization": "Bearer test-jwt"}
        )

    def test_parses_ws_url_with_explicit_port(self):
        message = _inference_message(policy_type="bc")
        message["msg"]["data_channel"]["url"] = "ws://127.0.0.1:8003/inference"

        config = _parse_inference_config(message)

        self.assertEqual(config.policy_type, "bc")
        self.assertEqual(config.policy_host, "ws://127.0.0.1:8003/inference")
        self.assertEqual(config.policy_port, 8003)
        self.assertEqual(config.policy_path, "/inference")

    def test_rejects_missing_url(self):
        message = _inference_message()
        del message["msg"]["data_channel"]["url"]

        with self.assertRaisesRegex(ValueError, "URL is required"):
            _parse_inference_config(message)

    def test_rejects_legacy_nested_message(self):
        message = {
            "cmd": "start_inference",
            "msg": {
                "msg": {"ws_url": "wss://inference.example/svc/3"},
                "token": "Bearer test-jwt",
            },
        }

        with self.assertRaisesRegex(ValueError, "URL is required"):
            _parse_inference_config(message)


class CoordinatorInferenceTest(unittest.IsolatedAsyncioTestCase):
    async def test_starts_lerobot_inference_without_logging_token(self):
        coordinator = object.__new__(Coordinator)
        coordinator.recording = False
        coordinator.replaying = False
        coordinator.inferring = False
        coordinator.daemon = SimpleNamespace(robot=object())
        coordinator.teleop = None
        coordinator.send_response = AsyncMock()

        with (
            patch("robodriver.core.coordinator.Inferencer") as inferencer_class,
            patch("robodriver.core.coordinator.logger") as logger,
        ):
            await coordinator._Coordinator__on_robot_command_handle(
                _inference_message()
            )

        inferencer = inferencer_class.return_value
        inferencer.connect.assert_called_once_with()
        inferencer.start.assert_called_once_with()
        config = inferencer_class.call_args.kwargs["infer_cfg"]
        self.assertEqual(config.policy_type, "lerobot")
        self.assertEqual(config.policy_host, "wss://inference.example/svc/3")
        self.assertTrue(coordinator.inferring)
        coordinator.send_response.assert_awaited_once_with(
            "start_inference", "success"
        )
        self.assertNotIn("test-jwt", str(logger.method_calls))


class BCPolicyURLTest(unittest.TestCase):
    def test_preserves_full_wss_url(self):
        policy = BCPolicy(
            host="wss://inference.example/svc/3",
            headers={"Authorization": "Bearer test-jwt"},
        )
        policy._connect_async = AsyncMock()

        try:
            policy._ensure_connected()
        finally:
            policy._loop.close()

        policy._connect_async.assert_awaited_once_with(
            "wss://inference.example/svc/3",
            {"Authorization": "Bearer test-jwt"},
        )


if __name__ == "__main__":
    unittest.main()
