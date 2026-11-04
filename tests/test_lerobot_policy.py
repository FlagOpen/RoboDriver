import unittest
from types import SimpleNamespace
from unittest.mock import call, patch

import numpy as np

from robodriver.core.inferencer import Inferencer
from robodriver.core.policies import LeRobotPolicy, create_policy, lerobot_policy


def _observation():
    return {
        "observation.state": np.arange(6, dtype=np.float64),
        "observation.images.image_top_rgb": np.full((4, 5, 3), 127, dtype=np.uint8),
        "observation.images.image_wrist": np.full((4, 5, 3), 64, dtype=np.uint8),
        "prompt": "pick up the block",
    }


class FakeClient:
    def __init__(self, responses):
        self._responses = list(responses)
        self.sent = []
        self.closed = False

    def send(self, data):
        self.sent.append(data)

    def recv(self):
        return self._responses.pop(0)

    def close(self):
        self.closed = True


class LeRobotPolicyTest(unittest.TestCase):
    def test_registry_creates_lerobot_policy(self):
        policy = create_policy(
            policy_type="lerobot",
            host="ws://127.0.0.1:8003",
            port=8003,
            path="",
            headers=None,
            api_key=None,
        )
        self.assertIsInstance(policy, LeRobotPolicy)

    def test_infer_preserves_feature_names_and_wire_format(self):
        metadata = {"action_dim": 6, "n_action_steps": 2, "action_fps": 30.0}
        actions = np.arange(12, dtype=np.float32).reshape(2, 6)
        client = FakeClient(
            [
                lerobot_policy._pack(metadata),
                lerobot_policy._pack(
                    {
                        "actions": actions,
                        "actual_steps": 2,
                        "action_fps": 30.0,
                    }
                ),
            ]
        )
        policy = LeRobotPolicy(host="ws://127.0.0.1:8003", api_key="secret")

        with patch("websockets.sync.client.connect", return_value=client) as connect:
            result = policy.infer(_observation())

        connect.assert_called_once_with(
            "ws://127.0.0.1:8003",
            compression=None,
            max_size=None,
            additional_headers={"Authorization": "Bearer secret"},
            open_timeout=15.0,
            ping_interval=20,
            ping_timeout=20,
        )
        request = lerobot_policy._unpack(client.sent[0])
        self.assertEqual(request["prompt"], "pick up the block")
        self.assertEqual(request["encoding"], "raw")
        self.assertEqual(request["sequence"], 0)
        self.assertEqual(
            set(request["images"]),
            {
                "observation.images.image_top",
                "observation.images.image_wrist",
            },
        )
        self.assertEqual(
            request["images"]["observation.images.image_top"].shape, (4, 5, 3)
        )
        self.assertEqual(
            request["images"]["observation.images.image_top"].dtype, np.uint8
        )
        self.assertEqual(request["state"].dtype, np.float32)
        np.testing.assert_array_equal(request["state"], np.arange(6))
        np.testing.assert_array_equal(result["actions"], actions)
        self.assertEqual(result["action_fps"], 30.0)
        self.assertEqual(policy.server_metadata, metadata)

    def test_converts_chw_float_image_to_hwc_uint8(self):
        obs = _observation()
        obs["observation.images.image_top_rgb"] = np.ones((3, 4, 5), dtype=np.float32)
        policy = LeRobotPolicy()

        request = policy._convert_observation(obs)

        image = request["images"]["observation.images.image_top"]
        self.assertEqual(image.shape, (4, 5, 3))
        self.assertEqual(image.dtype, np.uint8)
        np.testing.assert_array_equal(image, np.full((4, 5, 3), 255, dtype=np.uint8))

    def test_missing_state_is_rejected(self):
        obs = _observation()
        del obs["observation.state"]
        policy = LeRobotPolicy()

        with self.assertRaisesRegex(ValueError, "observation.state"):
            policy._convert_observation(obs)

    def test_server_error_is_raised_and_connection_closed(self):
        client = FakeClient(
            [
                lerobot_policy._pack({"model": "act"}),
                lerobot_policy._pack(
                    {"error": {"type": "ValueError", "message": "bad observation"}}
                ),
            ]
        )
        policy = LeRobotPolicy()

        with patch("websockets.sync.client.connect", return_value=client):
            with self.assertRaisesRegex(RuntimeError, "ValueError: bad observation"):
                policy.infer(_observation())

        self.assertTrue(client.closed)
        self.assertIsNone(policy.server_metadata)


class LeRobotActionExecutionTest(unittest.TestCase):
    def test_action_chunk_uses_server_action_fps(self):
        inferencer = object.__new__(Inferencer)
        inferencer.robot = SimpleNamespace(
            action_features={f"joint_{i}.pos": float for i in range(6)},
            send_action=lambda action: None,
        )
        inferencer.infer_cfg = SimpleNamespace(
            action_chunk_sleep=1.0,
            use_rerun=False,
        )
        inferencer.running = True

        with patch("robodriver.core.inferencer.time.sleep") as sleep:
            inferencer._execute_action_chunk(
                {
                    "actions": np.zeros((2, 6), dtype=np.float32),
                    "action_fps": 20.0,
                }
            )

        self.assertEqual(sleep.call_args_list, [call(0.05), call(0.05)])


if __name__ == "__main__":
    unittest.main()
