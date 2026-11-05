import unittest
from unittest.mock import patch

from openpi_client import msgpack_numpy
from openpi_client.websocket_client_policy import WebsocketClientPolicy
from robodriver.core.policies import (
    BCPolicy,
    FlagScalePolicy,
    LeRobotPolicy,
    OpenPIPolicy,
    create_policy,
)


HEADERS = {"Authorization": "Bearer test-jwt"}


class FakeClient:
    def __init__(self):
        self.closed = False

    def recv(self):
        return msgpack_numpy.packb({})

    def close(self):
        self.closed = True


class PolicyAuthTest(unittest.TestCase):
    def test_registry_accepts_common_headers_for_every_policy(self):
        policy_classes = {
            "bc": BCPolicy,
            "flagscale": FlagScalePolicy,
            "lerobot": LeRobotPolicy,
            "openpi": OpenPIPolicy,
        }

        for policy_type, policy_class in policy_classes.items():
            with self.subTest(policy_type=policy_type):
                policy = create_policy(
                    policy_type=policy_type,
                    host="127.0.0.1",
                    port=8003,
                    path="",
                    headers=HEADERS,
                    api_key=None,
                )
                self.assertIsInstance(policy, policy_class)

    def test_openpi_policy_forwards_headers(self):
        with patch(
            "openpi_client.websocket_client_policy.WebsocketClientPolicy"
        ) as client_class:
            policy = OpenPIPolicy(host="127.0.0.1", headers=HEADERS)
            policy._ensure_connected()

        client_class.assert_called_once_with(
            host="ws://127.0.0.1:8087/inference",
            port=None,
            api_key=None,
            headers=HEADERS,
        )

    def test_openpi_client_uses_bearer_header(self):
        with patch(
            "websockets.sync.client.connect", return_value=FakeClient()
        ) as connect:
            WebsocketClientPolicy(host="ws://127.0.0.1:8003", headers=HEADERS)

        self.assertEqual(
            connect.call_args.kwargs["additional_headers"], HEADERS
        )

    def test_flagscale_uses_bearer_header(self):
        policy = FlagScalePolicy(host="127.0.0.1", headers=HEADERS)
        with patch(
            "websockets.sync.client.connect", return_value=FakeClient()
        ) as connect:
            policy._ensure_connected()

        self.assertEqual(
            connect.call_args.kwargs["additional_headers"], HEADERS
        )

    def test_explicit_authorization_overrides_legacy_api_key(self):
        with patch(
            "websockets.sync.client.connect", return_value=FakeClient()
        ) as connect:
            WebsocketClientPolicy(
                host="ws://127.0.0.1:8003",
                api_key="legacy-key",
                headers=HEADERS,
            )

        self.assertEqual(
            connect.call_args.kwargs["additional_headers"], HEADERS
        )

    def test_legacy_api_key_behavior_is_preserved(self):
        with patch(
            "websockets.sync.client.connect", return_value=FakeClient()
        ) as connect:
            WebsocketClientPolicy(
                host="ws://127.0.0.1:8003", api_key="legacy-key"
            )

        self.assertEqual(
            connect.call_args.kwargs["additional_headers"],
            {"Authorization": "Api-Key legacy-key"},
        )


if __name__ == "__main__":
    unittest.main()
