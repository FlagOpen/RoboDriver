import threading
from typing import Any, Dict

from robodriver_robot_astribott1_aio_py.robot import AstribotT1AioPyRobot

from .config import AstribotT1AioSharpaPyRobotConfig
from .sharpa_hand import SharpaHands


class AstribotT1AioSharpaPyRobot(AstribotT1AioPyRobot):
    """AstriBot T1 body plus Sharpa hands, excluding the original grippers."""

    config_class = AstribotT1AioSharpaPyRobotConfig
    name = "astribott1_aio_sharpa_py"

    def __init__(self, config: AstribotT1AioSharpaPyRobotConfig):
        super().__init__(config)
        self._sharpa_lock = threading.RLock()
        self._sharpa_hands = SharpaHands(config.sharpa_hands)

    @property
    def observation_features(self) -> Dict[str, Any]:
        features = dict(super().observation_features)
        features.update(self._sharpa_hands.follower_features())
        return features

    @property
    def action_features(self) -> Dict[str, Any]:
        features = dict(super().action_features)
        features.update(self._sharpa_hands.leader_features())
        return features

    def connect(self, calibrate: bool = True) -> None:
        super().connect(calibrate=calibrate)
        try:
            with self._sharpa_lock:
                self._sharpa_hands.connect()
        except Exception:
            self.disconnect()
            raise

    def disconnect(self) -> None:
        try:
            with self._sharpa_lock:
                self._sharpa_hands.disconnect()
        finally:
            super().disconnect()

    def get_observation(self) -> Dict[str, Any]:
        observation = super().get_observation()
        with self._sharpa_lock:
            if self._sharpa_hands.connected:
                observation.update(self._sharpa_hands.get_observation())
        return observation

    def get_action(self) -> Dict[str, Any]:
        action = super().get_action()
        with self._sharpa_lock:
            if self._sharpa_hands.connected:
                action.update(self._sharpa_hands.get_action())
        return action

    def get_record_joints(self) -> tuple[Dict[str, float], Dict[str, float], Dict[str, int]]:
        observation, action, counts = super().get_record_joints()
        with self._sharpa_lock:
            if self._sharpa_hands.connected:
                hand_observation, hand_action = self._sharpa_hands.get_joint_snapshot()
                observation.update(hand_observation)
                action.update(hand_action)
        return observation, action, counts

    def send_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        body_action = {key: value for key, value in action.items() if not key.startswith("leader_hand_")}
        hand_action = {key: value for key, value in action.items() if key.startswith("leader_hand_")}
        sent = super().send_action(body_action) if body_action else {}
        if hand_action:
            with self._sharpa_lock:
                sent.update(self._sharpa_hands.send_action(hand_action))
        return sent