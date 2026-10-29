import unittest

import numpy as np

from robodriver.core.policies.bc_policy import BCRobotUtil


def _dual_arm_obs():
    """galaxea 风格观察：14 个关节标量 + 4 路双臂图像。"""
    obs = {f"joint_{i}.pos": float(i) for i in range(14)}
    for key in (
        "image_top_left",
        "image_top_right",
        "image_wrist_left",
        "image_wrist_right",
    ):
        obs[key] = np.zeros((480, 640, 3), dtype=np.uint8)
    return obs


def _so101_obs():
    """SO101 风格观察：6 个关节标量 + 2 路单臂图像。"""
    obs = {f"joint_{i}.pos": float(i) for i in range(6)}
    obs["image_top_rgb"] = np.zeros((480, 640, 3), dtype=np.uint8)
    obs["image_wrist"] = np.zeros((480, 640, 3), dtype=np.uint8)
    return obs


class ExtractStateTest(unittest.TestCase):
    def test_dual_arm_state_has_14_values(self):
        state = BCRobotUtil.extract_state(_dual_arm_obs())

        self.assertEqual(state.shape, (14,))
        self.assertEqual(state.dtype, np.float32)
        np.testing.assert_array_equal(state, np.arange(14, dtype=np.float32))

    def test_so101_state_has_6_values(self):
        state = BCRobotUtil.extract_state(_so101_obs())

        self.assertEqual(state.shape, (6,))
        np.testing.assert_array_equal(state, np.arange(6, dtype=np.float32))

    def test_numpy_scalar_joints_are_included(self):
        obs = {
            "joint_0.pos": np.float32(1.5),
            "joint_1.pos": np.float64(-2.0),
            "image_top_rgb": np.zeros((4, 4, 3), dtype=np.uint8),
        }

        state = BCRobotUtil.extract_state(obs)

        np.testing.assert_allclose(state, [1.5, -2.0])


class ExtractImagesTest(unittest.TestCase):
    def test_dual_arm_camera_mapping(self):
        images = BCRobotUtil.extract_images(_dual_arm_obs())

        self.assertEqual(
            set(images), {"top_left", "top_right", "wrist_left", "wrist_right"}
        )

    def test_so101_camera_mapping(self):
        images = BCRobotUtil.extract_images(_so101_obs())

        self.assertEqual(set(images), {"top", "wrist"})

    def test_transform_is_applied(self):
        obs = _so101_obs()

        images = BCRobotUtil.extract_images(obs, transform=lambda img: img.shape)

        self.assertEqual(images["top"], (480, 640, 3))


class ZipActionTest(unittest.TestCase):
    def test_action_is_zipped_with_feature_keys(self):
        action_features = {f"joint_{i}.pos": float for i in range(6)}

        result = BCRobotUtil.zip_action(np.arange(6, dtype=np.float32), action_features)

        self.assertEqual(list(result), list(action_features))
        np.testing.assert_array_equal(list(result.values()), np.arange(6))


if __name__ == "__main__":
    unittest.main()
