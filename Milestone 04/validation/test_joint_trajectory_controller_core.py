import unittest

import numpy as np

from joint_trajectory_controller_core import (
    DEFAULT_JOINT_NAMES,
    reorder_joint_state,
    tracking_error_metrics,
    validate_start_position,
    validate_start_velocity,
)


class TestJointTrajectoryControllerCore(unittest.TestCase):
    def test_joint_state_is_returned_in_expected_order(self):
        q, qdot = reorder_joint_state(
            DEFAULT_JOINT_NAMES,
            [1, 2, 3, 4, 5, 6],
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        )
        np.testing.assert_allclose(q, [1, 2, 3, 4, 5, 6])
        np.testing.assert_allclose(qdot, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6])

    def test_shuffled_joint_state_is_reordered_by_name(self):
        names = list(reversed(DEFAULT_JOINT_NAMES))
        q, qdot = reorder_joint_state(
            names, [6, 5, 4, 3, 2, 1], [0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
        )
        np.testing.assert_allclose(q, [1, 2, 3, 4, 5, 6])
        np.testing.assert_allclose(qdot, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6])

    def test_missing_joint_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "exactly 6 names"):
            reorder_joint_state(
                DEFAULT_JOINT_NAMES[:-1],
                [0] * 5,
                [0] * 5,
            )

    def test_duplicate_joint_name_is_rejected(self):
        names = list(DEFAULT_JOINT_NAMES)
        names[-1] = names[0]
        with self.assertRaisesRegex(ValueError, "duplicate"):
            reorder_joint_state(names, [0] * 6, [0] * 6)

    def test_unexpected_joint_name_is_rejected(self):
        names = list(DEFAULT_JOINT_NAMES)
        names[-1] = "unexpected_joint"
        with self.assertRaisesRegex(ValueError, "name mismatch"):
            reorder_joint_state(names, [0] * 6, [0] * 6)

    def test_non_finite_feedback_is_rejected(self):
        q = [0.0] * 6
        q[2] = float("nan")
        with self.assertRaisesRegex(ValueError, "non-finite"):
            reorder_joint_state(DEFAULT_JOINT_NAMES, q, [0.0] * 6)

    def test_start_pose_inside_tolerance_is_accepted(self):
        expected = np.zeros(6)
        actual = np.array([0.01, -0.02, 0.0, 0.03, 0.0, -0.01])
        maximum = validate_start_position(actual, expected, 0.05)
        self.assertAlmostEqual(maximum, 0.03)

    def test_start_pose_outside_tolerance_is_rejected(self):
        actual = np.zeros(6)
        expected = np.array([0.0, 0.0, 0.0, 0.06, 0.0, 0.0])
        with self.assertRaisesRegex(ValueError, "exceeds tolerance"):
            validate_start_position(actual, expected, 0.05)

    def test_invalid_start_tolerance_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "positive and finite"):
            validate_start_position(np.zeros(6), np.zeros(6), 0.0)

    def test_start_velocity_inside_tolerance_is_accepted(self):
        velocity = np.array([0.01, -0.02, 0.0, 0.03, 0.0, -0.01])
        maximum = validate_start_velocity(velocity, 0.05)
        self.assertAlmostEqual(maximum, 0.03)

    def test_start_velocity_outside_tolerance_is_rejected(self):
        velocity = np.array([0.0, 0.0, 0.0, 0.06, 0.0, 0.0])
        with self.assertRaisesRegex(ValueError, "exceeds tolerance"):
            validate_start_velocity(velocity, 0.05)

    def test_non_finite_start_velocity_is_rejected(self):
        velocity = np.zeros(6)
        velocity[1] = float("inf")
        with self.assertRaisesRegex(ValueError, "finite"):
            validate_start_velocity(velocity, 0.05)

    def test_tracking_metrics_are_correct(self):
        desired = np.zeros((3, 6))
        actual = np.array(
            [
                [0.1, 0, 0, 0, 0, 0],
                [-0.1, 0, 0, 0, 0, 0],
                [0.2, 0, 0, 0, 0, 0],
            ]
        )
        result = tracking_error_metrics(desired, actual)
        self.assertAlmostEqual(result["per_joint"][DEFAULT_JOINT_NAMES[0]]["mae_rad"], 0.4 / 3)
        self.assertAlmostEqual(result["per_joint"][DEFAULT_JOINT_NAMES[0]]["max_abs_error_rad"], 0.2)
        self.assertAlmostEqual(result["overall"]["max_abs_error_rad"], 0.2)

    def test_tracking_metrics_reject_empty_data(self):
        with self.assertRaisesRegex(ValueError, "At least one"):
            tracking_error_metrics(np.empty((0, 6)), np.empty((0, 6)))

    def test_tracking_metrics_reject_shape_mismatch(self):
        with self.assertRaisesRegex(ValueError, "same shape"):
            tracking_error_metrics(np.zeros((2, 6)), np.zeros((3, 6)))


if __name__ == "__main__":
    unittest.main()
