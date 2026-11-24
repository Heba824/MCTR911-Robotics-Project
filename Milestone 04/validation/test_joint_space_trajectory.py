"""Unit tests for the MCTR911 Milestone 04 joint-space trajectory module."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

TRAJECTORY_DIR = Path(__file__).resolve().parents[1] / "trajectory"
sys.path.insert(0, str(TRAJECTORY_DIR))

from joint_space_trajectory import QuinticJointTrajectory  # noqa: E402


class TestQuinticJointTrajectory(unittest.TestCase):
    def setUp(self) -> None:
        self.q0 = np.array([-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0.0])
        self.delta = np.array([-0.30, 0.16, -0.18, 0.12, -0.10, -0.20])
        self.qf = self.q0 + self.delta
        self.duration = 6.0
        self.trajectory = QuinticJointTrajectory(self.q0, self.qf, self.duration)

    def test_exact_start_and_goal_with_zero_endpoint_derivatives(self) -> None:
        start = self.trajectory.evaluate(0.0)
        goal = self.trajectory.evaluate(self.duration)
        np.testing.assert_allclose(start.position_rad, self.q0, atol=0.0, rtol=0.0)
        np.testing.assert_allclose(goal.position_rad, self.qf, atol=1e-14, rtol=0.0)
        np.testing.assert_allclose(start.velocity_rad_s, np.zeros(6), atol=0.0)
        np.testing.assert_allclose(goal.velocity_rad_s, np.zeros(6), atol=0.0)
        np.testing.assert_allclose(start.acceleration_rad_s2, np.zeros(6), atol=0.0)
        np.testing.assert_allclose(goal.acceleration_rad_s2, np.zeros(6), atol=0.0)

    def test_midpoint_progress_and_peak_velocity(self) -> None:
        point = self.trajectory.evaluate(self.duration / 2.0)
        np.testing.assert_allclose(point.progress, 0.5, atol=1e-14)
        np.testing.assert_allclose(point.position_rad, 0.5 * (self.q0 + self.qf), atol=1e-14)
        np.testing.assert_allclose(point.velocity_rad_s, 1.875 * self.delta / self.duration, atol=1e-14)
        np.testing.assert_allclose(point.acceleration_rad_s2, np.zeros(6), atol=1e-14)
        self.assertEqual(point.phase, "peak_velocity")

    def test_hold_after_move(self) -> None:
        point = self.trajectory.evaluate(self.duration + 1.25)
        np.testing.assert_allclose(point.position_rad, self.qf, atol=0.0)
        np.testing.assert_allclose(point.velocity_rad_s, np.zeros(6), atol=0.0)
        np.testing.assert_allclose(point.acceleration_rad_s2, np.zeros(6), atol=0.0)
        self.assertEqual(point.phase, "hold")

    def test_negative_time_clamps_to_start(self) -> None:
        point = self.trajectory.evaluate(-0.5)
        np.testing.assert_allclose(point.position_rad, self.q0, atol=0.0)
        np.testing.assert_allclose(point.velocity_rad_s, np.zeros(6), atol=0.0)
        self.assertEqual(point.phase, "before_start")

    def test_progress_is_monotonic_and_bounded(self) -> None:
        points = [self.trajectory.evaluate(t) for t in np.linspace(0.0, self.duration, 301)]
        progress = np.array([point.progress for point in points])
        self.assertTrue(np.all(progress >= 0.0))
        self.assertTrue(np.all(progress <= 1.0))
        self.assertTrue(np.all(np.diff(progress) >= -1e-14))

    def test_analytic_derivatives_match_finite_differences(self) -> None:
        t, h = 2.1, 1e-5
        p_minus = self.trajectory.evaluate(t - h)
        point = self.trajectory.evaluate(t)
        p_plus = self.trajectory.evaluate(t + h)
        velocity_fd = (p_plus.position_rad - p_minus.position_rad) / (2.0*h)
        acceleration_fd = (p_plus.velocity_rad_s - p_minus.velocity_rad_s) / (2.0*h)
        np.testing.assert_allclose(point.velocity_rad_s, velocity_fd, atol=2e-9, rtol=2e-7)
        np.testing.assert_allclose(point.acceleration_rad_s2, acceleration_fd, atol=2e-9, rtol=2e-7)

    def test_sampling_includes_exact_final_time_and_hold(self) -> None:
        samples = self.trajectory.sample(0.02, total_duration_s=8.0)
        self.assertEqual(len(samples), 401)
        self.assertAlmostEqual(samples[0].time_s, 0.0)
        self.assertAlmostEqual(samples[-1].time_s, 8.0)
        np.testing.assert_allclose(samples[-1].position_rad, self.qf, atol=0.0)
        np.testing.assert_allclose(samples[-1].velocity_rad_s, np.zeros(6), atol=0.0)
        self.assertEqual(samples[-1].phase, "hold")

    def test_analytical_peak_limits(self) -> None:
        np.testing.assert_allclose(
            self.trajectory.maximum_velocity_rad_s,
            1.875 * np.abs(self.delta) / self.duration,
        )
        np.testing.assert_allclose(
            self.trajectory.maximum_acceleration_rad_s2,
            (10.0 / np.sqrt(3.0)) * np.abs(self.delta) / self.duration**2,
        )

    def test_duration_for_velocity_and_acceleration_limits(self) -> None:
        vmax, amax = np.full(6, 0.20), np.full(6, 0.25)
        required = QuinticJointTrajectory.minimum_duration_for_limits(
            self.q0, self.qf, vmax, amax
        )
        candidate = QuinticJointTrajectory(self.q0, self.qf, required)
        self.assertTrue(np.all(candidate.maximum_velocity_rad_s <= vmax + 1e-12))
        self.assertTrue(np.all(candidate.maximum_acceleration_rad_s2 <= amax + 1e-12))

    def test_joint_limit_validation(self) -> None:
        self.trajectory.validate_joint_limits(np.full(6, -3.2), np.full(6, 3.2))
        with self.assertRaises(ValueError):
            self.trajectory.validate_joint_limits(np.full(6, -1.0), np.full(6, 3.2))

    def test_invalid_inputs_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            QuinticJointTrajectory(np.zeros(5), np.zeros(6), 2.0)
        with self.assertRaises(ValueError):
            QuinticJointTrajectory(np.zeros(6), np.zeros(6), 0.0)
        with self.assertRaises(ValueError):
            self.trajectory.evaluate(float("nan"))
        with self.assertRaises(ValueError):
            self.trajectory.sample(0.0)
        with self.assertRaises(ValueError):
            self.trajectory.sample(0.01, total_duration_s=5.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
