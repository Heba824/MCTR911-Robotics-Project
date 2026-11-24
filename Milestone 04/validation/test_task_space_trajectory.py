"""Unit tests for the MCTR911 Milestone 04 task-space trajectory module."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

TRAJECTORY_DIR = Path(__file__).resolve().parents[1] / "trajectory"
sys.path.insert(0, str(TRAJECTORY_DIR))

from task_space_trajectory import (  # noqa: E402
    QuinticTaskSpaceTrajectory,
    rotation_matrix_to_quaternion_xyzw,
    so3_exp,
    so3_log,
)


def make_pose(position, rotation=None):
    pose = np.eye(4)
    pose[:3, 3] = np.asarray(position, dtype=float)
    if rotation is not None:
        pose[:3, :3] = rotation
    return pose


def angular_velocity_from_rotation_difference(r_minus, r_plus, r_center, h):
    r_dot = (r_plus - r_minus) / (2.0 * h)
    skew = r_dot @ r_center.T
    skew = 0.5 * (skew - skew.T)
    return np.array([skew[2, 1], skew[0, 2], skew[1, 0]])


class TestTaskSpaceTrajectory(unittest.TestCase):
    def setUp(self):
        self.start = make_pose([0.2, -0.1, 0.5], so3_exp([0.1, -0.2, 0.05]))
        self.goal = make_pose(
            [0.45, 0.15, 0.42],
            so3_exp([-0.15, 0.12, 0.35]) @ self.start[:3, :3],
        )
        self.duration = 5.0
        self.trajectory = QuinticTaskSpaceTrajectory(self.start, self.goal, self.duration)

    def test_exact_start_and_goal_poses(self):
        start = self.trajectory.evaluate(0.0)
        goal = self.trajectory.evaluate(self.duration)
        np.testing.assert_allclose(start.pose_world, self.start, atol=1e-12)
        np.testing.assert_allclose(goal.pose_world, self.goal, atol=1e-12)

    def test_zero_endpoint_twist_and_acceleration(self):
        for point in (self.trajectory.evaluate(0.0), self.trajectory.evaluate(self.duration)):
            np.testing.assert_allclose(point.twist_world, np.zeros(6), atol=1e-14)
            np.testing.assert_allclose(point.acceleration_world, np.zeros(6), atol=1e-14)

    def test_midpoint_translation_and_quintic_progress(self):
        point = self.trajectory.evaluate(self.duration / 2.0)
        np.testing.assert_allclose(point.progress, 0.5, atol=1e-14)
        np.testing.assert_allclose(
            point.position_world_m,
            0.5 * (self.start[:3, 3] + self.goal[:3, 3]),
            atol=1e-14,
        )
        np.testing.assert_allclose(point.linear_acceleration_world_m_s2, np.zeros(3), atol=1e-14)

    def test_rotation_endpoints_and_shortest_angle(self):
        self.assertLessEqual(self.trajectory.rotation_angle_rad, np.pi + 1e-12)
        np.testing.assert_allclose(
            self.trajectory.evaluate(self.duration).pose_world[:3, :3],
            self.goal[:3, :3],
            atol=1e-12,
        )

    def test_angular_velocity_matches_rotation_finite_difference(self):
        t, h = 1.7, 1e-5
        minus = self.trajectory.evaluate(t - h).pose_world[:3, :3]
        center = self.trajectory.evaluate(t).pose_world[:3, :3]
        plus = self.trajectory.evaluate(t + h).pose_world[:3, :3]
        numeric = angular_velocity_from_rotation_difference(minus, plus, center, h)
        analytic = self.trajectory.evaluate(t).angular_velocity_world_rad_s
        np.testing.assert_allclose(analytic, numeric, atol=2e-9, rtol=2e-7)

    def test_linear_velocity_and_acceleration_match_finite_difference(self):
        t, h = 1.7, 1e-5
        minus = self.trajectory.evaluate(t - h)
        center = self.trajectory.evaluate(t)
        plus = self.trajectory.evaluate(t + h)
        velocity_fd = (plus.position_world_m - minus.position_world_m) / (2*h)
        acceleration_fd = (
            plus.linear_velocity_world_m_s - minus.linear_velocity_world_m_s
        ) / (2*h)
        np.testing.assert_allclose(center.linear_velocity_world_m_s, velocity_fd,
                                   atol=2e-9, rtol=2e-7)
        np.testing.assert_allclose(center.linear_acceleration_world_m_s2, acceleration_fd,
                                   atol=2e-9, rtol=2e-7)

    def test_angular_acceleration_matches_velocity_finite_difference(self):
        t, h = 1.7, 1e-5
        minus = self.trajectory.evaluate(t - h)
        center = self.trajectory.evaluate(t)
        plus = self.trajectory.evaluate(t + h)
        numeric = (
            plus.angular_velocity_world_rad_s - minus.angular_velocity_world_rad_s
        ) / (2*h)
        np.testing.assert_allclose(center.angular_acceleration_world_rad_s2,
                                   numeric, atol=2e-9, rtol=2e-7)

    def test_quaternion_is_normalized_and_ros_order(self):
        point = self.trajectory.evaluate(2.0)
        self.assertAlmostEqual(float(np.linalg.norm(point.quaternion_xyzw)), 1.0, places=12)
        reconstructed = rotation_matrix_to_quaternion_xyzw(point.pose_world[:3, :3])
        np.testing.assert_allclose(reconstructed, point.quaternion_xyzw, atol=1e-12)

    def test_hold_after_duration(self):
        point = self.trajectory.evaluate(self.duration + 1.0)
        np.testing.assert_allclose(point.pose_world, self.goal, atol=1e-12)
        np.testing.assert_allclose(point.twist_world, np.zeros(6), atol=1e-14)
        np.testing.assert_allclose(point.acceleration_world, np.zeros(6), atol=1e-14)
        self.assertEqual(point.phase, "hold")

    def test_sampling_includes_final_hold(self):
        points = self.trajectory.sample(0.02, total_duration_s=7.0)
        self.assertEqual(len(points), 351)
        self.assertAlmostEqual(points[-1].time_s, 7.0)
        self.assertEqual(points[-1].phase, "hold")

    def test_so3_log_exp_round_trip_and_shortest_path(self):
        phi = np.array([0.2, -0.3, 0.4])
        recovered = so3_log(so3_exp(phi))
        np.testing.assert_allclose(recovered, phi, atol=1e-12)

        rotation_270_z = so3_exp([0.0, 0.0, 1.5*np.pi])
        shortest = so3_log(rotation_270_z)
        self.assertAlmostEqual(float(np.linalg.norm(shortest)), 0.5*np.pi, places=10)

    def test_no_orientation_change_has_zero_angular_reference(self):
        same_rotation_goal = make_pose([0.3, 0.2, 0.6], self.start[:3, :3])
        trajectory = QuinticTaskSpaceTrajectory(self.start, same_rotation_goal, 3.0)
        point = trajectory.evaluate(1.2)
        np.testing.assert_allclose(point.angular_velocity_world_rad_s, np.zeros(3), atol=1e-14)
        np.testing.assert_allclose(point.angular_acceleration_world_rad_s2, np.zeros(3), atol=1e-14)

    def test_invalid_inputs_are_rejected(self):
        with self.assertRaises(ValueError):
            QuinticTaskSpaceTrajectory(np.eye(3), self.goal, 2.0)
        with self.assertRaises(ValueError):
            QuinticTaskSpaceTrajectory(self.start, self.goal, 0.0)
        invalid = self.goal.copy()
        invalid[0, 0] += 0.1
        with self.assertRaises(ValueError):
            QuinticTaskSpaceTrajectory(self.start, invalid, 2.0)
        with self.assertRaises(ValueError):
            self.trajectory.evaluate(float("nan"))
        with self.assertRaises(ValueError):
            self.trajectory.sample(0.0)
        with self.assertRaises(ValueError):
            self.trajectory.sample(0.01, total_duration_s=4.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
