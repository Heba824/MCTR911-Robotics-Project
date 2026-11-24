"""Unit tests for full-pose inverse kinematics and task-space trajectory mapping."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TRAJ_DIR = ROOT / 'Milestone 04' / 'trajectory'
M2_DIR = ROOT / 'Milestone 02' / 'kinematics'
M3_DIR = ROOT / 'Milestone 03' / 'kinematics'
for folder in (TRAJ_DIR, M2_DIR, M3_DIR):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

from ur5e_dh_fk import forward_kinematics  # noqa: E402
from ur5e_jacobian import geometric_jacobian  # noqa: E402
from task_space_trajectory import QuinticTaskSpaceTrajectory  # noqa: E402
from full_pose_ik import (  # noqa: E402
    HOME,
    JOINT_MAX,
    JOINT_MIN,
    full_pose_inverse_kinematics,
    pose_error_world,
)

JOINT_DELTA = np.array([-0.30, 0.16, -0.18, 0.12, -0.10, -0.20], dtype=float)


class TestFullPoseInverseKinematics(unittest.TestCase):
    def test_pose_error_zero_for_identical_pose(self):
        pose = forward_kinematics(HOME)
        p_err, r_err = pose_error_world(pose, pose)
        np.testing.assert_allclose(p_err, np.zeros(3), atol=1e-14)
        np.testing.assert_allclose(r_err, np.zeros(3), atol=1e-14)

    def test_pose_error_orientation_norm_is_shortest_rotation(self):
        from task_space_trajectory import so3_exp
        pose = np.eye(4)
        target = np.eye(4)
        target[:3, :3] = so3_exp([0.0, 0.0, 1.5 * np.pi])
        _, r_err = pose_error_world(pose, target)
        self.assertAlmostEqual(float(np.linalg.norm(r_err)), 0.5 * np.pi, places=10)

    def test_single_pose_ik_recovers_known_target(self):
        q_true = HOME + JOINT_DELTA
        pose = forward_kinematics(q_true)
        result = full_pose_inverse_kinematics(pose, HOME)
        self.assertTrue(result.success, msg=result.message)
        self.assertLessEqual(result.position_error_m, 1e-5)
        self.assertLessEqual(result.orientation_error_rad, 1e-5)
        pose_recovered = forward_kinematics(result.joint_angles)
        p_err, r_err = pose_error_world(pose_recovered, pose)
        self.assertLessEqual(float(np.linalg.norm(p_err)), 1e-5)
        self.assertLessEqual(float(np.linalg.norm(r_err)), 1e-5)

    def test_start_pose_fixed_point(self):
        pose = forward_kinematics(HOME)
        result = full_pose_inverse_kinematics(pose, HOME)
        self.assertTrue(result.success, msg=result.message)
        np.testing.assert_allclose(result.joint_angles, HOME, atol=1e-12)
        self.assertEqual(result.iterations, 0)

    def test_intermediate_task_space_samples_are_reachable(self):
        start_pose = forward_kinematics(HOME)
        goal_pose = forward_kinematics(HOME + JOINT_DELTA)
        trajectory = QuinticTaskSpaceTrajectory(start_pose, goal_pose, 6.0)
        # Coarse subset keeps the unit test quick while still checking continuity.
        samples = trajectory.sample(0.3, total_duration_s=6.0)
        q_seed = HOME.copy()
        max_step = 0.0
        for point in samples:
            result = full_pose_inverse_kinematics(point.pose_world, q_seed)
            self.assertTrue(result.success, msg=f't={point.time_s:.3f}s: {result.message}')
            self.assertLessEqual(result.position_error_m, 1e-5)
            self.assertLessEqual(result.orientation_error_rad, 1e-5)
            self.assertTrue(np.all(result.joint_angles >= JOINT_MIN - 1e-12))
            self.assertTrue(np.all(result.joint_angles <= JOINT_MAX + 1e-12))
            step = float(np.linalg.norm(result.joint_angles - q_seed))
            max_step = max(max_step, step)
            q_seed = result.joint_angles
        self.assertLess(max_step, 0.2)

    def test_geometric_jacobian_is_finite_at_solution(self):
        pose = forward_kinematics(HOME + JOINT_DELTA)
        result = full_pose_inverse_kinematics(pose, HOME)
        self.assertTrue(result.success, msg=result.message)
        jac = geometric_jacobian(result.joint_angles)
        self.assertEqual(jac.shape, (6, 6))
        self.assertTrue(np.all(np.isfinite(jac)))

    def test_invalid_inputs_are_rejected(self):
        pose = forward_kinematics(HOME)
        with self.assertRaises(ValueError):
            full_pose_inverse_kinematics(np.eye(3), HOME)
        with self.assertRaises(ValueError):
            full_pose_inverse_kinematics(pose, np.zeros(5))
        with self.assertRaises(ValueError):
            full_pose_inverse_kinematics(pose, HOME, max_iterations=0)
        with self.assertRaises(ValueError):
            full_pose_inverse_kinematics(pose, HOME, damping=-1.0)
        with self.assertRaises(ValueError):
            full_pose_inverse_kinematics(pose, HOME, position_weight=0.0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
