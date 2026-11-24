"""Unit tests for the MCTR911 UR5e motor-level position-servo controller."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

CONTROL_DIR = Path(__file__).resolve().parents[1] / "control"
sys.path.insert(0, str(CONTROL_DIR))

from motor_level_controller import (  # noqa: E402
    compute_position_servo_command,
    expected_tracking_torque,
    position_actuator_torque,
)


class TestMotorLevelPositionServo(unittest.TestCase):
    def setUp(self):
        self.kp = np.array([2000, 2000, 2000, 500, 500, 500], dtype=float)
        self.kd = np.array([400, 400, 400, 100, 100, 100], dtype=float)
        self.q_ref = np.array([-1.4, -1.2, 1.3, -1.1, -1.5, 0.2])
        self.qdot_ref = np.array([0.08, -0.04, 0.05, 0.03, -0.02, 0.06])
        self.tau_ff = np.array([3.0, -4.0, 2.5, 0.8, -0.6, 0.4])

    def test_feedforward_command_produces_requested_torque_at_reference(self):
        result = compute_position_servo_command(
            self.q_ref, self.qdot_ref, self.kp, self.kd,
            feedforward_torque_nm=self.tau_ff,
        )
        actual_torque = position_actuator_torque(
            result.command_rad, self.q_ref, self.qdot_ref, self.kp, self.kd
        )
        np.testing.assert_allclose(actual_torque, self.tau_ff, atol=1e-12, rtol=1e-12)

    def test_closed_loop_torque_equals_feedforward_plus_pd_error(self):
        q_actual = self.q_ref + np.array([0.01, -0.02, 0.015, -0.01, 0.008, -0.012])
        qdot_actual = self.qdot_ref + np.array([0.03, -0.02, 0.01, -0.015, 0.01, -0.025])
        result = compute_position_servo_command(
            self.q_ref, self.qdot_ref, self.kp, self.kd,
            feedforward_torque_nm=self.tau_ff,
        )
        measured_model_torque = position_actuator_torque(
            result.command_rad, q_actual, qdot_actual, self.kp, self.kd
        )
        expected = expected_tracking_torque(
            self.q_ref, self.qdot_ref, q_actual, qdot_actual,
            self.kp, self.kd, self.tau_ff,
        )
        np.testing.assert_allclose(measured_model_torque, expected, atol=1e-11, rtol=1e-12)

    def test_baseline_without_feedforward_is_reference_position(self):
        result = compute_position_servo_command(
            self.q_ref, self.qdot_ref, self.kp, self.kd
        )
        np.testing.assert_allclose(result.command_rad, self.q_ref, atol=0.0)
        np.testing.assert_allclose(result.feedforward_torque_nm, np.zeros(6), atol=0.0)
        self.assertFalse(np.any(result.clipped))

    def test_command_offset_compensates_actuator_velocity_damping(self):
        result = compute_position_servo_command(
            self.q_ref, self.qdot_ref, self.kp, self.kd,
            feedforward_torque_nm=np.zeros(6),
        )
        expected_offset = self.kd * self.qdot_ref / self.kp
        np.testing.assert_allclose(result.command_rad - self.q_ref, expected_offset, atol=1e-15)
        torque = position_actuator_torque(
            result.command_rad, self.q_ref, self.qdot_ref, self.kp, self.kd
        )
        np.testing.assert_allclose(torque, np.zeros(6), atol=1e-12)

    def test_control_range_clipping_is_reported(self):
        limits = np.column_stack((self.q_ref - 0.001, self.q_ref + 0.001))
        result = compute_position_servo_command(
            self.q_ref, self.qdot_ref, self.kp, self.kd,
            feedforward_torque_nm=self.tau_ff,
            control_range_rad=limits,
        )
        self.assertTrue(np.any(result.clipped))
        self.assertTrue(np.all(result.command_rad >= limits[:, 0]))
        self.assertTrue(np.all(result.command_rad <= limits[:, 1]))
        np.testing.assert_allclose(
            result.clipped,
            ~np.isclose(result.command_rad, result.unclipped_command_rad, rtol=0.0, atol=1e-12),
        )

    def test_zero_velocity_and_zero_feedforward_give_position_hold(self):
        result = compute_position_servo_command(
            self.q_ref, np.zeros(6), self.kp, self.kd,
            feedforward_torque_nm=np.zeros(6),
        )
        np.testing.assert_allclose(result.command_rad, self.q_ref, atol=0.0)

    def test_invalid_shapes_and_gains_are_rejected(self):
        with self.assertRaises(ValueError):
            compute_position_servo_command(np.zeros(5), np.zeros(6), self.kp, self.kd)
        with self.assertRaises(ValueError):
            compute_position_servo_command(self.q_ref, self.qdot_ref, np.zeros(6), self.kd)
        with self.assertRaises(ValueError):
            compute_position_servo_command(self.q_ref, self.qdot_ref, self.kp, -np.ones(6))
        with self.assertRaises(ValueError):
            compute_position_servo_command(
                self.q_ref, self.qdot_ref, self.kp, self.kd,
                feedforward_torque_nm=np.zeros(5),
            )

    def test_invalid_control_limits_are_rejected(self):
        with self.assertRaises(ValueError):
            compute_position_servo_command(
                self.q_ref, self.qdot_ref, self.kp, self.kd,
                control_range_rad=np.zeros((6, 3)),
            )
        invalid = np.column_stack((np.ones(6), np.zeros(6)))
        with self.assertRaises(ValueError):
            compute_position_servo_command(
                self.q_ref, self.qdot_ref, self.kp, self.kd,
                control_range_rad=invalid,
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
