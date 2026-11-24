"""Tests for M4 joint-reference CSV loading and quintic-Hermite interpolation."""
from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CONTROL_DIR = ROOT / "Milestone 04" / "control"
TRAJECTORY_DIR = ROOT / "Milestone 04" / "trajectory"
sys.path.insert(0, str(CONTROL_DIR))
sys.path.insert(0, str(TRAJECTORY_DIR))

from trajectory_csv import JointTrajectoryCSV  # noqa: E402
from joint_space_trajectory import QuinticJointTrajectory  # noqa: E402

JOINT_NAMES = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
)


class TestJointTrajectoryCSV(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.csv_path = (
            ROOT / "Milestone 04" / "Results" / "Joint_Space" /
            "joint_space_reference.csv"
        )
        cls.reference = JointTrajectoryCSV.load(cls.csv_path, JOINT_NAMES)

    def test_existing_reference_loads_with_expected_structure(self):
        self.assertEqual(self.reference.sample_count, 401)
        self.assertEqual(self.reference.joint_names, JOINT_NAMES)
        self.assertAlmostEqual(self.reference.start_time_s, 0.0)
        self.assertAlmostEqual(self.reference.end_time_s, 8.0)
        np.testing.assert_allclose(
            self.reference.start_position_rad,
            [-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0.0],
            atol=1e-12,
        )
        np.testing.assert_allclose(
            self.reference.goal_position_rad,
            [-1.8708, -1.4108, 1.3908, -1.4508, -1.6708, -0.2],
            atol=1e-12,
        )

    def test_interpolation_recovers_every_csv_knot(self):
        for index, time_s in enumerate(self.reference.times_s):
            point = self.reference.evaluate(float(time_s))
            np.testing.assert_allclose(
                point.position_rad, self.reference.positions_rad[index],
                rtol=0.0, atol=2e-12,
            )
            np.testing.assert_allclose(
                point.velocity_rad_s, self.reference.velocities_rad_s[index],
                rtol=0.0, atol=2e-10,
            )
            np.testing.assert_allclose(
                point.acceleration_rad_s2, self.reference.accelerations_rad_s2[index],
                rtol=0.0, atol=2e-8,
            )

    def test_interpolation_matches_original_analytic_quintic(self):
        q_start = self.reference.start_position_rad
        q_goal = self.reference.goal_position_rad
        analytic = QuinticJointTrajectory(q_start, q_goal, duration_s=6.0)

        for time_s in np.linspace(0.0, 8.0, 161):
            actual = self.reference.evaluate(float(time_s))
            expected = analytic.evaluate(float(time_s))
            np.testing.assert_allclose(
                actual.position_rad, expected.position_rad,
                rtol=0.0, atol=2e-10,
            )
            np.testing.assert_allclose(
                actual.velocity_rad_s, expected.velocity_rad_s,
                rtol=0.0, atol=2e-9,
            )
            np.testing.assert_allclose(
                actual.acceleration_rad_s2, expected.acceleration_rad_s2,
                rtol=0.0, atol=2e-8,
            )

    def test_reference_clamps_before_start_and_after_end(self):
        before = self.reference.evaluate(-1.0)
        start = self.reference.evaluate(0.0)
        after = self.reference.evaluate(9.0)
        end = self.reference.evaluate(8.0)

        np.testing.assert_allclose(before.position_rad, start.position_rad)
        np.testing.assert_allclose(before.velocity_rad_s, start.velocity_rad_s)
        np.testing.assert_allclose(after.position_rad, end.position_rad)
        np.testing.assert_allclose(after.velocity_rad_s, end.velocity_rad_s)
        self.assertEqual(before.phase, "before_start")
        self.assertEqual(after.phase, "after_end")

    def test_non_finite_evaluation_time_is_rejected(self):
        with self.assertRaises(ValueError):
            self.reference.evaluate(float("nan"))
        with self.assertRaises(ValueError):
            self.reference.evaluate(float("inf"))

    def _write_minimal_csv(self, directory: Path, times=(0.0, 1.0), bad_value=None):
        path = directory / "trajectory.csv"
        columns = ["time_s", "phase"]
        for name in JOINT_NAMES:
            columns.extend((
                f"q_desired_{name}_rad",
                f"qdot_desired_{name}_rad_s",
                f"qddot_desired_{name}_rad_s2",
            ))
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            for row_index, time_s in enumerate(times):
                row = {"time_s": time_s, "phase": "test"}
                for name in JOINT_NAMES:
                    row[f"q_desired_{name}_rad"] = 0.0
                    row[f"qdot_desired_{name}_rad_s"] = 0.0
                    row[f"qddot_desired_{name}_rad_s2"] = 0.0
                if bad_value is not None and row_index == 0:
                    row["q_desired_shoulder_pan_joint_rad"] = bad_value
                writer.writerow(row)
        return path

    def test_duplicate_or_decreasing_times_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write_minimal_csv(Path(tmp), times=(0.0, 0.0))
            with self.assertRaisesRegex(ValueError, "strictly increasing"):
                JointTrajectoryCSV.load(path, JOINT_NAMES)

            path = self._write_minimal_csv(Path(tmp), times=(0.0, 0.5, 0.4))
            with self.assertRaisesRegex(ValueError, "strictly increasing"):
                JointTrajectoryCSV.load(path, JOINT_NAMES)

    def test_non_finite_csv_values_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write_minimal_csv(Path(tmp), bad_value="nan")
            with self.assertRaisesRegex(ValueError, "non-finite"):
                JointTrajectoryCSV.load(path, JOINT_NAMES)

    def test_missing_required_columns_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "missing.csv"
            path.write_text("time_s,phase\n0,start\n1,hold\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "missing required columns"):
                JointTrajectoryCSV.load(path, JOINT_NAMES)

    def test_joint_name_contract_is_validated(self):
        with self.assertRaisesRegex(ValueError, "unique"):
            JointTrajectoryCSV.load(self.csv_path, ("same",) * 6)
        with self.assertRaisesRegex(ValueError, "Exactly 6"):
            JointTrajectoryCSV.load(self.csv_path, JOINT_NAMES[:5])


if __name__ == "__main__":
    unittest.main(verbosity=2)
