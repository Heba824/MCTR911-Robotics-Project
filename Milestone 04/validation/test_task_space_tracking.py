"""Validation for the M4 task-space reference adapter and Cartesian analyzer."""
from __future__ import annotations

import csv
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for directory in (
    ROOT / "Milestone 02" / "kinematics",
    ROOT / "Milestone 03" / "kinematics",
    ROOT / "Milestone 04" / "trajectory",
    ROOT / "Milestone 04" / "control",
    ROOT / "Milestone 04" / "analysis",
):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from convert_task_space_ik_to_joint_reference import (  # noqa: E402
    convert_task_space_ik_to_joint_reference,
)
from analyze_task_space_tracking import (  # noqa: E402
    analyze_task_space_tracking,
    quaternion_xyzw_to_matrix,
)
from trajectory_csv import JointTrajectoryCSV  # noqa: E402
from ur5e_dh_fk import JOINT_NAMES  # noqa: E402

TASK_DIR = ROOT / "Milestone 04" / "Results" / "Task_Space"
IK_CSV = TASK_DIR / "task_space_joint_trajectory.csv"
TASK_REFERENCE = TASK_DIR / "task_space_reference.csv"


class TestTaskSpaceReferenceAdapter(unittest.TestCase):
    def test_existing_ik_csv_converts_to_controller_schema(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "task_space_joint_reference.csv"
            summary = convert_task_space_ik_to_joint_reference(
                IK_CSV, TASK_REFERENCE, output
            )
            self.assertEqual(summary["sample_count"], 401)
            self.assertLess(summary["maximum_ik_position_residual_m"], 1e-3)
            self.assertLess(summary["maximum_ik_orientation_residual_rad"], 1e-3)
            reference = JointTrajectoryCSV.load(output, JOINT_NAMES)
            self.assertEqual(reference.sample_count, 401)
            self.assertAlmostEqual(reference.start_time_s, 0.0)
            self.assertAlmostEqual(reference.end_time_s, 8.0)
            np.testing.assert_allclose(
                reference.start_position_rad,
                [-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0.0],
                atol=2e-6,
            )
            np.testing.assert_allclose(
                reference.goal_position_rad,
                [-1.8708, -1.4108, 1.3908, -1.4508, -1.6708, -0.2],
                atol=2e-5,
            )

    def test_adapter_preserves_every_ik_knot(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "reference.csv"
            convert_task_space_ik_to_joint_reference(IK_CSV, TASK_REFERENCE, output)
            reference = JointTrajectoryCSV.load(output, JOINT_NAMES)
            with IK_CSV.open("r", newline="", encoding="utf-8") as stream:
                source_rows = list(csv.DictReader(stream))
            for index, row in enumerate(source_rows):
                point = reference.evaluate(float(row["time_s"]))
                expected_q = np.array([float(row[f"q{i}_rad"]) for i in range(1, 7)])
                expected_qdot = np.array([float(row[f"qdot{i}_rad_s"]) for i in range(1, 7)])
                expected_qddot = np.array([float(row[f"qddot{i}_rad_s2"]) for i in range(1, 7)])
                np.testing.assert_allclose(point.position_rad, expected_q, atol=2e-10, rtol=0.0)
                np.testing.assert_allclose(point.velocity_rad_s, expected_qdot, atol=2e-8, rtol=0.0)
                np.testing.assert_allclose(point.acceleration_rad_s2, expected_qddot, atol=2e-6, rtol=0.0)

    def test_mismatched_reference_times_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            task_copy = Path(temporary) / "task_reference.csv"
            shutil.copy2(TASK_REFERENCE, task_copy)
            with task_copy.open("r", newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
                fields = list(rows[0].keys())
            rows[1]["time_s"] = str(float(rows[1]["time_s"]) + 0.001)
            with task_copy.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
            with self.assertRaisesRegex(ValueError, "time samples do not match"):
                convert_task_space_ik_to_joint_reference(IK_CSV, task_copy, Path(temporary) / "out.csv")

    def test_large_ik_residual_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            ik_copy = Path(temporary) / "ik.csv"
            shutil.copy2(IK_CSV, ik_copy)
            with ik_copy.open("r", newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
                fields = list(rows[0].keys())
            rows[10]["position_error_m"] = "0.02"
            with ik_copy.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
            with self.assertRaisesRegex(ValueError, "exceeds the configured limit"):
                convert_task_space_ik_to_joint_reference(ik_copy, TASK_REFERENCE, Path(temporary) / "out.csv")


class TestTaskSpaceTrackingAnalysis(unittest.TestCase):
    def _write_ideal_tracking_csv(self, path: Path, *, inject_nan: bool = False) -> None:
        with IK_CSV.open("r", newline="", encoding="utf-8") as stream:
            ik_rows = list(csv.DictReader(stream))
        fields = ["wall_time_s", "trajectory_time_s", "phase"]
        for name in JOINT_NAMES:
            fields.extend((f"q_actual_{name}_rad", f"qdot_actual_{name}_rad_s"))
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for index, source in enumerate(ik_rows):
                row = {
                    "wall_time_s": float(source["time_s"]) + 0.001,
                    "trajectory_time_s": source["time_s"],
                    "phase": source["phase"],
                }
                for joint_index, name in enumerate(JOINT_NAMES, start=1):
                    q = source[f"q{joint_index}_rad"]
                    if inject_nan and index == 5 and joint_index == 2:
                        q = "nan"
                    row[f"q_actual_{name}_rad"] = q
                    row[f"qdot_actual_{name}_rad_s"] = source[f"qdot{joint_index}_rad_s"]
                writer.writerow(row)

    def test_ik_solution_reconstructs_desired_cartesian_path(self):
        with tempfile.TemporaryDirectory() as temporary:
            temporary_path = Path(temporary)
            tracking = temporary_path / "ideal_tracking.csv"
            output_csv = temporary_path / "cartesian_tracking.csv"
            output_json = temporary_path / "metrics.json"
            self._write_ideal_tracking_csv(tracking)
            csv_path, json_path, metrics = analyze_task_space_tracking(
                tracking, TASK_REFERENCE, output_csv, output_json
            )
            self.assertEqual(csv_path, output_csv.resolve())
            self.assertEqual(json_path, output_json.resolve())
            self.assertEqual(metrics["sample_count"], 401)
            self.assertLess(metrics["overall"]["position"]["max_norm_m"], 2e-5)
            self.assertLess(metrics["overall"]["orientation"]["max_norm_rad"], 2e-5)
            self.assertTrue(output_csv.is_file())
            self.assertTrue(output_json.is_file())
            saved = json.loads(output_json.read_text(encoding="utf-8"))
            self.assertEqual(saved["end_effector_frame"], "attachment_site")
            self.assertEqual(saved["coordinate_frame"], "world")

    def test_non_finite_actual_joint_state_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            tracking = Path(temporary) / "tracking.csv"
            self._write_ideal_tracking_csv(tracking, inject_nan=True)
            with self.assertRaisesRegex(ValueError, "Non-finite"):
                analyze_task_space_tracking(
                    tracking, TASK_REFERENCE,
                    Path(temporary) / "out.csv", Path(temporary) / "out.json"
                )

    def test_quaternion_conversion_is_invariant_to_sign(self):
        quaternion = np.array([0.0, 0.0, 0.0, 1.0])
        rotation_positive = quaternion_xyzw_to_matrix(quaternion)
        rotation_negative = quaternion_xyzw_to_matrix(-quaternion)
        np.testing.assert_allclose(rotation_positive, np.eye(3), atol=1e-12)
        np.testing.assert_allclose(rotation_negative, rotation_positive, atol=1e-12)

    def test_reference_samples_are_checked_against_quintic_task_space_model(self):
        with tempfile.TemporaryDirectory() as temporary:
            tracking = Path(temporary) / "tracking.csv"
            task_copy = Path(temporary) / "task_reference.csv"
            self._write_ideal_tracking_csv(tracking)
            shutil.copy2(TASK_REFERENCE, task_copy)
            with task_copy.open("r", newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
                fields = list(rows[0].keys())
            rows[25]["x_desired_m"] = str(float(rows[25]["x_desired_m"]) + 0.01)
            with task_copy.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
            with self.assertRaisesRegex(ValueError, "does not match the project's quintic"):
                analyze_task_space_tracking(
                    tracking, task_copy,
                    Path(temporary) / "out.csv", Path(temporary) / "out.json"
                )

    def test_analyzer_rejects_tracking_time_outside_reference(self):
        with tempfile.TemporaryDirectory() as temporary:
            tracking = Path(temporary) / "tracking.csv"
            self._write_ideal_tracking_csv(tracking)
            with tracking.open("r", newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
                fields = list(rows[0].keys())
            rows[-1]["trajectory_time_s"] = "8.5"
            with tracking.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
            with self.assertRaisesRegex(ValueError, "outside the task-space reference interval"):
                analyze_task_space_tracking(
                    tracking, TASK_REFERENCE,
                    Path(temporary) / "out.csv", Path(temporary) / "out.json"
                )


if __name__ == "__main__":
    unittest.main(verbosity=2)
