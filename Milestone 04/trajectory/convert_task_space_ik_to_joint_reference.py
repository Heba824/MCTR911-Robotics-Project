#!/usr/bin/env python3
"""Convert the full-pose IK result into the existing M4 joint-reference CSV schema.

The original task-space and IK CSV files are read-only inputs.  This adapter
creates a separate reference file whose q/qdot/qddot columns use the exact
joint-name contract expected by Milestone 04/control/trajectory_csv.py.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
M2_KINEMATICS = ROOT / "Milestone 02" / "kinematics"
M4_TRAJECTORY = ROOT / "Milestone 04" / "trajectory"
for directory in (M2_KINEMATICS, M4_TRAJECTORY):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from ur5e_dh_fk import JOINT_NAMES  # noqa: E402
from full_pose_ik import JOINT_MIN, JOINT_MAX  # noqa: E402

DEFAULT_IK_CSV = ROOT / "Milestone 04" / "Results" / "Task_Space" / "task_space_joint_trajectory.csv"
DEFAULT_TASK_REFERENCE = ROOT / "Milestone 04" / "Results" / "Task_Space" / "task_space_reference.csv"
DEFAULT_OUTPUT = ROOT / "Milestone 04" / "Results" / "Task_Space" / "task_space_joint_reference.csv"

TASK_POSE_COLUMNS = (
    "x_desired_m", "y_desired_m", "z_desired_m",
    "qx_desired", "qy_desired", "qz_desired", "qw_desired",
)


def _read_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Input CSV does not exist: {path}")
    with path.open("r", newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        rows = list(reader)
        if not rows:
            raise ValueError(f"CSV contains no data rows: {path}")
        return list(reader.fieldnames), rows


def _numeric_column(rows: list[dict[str, str]], name: str, path: Path) -> np.ndarray:
    try:
        values = np.asarray([float(row[name]) for row in rows], dtype=float)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Missing or invalid numeric column '{name}' in {path}") from exc
    if values.ndim != 1 or not np.all(np.isfinite(values)):
        raise ValueError(f"Column '{name}' contains non-finite values in {path}")
    return values


def _require_columns(fieldnames: Iterable[str], required: Iterable[str], path: Path) -> None:
    available = set(fieldnames)
    missing = [name for name in required if name not in available]
    if missing:
        raise ValueError(f"{path} is missing required columns: {', '.join(missing)}")


def convert_task_space_ik_to_joint_reference(
    ik_csv: str | Path = DEFAULT_IK_CSV,
    task_reference_csv: str | Path = DEFAULT_TASK_REFERENCE,
    output_csv: str | Path = DEFAULT_OUTPUT,
    *,
    max_ik_position_error_m: float = 1e-3,
    max_ik_orientation_error_rad: float = 1e-3,
) -> dict:
    """Validate and convert IK samples to the controller's named-joint schema."""
    ik_path = Path(ik_csv).expanduser().resolve()
    task_path = Path(task_reference_csv).expanduser().resolve()
    output_path = Path(output_csv).expanduser().resolve()

    if not np.isfinite(max_ik_position_error_m) or max_ik_position_error_m <= 0.0:
        raise ValueError("max_ik_position_error_m must be positive and finite")
    if not np.isfinite(max_ik_orientation_error_rad) or max_ik_orientation_error_rad <= 0.0:
        raise ValueError("max_ik_orientation_error_rad must be positive and finite")
    if output_path in (ik_path, task_path):
        raise ValueError("Output path must be different from both source CSV files")

    ik_fields, ik_rows = _read_rows(ik_path)
    task_fields, task_rows = _read_rows(task_path)

    ik_required = ["sample_index", "time_s", "phase", "progress",
                   "position_error_m", "orientation_error_rad", "min_singular_value",
                   "joint_step_from_previous_rad"]
    for index in range(1, 7):
        ik_required.extend((f"q{index}_rad", f"qdot{index}_rad_s", f"qddot{index}_rad_s2"))
    ik_required.extend(TASK_POSE_COLUMNS)
    task_required = ["time_s", "phase", *TASK_POSE_COLUMNS]
    _require_columns(ik_fields, ik_required, ik_path)
    _require_columns(task_fields, task_required, task_path)

    if len(ik_rows) != len(task_rows):
        raise ValueError(
            "IK and task-space reference row counts differ: "
            f"{len(ik_rows)} versus {len(task_rows)}"
        )

    ik_times = _numeric_column(ik_rows, "time_s", ik_path)
    task_times = _numeric_column(task_rows, "time_s", task_path)
    if ik_times.size < 2 or not np.isclose(ik_times[0], 0.0, atol=1e-9, rtol=0.0):
        raise ValueError("IK reference must contain at least two rows and start at time zero")
    if np.any(np.diff(ik_times) <= 0.0) or np.any(np.diff(task_times) <= 0.0):
        raise ValueError("Both reference time columns must be strictly increasing")
    if not np.allclose(ik_times, task_times, atol=1e-9, rtol=0.0):
        raise ValueError("IK and task-space reference time samples do not match")

    ik_phases = [row["phase"].strip() for row in ik_rows]
    task_phases = [row["phase"].strip() for row in task_rows]
    if ik_phases != task_phases:
        raise ValueError("IK and task-space reference phase labels do not match")

    # Ensure the pose copied into the IK result is exactly the pose that the
    # task-space trajectory generator produced (within CSV floating precision).
    for column in TASK_POSE_COLUMNS:
        ik_values = _numeric_column(ik_rows, column, ik_path)
        task_values = _numeric_column(task_rows, column, task_path)
        if not np.allclose(ik_values, task_values, atol=1e-9, rtol=0.0):
            raise ValueError(f"IK and task-space pose data disagree in column '{column}'")

    progress_values = _numeric_column(ik_rows, "progress", ik_path)
    sample_indices = _numeric_column(ik_rows, "sample_index", ik_path)
    if not np.allclose(sample_indices, np.arange(len(ik_rows)), atol=0.0, rtol=0.0):
        raise ValueError("IK sample_index values must be consecutive and start at zero")
    if np.any(progress_values < -1e-9) or np.any(progress_values > 1.0 + 1e-9):
        raise ValueError("IK progress values must lie within [0, 1]")

    position_residual = _numeric_column(ik_rows, "position_error_m", ik_path)
    orientation_residual = _numeric_column(ik_rows, "orientation_error_rad", ik_path)
    if np.any(position_residual < 0.0) or np.any(orientation_residual < 0.0):
        raise ValueError("IK pose residual columns must be non-negative")
    max_position_residual = float(np.max(position_residual))
    max_orientation_residual = float(np.max(orientation_residual))
    if max_position_residual > max_ik_position_error_m:
        raise ValueError(
            f"IK position residual {max_position_residual:.6g} m exceeds "
            f"the configured limit {max_ik_position_error_m:.6g} m"
        )
    if max_orientation_residual > max_ik_orientation_error_rad:
        raise ValueError(
            f"IK orientation residual {max_orientation_residual:.6g} rad exceeds "
            f"the configured limit {max_ik_orientation_error_rad:.6g} rad"
        )

    minimum_singular_values = _numeric_column(ik_rows, "min_singular_value", ik_path)
    joint_steps = _numeric_column(ik_rows, "joint_step_from_previous_rad", ik_path)
    if np.any(minimum_singular_values <= 0.0):
        raise ValueError("IK minimum singular values must be positive")
    if np.any(joint_steps < 0.0):
        raise ValueError("IK joint-step diagnostics must be non-negative")

    positions = np.column_stack([_numeric_column(ik_rows, f"q{i}_rad", ik_path) for i in range(1, 7)])
    velocities = np.column_stack([_numeric_column(ik_rows, f"qdot{i}_rad_s", ik_path) for i in range(1, 7)])
    accelerations = np.column_stack([_numeric_column(ik_rows, f"qddot{i}_rad_s2", ik_path) for i in range(1, 7)])
    if np.any(positions < JOINT_MIN[None, :] - 1e-9) or np.any(positions > JOINT_MAX[None, :] + 1e-9):
        raise ValueError("IK joint positions violate the configured UR5e joint limits")

    fields = ["time_s", "phase", "progress", "sample_index"]
    for joint_name in JOINT_NAMES:
        fields.extend((
            f"q_desired_{joint_name}_rad",
            f"qdot_desired_{joint_name}_rad_s",
            f"qddot_desired_{joint_name}_rad_s2",
        ))
    fields.extend(TASK_POSE_COLUMNS)
    fields.extend(("position_error_m", "orientation_error_rad", "min_singular_value", "joint_step_from_previous_rad"))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row_index, source_row in enumerate(ik_rows):
            row: dict[str, float | str | int] = {
                "time_s": float(ik_times[row_index]),
                "phase": ik_phases[row_index],
                "progress": float(progress_values[row_index]),
                "sample_index": int(sample_indices[row_index]),
            }
            for joint_index, joint_name in enumerate(JOINT_NAMES):
                row[f"q_desired_{joint_name}_rad"] = float(positions[row_index, joint_index])
                row[f"qdot_desired_{joint_name}_rad_s"] = float(velocities[row_index, joint_index])
                row[f"qddot_desired_{joint_name}_rad_s2"] = float(accelerations[row_index, joint_index])
            for column in TASK_POSE_COLUMNS:
                row[column] = float(source_row[column])
            for column in ("position_error_m", "orientation_error_rad", "min_singular_value", "joint_step_from_previous_rad"):
                row[column] = float(source_row[column])
            writer.writerow(row)

    summary = {
        "output_csv": str(output_path),
        "ik_source_csv": str(ik_path),
        "task_space_reference_csv": str(task_path),
        "sample_count": int(len(ik_rows)),
        "start_time_s": float(ik_times[0]),
        "end_time_s": float(ik_times[-1]),
        "joint_order": list(JOINT_NAMES),
        "maximum_ik_position_residual_m": max_position_residual,
        "maximum_ik_orientation_residual_rad": max_orientation_residual,
        "minimum_jacobian_singular_value": float(np.min(minimum_singular_values)),
        "maximum_joint_step_rad": float(np.max(joint_steps)),
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ik-csv", type=Path, default=DEFAULT_IK_CSV)
    parser.add_argument("--task-reference", type=Path, default=DEFAULT_TASK_REFERENCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-ik-position-error-m", type=float, default=1e-3)
    parser.add_argument("--max-ik-orientation-error-rad", type=float, default=1e-3)
    args = parser.parse_args()
    summary = convert_task_space_ik_to_joint_reference(
        args.ik_csv,
        args.task_reference,
        args.output,
        max_ik_position_error_m=args.max_ik_position_error_m,
        max_ik_orientation_error_rad=args.max_ik_orientation_error_rad,
    )
    print("Task-space IK reference converted successfully.")
    print(f"Samples: {summary['sample_count']} | duration: {summary['end_time_s']:.3f} s")
    print(f"Max IK position residual: {summary['maximum_ik_position_residual_m']:.3e} m")
    print(f"Max IK orientation residual: {summary['maximum_ik_orientation_residual_rad']:.3e} rad")
    print(f"Minimum Jacobian singular value: {summary['minimum_jacobian_singular_value']:.3e}")
    print(f"Maximum joint step: {summary['maximum_joint_step_rad']:.3e} rad")
    print("Joint order:", ", ".join(summary["joint_order"]))
    print("Controller reference:", summary["output_csv"])


if __name__ == "__main__":
    main()
