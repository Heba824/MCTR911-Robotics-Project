#!/usr/bin/env python3
"""Reconstruct and evaluate the UR5e Cartesian end-effector tracking path.

Actual end-effector poses are reconstructed from measured ROS 2 joint feedback
using the project's validated DH forward kinematics. The desired Cartesian
reference is reconstructed from the existing quintic task-space generator and
validated against every stored CSV sample before being evaluated at controller
time. Orientation error follows the project's world-frame convention
Log(R_desired R_actual^T).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
M2_KINEMATICS = ROOT / "Milestone 02" / "kinematics"
M3_KINEMATICS = ROOT / "Milestone 03" / "kinematics"
M4_TRAJECTORY = ROOT / "Milestone 04" / "trajectory"
for directory in (M2_KINEMATICS, M3_KINEMATICS, M4_TRAJECTORY):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from ur5e_dh_fk import JOINT_NAMES, forward_kinematics  # noqa: E402
from ur5e_jacobian import geometric_jacobian  # noqa: E402
from task_space_trajectory import (  # noqa: E402
    QuinticTaskSpaceTrajectory,
    so3_log,
    rotation_matrix_to_quaternion_xyzw,
)

DEFAULT_TASK_REFERENCE = ROOT / "Milestone 04" / "Results" / "Task_Space" / "task_space_reference.csv"
DEFAULT_OUTPUT_DIRECTORY = ROOT / "Milestone 04" / "Results" / "Tracking" / "Task_Space"

POSITION_COLUMNS = ("x_desired_m", "y_desired_m", "z_desired_m")
QUATERNION_COLUMNS = ("qx_desired", "qy_desired", "qz_desired", "qw_desired")
LINEAR_VELOCITY_COLUMNS = ("vx_desired_m_s", "vy_desired_m_s", "vz_desired_m_s")
ANGULAR_VELOCITY_COLUMNS = ("wx_desired_rad_s", "wy_desired_rad_s", "wz_desired_rad_s")
LINEAR_ACCELERATION_COLUMNS = ("ax_desired_m_s2", "ay_desired_m_s2", "az_desired_m_s2")
ANGULAR_ACCELERATION_COLUMNS = ("alphax_desired_rad_s2", "alphay_desired_rad_s2", "alphaz_desired_rad_s2")


def _read_csv(path: str | Path) -> tuple[list[str], list[dict[str, str]]]:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"CSV file does not exist: {source}")
    with source.open("r", newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {source}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"CSV contains no data rows: {source}")
    return list(reader.fieldnames), rows


def _numeric_matrix(rows: list[dict[str, str]], columns: Iterable[str], source: str) -> np.ndarray:
    names = tuple(columns)
    try:
        values = np.asarray([[float(row[name]) for name in names] for row in rows], dtype=float)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Missing or invalid numeric data in {source}: {exc}") from exc
    if values.ndim != 2 or values.shape[1] != len(names) or not np.all(np.isfinite(values)):
        raise ValueError(f"Non-finite or malformed numeric data in {source}")
    return values


def quaternion_xyzw_to_matrix(quaternion_xyzw: Iterable[float]) -> np.ndarray:
    """Convert a normalized or nonzero XYZW quaternion to a rotation matrix."""
    quaternion = np.asarray(quaternion_xyzw, dtype=float)
    if quaternion.shape != (4,) or not np.all(np.isfinite(quaternion)):
        raise ValueError("Quaternion must contain four finite XYZW values")
    norm = float(np.linalg.norm(quaternion))
    if norm < 1e-12:
        raise ValueError("Quaternion norm is too small")
    x, y, z, w = quaternion / norm
    return np.array([
        [1.0 - 2.0 * (y*y + z*z), 2.0 * (x*y - z*w), 2.0 * (x*z + y*w)],
        [2.0 * (x*y + z*w), 1.0 - 2.0 * (x*x + z*z), 2.0 * (y*z - x*w)],
        [2.0 * (x*z - y*w), 2.0 * (y*z + x*w), 1.0 - 2.0 * (x*x + y*y)],
    ], dtype=float)


def _component_metrics(error: np.ndarray, labels: tuple[str, ...], unit: str) -> dict:
    result = {}
    for axis, label in enumerate(labels):
        values = error[:, axis]
        result[label] = {
            f"rmse_{unit}": float(np.sqrt(np.mean(values**2))),
            f"mae_{unit}": float(np.mean(np.abs(values))),
            f"max_abs_error_{unit}": float(np.max(np.abs(values))),
            f"final_error_{unit}": float(values[-1]),
        }
    return result


def _vector_metrics(error: np.ndarray, *, unit: str, labels: tuple[str, ...]) -> dict:
    norms = np.linalg.norm(error, axis=1)
    return {
        f"rmse_norm_{unit}": float(np.sqrt(np.mean(norms**2))),
        f"mae_norm_{unit}": float(np.mean(norms)),
        f"max_norm_{unit}": float(np.max(norms)),
        f"final_norm_{unit}": float(norms[-1]),
        "components": _component_metrics(error, labels, unit),
    }


def _phase_metrics(phases: np.ndarray, position_error: np.ndarray, orientation_error: np.ndarray,
                   linear_velocity_error: np.ndarray, angular_velocity_error: np.ndarray) -> dict:
    output = {}
    for phase in dict.fromkeys(str(value) for value in phases):
        mask = phases == phase
        if not np.any(mask):
            continue
        output[phase] = {
            "sample_count": int(np.count_nonzero(mask)),
            "position": _vector_metrics(position_error[mask], unit="m", labels=("x", "y", "z")),
            "orientation": _vector_metrics(orientation_error[mask], unit="rad", labels=("x", "y", "z")),
            "linear_velocity": _vector_metrics(linear_velocity_error[mask], unit="m_s", labels=("x", "y", "z")),
            "angular_velocity": _vector_metrics(angular_velocity_error[mask], unit="rad_s", labels=("x", "y", "z")),
        }
    return output


def analyze_task_space_tracking(
    tracking_csv: str | Path,
    task_reference_csv: str | Path = DEFAULT_TASK_REFERENCE,
    output_csv: str | Path | None = None,
    output_json: str | Path | None = None,
) -> tuple[Path, Path, dict]:
    """Reconstruct actual FK poses and write Cartesian tracking data/metrics."""
    tracking_path = Path(tracking_csv).expanduser().resolve()
    reference_path = Path(task_reference_csv).expanduser().resolve()
    tracking_fields, tracking_rows = _read_csv(tracking_path)
    reference_fields, reference_rows = _read_csv(reference_path)

    required_tracking = ["wall_time_s", "trajectory_time_s", "phase"]
    for joint_name in JOINT_NAMES:
        required_tracking.extend((f"q_actual_{joint_name}_rad", f"qdot_actual_{joint_name}_rad_s"))
    required_reference = ["time_s", "phase", "progress", *POSITION_COLUMNS, *QUATERNION_COLUMNS,
                          *LINEAR_VELOCITY_COLUMNS, *ANGULAR_VELOCITY_COLUMNS,
                          *LINEAR_ACCELERATION_COLUMNS, *ANGULAR_ACCELERATION_COLUMNS]
    for fields, required, source in ((tracking_fields, required_tracking, tracking_path),
                                     (reference_fields, required_reference, reference_path)):
        missing = [column for column in required if column not in fields]
        if missing:
            raise ValueError(f"{source} is missing required columns: {', '.join(missing)}")

    wall_time = _numeric_matrix(tracking_rows, ("wall_time_s",), str(tracking_path))[:, 0]
    query_time = _numeric_matrix(tracking_rows, ("trajectory_time_s",), str(tracking_path))[:, 0]
    reference_time = _numeric_matrix(reference_rows, ("time_s",), str(reference_path))[:, 0]
    if wall_time.size < 2 or np.any(np.diff(wall_time) <= 0.0):
        raise ValueError("Tracking wall_time_s must contain at least two strictly increasing samples")
    if np.any(np.diff(query_time) < -1e-10):
        raise ValueError("Tracking trajectory_time_s must be non-decreasing")
    if reference_time.size < 2 or np.any(np.diff(reference_time) <= 0.0):
        raise ValueError("Task-space reference time_s must be strictly increasing")
    if not np.isclose(reference_time[0], 0.0, atol=1e-9, rtol=0.0):
        raise ValueError("Task-space reference must start at time_s = 0")
    time_tolerance = 1e-6
    if query_time[0] < reference_time[0] - time_tolerance or query_time[-1] > reference_time[-1] + time_tolerance:
        raise ValueError(
            "Tracking trajectory time lies outside the task-space reference interval: "
            f"[{query_time[0]:.9f}, {query_time[-1]:.9f}] versus "
            f"[{reference_time[0]:.9f}, {reference_time[-1]:.9f}]"
        )
    query_time = np.clip(query_time, reference_time[0], reference_time[-1])

    reference_position = _numeric_matrix(reference_rows, POSITION_COLUMNS, str(reference_path))
    reference_quaternion = _numeric_matrix(reference_rows, QUATERNION_COLUMNS, str(reference_path))
    reference_linear_velocity = _numeric_matrix(reference_rows, LINEAR_VELOCITY_COLUMNS, str(reference_path))
    reference_angular_velocity = _numeric_matrix(reference_rows, ANGULAR_VELOCITY_COLUMNS, str(reference_path))
    reference_linear_acceleration = _numeric_matrix(reference_rows, LINEAR_ACCELERATION_COLUMNS, str(reference_path))
    reference_angular_acceleration = _numeric_matrix(reference_rows, ANGULAR_ACCELERATION_COLUMNS, str(reference_path))
    reference_progress = _numeric_matrix(reference_rows, ("progress",), str(reference_path))[:, 0]
    quaternion_norms = np.linalg.norm(reference_quaternion, axis=1)
    if np.any(quaternion_norms < 1e-12) or np.any(np.abs(quaternion_norms - 1.0) > 1e-3):
        raise ValueError("Task-space reference contains invalid or non-unit quaternions")
    if np.any(reference_progress < -1e-9) or np.any(reference_progress > 1.0 + 1e-9):
        raise ValueError("Task-space reference progress must lie within [0, 1]")
    if np.any(np.diff(reference_progress) < -1e-9):
        raise ValueError("Task-space reference progress must be non-decreasing")
    reference_quaternion = reference_quaternion / quaternion_norms[:, None]

    # Reconstruct the exact task-space generator used for this project instead
    # of introducing a piecewise-linear approximation between the 20 ms CSV
    # samples. The stored CSV is checked against the reconstructed model first.
    completed_indices = np.flatnonzero(reference_progress >= 1.0 - 1e-10)
    if completed_indices.size == 0:
        raise ValueError("Task-space reference never reaches progress=1.0")
    movement_duration_s = float(reference_time[int(completed_indices[0])])
    if movement_duration_s <= 0.0:
        raise ValueError("Task-space movement duration must be positive")

    start_pose = np.eye(4, dtype=float)
    start_pose[:3, 3] = reference_position[0]
    start_pose[:3, :3] = quaternion_xyzw_to_matrix(reference_quaternion[0])
    goal_pose = np.eye(4, dtype=float)
    goal_pose[:3, 3] = reference_position[-1]
    goal_pose[:3, :3] = quaternion_xyzw_to_matrix(reference_quaternion[-1])
    desired_trajectory = QuinticTaskSpaceTrajectory(start_pose, goal_pose, movement_duration_s)

    reference_position_residual = []
    reference_orientation_residual = []
    reference_twist_residual = []
    reference_acceleration_residual = []
    reference_progress_residual = []
    for index, sample_time in enumerate(reference_time):
        point = desired_trajectory.evaluate(float(sample_time))
        reference_position_residual.append(float(np.linalg.norm(point.position_world_m - reference_position[index])))
        reference_orientation_residual.append(float(np.linalg.norm(
            so3_log(point.pose_world[:3, :3] @ quaternion_xyzw_to_matrix(reference_quaternion[index]).T)
        )))
        reference_twist = np.concatenate((reference_linear_velocity[index], reference_angular_velocity[index]))
        reference_acceleration = np.concatenate((reference_linear_acceleration[index], reference_angular_acceleration[index]))
        reference_twist_residual.append(float(np.linalg.norm(point.twist_world - reference_twist)))
        reference_acceleration_residual.append(float(np.linalg.norm(point.acceleration_world - reference_acceleration)))
        reference_progress_residual.append(abs(float(point.progress) - float(reference_progress[index])))

    reference_model_validation = {
        "maximum_position_residual_m": float(np.max(reference_position_residual)),
        "maximum_orientation_residual_rad": float(np.max(reference_orientation_residual)),
        "maximum_twist_residual": float(np.max(reference_twist_residual)),
        "maximum_acceleration_residual": float(np.max(reference_acceleration_residual)),
        "maximum_progress_residual": float(np.max(reference_progress_residual)),
        "movement_duration_s": movement_duration_s,
    }
    validation_tolerance = 1e-7
    if any(value > validation_tolerance for key, value in reference_model_validation.items()
           if key != "movement_duration_s"):
        raise ValueError(
            "Stored task-space CSV does not match the project's quintic task-space trajectory model: "
            f"{reference_model_validation}"
        )

    desired_points = [desired_trajectory.evaluate(float(sample_time)) for sample_time in query_time]
    desired_position = np.asarray([point.position_world_m for point in desired_points], dtype=float)
    desired_quaternion = np.asarray([point.quaternion_xyzw for point in desired_points], dtype=float)
    desired_linear_velocity = np.asarray([point.linear_velocity_world_m_s for point in desired_points], dtype=float)
    desired_angular_velocity = np.asarray([point.angular_velocity_world_rad_s for point in desired_points], dtype=float)

    actual_joint_position = _numeric_matrix(
        tracking_rows, (f"q_actual_{name}_rad" for name in JOINT_NAMES), str(tracking_path)
    )
    actual_joint_velocity = _numeric_matrix(
        tracking_rows, (f"qdot_actual_{name}_rad_s" for name in JOINT_NAMES), str(tracking_path)
    )

    sample_count = len(tracking_rows)
    actual_position = np.empty((sample_count, 3), dtype=float)
    actual_quaternion = np.empty((sample_count, 4), dtype=float)
    actual_linear_velocity = np.empty((sample_count, 3), dtype=float)
    actual_angular_velocity = np.empty((sample_count, 3), dtype=float)
    position_error = np.empty((sample_count, 3), dtype=float)
    orientation_error = np.empty((sample_count, 3), dtype=float)
    for index, (q, qdot) in enumerate(zip(actual_joint_position, actual_joint_velocity)):
        pose = forward_kinematics(q)
        actual_position[index] = pose[:3, 3]
        actual_quaternion[index] = rotation_matrix_to_quaternion_xyzw(pose[:3, :3])
        actual_twist = geometric_jacobian(q) @ qdot
        actual_linear_velocity[index] = actual_twist[:3]
        actual_angular_velocity[index] = actual_twist[3:]
        desired_rotation = quaternion_xyzw_to_matrix(desired_quaternion[index])
        orientation_error[index] = so3_log(desired_rotation @ pose[:3, :3].T)
        position_error[index] = desired_position[index] - actual_position[index]

    linear_velocity_error = desired_linear_velocity - actual_linear_velocity
    angular_velocity_error = desired_angular_velocity - actual_angular_velocity
    position_error_norm = np.linalg.norm(position_error, axis=1)
    orientation_error_norm = np.linalg.norm(orientation_error, axis=1)
    linear_velocity_error_norm = np.linalg.norm(linear_velocity_error, axis=1)
    angular_velocity_error_norm = np.linalg.norm(angular_velocity_error, axis=1)

    output_fields = ["wall_time_s", "trajectory_time_s", "phase"]
    for axis in "xyz":
        output_fields.extend((f"{axis}_desired_m", f"{axis}_actual_m", f"{axis}_error_m"))
    output_fields.append("position_error_m")
    for component in ("qx", "qy", "qz", "qw"):
        output_fields.extend((f"{component}_desired", f"{component}_actual"))
    output_fields.extend(("orientation_error_x_rad", "orientation_error_y_rad", "orientation_error_z_rad", "orientation_error_rad"))
    for axis in "xyz":
        output_fields.extend((f"v{axis}_desired_m_s", f"v{axis}_actual_m_s", f"v{axis}_error_m_s"))
    output_fields.append("linear_velocity_error_m_s")
    for axis in "xyz":
        output_fields.extend((f"w{axis}_desired_rad_s", f"w{axis}_actual_rad_s", f"w{axis}_error_rad_s"))
    output_fields.append("angular_velocity_error_rad_s")
    for joint_name in JOINT_NAMES:
        output_fields.extend((f"q_actual_{joint_name}_rad", f"qdot_actual_{joint_name}_rad_s"))

    if output_csv is None or output_json is None:
        output_dir = DEFAULT_OUTPUT_DIRECTORY
        output_dir.mkdir(parents=True, exist_ok=True)
        stem = tracking_path.stem
        if stem.startswith("joint_space_tracking_"):
            stem = "task_space_tracking_" + stem[len("joint_space_tracking_"):]
        else:
            stem = "task_space_tracking_" + stem
        if output_csv is None:
            output_csv = output_dir / f"{stem}.csv"
        if output_json is None:
            output_json = output_dir / f"{stem}_metrics.json"
    output_csv_path = Path(output_csv).expanduser().resolve()
    output_json_path = Path(output_json).expanduser().resolve()
    if output_csv_path in (tracking_path, reference_path) or output_json_path in (tracking_path, reference_path):
        raise ValueError("Analysis outputs must not overwrite either source CSV")
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    output_json_path.parent.mkdir(parents=True, exist_ok=True)

    with output_csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=output_fields)
        writer.writeheader()
        for index, source_row in enumerate(tracking_rows):
            row: dict[str, float | str] = {
                "wall_time_s": float(wall_time[index]),
                "trajectory_time_s": float(query_time[index]),
                "phase": source_row["phase"],
                "position_error_m": float(position_error_norm[index]),
                "orientation_error_x_rad": float(orientation_error[index, 0]),
                "orientation_error_y_rad": float(orientation_error[index, 1]),
                "orientation_error_z_rad": float(orientation_error[index, 2]),
                "orientation_error_rad": float(orientation_error_norm[index]),
                "linear_velocity_error_m_s": float(linear_velocity_error_norm[index]),
                "angular_velocity_error_rad_s": float(angular_velocity_error_norm[index]),
            }
            for axis_index, axis in enumerate("xyz"):
                row[f"{axis}_desired_m"] = float(desired_position[index, axis_index])
                row[f"{axis}_actual_m"] = float(actual_position[index, axis_index])
                row[f"{axis}_error_m"] = float(position_error[index, axis_index])
                row[f"v{axis}_desired_m_s"] = float(desired_linear_velocity[index, axis_index])
                row[f"v{axis}_actual_m_s"] = float(actual_linear_velocity[index, axis_index])
                row[f"v{axis}_error_m_s"] = float(linear_velocity_error[index, axis_index])
                row[f"w{axis}_desired_rad_s"] = float(desired_angular_velocity[index, axis_index])
                row[f"w{axis}_actual_rad_s"] = float(actual_angular_velocity[index, axis_index])
                row[f"w{axis}_error_rad_s"] = float(angular_velocity_error[index, axis_index])
            for component_index, component in enumerate(("qx", "qy", "qz", "qw")):
                row[f"{component}_desired"] = float(desired_quaternion[index, component_index])
                row[f"{component}_actual"] = float(actual_quaternion[index, component_index])
            for joint_index, joint_name in enumerate(JOINT_NAMES):
                row[f"q_actual_{joint_name}_rad"] = float(actual_joint_position[index, joint_index])
                row[f"qdot_actual_{joint_name}_rad_s"] = float(actual_joint_velocity[index, joint_index])
            writer.writerow(row)

    phases = np.asarray([row["phase"] for row in tracking_rows], dtype=str)
    movement_mask = phases != "hold"
    hold_mask = phases == "hold"
    metrics = {
        "tracking_csv": str(tracking_path),
        "task_space_reference_csv": str(reference_path),
        "cartesian_tracking_csv": str(output_csv_path),
        "sample_count": int(sample_count),
        "trajectory_time_start_s": float(query_time[0]),
        "trajectory_time_end_s": float(query_time[-1]),
        "wall_time_duration_s": float(wall_time[-1] - wall_time[0]),
        "end_effector_frame": "attachment_site",
        "coordinate_frame": "world",
        "position_error_sign_convention": "desired_position - actual_position",
        "orientation_error_convention": "Log(R_desired R_actual^T), world-frame rotation vector",
        "velocity_error_sign_convention": "desired_twist - actual_twist",
        "overall": {
            "position": _vector_metrics(position_error, unit="m", labels=("x", "y", "z")),
            "orientation": _vector_metrics(orientation_error, unit="rad", labels=("x", "y", "z")),
            "orientation_rmse_deg": float(np.rad2deg(np.sqrt(np.mean(orientation_error_norm**2)))),
            "orientation_mae_deg": float(np.rad2deg(np.mean(orientation_error_norm))),
            "orientation_max_deg": float(np.rad2deg(np.max(orientation_error_norm))),
            "linear_velocity": _vector_metrics(linear_velocity_error, unit="m_s", labels=("x", "y", "z")),
            "angular_velocity": _vector_metrics(angular_velocity_error, unit="rad_s", labels=("x", "y", "z")),
        },
        "movement": {
            "sample_count": int(np.count_nonzero(movement_mask)),
            "position": _vector_metrics(position_error[movement_mask], unit="m", labels=("x", "y", "z")),
            "orientation": _vector_metrics(orientation_error[movement_mask], unit="rad", labels=("x", "y", "z")),
            "linear_velocity": _vector_metrics(linear_velocity_error[movement_mask], unit="m_s", labels=("x", "y", "z")),
            "angular_velocity": _vector_metrics(angular_velocity_error[movement_mask], unit="rad_s", labels=("x", "y", "z")),
        },
        "hold": None if not np.any(hold_mask) else {
            "sample_count": int(np.count_nonzero(hold_mask)),
            "position": _vector_metrics(position_error[hold_mask], unit="m", labels=("x", "y", "z")),
            "orientation": _vector_metrics(orientation_error[hold_mask], unit="rad", labels=("x", "y", "z")),
            "linear_velocity": _vector_metrics(linear_velocity_error[hold_mask], unit="m_s", labels=("x", "y", "z")),
            "angular_velocity": _vector_metrics(angular_velocity_error[hold_mask], unit="rad_s", labels=("x", "y", "z")),
        },
        "initial_condition_error": {
            "position_error_m": float(position_error_norm[0]),
            "orientation_error_rad": float(orientation_error_norm[0]),
        },
        "final_error": {
            "position_error_m": float(position_error_norm[-1]),
            "orientation_error_rad": float(orientation_error_norm[-1]),
        },
        "by_phase": _phase_metrics(phases, position_error, orientation_error,
                                    linear_velocity_error, angular_velocity_error),
        "units": {
            "position": "m", "orientation": "rad", "linear_velocity": "m/s",
            "angular_velocity": "rad/s", "joint_position": "rad", "joint_velocity": "rad/s",
        },
        "reference_model_validation": reference_model_validation,
        "method": {
            "actual_pose": "Milestone 02 standard-DH forward_kinematics(actual_joint_position)",
            "actual_twist": "Milestone 03 geometric_jacobian(actual_joint_position) @ actual_joint_velocity",
            "desired_pose_and_twist": "QuinticTaskSpaceTrajectory reconstructed from the stored task-space CSV endpoints and movement duration, after validating every stored sample",
            "orientation_error": "Log(R_desired R_actual^T), world-frame rotation vector",
        },
    }
    with output_json_path.open("w", encoding="utf-8") as stream:
        json.dump(metrics, stream, indent=2, allow_nan=False)
    return output_csv_path, output_json_path, metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracking-csv", type=Path, required=True,
                        help="Joint-space tracking CSV recorded by the M4 ROS 2 controller")
    parser.add_argument("--task-space-reference", type=Path, default=DEFAULT_TASK_REFERENCE)
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()
    csv_path, json_path, metrics = analyze_task_space_tracking(
        args.tracking_csv, args.task_space_reference, args.output_csv, args.output_json
    )
    overall = metrics["overall"]
    print("Task-space tracking analysis completed.")
    print(f"Samples: {metrics['sample_count']} | wall duration: {metrics['wall_time_duration_s']:.3f} s")
    print(f"Position RMSE: {overall['position']['rmse_norm_m']:.6e} m")
    print(f"Position maximum error: {overall['position']['max_norm_m']:.6e} m")
    print(f"Orientation RMSE: {overall['orientation']['rmse_norm_rad']:.6e} rad "
          f"({overall['orientation_rmse_deg']:.6e} deg)")
    print(f"Linear velocity RMSE: {overall['linear_velocity']['rmse_norm_m_s']:.6e} m/s")
    print(f"Angular velocity RMSE: {overall['angular_velocity']['rmse_norm_rad_s']:.6e} rad/s")
    print(f"Reference model max position residual: {metrics['reference_model_validation']['maximum_position_residual_m']:.3e} m")
    print(f"Cartesian tracking CSV: {csv_path}")
    print(f"Metrics JSON: {json_path}")


if __name__ == "__main__":
    main()
