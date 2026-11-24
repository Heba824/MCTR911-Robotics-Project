#!/usr/bin/env python3
"""Plot Cartesian pose and velocity tracking reconstructed from ROS 2 feedback."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "Milestone 04" / "Results" / "Tracking" / "Task_Space"
DEFAULT_OUTPUT = ROOT / "Milestone 04" / "Figures" / "Task_Space" / "Tracking"


def _load(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"Cartesian tracking CSV does not exist: {path}")
    with path.open("r", newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ValueError("Cartesian tracking CSV has no header")
        rows = list(reader)
    if not rows:
        raise ValueError("Cartesian tracking CSV contains no data rows")
    data: dict[str, np.ndarray] = {}
    for column in reader.fieldnames:
        try:
            data[column] = np.asarray([float(row[column]) for row in rows], dtype=float)
        except (TypeError, ValueError):
            continue
    required = ("trajectory_time_s", "x_desired_m", "x_actual_m", "y_desired_m", "y_actual_m",
                "z_desired_m", "z_actual_m", "position_error_m", "orientation_error_rad",
                "linear_velocity_error_m_s", "angular_velocity_error_rad_s")
    missing = [column for column in required if column not in data]
    if missing:
        raise ValueError("Cartesian tracking CSV is missing columns: " + ", ".join(missing))
    if any(not np.all(np.isfinite(data[column])) for column in required):
        raise ValueError("Cartesian tracking CSV contains non-finite plotting data")
    return data


def plot_task_space_tracking(input_csv: str | Path, output_directory: str | Path) -> list[Path]:
    source = Path(input_csv).expanduser().resolve()
    output_dir = Path(output_directory).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    data = _load(source)
    time_s = data["trajectory_time_s"]
    outputs: list[Path] = []

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    for axis_index, axis_name in enumerate("xyz"):
        axes[axis_index].plot(time_s, data[f"{axis_name}_desired_m"], label="Desired")
        axes[axis_index].plot(time_s, data[f"{axis_name}_actual_m"], linestyle="--", label="Actual")
        axes[axis_index].set_ylabel(f"{axis_name.upper()} [m]")
        axes[axis_index].grid(True, alpha=0.3)
        axes[axis_index].legend()
    axes[-1].set_xlabel("Trajectory time [s]")
    fig.suptitle("Task-space position tracking")
    fig.tight_layout()
    path = output_dir / "task_space_position_tracking.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    outputs.append(path)

    fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
    for axis_index, axis_name in enumerate("xyz"):
        axes[axis_index].plot(time_s, data[f"{axis_name}_error_m"])
        axes[axis_index].set_ylabel(f"e{axis_name.upper()} [m]")
        axes[axis_index].grid(True, alpha=0.3)
    axes[3].plot(time_s, data["position_error_m"], label="3D position error norm")
    axes[3].set_ylabel("Norm [m]")
    axes[3].set_xlabel("Trajectory time [s]")
    axes[3].grid(True, alpha=0.3)
    axes[3].legend()
    fig.suptitle("Task-space position error")
    fig.tight_layout()
    path = output_dir / "task_space_position_error.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    outputs.append(path)

    fig = plt.figure(figsize=(9, 7))
    axis = fig.add_subplot(111, projection="3d")
    axis.plot(data["x_desired_m"], data["y_desired_m"], data["z_desired_m"], label="Desired path")
    axis.plot(data["x_actual_m"], data["y_actual_m"], data["z_actual_m"], linestyle="--", label="Actual path")
    axis.scatter(data["x_desired_m"][0], data["y_desired_m"][0], data["z_desired_m"][0], marker="o", label="Start")
    axis.scatter(data["x_desired_m"][-1], data["y_desired_m"][-1], data["z_desired_m"][-1], marker="^", label="Goal")
    axis.set_xlabel("X [m]")
    axis.set_ylabel("Y [m]")
    axis.set_zlabel("Z [m]")
    axis.set_title("Desired versus actual Cartesian path")
    axis.legend()
    fig.tight_layout()
    path = output_dir / "task_space_cartesian_path_tracking.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    outputs.append(path)

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    axes[0].plot(time_s, data["orientation_error_rad"], label="Orientation error angle")
    axes[0].set_ylabel("Angle [rad]")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    axes[1].plot(time_s, data["linear_velocity_error_m_s"], label="Linear velocity error")
    axes[1].set_ylabel("[m/s]")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    axes[2].plot(time_s, data["angular_velocity_error_rad_s"], label="Angular velocity error")
    axes[2].set_ylabel("[rad/s]")
    axes[2].set_xlabel("Trajectory time [s]")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()
    fig.suptitle("Task-space orientation and velocity tracking errors")
    fig.tight_layout()
    path = output_dir / "task_space_orientation_velocity_errors.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    outputs.append(path)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True, help="Cartesian tracking CSV from analyze_task_space_tracking.py")
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    outputs = plot_task_space_tracking(args.csv, args.outdir)
    for path in outputs:
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()
