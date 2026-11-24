"""Plot the Cartesian pose, twist and acceleration reference CSV."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def read_csv(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"Task-space CSV not found: {path}")
    with path.open("r", newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"Task-space CSV is empty: {path}")
    columns = rows[0].keys()
    return {
        key: np.asarray([float(row[key]) for row in rows], dtype=float)
        for key in columns if key not in {"phase"}
    }


def save_time_plot(data, output_dir, keys, labels, ylabel, filename, title):
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for key, label in zip(keys, labels):
        ax.plot(data["time_s"], data[key], label=label, linewidth=1.8)
    ax.set_title(title)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=3, loc="best")
    fig.tight_layout()
    path = output_dir / filename
    fig.savefig(path, dpi=180)
    plt.close(fig)
    print("Saved:", path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv", type=Path,
        default=ROOT / "Milestone 04" / "Results" / "Task_Space" / "task_space_reference.csv",
    )
    parser.add_argument(
        "--outdir", type=Path,
        default=ROOT / "Milestone 04" / "Figures" / "Task_Space",
    )
    args = parser.parse_args()

    data = read_csv(args.csv.expanduser().resolve())
    outdir = args.outdir.expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    save_time_plot(
        data, outdir,
        ["x_desired_m", "y_desired_m", "z_desired_m"],
        ["x", "y", "z"], "Position [m]",
        "task_space_position_reference.png",
        "UR5e task-space position reference",
    )
    save_time_plot(
        data, outdir,
        ["vx_desired_m_s", "vy_desired_m_s", "vz_desired_m_s"],
        ["vx", "vy", "vz"], "Linear velocity [m/s]",
        "task_space_linear_velocity_reference.png",
        "UR5e task-space linear velocity reference",
    )
    save_time_plot(
        data, outdir,
        ["wx_desired_rad_s", "wy_desired_rad_s", "wz_desired_rad_s"],
        ["wx", "wy", "wz"], "Angular velocity [rad/s]",
        "task_space_angular_velocity_reference.png",
        "UR5e task-space angular velocity reference (world frame)",
    )
    save_time_plot(
        data, outdir,
        ["ax_desired_m_s2", "ay_desired_m_s2", "az_desired_m_s2"],
        ["ax", "ay", "az"], "Linear acceleration [m/s²]",
        "task_space_linear_acceleration_reference.png",
        "UR5e task-space linear acceleration reference",
    )
    save_time_plot(
        data, outdir,
        ["alphax_desired_rad_s2", "alphay_desired_rad_s2", "alphaz_desired_rad_s2"],
        ["alpha x", "alpha y", "alpha z"], "Angular acceleration [rad/s²]",
        "task_space_angular_acceleration_reference.png",
        "UR5e task-space angular acceleration reference (world frame)",
    )

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(data["x_desired_m"], data["y_desired_m"], data["z_desired_m"],
            linewidth=2.2, label="Desired Cartesian path")
    ax.scatter(data["x_desired_m"][0], data["y_desired_m"][0], data["z_desired_m"][0],
               marker="o", s=55, label="Start")
    ax.scatter(data["x_desired_m"][-1], data["y_desired_m"][-1], data["z_desired_m"][-1],
               marker="^", s=65, label="Goal")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")
    ax.set_title("UR5e desired end-effector Cartesian path")
    ax.legend()
    fig.tight_layout()
    path = outdir / "task_space_cartesian_path.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    print("Saved:", path)


if __name__ == "__main__":
    main()
