"""Generate Milestone 03 smooth-motion figures, including two acceleration estimates."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CSV = ROOT / "Milestone 03" / "Results" / "Smooth_Motion" / "smooth_motion.csv"
DEFAULT_OUT = ROOT / "Milestone 03" / "Figures" / "Smooth_Motion"


def load_rows(path: Path) -> dict[str, np.ndarray]:
    with path.open("r", newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) < 3:
        raise ValueError(f"Expected at least 3 simulation samples in {path}")
    data: dict[str, np.ndarray] = {}
    for key in rows[0].keys():
        values = [row[key] for row in rows]
        try:
            data[key] = np.asarray([float(value) for value in values], dtype=float)
        except (TypeError, ValueError):
            data[key] = np.asarray(values, dtype=object)
    return data


def _has(data: dict[str, np.ndarray], key: str) -> bool:
    return key in data and np.asarray(data[key]).size > 0


def save_speed_plot(data: dict[str, np.ndarray], out: Path) -> None:
    t = data["time_s"]
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    axes[0].plot(t, data["target_joint_speed_norm_rad_s"], "--", label="Reference joint-speed norm")
    axes[0].plot(t, data["actual_joint_speed_norm_rad_s"], label="Actual MuJoCo joint-speed norm")
    move_end = float(data["move_duration_s"][0])
    axes[0].axvline(move_end, linestyle=":", label="Target reached / hold")
    axes[0].set_ylabel("Joint speed (rad/s)")
    axes[0].set_title("Joint-space speed: acceleration and deceleration")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(t, data["target_ee_linear_speed_m_s"], "--", label="Reference end-effector linear speed")
    axes[1].plot(t, data["actual_ee_linear_speed_m_s"], label="Actual end-effector linear speed")
    axes[1].axvline(move_end, linestyle=":", label="Target reached / hold")
    axes[1].set_xlabel("Simulation time (s)")
    axes[1].set_ylabel("Linear speed (m/s)")
    axes[1].set_title("End-effector linear speed")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(out / "smooth_motion_speed_profile.png", dpi=180)
    plt.close(fig)


def save_acceleration_plot(data: dict[str, np.ndarray], out: Path) -> None:
    """Show reference, instantaneous qacc, and sampled-velocity acceleration."""
    t = data["time_s"]
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    axes[0].plot(t, data["target_joint_acceleration_norm_rad_s2"], "--", label="Reference joint-acceleration norm")
    axes[0].plot(t, data["actual_joint_acceleration_norm_rad_s2"], label="Instantaneous MuJoCo qacc norm")
    if _has(data, "fd_actual_joint_acceleration_norm_rad_s2"):
        axes[0].plot(t, data["fd_actual_joint_acceleration_norm_rad_s2"], label="Finite-difference qvel acceleration norm")
    move_end = float(data["move_duration_s"][0])
    axes[0].axvline(move_end, linestyle=":", label="Target reached / hold")
    axes[0].set_ylabel("Joint acceleration (rad/s²)")
    axes[0].set_title("Joint-space acceleration: two measured definitions")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(t, data["target_ee_linear_acceleration_m_s2"], "--", label="Reference EE linear acceleration")
    axes[1].plot(t, data["actual_ee_linear_acceleration_m_s2"], label="From instantaneous qacc")
    if _has(data, "fd_actual_ee_linear_acceleration_m_s2"):
        axes[1].plot(t, data["fd_actual_ee_linear_acceleration_m_s2"], label="From finite-difference qvel")
    axes[1].axvline(move_end, linestyle=":", label="Target reached / hold")
    axes[1].set_xlabel("Simulation time (s)")
    axes[1].set_ylabel("Linear acceleration (m/s²)")
    axes[1].set_title("End-effector linear acceleration magnitude")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(out / "smooth_motion_acceleration_profile.png", dpi=180)
    plt.close(fig)


def save_joint_acceleration_comparison(data: dict[str, np.ndarray], out: Path) -> None:
    """Plot per-joint reference, post-mj_forward qacc, and FD qddot."""
    if not _has(data, "fd_actual_qddot1_rad_s2"):
        return
    t = data["time_s"]
    fig, axes = plt.subplots(3, 2, figsize=(13, 10), sharex=True)
    for i, ax in enumerate(axes.flat, start=1):
        ax.plot(t, data[f"target_qddot{i}_rad_s2"], "--", label="Reference")
        ax.plot(t, data[f"actual_qddot{i}_rad_s2"], alpha=0.85, label="Instantaneous qacc")
        ax.plot(t, data[f"fd_actual_qddot{i}_rad_s2"], linewidth=1.1, label="FD of actual qvel")
        ax.set_title(f"Joint {i}")
        ax.set_ylabel("Acceleration (rad/s²)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    for ax in axes[-1, :]:
        ax.set_xlabel("Simulation time (s)")
    fig.suptitle("Joint acceleration: reference vs two measured definitions", y=1.01)
    fig.tight_layout()
    fig.savefig(out / "smooth_motion_joint_acceleration_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_ee_acceleration_comparison(data: dict[str, np.ndarray], out: Path) -> None:
    """Plot Cartesian acceleration computed from qacc and FD-derived qddot."""
    if not _has(data, "fd_actual_ee_linear_acceleration_m_s2"):
        return
    t = data["time_s"]
    move_end = float(data["move_duration_s"][0])
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    axes[0].plot(t, data["target_ee_linear_acceleration_m_s2"], "--", label="Reference")
    axes[0].plot(t, data["actual_ee_linear_acceleration_m_s2"], label="From instantaneous qacc")
    axes[0].plot(t, data["fd_actual_ee_linear_acceleration_m_s2"], label="From FD of actual qvel")
    axes[0].axvline(move_end, linestyle=":", label="Target reached / hold")
    axes[0].set_ylabel("Linear acceleration (m/s²)")
    axes[0].set_title("End-effector linear acceleration magnitude")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(t, data["target_ee_angular_acceleration_rad_s2"], "--", label="Reference")
    axes[1].plot(t, data["actual_ee_angular_acceleration_rad_s2"], label="From instantaneous qacc")
    axes[1].plot(t, data["fd_actual_ee_angular_acceleration_rad_s2"], label="From FD of actual qvel")
    axes[1].axvline(move_end, linestyle=":", label="Target reached / hold")
    axes[1].set_xlabel("Simulation time (s)")
    axes[1].set_ylabel("Angular acceleration (rad/s²)")
    axes[1].set_title("End-effector angular acceleration magnitude")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(out / "smooth_motion_ee_acceleration_comparison.png", dpi=180)
    plt.close(fig)


def save_acceleration_diagnostic_plot(data: dict[str, np.ndarray], out: Path) -> None:
    if not _has(data, "qacc_vs_fd_joint_acceleration_error_norm_rad_s2"):
        return
    t = data["time_s"]
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(t, data["joint_acceleration_tracking_error_norm_rad_s2"], label="Instantaneous qacc vs reference")
    ax.plot(t, data["fd_joint_acceleration_tracking_error_norm_rad_s2"], label="FD qvel vs reference")
    ax.plot(t, data["qacc_vs_fd_joint_acceleration_error_norm_rad_s2"], label="Instantaneous qacc vs FD qvel")
    ax.set_xlabel("Simulation time (s)")
    ax.set_ylabel("Joint acceleration error norm (rad/s²)")
    ax.set_title("Acceleration measurement and tracking diagnostics")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "smooth_motion_acceleration_measurement_diagnostics.png", dpi=180)
    plt.close(fig)


def save_joint_tracking_plot(data: dict[str, np.ndarray], out: Path) -> None:
    t = data["time_s"]
    fig, axes = plt.subplots(3, 2, figsize=(13, 10), sharex=True)
    for i, ax in enumerate(axes.flat, start=1):
        ax.plot(t, data[f"target_q{i}_rad"], "--", label="Target")
        ax.plot(t, data[f"actual_q{i}_rad"], label="Actual MuJoCo")
        ax.set_title(f"Joint {i}")
        ax.set_ylabel("Angle (rad)")
        ax.grid(True, alpha=0.3)
        ax.legend()
    for ax in axes[-1, :]:
        ax.set_xlabel("Simulation time (s)")
    fig.suptitle("Joint-position tracking during smooth motion", y=1.01)
    fig.tight_layout()
    fig.savefig(out / "smooth_motion_joint_tracking.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_error_plot(data: dict[str, np.ndarray], out: Path) -> None:
    t = data["time_s"]
    metrics = [
        ("position_error_m", 1e-5, "Position / 1e-5 m"),
        ("linear_velocity_error_m_s", 1e-4, "Linear velocity / 1e-4 m/s"),
        ("angular_velocity_error_rad_s", 1e-4, "Angular velocity / 1e-4 rad/s"),
        ("linear_acceleration_error_m_s2", 1e-3, "Linear acceleration / 1e-3 m/s²"),
        ("angular_acceleration_error_rad_s2", 1e-3, "Angular acceleration / 1e-3 rad/s²"),
    ]
    fig, ax = plt.subplots(figsize=(11, 6))
    for key, tolerance, label in metrics:
        normalized = np.maximum(data[key] / tolerance, 1e-15)
        ax.semilogy(t, normalized, label=label)
    ax.axhline(1.0, linestyle="--", label="Configured tolerance")
    ax.set_xlabel("Simulation time (s)")
    ax.set_ylabel("Error / tolerance (dimensionless)")
    ax.set_title("Python–MuJoCo kinematic agreement (normalized errors)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "smooth_motion_kinematic_errors.png", dpi=180)
    plt.close(fig)


def save_path_plot(data: dict[str, np.ndarray], out: Path) -> None:
    x = data["mujoco_position_x_m"]
    y = data["mujoco_position_y_m"]
    z = data["mujoco_position_z_m"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(x, y, label="Attachment-site path")
    axes[0].scatter([x[0]], [y[0]], marker="o", label="Start")
    axes[0].scatter([x[-1]], [y[-1]], marker="x", label="End")
    axes[0].set_xlabel("World X (m)")
    axes[0].set_ylabel("World Y (m)")
    axes[0].set_title("End-effector path (top view)")
    axes[0].axis("equal")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(data["time_s"], z, label="World Z")
    axes[1].set_xlabel("Simulation time (s)")
    axes[1].set_ylabel("World Z (m)")
    axes[1].set_title("End-effector height")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(out / "smooth_motion_end_effector_path.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    csv_path = args.csv.expanduser().resolve()
    out = args.outdir.expanduser().resolve()
    if not csv_path.is_file():
        parser.error(f"CSV not found: {csv_path}")
    out.mkdir(parents=True, exist_ok=True)
    data = load_rows(csv_path)

    save_speed_plot(data, out)
    save_acceleration_plot(data, out)
    save_joint_acceleration_comparison(data, out)
    save_ee_acceleration_comparison(data, out)
    save_acceleration_diagnostic_plot(data, out)
    save_joint_tracking_plot(data, out)
    save_error_plot(data, out)
    save_path_plot(data, out)
    print(f"Saved updated smooth-motion figures to: {out}")
    if not _has(data, "fd_actual_qddot1_rad_s2"):
        print("NOTE: This CSV predates the dual-acceleration update; rerun the motion script for the new acceleration comparison plots.")


if __name__ == "__main__":
    main()
