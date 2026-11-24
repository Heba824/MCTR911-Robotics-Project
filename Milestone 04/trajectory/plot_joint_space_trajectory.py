"""Plot desired UR5e joint position, velocity and acceleration references."""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
FK_DIR = ROOT / "Milestone 02" / "kinematics"
if str(FK_DIR) not in sys.path:
    sys.path.insert(0, str(FK_DIR))

from ur5e_dh_fk import JOINT_NAMES  # noqa: E402


def read_reference(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"Trajectory CSV not found: {path}")
    with path.open("r", newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"Trajectory CSV is empty: {path}")
    data = {"time_s": np.asarray([float(row["time_s"]) for row in rows], dtype=float)}
    for prefix, suffix in (("q", "rad"), ("qdot", "rad_s"), ("qddot", "rad_s2")):
        for joint in JOINT_NAMES:
            column = f"{prefix}_desired_{joint}_{suffix}"
            if column not in rows[0]:
                raise ValueError(f"Required CSV column is missing: {column}")
            data[column] = np.asarray([float(row[column]) for row in rows], dtype=float)
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv", type=Path,
        default=ROOT / "Milestone 04" / "Results" / "Joint_Space" / "joint_space_reference.csv",
    )
    parser.add_argument(
        "--outdir", type=Path,
        default=ROOT / "Milestone 04" / "Figures" / "Joint_Space",
    )
    args = parser.parse_args()

    data = read_reference(args.csv.expanduser().resolve())
    output_dir = args.outdir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    specs = (
        ("q", "rad", "Joint position [rad]", "joint_position_reference.png"),
        ("qdot", "rad_s", "Joint velocity [rad/s]", "joint_velocity_reference.png"),
        ("qddot", "rad_s2", "Joint acceleration [rad/s²]", "joint_acceleration_reference.png"),
    )
    for prefix, suffix, ylabel, filename in specs:
        fig, ax = plt.subplots(figsize=(12, 6))
        for joint in JOINT_NAMES:
            column = f"{prefix}_desired_{joint}_{suffix}"
            label = joint.removesuffix("_joint").replace("_", " ")
            ax.plot(data["time_s"], data[column], label=label, linewidth=1.8)
        ax.set_title(f"UR5e joint-space quintic reference — {ylabel}")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend(ncol=3, loc="best")
        fig.tight_layout()
        destination = output_dir / filename
        fig.savefig(destination, dpi=180)
        plt.close(fig)
        print("Saved:", destination)


if __name__ == "__main__":
    main()
