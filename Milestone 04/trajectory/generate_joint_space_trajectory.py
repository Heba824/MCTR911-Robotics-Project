"""Generate a reproducible UR5e joint-space quintic reference CSV.

Run from the repository root:
  python3 "Milestone 04/trajectory/generate_joint_space_trajectory.py"

Defaults reproduce the conservative six-joint movement used in the M3
smooth-motion demonstration. This script does not launch MuJoCo or publish
robot commands.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
FK_DIR = ROOT / "Milestone 02" / "kinematics"
if str(FK_DIR) not in sys.path:
    sys.path.insert(0, str(FK_DIR))

from ur5e_dh_fk import JOINT_NAMES  # noqa: E402
from joint_space_trajectory import QuinticJointTrajectory  # noqa: E402

HOME = np.array([-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0.0], dtype=float)
JOINT_DELTA = np.array([-0.30, 0.16, -0.18, 0.12, -0.10, -0.20], dtype=float)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, default=6.0)
    parser.add_argument("--hold-duration", type=float, default=2.0)
    parser.add_argument("--sample-period", type=float, default=0.02)
    parser.add_argument("--start", type=float, nargs=6, default=None,
                        metavar=("Q1", "Q2", "Q3", "Q4", "Q5", "Q6"))
    parser.add_argument("--goal", type=float, nargs=6, default=None,
                        metavar=("Q1", "Q2", "Q3", "Q4", "Q5", "Q6"))
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "Milestone 04" / "Results" / "Joint_Space" / "joint_space_reference.csv",
    )
    args = parser.parse_args()

    if args.duration <= 0.0 or not np.isfinite(args.duration):
        parser.error("--duration must be positive and finite")
    if args.hold_duration < 0.0 or not np.isfinite(args.hold_duration):
        parser.error("--hold-duration must be non-negative and finite")
    if args.sample_period <= 0.0 or not np.isfinite(args.sample_period):
        parser.error("--sample-period must be positive and finite")
    if (args.start is None) != (args.goal is None):
        parser.error("Specify both --start and --goal, or neither")

    q_start = HOME.copy() if args.start is None else np.asarray(args.start, dtype=float)
    q_goal = HOME + JOINT_DELTA if args.goal is None else np.asarray(args.goal, dtype=float)
    trajectory = QuinticJointTrajectory(q_start, q_goal, args.duration)
    samples = trajectory.sample(args.sample_period, args.duration + args.hold_duration)

    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["time_s", "phase", "progress"]
    for prefix, suffix in (("q", "rad"), ("qdot", "rad_s"), ("qddot", "rad_s2")):
        fieldnames.extend(f"{prefix}_desired_{name}_{suffix}" for name in JOINT_NAMES)

    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for point in samples:
            row = {"time_s": point.time_s, "phase": point.phase, "progress": point.progress}
            for i, name in enumerate(JOINT_NAMES):
                row[f"q_desired_{name}_rad"] = point.position_rad[i]
                row[f"qdot_desired_{name}_rad_s"] = point.velocity_rad_s[i]
                row[f"qddot_desired_{name}_rad_s2"] = point.acceleration_rad_s2[i]
            writer.writerow(row)

    print("Joint-space trajectory generated successfully.")
    print("Joint order:", ", ".join(JOINT_NAMES))
    print("Start [rad]:", np.array2string(q_start, precision=5))
    print("Goal  [rad]:", np.array2string(q_goal, precision=5))
    print(f"Movement: {args.duration:.3f} s | hold: {args.hold_duration:.3f} s")
    print(f"Sample period: {args.sample_period:.4f} s | samples: {len(samples)}")
    print("Peak |qdot| [rad/s]:", np.array2string(trajectory.maximum_velocity_rad_s, precision=6))
    print("Peak |qddot| [rad/s^2]:", np.array2string(trajectory.maximum_acceleration_rad_s2, precision=6))
    print("CSV:", output)


if __name__ == "__main__":
    main()
