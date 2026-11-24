"""Generate a Cartesian UR5e pose trajectory CSV from two known joint poses.

The endpoint poses are calculated with the existing, validated Milestone 02
standard-DH forward-kinematics implementation. The trajectory between them is
a straight line in world-frame position and a shortest fixed-world-axis
orientation interpolation. Intermediate full-pose IK feasibility and
collision clearance are validated in a later Milestone 04 stage.
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

from ur5e_dh_fk import JOINT_NAMES, forward_kinematics  # noqa: E402
from task_space_trajectory import QuinticTaskSpaceTrajectory  # noqa: E402

HOME = np.array([-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0.0], dtype=float)
JOINT_DELTA = np.array([-0.30, 0.16, -0.18, 0.12, -0.10, -0.20], dtype=float)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, default=6.0)
    parser.add_argument("--hold-duration", type=float, default=2.0)
    parser.add_argument("--sample-period", type=float, default=0.02)
    parser.add_argument("--start-joints", type=float, nargs=6, default=None,
                        metavar=("Q1", "Q2", "Q3", "Q4", "Q5", "Q6"))
    parser.add_argument("--goal-joints", type=float, nargs=6, default=None,
                        metavar=("Q1", "Q2", "Q3", "Q4", "Q5", "Q6"))
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "Milestone 04" / "Results" / "Task_Space" / "task_space_reference.csv",
    )
    args = parser.parse_args()

    if args.duration <= 0.0 or not np.isfinite(args.duration):
        parser.error("--duration must be positive and finite")
    if args.hold_duration < 0.0 or not np.isfinite(args.hold_duration):
        parser.error("--hold-duration must be non-negative and finite")
    if args.sample_period <= 0.0 or not np.isfinite(args.sample_period):
        parser.error("--sample-period must be positive and finite")
    if (args.start_joints is None) != (args.goal_joints is None):
        parser.error("Specify both --start-joints and --goal-joints, or neither")

    q_start = HOME.copy() if args.start_joints is None else np.asarray(args.start_joints, dtype=float)
    q_goal = HOME + JOINT_DELTA if args.goal_joints is None else np.asarray(args.goal_joints, dtype=float)
    if q_start.shape != (6,) or q_goal.shape != (6,):
        parser.error("Start and goal must each contain six joint angles")
    if not np.all(np.isfinite(q_start)) or not np.all(np.isfinite(q_goal)):
        parser.error("Joint angles must be finite")

    start_pose = forward_kinematics(q_start)
    goal_pose = forward_kinematics(q_goal)
    trajectory = QuinticTaskSpaceTrajectory(start_pose, goal_pose, args.duration)
    samples = trajectory.sample(args.sample_period, args.duration + args.hold_duration)

    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    fields = [
        "time_s", "phase", "progress",
        "x_desired_m", "y_desired_m", "z_desired_m",
        "qx_desired", "qy_desired", "qz_desired", "qw_desired",
        "vx_desired_m_s", "vy_desired_m_s", "vz_desired_m_s",
        "wx_desired_rad_s", "wy_desired_rad_s", "wz_desired_rad_s",
        "ax_desired_m_s2", "ay_desired_m_s2", "az_desired_m_s2",
        "alphax_desired_rad_s2", "alphay_desired_rad_s2", "alphaz_desired_rad_s2",
    ]

    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for point in samples:
            row = {
                "time_s": point.time_s,
                "phase": point.phase,
                "progress": point.progress,
                "x_desired_m": point.position_world_m[0],
                "y_desired_m": point.position_world_m[1],
                "z_desired_m": point.position_world_m[2],
                "qx_desired": point.quaternion_xyzw[0],
                "qy_desired": point.quaternion_xyzw[1],
                "qz_desired": point.quaternion_xyzw[2],
                "qw_desired": point.quaternion_xyzw[3],
                "vx_desired_m_s": point.linear_velocity_world_m_s[0],
                "vy_desired_m_s": point.linear_velocity_world_m_s[1],
                "vz_desired_m_s": point.linear_velocity_world_m_s[2],
                "wx_desired_rad_s": point.angular_velocity_world_rad_s[0],
                "wy_desired_rad_s": point.angular_velocity_world_rad_s[1],
                "wz_desired_rad_s": point.angular_velocity_world_rad_s[2],
                "ax_desired_m_s2": point.linear_acceleration_world_m_s2[0],
                "ay_desired_m_s2": point.linear_acceleration_world_m_s2[1],
                "az_desired_m_s2": point.linear_acceleration_world_m_s2[2],
                "alphax_desired_rad_s2": point.angular_acceleration_world_rad_s2[0],
                "alphay_desired_rad_s2": point.angular_acceleration_world_rad_s2[1],
                "alphaz_desired_rad_s2": point.angular_acceleration_world_rad_s2[2],
            }
            writer.writerow(row)

    print("Task-space trajectory generated successfully.")
    print("Joint order used to derive endpoint poses:", ", ".join(JOINT_NAMES))
    print("Start joint configuration [rad]:", np.array2string(q_start, precision=5))
    print("Goal joint configuration  [rad]:", np.array2string(q_goal, precision=5))
    print("Start position [m]:", np.array2string(start_pose[:3, 3], precision=6))
    print("Goal position  [m]:", np.array2string(goal_pose[:3, 3], precision=6))
    print("Cartesian displacement [m]:",
          np.array2string(goal_pose[:3, 3] - start_pose[:3, 3], precision=6))
    print(f"Shortest orientation change: {np.rad2deg(trajectory.rotation_angle_rad):.4f} deg")
    print(f"Movement: {args.duration:.3f} s | hold: {args.hold_duration:.3f} s")
    print(f"Sample period: {args.sample_period:.4f} s | samples: {len(samples)}")
    print("CSV:", output)
    print("NOTE: intermediate-pose IK feasibility and collision clearance are not yet certified.")


if __name__ == "__main__":
    main()
