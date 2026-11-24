"""Map the Milestone 04 task-space reference to a joint-space trajectory.

For each desired Cartesian sample:
1. Solve full-pose inverse kinematics for the desired pose using the previous
   joint solution as the seed.
2. Solve inverse velocity kinematics for the desired world-frame twist.
3. Solve inverse acceleration kinematics for the desired world-frame twist derivative.
4. Record joint values, reconstructed FK pose, and tracking residuals.

This script certifies kinematic reachability only under the implemented model.
It does not yet run MuJoCo, command ROS 2, or perform collision checking.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
M2_DIR = ROOT / 'Milestone 02' / 'kinematics'
M3_DIR = ROOT / 'Milestone 03' / 'kinematics'
THIS_DIR = Path(__file__).resolve().parent
for folder in (M2_DIR, M3_DIR, THIS_DIR):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

from ur5e_dh_fk import JOINT_NAMES, forward_kinematics  # noqa: E402
from ur5e_jacobian import geometric_jacobian  # noqa: E402
from ur5e_velocity_kinematics import inverse_velocity  # noqa: E402
from ur5e_acceleration_kinematics import inverse_acceleration  # noqa: E402
from task_space_trajectory import QuinticTaskSpaceTrajectory  # noqa: E402
from full_pose_ik import HOME, JOINT_MAX, JOINT_MIN, full_pose_inverse_kinematics, pose_error_world  # noqa: E402

JOINT_DELTA = np.array([-0.30, 0.16, -0.18, 0.12, -0.10, -0.20], dtype=float)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--duration', type=float, default=6.0)
    parser.add_argument('--hold-duration', type=float, default=2.0)
    parser.add_argument('--sample-period', type=float, default=0.02)
    parser.add_argument('--start-joints', type=float, nargs=6, default=None,
                        metavar=('Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6'))
    parser.add_argument('--goal-joints', type=float, nargs=6, default=None,
                        metavar=('Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6'))
    parser.add_argument('--ik-damping', type=float, default=1e-4)
    parser.add_argument('--ik-max-iterations', type=int, default=250)
    parser.add_argument('--ik-max-step', type=float, default=0.15)
    parser.add_argument('--position-weight', type=float, default=1.0)
    parser.add_argument('--orientation-weight', type=float, default=1.0)
    parser.add_argument('--velocity-damping', type=float, default=1e-5)
    parser.add_argument('--acceleration-damping', type=float, default=1e-5)
    parser.add_argument(
        '--output', type=Path,
        default=ROOT / 'Milestone 04' / 'Results' / 'Task_Space' / 'task_space_joint_trajectory.csv'
    )
    args = parser.parse_args()

    q_start = HOME.copy() if args.start_joints is None else np.asarray(args.start_joints, dtype=float)
    q_goal = HOME + JOINT_DELTA if args.goal_joints is None else np.asarray(args.goal_joints, dtype=float)
    if q_start.shape != (6,) or q_goal.shape != (6,):
        parser.error('Start and goal joint arrays must each have shape (6,)')
    if np.any(q_start < JOINT_MIN) or np.any(q_start > JOINT_MAX):
        parser.error('Start joint configuration violates configured limits')
    if np.any(q_goal < JOINT_MIN) or np.any(q_goal > JOINT_MAX):
        parser.error('Goal joint configuration violates configured limits')

    start_pose = forward_kinematics(q_start)
    goal_pose = forward_kinematics(q_goal)
    trajectory = QuinticTaskSpaceTrajectory(start_pose, goal_pose, args.duration)
    samples = trajectory.sample(args.sample_period, args.duration + args.hold_duration)

    rows = []
    q_seed = q_start.copy()
    max_position_error = 0.0
    max_orientation_error = 0.0
    min_sigma_overall = float('inf')
    max_step_between_samples = 0.0

    for index, point in enumerate(samples):
        ik = full_pose_inverse_kinematics(
            point.pose_world,
            q_seed,
            position_weight=args.position_weight,
            orientation_weight=args.orientation_weight,
            max_iterations=args.ik_max_iterations,
            max_step_rad=args.ik_max_step,
            damping=args.ik_damping,
        )
        if not ik.success:
            raise RuntimeError(
                f'IK failed at sample {index} (t={point.time_s:.3f} s): {ik.message}; '
                f'position_error={ik.position_error_m:.3e} m, '
                f'orientation_error={ik.orientation_error_rad:.3e} rad'
            )

        q = ik.joint_angles
        qdot = inverse_velocity(q, point.twist_world, damping=args.velocity_damping)
        qddot = inverse_acceleration(q, qdot, point.acceleration_world, damping=args.acceleration_damping)
        pose_fk = forward_kinematics(q)
        p_err, r_err = pose_error_world(pose_fk, point.pose_world)
        position_error = float(np.linalg.norm(p_err))
        orientation_error = float(np.linalg.norm(r_err))
        singular_values = np.linalg.svd(geometric_jacobian(q), compute_uv=False)
        min_sigma = float(np.min(singular_values))
        if index == 0:
            step_from_previous = 0.0
        else:
            step_from_previous = float(np.linalg.norm(q - q_seed))
        q_seed = q.copy()

        max_position_error = max(max_position_error, position_error)
        max_orientation_error = max(max_orientation_error, orientation_error)
        min_sigma_overall = min(min_sigma_overall, min_sigma)
        max_step_between_samples = max(max_step_between_samples, step_from_previous)

        row = {
            'sample_index': index,
            'time_s': point.time_s,
            'phase': point.phase,
            'progress': point.progress,
            'x_desired_m': point.position_world_m[0],
            'y_desired_m': point.position_world_m[1],
            'z_desired_m': point.position_world_m[2],
            'qx_desired': point.quaternion_xyzw[0],
            'qy_desired': point.quaternion_xyzw[1],
            'qz_desired': point.quaternion_xyzw[2],
            'qw_desired': point.quaternion_xyzw[3],
            'x_fk_m': pose_fk[0, 3],
            'y_fk_m': pose_fk[1, 3],
            'z_fk_m': pose_fk[2, 3],
            'position_error_m': position_error,
            'orientation_error_rad': orientation_error,
            'ik_iterations': ik.iterations,
            'min_singular_value': min_sigma,
            'joint_step_from_previous_rad': step_from_previous,
        }
        for i in range(6):
            row[f'q{i+1}_rad'] = q[i]
            row[f'qdot{i+1}_rad_s'] = qdot[i]
            row[f'qddot{i+1}_rad_s2'] = qddot[i]
        rows.append(row)

    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print('Task-space to joint-space mapping generated successfully.')
    print('Joint order:', ', '.join(JOINT_NAMES))
    print('Start q [rad]:', np.array2string(q_start, precision=5))
    print('Goal  q [rad]:', np.array2string(q_goal, precision=5))
    print(f'Movement: {args.duration:.3f} s | hold: {args.hold_duration:.3f} s')
    print(f'Sample period: {args.sample_period:.4f} s | samples: {len(rows)}')
    print(f'Max FK position error: {max_position_error:.3e} m')
    print(f'Max FK orientation error: {max_orientation_error:.3e} rad')
    print(f'Min Jacobian singular value along path: {min_sigma_overall:.3e}')
    print(f'Max joint-step norm between consecutive samples: {max_step_between_samples:.3e} rad')
    print('CSV:', output)
    print('NOTE: this certifies model-based kinematic reachability only; collision checks are still pending.')


if __name__ == '__main__':
    main()
