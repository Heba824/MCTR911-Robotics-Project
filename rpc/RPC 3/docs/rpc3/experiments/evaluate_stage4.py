"""Plot Stage 3/4 tracking and run an offline Panda singularity comparison."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mujoco
import numpy as np

from panda_kinematics.cartesian_flc import damped_inverse, joint_velocity
from panda_kinematics.panda_model import DEFAULT_SCENE
from panda_kinematics.trajectory_node import Q_MAX, Q_MIN, compute_jacobian, forward_kinematics


MODES = ('model_only', 'joint_flc', 'cartesian_flc')
LABELS = ('Model only', 'Joint FLC', 'Cartesian FLC')


def read_rows(path):
    with path.open(newline='') as file:
        return list(csv.DictReader(file))


def matrix(rows, key):
    return np.asarray([[float(row[f'{key}_{joint}']) for joint in range(1, 8)]
                       for row in rows])


def write_rows(path, rows):
    with path.open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def tracking(input_path, output):
    joint_fig, joint_axes = plt.subplots(4, 2, figsize=(11, 9), sharex=True)
    xyz_fig, xyz_axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    error_fig, error_axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    table = []
    for mode, label in zip(MODES, LABELS):
        paths = sorted([*input_path.rglob(f'{mode}.csv'),
                        *(p for p in input_path.rglob('tracking.csv')
                          if p.parent.name.endswith(f'_{mode}'))])
        for path in paths:
            rows = [row for row in read_rows(path) if row['phase'] == 'tracking']
            if not rows:
                continue
            time_key = 'sim_time' if 'sim_time' in rows[0] else 'reference_s'
            t = np.asarray([float(row[time_key]) for row in rows])
            if not np.isclose(t[-1], 4.):
                continue
            actual, desired = matrix(rows, 'q'), matrix(rows, 'q_des')
            error = desired - actual
            xyz, xyz_des = (np.asarray([forward_kinematics(q)[:3, 3] for q in poses])
                            for poses in (actual, desired))
            distance = np.linalg.norm(xyz_des - xyz, axis=1)
            torque = matrix(rows, 'tau_command')
            table.append({
                'mode': mode, 'source': str(path), 'samples': len(rows),
                'joint_rms_rad': float(np.sqrt(np.mean(error**2))),
                'joint_peak_rad': float(abs(error).max()),
                'joint_final_max_rad': float(abs(error[-1]).max()),
                'xyz_rms_m': float(np.sqrt(np.mean(distance**2))),
                'xyz_peak_m': float(distance.max()), 'xyz_final_m': float(distance[-1]),
                'peak_absolute_command_Nm': float(abs(torque).max()),
            })
            for i, ax in enumerate(joint_axes.flat[:7]):
                if len(table) == 1:
                    ax.plot(t, desired[:, i], 'k--', label='Desired')
                ax.plot(t, actual[:, i], label=label)
                ax.set_ylabel(f'Joint {i+1} (rad)')
            for i, ax in enumerate(xyz_axes):
                if len(table) == 1:
                    ax.plot(t, xyz_des[:, i], 'k--', label='Desired')
                ax.plot(t, xyz[:, i], label=label)
                ax.set_ylabel(f'{"XYZ"[i]} (m)')
            error_axes[0].plot(t, distance*1000, label=label)
            error_axes[1].plot(t, np.max(abs(torque), axis=1), label=label)
    if not table:
        raise ValueError('No complete tracking traces ending at the four-second reference')
    joint_axes.flat[-1].axis('off')
    error_axes[0].set_ylabel('XYZ error norm (mm)')
    error_axes[1].set_ylabel('Peak arm command (N·m)')
    for fig, axes, filename in (
            (joint_fig, joint_axes.flat, 'joint_tracking.png'),
            (xyz_fig, xyz_axes, 'xyz_tracking.png'),
            (error_fig, error_axes, 'errors_and_torques.png')):
        axes = list(axes)
        axes[0].legend()
        for ax in axes:
            ax.grid(alpha=.25)
            ax.set_xlabel('Reference time (s)')
        fig.tight_layout()
        fig.savefig(output / filename, dpi=150)
        plt.close(fig)
    write_rows(output / 'comparison.csv', table)
    return table


def singularity(scene, output):
    rng = np.random.default_rng(911)
    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    arm = [model.joint(f'joint{i}').qposadr[0] for i in range(1, 8)]
    mujoco.mj_resetDataKeyframe(model, data, model.key('home').id)
    candidate = None
    for q in rng.uniform(Q_MIN+.04, Q_MAX-.04, (6000, 7)):
        jac = compute_jacobian(q)
        u, singular_values, _ = np.linalg.svd(jac, full_matrices=False)
        if candidate is not None and singular_values[-1] >= candidate[0]:
            continue
        data.qpos[arm] = q
        mujoco.mj_forward(model, data)
        if data.ncon == 0:
            candidate = (singular_values[-1], q, jac, u[:, -1])
    if candidate is None or candidate[0] >= .02:
        raise RuntimeError('Seeded search found no collision-free near-singular pose')

    _, q, jac, direction = candidate
    damped, _, spectrum, damping = damped_inverse(jac)
    undamped = np.linalg.pinv(jac)
    rows = []
    for speed in np.linspace(.01, .08, 8):
        desired = speed*direction
        bounded, _, _, limited = joint_velocity(q, q, np.zeros(7), jac, desired)
        velocities = (undamped @ desired, damped @ desired, bounded)
        rows.append({
            'desired_speed_m_s': float(speed),
            **{f'{label}_joint_speed_norm_rad_s': float(np.linalg.norm(v))
               for label, v in zip(('undamped', 'damped', 'bounded'), velocities)},
            **{f'{label}_task_residual_m_s': float(np.linalg.norm(jac @ v - desired))
               for label, v in zip(('undamped', 'damped', 'bounded'), velocities)},
            'joint_limit_activations': int(limited),
        })
    write_rows(output / 'singularity.csv', rows)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for label in ('undamped', 'damped', 'bounded'):
        ax.semilogy([r['desired_speed_m_s'] for r in rows],
                    [r[f'{label}_joint_speed_norm_rad_s'] for r in rows], label=label)
    ax.set(xlabel='Requested XYZ speed along weakest direction (m/s)',
           ylabel='Joint-speed norm (rad/s)')
    ax.legend()
    ax.grid(alpha=.25)
    fig.tight_layout()
    fig.savefig(output / 'singularity.png', dpi=150)
    plt.close(fig)
    return {'seed': 911, 'samples': 6000, 'contact_count_at_pose': 0, 'q_rad': q.tolist(),
            'sigma': spectrum.tolist(), 'damping': damping,
            'comparison_at_max_speed': rows[-1]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--scene', type=Path, default=DEFAULT_SCENE)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    report = {'tracking': tracking(args.input, args.output),
              'singularity': singularity(args.scene, args.output)}
    (args.output / 'evaluation.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
