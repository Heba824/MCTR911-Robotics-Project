"""Plot the joint trajectory produced from the task-space IK mapping."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def read_csv(path: Path) -> tuple[dict[str, np.ndarray], list[str]]:
    if not path.is_file():
        raise FileNotFoundError(f'CSV not found: {path}')
    with path.open('r', newline='', encoding='utf-8') as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f'CSV is empty: {path}')
    numeric = {}
    for key in rows[0].keys():
        if key == 'phase':
            continue
        numeric[key] = np.asarray([float(row[key]) for row in rows], dtype=float)
    phases = [row['phase'] for row in rows]
    return numeric, phases


def save_plot(data, keys, labels, ylabel, title, output_path):
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for key, label in zip(keys, labels):
        ax.plot(data['time_s'], data[key], label=label, linewidth=1.8)
    ax.set_xlabel('Time [s]')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=3, loc='best')
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    print('Saved:', output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv', type=Path,
                        default=ROOT / 'Milestone 04' / 'Results' / 'Task_Space' / 'task_space_joint_trajectory.csv')
    parser.add_argument('--outdir', type=Path,
                        default=ROOT / 'Milestone 04' / 'Figures' / 'Task_Space')
    args = parser.parse_args()

    data, _ = read_csv(args.csv.expanduser().resolve())
    outdir = args.outdir.expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    save_plot(data, [f'q{i}_rad' for i in range(1, 7)], [f'q{i}' for i in range(1, 7)],
              'Joint position [rad]', 'Joint trajectory recovered from task-space IK',
              outdir / 'task_space_joint_positions.png')
    save_plot(data, [f'qdot{i}_rad_s' for i in range(1, 7)], [f'qdot{i}' for i in range(1, 7)],
              'Joint velocity [rad/s]', 'Joint velocities from inverse velocity kinematics',
              outdir / 'task_space_joint_velocities.png')
    save_plot(data, [f'qddot{i}_rad_s2' for i in range(1, 7)], [f'qddot{i}' for i in range(1, 7)],
              'Joint acceleration [rad/s²]', 'Joint accelerations from inverse acceleration kinematics',
              outdir / 'task_space_joint_accelerations.png')
    save_plot(data, ['position_error_m', 'orientation_error_rad'], ['position error', 'orientation error'],
              'Error magnitude', 'Full-pose IK reconstruction errors',
              outdir / 'task_space_ik_errors.png')
    save_plot(data, ['min_singular_value', 'joint_step_from_previous_rad'],
              ['min singular value', 'joint step from previous'],
              'Diagnostic value', 'Path conditioning and continuity diagnostics',
              outdir / 'task_space_ik_diagnostics.png')


if __name__ == '__main__':
    main()
