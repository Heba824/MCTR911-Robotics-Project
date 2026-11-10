"""Create report-ready plots from physically stepped MuJoCo motion CSV.

Usage:
  python 'Milestone 03/visualization/plot_motion_results.py'
  python 'Milestone 03/visualization/plot_motion_results.py' --csv 'Milestone 03/Results/live_motion.csv'
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv', type=Path, default=ROOT / 'Milestone 03' / 'Results' / 'live_motion.csv')
    parser.add_argument('--outdir', type=Path, default=ROOT / 'Milestone 03' / 'Figures')
    args = parser.parse_args()
    with args.csv.expanduser().open(newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    if not rows:
        parser.error('CSV contains no samples')
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit('Matplotlib is missing. Install it into your existing venv: pip install matplotlib') from exc
    t = np.array([float(r['time_s']) for r in rows])
    def series(col: str) -> np.ndarray:
        return np.array([float(r[col]) for r in rows])
    outdir = args.outdir.expanduser()
    outdir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True, layout='constrained')
    for ax, suffix, units in zip(axes, 'xyz', ['m', 'm', 'm']):
        ax.plot(t, series(f'fk_{suffix}_m'), label='Python FK', linewidth=2)
        ax.plot(t, series(f'mujoco_{suffix}_m'), label='MuJoCo site', linestyle='--')
        ax.set_ylabel(f'{suffix.upper()} ({units})')
        ax.grid(alpha=0.25)
    axes[0].legend(loc='best')
    axes[-1].set_xlabel('Simulated time (s)')
    fig.suptitle('Moving UR5e attachment_site: Position Comparison')
    fig.savefig(outdir / 'motion_position_comparison.png', dpi=160)
    plt.close(fig)

    for title, keys, fname, unit in (
        ('Linear Velocity', ['vx', 'vy', 'vz'], 'motion_linear_velocity_comparison.png', 'm/s'),
        ('Angular Velocity', ['wx', 'wy', 'wz'], 'motion_angular_velocity_comparison.png', 'rad/s'),
        ('Linear Acceleration', ['ax', 'ay', 'az'], 'motion_linear_acceleration_comparison.png', 'm/s²'),
        ('Angular Acceleration', ['alphax', 'alphay', 'alphaz'], 'motion_angular_acceleration_comparison.png', 'rad/s²'),
    ):
        fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True, layout='constrained')
        for ax, component in zip(axes, keys):
            ax.plot(t, series('python_' + component), label='Python formula', linewidth=2)
            ax.plot(t, series('mujoco_' + component), '--', label='MuJoCo-derived')
            ax.set_ylabel(f'{component} ({unit})')
            ax.grid(alpha=0.25)
        axes[0].legend(loc='best')
        axes[-1].set_xlabel('Simulated time (s)')
        fig.suptitle('Moving UR5e attachment_site: ' + title)
        fig.savefig(outdir / fname, dpi=160)
        plt.close(fig)

    keys = ['position_error_m', 'linear_velocity_error_m_s', 'angular_velocity_error_rad_s',
            'linear_acceleration_error_m_s2', 'angular_acceleration_error_rad_s2']
    fig, axes = plt.subplots(3, 2, figsize=(12, 9), sharex=True, layout='constrained')
    for ax, key in zip(axes.flat, keys):
        ax.plot(t, series(key))
        ax.set_title(key.replace('_', ' '))
        ax.grid(alpha=0.25)
        ax.set_xlabel('Simulated time (s)')
    axes.flat[-1].axis('off')
    fig.suptitle('Actual-Motion Kinematics: Numerical Agreement Errors')
    fig.savefig(outdir / 'motion_kinematics_errors.png', dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4), layout='constrained')
    for i in range(1, 7):
        ax.plot(t, series(f'q{i}_rad') - series(f'target_q{i}_rad'), label=f'joint {i}')
    ax.set(xlabel='Simulated time (s)', ylabel='Actual minus command (rad)',
           title='Position Actuator: Actual Joint Tracking Error')
    ax.grid(alpha=0.25)
    ax.legend(ncol=3, fontsize=9)
    fig.savefig(outdir / 'motion_joint_tracking.png', dpi=160)
    plt.close(fig)
    print(f'Saved 7 plots to: {outdir.resolve()}')

if __name__ == '__main__':
    main()
