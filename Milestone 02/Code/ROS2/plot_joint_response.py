#!/usr/bin/env python3
"""Plot desired/actual joint angles and tracking error from MCTR911 UR5e logs.

Supports constant_controller.py, ur5e_signal_generator.py, slider_controller.py.
Run: python3 plot_joint_response.py --input "Milestone 02/Figures/<file>.csv" --joint 1
"""
import argparse
import csv
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description='Plot UR5e joint target tracking data.')
    parser.add_argument('--input', type=Path, required=True, help='CSV log from constant, signal, or slider controller')
    parser.add_argument('--joint', type=int, default=1, choices=range(1, 7), metavar='1..6', help='Joint number (1 through 6)')
    parser.add_argument('--output-dir', type=Path, default=None, help='Where to write PNG graphs (default: CSV parent / Plots)')
    args = parser.parse_args()

    desired_key = f'desired_j{args.joint}_rad'
    actual_key = f'actual_j{args.joint}_rad'
    times, desired, actual = [], [], []
    with args.input.open(newline='', encoding='utf-8-sig') as infile:
        reader = csv.DictReader(infile)
        required = {'time_s', desired_key, actual_key}
        missing = required - set(reader.fieldnames or [])
        if missing:
            parser.error(f'Missing columns in {args.input.name}: {", ".join(sorted(missing))}')
        for row in reader:
            try:
                t = float(row['time_s'])
                d = float(row[desired_key])
                a = float(row[actual_key])
            except (ValueError, TypeError, KeyError):
                continue
            if all(map(math.isfinite, (t, d, a))):
                times.append(t)
                desired.append(d)
                actual.append(a)
    if not times:
        parser.error('CSV contains no usable data points.')
    errors = [d - a for d, a in zip(desired, actual)]
    mae = sum(abs(e) for e in errors) / len(errors)
    rmse = math.sqrt(sum(e * e for e in errors) / len(errors))
    max_abs = max(abs(e) for e in errors)

    output_dir = args.output_dir or (args.input.parent / 'Plots')
    output_dir.mkdir(parents=True, exist_ok=True)
    name = f'{args.input.stem}_j{args.joint}'
    title = f'{args.input.stem.replace("_", " ")} — J{args.joint}'

    fig, ax = plt.subplots(figsize=(10, 4.7))
    ax.plot(times, desired, label='Desired position')
    ax.plot(times, actual, label='Actual position')
    ax.set(title=title, xlabel='Time (s)', ylabel='Joint angle (rad)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    tracking_png = output_dir / f'{name}_tracking.png'
    fig.savefig(tracking_png, dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4.7))
    ax.plot(times, errors, label='Desired - actual')
    ax.axhline(0, linestyle='--', linewidth=1, alpha=0.6)
    ax.set(title=f'{title} — tracking error', xlabel='Time (s)', ylabel='Tracking error (rad)')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    error_png = output_dir / f'{name}_error.png'
    fig.savefig(error_png, dpi=160)
    plt.close(fig)

    print(f'Source: {args.input}')
    print(f'Joint: J{args.joint}; samples: {len(times)}; time range: {times[0]:.2f} to {times[-1]:.2f} s')
    print(f'MAE: {mae:.5f} rad | RMSE: {rmse:.5f} rad | max absolute error: {max_abs:.5f} rad')
    print(f'Final measured error: {errors[-1]:+.5f} rad')
    print(f'Saved: {tracking_png}\nSaved: {error_png}')


if __name__ == '__main__':
    main()
