#!/usr/bin/env python3
"""Generate dependency-free SVG error plot comparing the two *separate* RPC2 runs."""
import argparse
import csv
import html
import json
from pathlib import Path


def get_samples(path):
    with path.open(newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    result = []
    for row in rows:
        if row['phase'] == 'MOVING':
            result.append((float(row['t_wall_s']),
                           float(row['max_abs_joint_error_rad']) * 1000.0))
    return result


def svg_polyline(data, max_time, max_err, left, top, width, height):
    if not data:
        return ''
    # Less than 850 points per series keeps SVG readable.
    interval = max(1, len(data) // 850)
    data = data[::interval] + ([data[-1]] if data[-1] != data[::interval][-1] else [])
    coords = []
    for t, error in data:
        x = left + width * t / max_time
        y = top + height - height * error / max_err
        coords.append(f'{x:.2f},{y:.2f}')
    return ' '.join(coords)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dir', default='~/ros2_ws/rpc2_comparison_results')
    args = parser.parse_args()
    root = Path(args.dir).expanduser().resolve()
    modes = ('position', 'velocity')
    colors = {'position': '#2463b4', 'velocity': '#cf5d29'}
    data = {}
    summaries = {}
    for mode in modes:
        csv_path = root / f'{mode}_tracking.csv'
        json_path = root / f'{mode}_summary.json'
        if not csv_path.exists() or not json_path.exists():
            print(f'Missing {csv_path.name} or {json_path.name}. Run both modes first.')
            return 1
        data[mode] = get_samples(csv_path)
        summaries[mode] = json.loads(json_path.read_text(encoding='utf-8'))
    if any(not data[mode] for mode in modes):
        print('Missing moving samples; cannot generate comparison.')
        return 1
    config = [(s['cycles'], s['leg_duration_s'], s['endpoint_hold_s']) for s in summaries.values()]
    if config[0] != config[1]:
        print('WARNING: runs have different motion parameters. Do not describe as controlled comparison.')
    max_time = max(v[-1][0] for v in data.values()) * 1.015
    max_err = max(max(e for _, e in v) for v in data.values()) * 1.12
    max_err = max(1.0, max_err)
    width, height, left, top = 1110, 570, 75, 145
    plot_w, plot_h = width - 145, height - 245
    chunks = [f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="white" />
<text x="{left}" y="42" font-family="Arial,sans-serif" font-size="27" font-weight="bold" fill="#17263c">RPC 2 — POSITION vs VELOCITY</text>
<text x="{left}" y="69" font-family="Arial,sans-serif" font-size="15" fill="#4a5971">Team 1 repeated reach-and-return | Actual joint angles compared with current commanded target</text>
''']
    for i in range(5):
        val = max_err * i / 4
        y = top + plot_h - plot_h * i / 4
        chunks.append(f'<line x1="{left}" y1="{y:.2f}" x2="{left+plot_w}" y2="{y:.2f}" stroke="#e3e9f0" />')
        chunks.append(f'<text x="{left-10}" y="{y+5:.2f}" text-anchor="end" font-family="Arial" font-size="13" fill="#61718b">{val:.1f}</text>')
    for i in range(7):
        val = max_time * i / 6
        x = left + plot_w * i / 6
        chunks.append(f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{top+plot_h}" stroke="#edf0f4" />')
        chunks.append(f'<text x="{x:.2f}" y="{top+plot_h+24}" text-anchor="middle" font-family="Arial" font-size="13" fill="#61718b">{val:.0f}</text>')
    chunks.append(f'<text x="{left+plot_w/2}" y="{top+plot_h+55}" text-anchor="middle" font-family="Arial" font-size="15" fill="#324760">Wall-clock time since start (s)</text>')
    chunks.append(f'<text transform="translate(20 {top+plot_h/2}) rotate(-90)" text-anchor="middle" font-family="Arial" font-size="15" fill="#324760">Largest joint error (mrad)</text>')
    for mode in modes:
        pts = svg_polyline(data[mode], max_time, max_err, left, top, plot_w, plot_h)
        chunks.append(f'<polyline fill="none" stroke="{colors[mode]}" stroke-width="1.8" points="{pts}" />')
    footer_y = top + plot_h + 95
    for i, mode in enumerate(modes):
        xx = left + i * 475
        s = summaries[mode]
        chunks.append(f'<line x1="{xx}" y1="{footer_y}" x2="{xx+32}" y2="{footer_y}" stroke="{colors[mode]}" stroke-width="4" />')
        chunks.append(f'<text x="{xx+42}" y="{footer_y+5}" font-family="Arial" font-size="14" fill="#26374c">{mode.upper()} — RMSE {s["motion_joint_rmse_rad"]*1000:.1f} mrad; max {s["motion_max_joint_error_rad"]*1000:.1f} mrad</text>')
    chunks.append(f'<text x="{left}" y="{height-19}" font-family="Arial" font-size="12" fill="#65748c">Independent simulation runs; error is against current published/integrated actuator target, not a clock-aligned ideal reference.</text>')
    chunks.append('</svg>')
    output = root / 'position_vs_velocity.svg'
    output.write_text('\n'.join(chunks), encoding='utf-8')
    print(f'Graph saved: {output}')
    print('------- Side-by-side quantitative results -------')
    for mode in modes:
        s = summaries[mode]
        print(f'{mode.upper():8s} motion RMSE: {s["motion_joint_rmse_rad"]:.5f} rad; '
              f'max error: {s["motion_max_joint_error_rad"]:.5f} rad; '
              f'final home error: {s.get("final_home_max_joint_error_rad", float("nan")):.5f} rad')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
