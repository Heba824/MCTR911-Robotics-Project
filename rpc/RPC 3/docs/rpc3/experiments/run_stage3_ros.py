"""Run paired Stage 3 trials over ROS topics using installed Jazzy executables."""

import argparse
import csv
from contextlib import contextmanager
import json
import os
from pathlib import Path
import signal
from statistics import mean
import subprocess

from ament_index_python.packages import get_package_prefix


@contextmanager
def running(command, log_path):
    with log_path.open('w') as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        try:
            yield process
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGINT)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()


def read_csv(path):
    if not path.exists():
        return []
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def trace_metrics(trace, tracking, shutdown_ros_ns):
    """Join command and state timing using sequence labels and local clocks."""
    receipts = [row for row in trace if row['event'] == 'valid_command']
    states = [row for row in trace if row['event'] == 'state']
    watchdog = [row for row in trace if row['event'] == 'watchdog']
    renders = [row for row in trace if row['event'] == 'render']

    def gaps(rows):
        stamps = [int(row['monotonic_ns']) for row in rows]
        return [(right-left)/1e6 for left, right in zip(stamps, stamps[1:])]

    def stats(values):
        return {'mean': mean(values), 'maximum': max(values)} if values else None

    by_sequence = {int(row['sequence']): row for row in receipts if int(row['sequence']) > 0}
    paired = [(row, by_sequence[int(row['command_sequence'])]) for row in tracking
              if int(row['command_sequence']) in by_sequence]
    endpoint_received = any(row['phase'] == 'tracking' and float(row['reference_s']) == 4.0
                            for row, _ in paired)
    valid_gaps, state_gaps = gaps(receipts), gaps(states)
    return {
        'trace_available': bool(trace), 'valid_receipts': len(receipts),
        'recorded_commands_matched': len(paired), 'endpoint_received': endpoint_received,
        'max_valid_receipt_gap_ms': max(valid_gaps, default=None),
        'bridge_state_rate_hz': 1000 / mean(state_gaps) if state_gaps else None,
        'bridge_state_gap_ms': stats(state_gaps),
        'bridge_physics_ms': stats([int(r['physics_ns'])/1e6 for r in states]),
        'bridge_render_ms': stats([int(r['callback_ns'])/1e6 for r in renders]),
        'bridge_rejected_stale': sum(row['event'] == 'rejected_stale' for row in trace),
        **{f'bridge_{field.removesuffix("_ns")}_ms': stats([
            int(r[field])/1e6 for r in states if r.get(field)
        ]) for field in ('pre_status_ns', 'status_publish_ns', 'state_publish_ns', 'callback_cpu_ns')},
        'bridge_state_callback_ms': stats([int(r['callback_ns'])/1e6 for r in states]),
        'bridge_command_callback_ms': stats([int(r['callback_ns'])/1e6 for r in receipts]),
        'publish_to_bridge_callback_ms': stats([
            (int(r['ros_ns']) - int(c['publish_ros_ns']))/1e6 for c, r in paired
        ]),
        'middleware_source_to_receive_ms': stats([
            (int(r['dds_receive_ns'])-int(r['dds_source_ns']))/1e6 for r in receipts
            if 0 < int(r['dds_source_ns']) <= int(r['dds_receive_ns'])
        ]),
        'bridge_watchdogs_before_shutdown': sum(
            shutdown_ros_ns is None or int(r['ros_ns']) < shutdown_ros_ns for r in watchdog),
        'bridge_watchdogs_total': len(watchdog),
    }


def trial(args, mode, directory):
    directory.mkdir(parents=True, exist_ok=True)
    trace_path = directory / 'bridge_timing.csv'
    bridge = ['ros2', 'launch', 'mujoco_ros2', 'rpc3_torque.launch.py',
              f'scene:={args.scene}', f'headless:={str(not args.gui).lower()}', 'namespace:=rpc3', f'dds_publication_mode:={args.dds_publication_mode}',
              f'torque_trace_csv:={trace_path}']
    executable = (Path(get_package_prefix('panda_kinematics'))
                  / 'lib/panda_kinematics/controller_node')
    csv_path = directory / 'tracking.csv'
    controller = [str(executable), '--ros-args', '-r', '__ns:=/rpc3',
                  '-p', f'scene:={args.scene}', '-p', f'control_mode:={mode}',
                  '-p', f'results_csv:={csv_path}']
    if args.bridge_cpus:
        bridge = ['taskset', '-c', args.bridge_cpus, *bridge]
    if args.controller_cpus:
        controller = ['taskset', '-c', args.controller_cpus, *controller]
    with running(bridge, directory / 'bridge.log') as simulation:
        with running(controller, directory / 'controller.log') as control:
            try:
                control.wait(timeout=args.seconds)
            except subprocess.TimeoutExpired:
                pass
        bridge_alive = simulation.poll() is None

    summary = csv_path.with_suffix('.json')
    if not summary.exists():
        return {'passed': False, 'error': 'Controller did not write tracking diagnostics'}
    result = json.loads(summary.read_text())
    tracking = read_csv(csv_path)
    result.update(trace_metrics(read_csv(trace_path), tracking, result.get('shutdown_ros_ns')))
    result['command_torques_within_limits'] = bool(tracking) and all(
        abs(float(row[f'tau_command_{i}'])) <= (87 if i <= 4 else 12)
        for row in tracking for i in range(1, 8)
    )
    result['passed'] = (
        bridge_alive and not result['fault'] and result['bridge_status'] == 'active'
        and result['tracking_endpoint_sent'] and result['watchdog_events_observed'] == 0
        and result.get('terminal_hold_commands', 0) > 0
        and result['endpoint_received'] and result['command_torques_within_limits']
        and result['bridge_watchdogs_before_shutdown'] == 0
    )
    if not result['passed'] and not result['fault']:
        logs = ((directory / 'controller.log').read_text()
                + (directory / 'bridge.log').read_text())
        if 'Error creating socket: Operation not permitted' in logs:
            result['error'] = 'DDS socket creation denied'
        elif not tracking:
            result['error'] = 'No tracking commands recorded'
        elif not result['endpoint_received']:
            result['error'] = 'No receipt for the trajectory endpoint'
        else:
            result['error'] = 'Trial failed a terminal hold, torque, or watchdog check'
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--dds-publication-mode', choices=('ASYNCHRONOUS', 'SYNCHRONOUS'),
                        default='ASYNCHRONOUS')
    parser.add_argument('--gui', action='store_true', help='Open the TORQUE graphical bridge')
    parser.add_argument('--modes', nargs='+', choices=('model_only', 'joint_flc', 'cartesian_flc'),
                        default=('model_only', 'joint_flc'))
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--bridge-cpus', help='Optional Linux taskset CPU list')
    parser.add_argument('--controller-cpus', help='Optional Linux taskset CPU list')
    parser.add_argument('--seconds', type=float, default=10)
    args = parser.parse_args()
    if args.repeats < 1 or args.seconds < 6:
        parser.error('Use at least one repeat and six seconds per trial')
    args.output.mkdir(parents=True, exist_ok=False)
    args.scene = args.scene.expanduser().resolve()
    results = []
    for i in range(1, args.repeats + 1):
        for mode in args.modes:
            directory = args.output / f'{i:02d}_{mode}'
            results.append({'repeat': i, 'mode': mode, **trial(args, mode, directory)})
    (args.output / 'summary.json').write_text(json.dumps(results, indent=2) + '\n')
    for row in results:
        print(row['repeat'], row['mode'], 'PASS' if row['passed'] else 'FAIL',
              'commands=', row.get('commands'), 'fault=', row.get('fault') or row.get('error'))
    return int(not all(row['passed'] for row in results))


if __name__ == '__main__':
    raise SystemExit(main())
