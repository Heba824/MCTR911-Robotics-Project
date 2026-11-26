"""Run Stage 3 against actual headless bridge callbacks using a test-only pipe."""

import argparse
import csv
import json
from pathlib import Path
import subprocess
import time

import numpy as np

from panda_kinematics.dynamics import PandaDynamics
from panda_kinematics.cartesian_flc import CARTESIAN_FIELDS, CartesianController
from panda_kinematics.control_timing import record_timing, timing_start
from panda_kinematics.joint_control import JointController, warm_controller
from panda_kinematics.panda_model import DEFAULT_SCENE, PandaModel
from panda_kinematics.trajectory_node import Q_END, Q_MAX, Q_MIN, Q_START, forward_kinematics, plan_trajectory


def read_state(stream):
    for line in stream:
        if line.startswith('STATE '):
            break
    else:
        raise RuntimeError('Bridge exited before returning a state')
    values = np.fromstring(line.removeprefix('STATE '), sep=' ')
    if values.shape != (24,) or not np.all(np.isfinite(values)):
        raise RuntimeError(f'Invalid bridge state: {line!r}')
    return values[0], int(values[1]), int(values[2]), *np.split(values[3:], 3)


def run(args, mode, dynamics):
    times, desired, velocities, accelerations = plan_trajectory(Q_START, Q_END, 4.0, 0.01)
    controller = CartesianController(dynamics) if mode == 'cartesian_flc' else JointController(dynamics, mode)
    command = [str(args.bridge), str(args.scene), '--exchange',
               '--ros-args', '-p', 'control_mode:=TORQUE', '-p', 'headless:=true']
    fields = ('q', 'dq', 'q_des', 'dq_des', 'ddq_des', 'error',
              'tau_requested', 'tau_command', 'tau_applied')
    header = ['sim_time', 'phase', 'contact_count', 'watchdog_status', 'compute_ms', 'saturated']
    header += [f'{field}_{i}' for field in fields for i in range(1, 8)]
    profile_fields = tuple(f'{name}_{unit}' for name in (
        'dynamics', 'reference_and_model_numpy', 'correction', 'torque_limits_numpy',
        'total', 'cycle_work', 'fuzzy_memberships', 'fuzzy_rules',
        'fuzzy_aggregation', 'fuzzy_centroid',
    ) for unit in ('ms', 'cpu_ms'))
    header.extend((*profile_fields, 'send_offset_from_tick_ms'))
    header.extend(f'{field}_{axis}' for field in ('hand', 'hand_des', 'hand_error')
                  for axis in 'xyz')
    header.extend(CARTESIAN_FIELDS)
    errors, xyz_errors, profiles, send_gaps, send_offsets = [], [], [], [], []
    last_send = None
    watchdog_events = 0
    applied_error = saturation = 0
    start = time.perf_counter()
    with subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                          text=True, bufsize=1) as bridge:
        try:
            now, state, contacts, q, dq, _ = read_state(bridge.stdout)
            if state != 0 or not np.allclose(q, Q_START, atol=1e-6, rtol=0):
                raise RuntimeError('Bridge did not start paused at the expected home')
            warmup = warm_controller(controller, (times, desired, velocities, accelerations))
            with (args.output / f'{mode}.csv').open('w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(header)
                schedule_start = time.perf_counter()
                for i in range(551):
                    if args.pace:
                        time.sleep(max(0, schedule_start + i*0.01 - time.perf_counter()))
                    cycle_start = timing_start()
                    reference = min(i, len(times) - 1)
                    qd, dqd, ddqd = desired[reference], velocities[reference], accelerations[reference]
                    phase = 'tracking' if i <= 400 else 'observation' if i <= 500 else 'terminal_hold'
                    if (not np.isclose(now, i*0.01, atol=1e-9) or contacts
                            or np.any(q < Q_MIN) or np.any(q > Q_MAX)
                            or np.max(abs(qd - q)) > 0.5):
                        raise RuntimeError(f'Unsafe or unexpected bridge state at {now:.3f} s')
                    t0 = timing_start()
                    profile = {}
                    result = controller.compute(q, dq, qd, dqd, ddqd,
                                                terminal_hold=phase == 'terminal_hold', profile=profile)
                    record_timing(profile, 'total', t0)
                    send_at = time.perf_counter()
                    offset = (send_at - schedule_start - i*0.01)*1000 if args.pace else None
                    if offset is not None:
                        send_offsets.append(offset)
                    if last_send is not None:
                        send_gaps.append((send_at - last_send)*1000)
                    last_send = send_at
                    bridge.stdin.write(' '.join(map(str, [*result.command, 255, 0])) + '\n')
                    next_time, state, contacts, next_q, next_dq, effort = read_state(bridge.stdout)
                    record_timing(profile, 'cycle_work', cycle_start)
                    if state not in (1, 2):
                        raise RuntimeError(f'Bridge did not activate at {now:.3f} s: {state}')
                    watchdog_events += int(state == 2)
                    clipped = np.count_nonzero(result.requested != result.command)
                    saturation += clipped
                    applied_error = max(applied_error, float(np.max(abs(result.command - effort))))
                    profiles.append(profile)
                    errors.append(qd - q)
                    xyz, xyz_des = (forward_kinematics(v)[:3, 3] for v in (q, qd))
                    xyz_errors.append(xyz_des - xyz)
                    writer.writerow([now, phase, contacts, state, profile['total_ms'], clipped, *q, *dq,
                                     *qd, *dqd, *ddqd, *(qd - q), *result.requested,
                                     *result.command, *effort,
                                     *(profile.get(field) for field in profile_fields), offset,
                                     *xyz, *xyz_des, *(xyz_des - xyz),
                                     *(profile.get(field) for field in CARTESIAN_FIELDS)])
                    if state != 1:
                        break
                    now, q, dq = next_time, next_q, next_dq
        finally:
            bridge.stdin.close()
            bridge.wait(timeout=5)
    if bridge.returncode or (not watchdog_events and applied_error > 1e-10):
        raise RuntimeError(f'Bridge exit={bridge.returncode}, torque mismatch={applied_error}')
    errors = np.asarray(errors)
    tracking = errors[:401]
    xyz_tracking = np.linalg.norm(np.asarray(xyz_errors[:401]), axis=1)
    cartesian = [p for p in profiles if 'jacobian_sigma_3' in p]
    cartesian_summary = {
        'minimum_sigma': float(min(p['jacobian_sigma_3'] for p in cartesian)),
        'maximum_damping': float(max(p['jacobian_damping'] for p in cartesian)),
        'cartesian_limit_counts': {key: int(sum(p[key] for p in cartesian))
                                   for key in CARTESIAN_FIELDS[-4:]},
    } if cartesian else {}

    def statistics(values):
        return dict(zip(('mean', 'p95', 'p99', 'maximum'),
                        (float(np.mean(values)), *np.percentile(values, [95, 99]), max(values))))

    latencies = [profile['total_ms'] for profile in profiles]
    worst = max(profiles, key=lambda profile: profile['total_ms'])
    return {
        'mode': mode, **cartesian_summary,
        'xyz_rms_m': float(np.sqrt(np.mean(xyz_tracking**2))),
        'xyz_peak_m': float(xyz_tracking.max()),
        'xyz_final_m': float(xyz_tracking[-1]) if len(xyz_tracking) == 401 else None, 'wall_seconds_including_bridge_startup': time.perf_counter() - start,
        'rms_rad_by_joint': np.sqrt(np.mean(tracking**2, axis=0)).tolist(),
        'peak_rad_by_joint': np.max(abs(tracking), axis=0).tolist(),
        'tracking_final_rad_by_joint': abs(errors[400]).tolist() if len(errors) > 400 else None,
        'observation_final_rad_by_joint': abs(errors[500]).tolist() if len(errors) > 500 else None,
        'terminal_hold_final_rad_by_joint': abs(errors[550]).tolist() if len(errors) > 550 else None,
        'applied_torque_max_discrepancy_Nm': applied_error,
        'saturated_joint_samples': int(saturation),
        'watchdog_status': 'timed_out' if watchdog_events else 'active',
        'watchdog_events': watchdog_events, 'cycles': len(profiles), 'paced': args.pace,
        'compute_ms': statistics(latencies),
        'compute_over_10ms': sum(value > 10 for value in latencies),
        'send_deadline_misses': sum(offset > 10 for offset in send_offsets) if args.pace else None,
        'cycle_work_over_10ms': sum(row['cycle_work_ms'] > 10 for row in profiles),
        'send_gap_ms': statistics(send_gaps) if send_gaps else None,
        'first_five_compute_ms': latencies[:5], 'worst_compute_profile': worst,
        'warmup_ms': [sample['total_ms'] for sample in warmup],
        'component_timings': {
            field: statistics([profile[field] for profile in profiles if field in profile])
            for field in profile_fields if any(field in profile for profile in profiles)
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bridge', type=Path, required=True)
    parser.add_argument('--scene', type=Path, default=DEFAULT_SCENE)
    parser.add_argument('--pace', action='store_true', help='Schedule commands at nominal 100 Hz')
    parser.add_argument('--modes', nargs='+', choices=('model_only', 'joint_flc', 'cartesian_flc'),
                        default=('model_only', 'joint_flc'))
    parser.add_argument('--repeats', type=int, default=1)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    dynamics = PandaDynamics(PandaModel.from_scene(args.scene))
    if args.repeats < 1:
        parser.error('--repeats must be positive')
    output = args.output
    results = []
    for repeat in range(1, args.repeats + 1):
        args.output = output / f'run_{repeat:02d}' if args.repeats > 1 else output
        args.output.mkdir(parents=True, exist_ok=True)
        for mode in args.modes:
            results.append({'repeat': repeat, **run(args, mode, dynamics)})
    report = json.dumps(results, indent=2)
    (output / 'summary.json').write_text(report + '\n')
    print(report)
    return int(any(result['watchdog_events'] for result in results))


if __name__ == '__main__':
    raise SystemExit(main())
