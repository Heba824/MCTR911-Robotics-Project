#!/usr/bin/env python3
"""Long-form RPC 2 Team 1 comparison: repeated quintic reach-and-return.

Separate fresh simulations for POSITION and VELOCITY; exactly the same joint-space
paths, segment durations and ROS topic rates for both. Does not modify the
instructor template. Uses the existing, verified plan_trajectory() implementation.

Output: CSV + JSON for each mode, allowing a fair tracking-error graph.
"""
import argparse
import csv
import json
import math
from pathlib import Path
import time

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray

from panda_kinematics.trajectory_node import (
    Q_START, Q_END, Q_MIN, Q_MAX, TRAJ_DT, GRIPPER_CTRL, plan_trajectory,
)


def build_segments(cycles, leg_duration):
    """Generate trajectories using the already-tested instructor math."""
    segments = []
    for cycle in range(1, cycles + 1):
        for label, q0, q1 in [('reach', Q_START, Q_END), ('return', Q_END, Q_START)]:
            t, q, dq, ddq = plan_trajectory(q0, q1, leg_duration, TRAJ_DT)
            assert len(t) == len(q) == len(dq) == len(ddq)
            if not (np.all(q >= Q_MIN - 1e-10) and np.all(q <= Q_MAX + 1e-10)):
                raise ValueError('A planned sample violates the joint limits')
            segments.append({'cycle': cycle, 'leg': label,
                             'q': q, 'dq': dq, 'target': q1.copy()})
    return segments


class RepeatedComparison(Node):
    def __init__(self, args):
        super().__init__('rpc2_repeated_comparison')
        self.args = args
        self.segments = build_segments(args.cycles, args.leg_duration)
        self.publisher = self.create_publisher(Float64MultiArray, '/joint_commands', 10)
        self.subscription = self.create_subscription(
            JointState, '/joint_state', self.on_state, 10)
        self.timer = self.create_timer(TRAJ_DT, self.tick)
        self.stage = 'WAITING'
        self.current_segment = 0
        self.sample_idx = 0
        self.hold_ticks_remaining = 0
        self.reference = Q_START.copy()
        self.actual = None
        self.start_wall = time.monotonic()
        self.t0 = None
        self.done = False
        self.failed = False
        self.samples = []
        self.checkpoints = []
        self.n_commands = 0
        self.last_progress = -1
        self.get_logger().info(
            f'Comparative demo: {args.mode}, {args.cycles} cycles, '
            f'{args.leg_duration:.2f}s per leg, {args.hold:.2f}s hold.'
        )
        self.get_logger().info('Will not start until the Panda is at HOME.')

    def on_state(self, msg):
        if len(msg.position) < 7:
            return
        self.actual = np.asarray(msg.position[:7], dtype=float)
        if self.t0 is None:
            return
        t = time.monotonic() - self.t0
        err = self.actual - self.reference
        seg = self.segments[min(self.current_segment, len(self.segments) - 1)]
        self.samples.append([
            t, self.stage, seg['cycle'], seg['leg'],
            *self.actual.tolist(), *self.reference.tolist(), *err.tolist(),
            float(np.max(np.abs(err))),
        ])

    def publish(self, arm):
        msg = Float64MultiArray()
        msg.data = np.asarray(arm, dtype=float).tolist() + [float(GRIPPER_CTRL), 0.0]
        self.publisher.publish(msg)
        self.n_commands += 1

    def hold(self, target):
        self.publish(target if self.args.mode == 'POSITION' else np.zeros(7))

    def tick(self):
        if self.done:
            return
        if self.stage == 'WAITING':
            if self.actual is None:
                if time.monotonic() - self.start_wall > 45.0:
                    self.get_logger().error('No joint states received after 45s. Check bridge log.')
                    self.failed, self.done = True, True
                return
            home_err = np.max(np.abs(self.actual - Q_START))
            if home_err > 0.06:
                self.get_logger().error(
                    f'Robot not at HOME (max error {home_err:.3f}rad). '
                    'Stop and restart MuJoCo before this test.')
                self.failed, self.done = True, True
                return
            self.t0 = time.monotonic()
            self.stage = 'MOVING'
            self.get_logger().info('HOME verified; starting repeated motion.')

        seg = self.segments[self.current_segment]
        if self.stage == 'MOVING':
            if self.sample_idx < len(seg['q']):
                q = seg['q'][self.sample_idx]
                dq = seg['dq'][self.sample_idx]
                if self.args.mode == 'POSITION':
                    self.reference = q.copy()
                    self.publish(q)
                else:
                    # Mirrors the fixed bridge: each published dq is integrated
                    # for exactly the original TRAJ_DT=0.05 s.
                    self.reference += dq * TRAJ_DT
                    self.publish(dq)
                if self.sample_idx == 0:
                    self.get_logger().info(
                        f"Cycle {seg['cycle']}/{self.args.cycles}: {seg['leg']} started")
                self.sample_idx += 1
                return
            self.stage = 'HOLD'
            self.hold_ticks_remaining = max(1, round(self.args.hold / TRAJ_DT))

        if self.stage == 'HOLD':
            self.hold(seg['target'])
            self.hold_ticks_remaining -= 1
            if self.hold_ticks_remaining > 0:
                return
            if self.actual is not None:
                endpoint_err = np.abs(self.actual - seg['target'])
                record = {'cycle': seg['cycle'], 'leg': seg['leg'],
                          't_wall_s': round(time.monotonic() - self.t0, 4),
                          'max_endpoint_error_rad': float(np.max(endpoint_err)),
                          'endpoint_joint_rmse_rad': float(np.sqrt(np.mean(endpoint_err**2)))}
                self.checkpoints.append(record)
                self.get_logger().info(
                    f"{seg['leg']} {seg['cycle']}: max endpoint error = "
                    f"{record['max_endpoint_error_rad']:.5f} rad")
            self.current_segment += 1
            self.sample_idx = 0
            if self.current_segment < len(self.segments):
                self.stage = 'MOVING'
                return
            self.current_segment = len(self.segments) - 1
            self.stage = 'FINAL_SETTLE'
            self.hold_ticks_remaining = max(1, round(self.args.final_settle / TRAJ_DT))
            self.get_logger().info('All cycles done; allowing final settling.')
            return

        if self.stage == 'FINAL_SETTLE':
            self.hold(Q_START)
            self.hold_ticks_remaining -= 1
            if self.hold_ticks_remaining <= 0:
                self.save()
                self.done = True

    def save(self):
        root = Path(self.args.output).expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)
        stem = self.args.mode.lower()
        csv_path = root / f'{stem}_tracking.csv'
        json_path = root / f'{stem}_summary.json'
        header = (['t_wall_s', 'phase', 'cycle', 'leg'] +
                  [f'actual_j{i}_rad' for i in range(1, 8)] +
                  [f'reference_j{i}_rad' for i in range(1, 8)] +
                  [f'error_j{i}_rad' for i in range(1, 8)] +
                  ['max_abs_joint_error_rad'])
        with csv_path.open('w', newline='', encoding='utf-8') as handle:
            writer = csv.writer(handle)
            writer.writerow(header)
            writer.writerows(self.samples)

        summary = {
            'mode': self.args.mode, 'cycles': self.args.cycles,
            'leg_duration_s': self.args.leg_duration,
            'endpoint_hold_s': self.args.hold,
            'wall_duration_s': None if self.t0 is None else time.monotonic() - self.t0,
            'number_of_joint_state_samples': len(self.samples),
            'commands_sent': self.n_commands,
            'reference_definition': (
                'last published position target' if self.args.mode == 'POSITION'
                else 'cumulative 0.05s integration of published velocity samples'
            ),
            'checkpoint_errors': self.checkpoints,
        }
        if self.samples:
            moving = [s for s in self.samples if s[1] == 'MOVING']
            selected = moving if moving else self.samples
            err = np.array([s[18:25] for s in selected], dtype=float)
            summary['motion_joint_rmse_rad'] = float(np.sqrt(np.mean(err**2)))
            summary['motion_max_joint_error_rad'] = float(np.max(np.abs(err)))
            summary['motion_samples'] = len(selected)
            measured = np.array([s[4:11] for s in self.samples], dtype=float)
            summary['any_measured_joint_out_of_limits'] = bool(np.any(
                (measured < Q_MIN[None, :] - 1e-6) |
                (measured > Q_MAX[None, :] + 1e-6)))
        if self.actual is not None:
            final_err = self.actual - Q_START
            summary['final_home_max_joint_error_rad'] = float(np.max(np.abs(final_err)))
            summary['final_home_joint_rmse_rad'] = float(np.sqrt(np.mean(final_err**2)))
            summary['final_measured_q_rad'] = self.actual.tolist()
        with json_path.open('w', encoding='utf-8') as handle:
            json.dump(summary, handle, indent=2)
        self.get_logger().info('FINAL METRICS: ' + json.dumps({
            k: v for k, v in summary.items() if k in
            ('mode', 'motion_joint_rmse_rad', 'motion_max_joint_error_rad',
             'final_home_max_joint_error_rad', 'any_measured_joint_out_of_limits')}, indent=2))
        self.get_logger().info(f'Saved {csv_path} and {json_path}')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', required=True, choices=('POSITION', 'VELOCITY'))
    p.add_argument('--cycles', type=int, default=4,
                   help='Number of forward/return cycles (default 4)')
    p.add_argument('--leg-duration', type=float, default=2.0,
                   help='Each forward or return motion, seconds (default 2.0)')
    p.add_argument('--hold', type=float, default=0.4,
                   help='Pause at each reversal, seconds (default 0.4)')
    p.add_argument('--final-settle', type=float, default=1.0,
                   help='Extra final settling time, seconds (default 1.0)')
    p.add_argument('--output', default='~/ros2_ws/rpc2_comparison_results')
    args = p.parse_args()
    if not (1 <= args.cycles <= 12):
        p.error('--cycles must be 1..12')
    for arg in ('leg_duration', 'hold', 'final_settle'):
        value = getattr(args, arg)
        if not (math.isfinite(value) and value >= TRAJ_DT):
            p.error(f'--{arg.replace("_", "-")} must be finite and >= {TRAJ_DT} s')
        if abs(value / TRAJ_DT - round(value / TRAJ_DT)) > 1e-8:
            p.error(f'--{arg.replace("_", "-")} must be a multiple of {TRAJ_DT} s')
    rclpy.init()
    node = None
    try:
        node = RepeatedComparison(args)
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=0.2)
        if node.failed:
            return 1
        return 0
    except KeyboardInterrupt:
        print('Demo interrupted; no results saved.')
        return 130
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    raise SystemExit(main())
