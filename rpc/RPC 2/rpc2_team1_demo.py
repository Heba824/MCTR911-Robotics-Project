#!/usr/bin/env python3
"""RPC 2 Team 1 MuJoCo demonstration (external helper; no template edits).

Uses the already-verified plan_trajectory() from panda_kinematics.trajectory_node.
For each run, start a FRESH MuJoCo simulation in the corresponding control mode.

Example:
    python3 ~/ros2_ws/rpc2_team1_demo.py --mode VELOCITY
    python3 ~/ros2_ws/rpc2_team1_demo.py --mode POSITION --return-home

This is an 'as-if pick-and-place' reach-and-lift demonstration. It does not
simulate attaching, grasping, or carrying an object.
"""

import argparse
import math
import sys
import time

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray

from panda_kinematics.trajectory_node import (
    Q_START, Q_END, Q_MIN, Q_MAX, GRIPPER_CTRL,
    TRAJ_DURATION, TRAJ_DT, plan_trajectory, forward_kinematics,
)


class Team1Demo(Node):
    def __init__(self, mode: str, duration: float, hold: float, return_home: bool):
        super().__init__('rpc2_team1_demo')
        self.mode = mode
        self.duration = duration
        self.hold = hold
        self.return_home = return_home
        self.finished = False
        self.latest_q = None
        self.start_deadline = time.monotonic() + 15.0
        self.started = False
        self.segment_index = 0
        self.sample_index = 0
        self.hold_steps_remaining = 0
        self.phase = 'WAITING'
        self.last_status_time = 0.0
        self.segments = []

        for name, a, b in ([('reach and lift', Q_START, Q_END)] +
                           ([('return to neutral', Q_END, Q_START)] if return_home else [])):
            times, q, dq, ddq = plan_trajectory(a, b, duration, TRAJ_DT)
            assert q.shape == dq.shape == ddq.shape
            assert q.shape[1] == 7
            assert np.all(q >= Q_MIN - 1e-10) and np.all(q <= Q_MAX + 1e-10)
            self.segments.append((name, q, dq, b))

        # Optional topic isolation for running two simulators side by side.
        # With RPC2_TOPIC_NS unset, preserve the original single-window behavior.
        import os
        topic_ns = os.environ.get('RPC2_TOPIC_NS', '').strip('/')
        topic_prefix = f'/{topic_ns}/' if topic_ns else ''
        self.publisher = self.create_publisher(
            Float64MultiArray, topic_prefix + 'joint_commands', 10)
        self.subscription = self.create_subscription(
            JointState, topic_prefix + 'joint_state', self.on_joint_state, 10)
        self.timer = self.create_timer(TRAJ_DT, self.tick)
        self.get_logger().info(
            'TEAM 1 DEMO: mode=%s, forward motion=%.1fs, hold=%.1fs, return=%s' %
            (mode, duration, hold, return_home))
        self.get_logger().info('Start q (rad): ' + np.array2string(Q_START, precision=4))
        self.get_logger().info('Goal  q (rad): ' + np.array2string(Q_END, precision=4))
        start_xyz = forward_kinematics(Q_START)[:3, 3]
        end_xyz = forward_kinematics(Q_END)[:3, 3]
        self.get_logger().info('Hand displacement (cm): ' +
                               np.array2string(100 * (end_xyz - start_xyz), precision=2))

    def on_joint_state(self, msg: JointState):
        if len(msg.position) >= 7:
            self.latest_q = np.asarray(msg.position[:7], dtype=float)

    def publish_sample(self, values):
        cmd = Float64MultiArray()
        cmd.data = [float(v) for v in values] + [float(GRIPPER_CTRL), 0.0]
        self.publisher.publish(cmd)

    def hold_command(self, q_target):
        # Position mode holds target angles; velocity mode integrates nothing.
        values = q_target if self.mode == 'POSITION' else np.zeros(7)
        self.publish_sample(values)

    def report_endpoint(self, target):
        if self.latest_q is None:
            self.get_logger().warn('No joint state feedback at endpoint.')
            return
        err = np.abs(self.latest_q - target)
        self.get_logger().info('Measured q (rad): ' +
                               np.array2string(self.latest_q, precision=5))
        self.get_logger().info('Endpoint: max |q_error|=%.5f rad; RMSE=%.5f rad' %
                               (np.max(err), math.sqrt(np.mean(err ** 2))))

    def tick(self):
        if self.finished:
            return

        if not self.started:
            if self.latest_q is None:
                if time.monotonic() > self.start_deadline:
                    self.get_logger().error(
                        'No /joint_state received. Is MuJoCo running?')
                    self.finished = True
                return
            max_home_error = float(np.max(np.abs(self.latest_q - Q_START)))
            if max_home_error > 0.06:
                self.get_logger().error(
                    'Robot is not at HOME (max error %.3f rad). Stop/restart MuJoCo '
                    'to reset to the home keyframe before starting the demonstration.' %
                    max_home_error)
                self.finished = True
                return
            self.started = True
            self.phase = 'MOVING'
            self.get_logger().info('MuJoCo is ready. Beginning visible reach-and-lift.')

        name, q, dq, target = self.segments[self.segment_index]
        if self.phase == 'MOVING':
            if self.sample_index < len(q):
                values = q[self.sample_index] if self.mode == 'POSITION' else dq[self.sample_index]
                self.publish_sample(values)
                if self.sample_index % 20 == 0:
                    pct = 100 * self.sample_index / (len(q) - 1)
                    self.get_logger().info('%s: %.0f%%' % (name, pct))
                self.sample_index += 1
                return
            self.phase = 'HOLDING'
            self.hold_steps_remaining = max(1, int(round(self.hold / TRAJ_DT)))
            self.get_logger().info('Reached end of planned %s; holding target.' % name)

        if self.phase == 'HOLDING':
            self.hold_command(target)
            self.hold_steps_remaining -= 1
            if self.hold_steps_remaining > 0:
                return
            self.report_endpoint(target)
            self.segment_index += 1
            if self.segment_index >= len(self.segments):
                self.get_logger().info('Team 1 demonstration finished. MuJoCo remains open.')
                self.finished = True
                return
            self.sample_index = 0
            self.phase = 'MOVING'
            self.get_logger().info('Beginning return-to-home replay.')


def main():
    parser = argparse.ArgumentParser(description='RPC2 Team 1 animated MuJoCo reach-and-lift')
    parser.add_argument('--mode', choices=['POSITION', 'VELOCITY'], required=True,
                        help='Must match control_mode in mujoco_ros2/launch/example.py')
    parser.add_argument('--duration', type=float, default=TRAJ_DURATION,
                        help='Duration of each leg, seconds (default matches assignment: 4.0)')
    parser.add_argument('--hold', type=float, default=2.0,
                        help='Seconds to hold/observe each endpoint (default: 2.0)')
    parser.add_argument('--return-home', action='store_true',
                        help='After the Team 1 reach, replay the reverse trajectory to home')
    args = parser.parse_args()

    if not (math.isfinite(args.duration) and args.duration >= TRAJ_DT and
            math.isfinite(args.hold) and args.hold >= TRAJ_DT):
        parser.error('--duration and --hold must be finite and at least 0.05 seconds')
    if abs(args.duration / TRAJ_DT - round(args.duration / TRAJ_DT)) > 1e-9:
        parser.error('--duration must be a multiple of 0.05 seconds')

    rclpy.init()
    node = None
    try:
        node = Team1Demo(args.mode, args.duration, args.hold, args.return_home)
        while rclpy.ok() and not node.finished:
            rclpy.spin_once(node, timeout_sec=0.3)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
