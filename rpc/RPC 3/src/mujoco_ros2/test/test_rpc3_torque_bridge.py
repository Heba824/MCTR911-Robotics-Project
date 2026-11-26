"""Headless ROS integration checks. Run with the built bridge and Jazzy sourced."""

import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import subprocess
import time

from ament_index_python.packages import get_package_prefix
import numpy as np
import rclpy
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray, String

from panda_kinematics.dynamics import PandaDynamics
from panda_kinematics.panda_model import DEFAULT_SCENE, PandaModel


@contextmanager
def bridge(scene, mode):
    executable = Path(get_package_prefix('mujoco_ros2')) / 'lib/mujoco_ros2/mujoco_node'
    process = subprocess.Popen([
        str(executable), str(scene),
        '--ros-args', '-r', '__ns:=/rpc3_test',
        '-p', f'control_mode:={mode}', '-p', 'headless:=true',
    ])
    node = rclpy.create_node('smoke_test', namespace='/rpc3_test')
    try:
        yield node, process
    finally:
        node.destroy_node()
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


class Client:
    def __init__(self, node, process):
        self.node, self.process = node, process
        self.states, self.statuses = [], []
        self.state_sub = node.create_subscription(JointState, 'joint_state', self.states.append, 20)
        self.status_sub = node.create_subscription(
            String, 'torque_status', self.record_status, 20)
        self.commands = node.create_publisher(Float64MultiArray, 'joint_commands', 10)
        self.wait(lambda: self.states and self.commands.get_subscription_count(), 10)

    def record_status(self, msg):
        if not self.statuses or self.statuses[-1] != msg.data:
            self.statuses.append(msg.data)
            self.status_changed = time.monotonic()

    def wait(self, condition, timeout=2):
        end = time.monotonic() + timeout
        while not condition():
            assert self.process.poll() is None, f'Bridge exited with {self.process.returncode}'
            assert time.monotonic() < end, 'Timed out waiting for bridge observation'
            rclpy.spin_once(self.node, timeout_sec=0.005)

    def arm(self):
        return np.asarray(self.states[-1].position[:7])

    def effort(self):
        return np.asarray(self.states[-1].effort[:7])

    def send(self, command):
        self.commands.publish(Float64MultiArray(data=list(map(float, command))))

    def stream(self, command, seconds):
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            self.send(command)
            next_send = time.monotonic() + 0.01
            self.wait(lambda: time.monotonic() >= next_send, 1)


def test_torque(scene):
    dynamics = PandaDynamics(PandaModel.from_scene(scene))
    with bridge(scene, 'TORQUE') as (node, process):
        client = Client(node, process)
        client.wait(lambda: 'waiting' in client.statuses)
        home = client.arm().copy()
        np.testing.assert_array_equal(client.states[-1].name[:7], [f'joint{i}' for i in range(1, 8)])
        client.stream([0] * 8, 0.15)
        assert client.statuses[-1] == 'waiting'
        np.testing.assert_array_equal(client.arm(), home)

        gravity = dynamics.gravity(home)
        hold = [*gravity, 255, 0]
        start = len(client.states)
        client.stream(hold, 2.0)
        states = client.states[start:]
        deviations = np.max(np.abs(np.array([state.position[:7] for state in states]) - home), axis=0)
        assert max(deviations) < 0.01, deviations
        stamps = [state.header.stamp.sec + state.header.stamp.nanosec * 1e-9 for state in states]
        rate = (len(stamps) - 1) / (stamps[-1] - stamps[0])
        np.testing.assert_allclose(client.effort(), gravity, rtol=0, atol=1e-10)

        pulses = []
        for i in range(7):
            for sign in (1, -1):
                command = gravity.copy()
                command[i] += sign * 0.5
                before = client.states[-1].velocity[i]
                client.send([*command, 255, 0])
                client.wait(lambda: np.allclose(client.effort(), command, atol=1e-10, rtol=0))
                after = client.states[-1].velocity[i]
                assert sign * (after - before) > 0, (i, sign, before, after)
                pulses.append([i + 1, sign, command[i], client.effort()[i], after - before])
                client.stream(hold, 0.04)

        # Accept both exact signed limits, then recover immediately.
        limits = np.array([87, 87, 87, 87, 12, 12, 12])
        for sign in (1, -1):
            client.send([*(sign * limits), 255, 0])
            client.wait(lambda: np.array_equal(client.effort(), sign * limits))
            client.stream(hold, 0.03)

        invalid = [[0] * 8, [0] * 10, [np.nan, *hold[1:]],
                   [*hold[:6], 12.001, 0, 0], [*hold[:7], 256, 0],
                   [*hold[:7], 0, 1], [np.inf, *hold[1:]]]
        for packet in invalid:
            client.stream(hold, 0.02)
            start = len(client.states)
            client.send(packet)
            client.wait(lambda: len(client.states) >= start + 3)
            for state in client.states[start:]:
                np.testing.assert_allclose(state.effort[:7], gravity, atol=1e-10, rtol=0)
                assert min(state.position[7:]) > 0.035

        client.stream([*gravity, 0, 0], 0.25)
        fingers = client.states[-1].position[7:]
        assert max(fingers) < 0.025, fingers
        np.testing.assert_allclose(client.effort(), gravity, atol=1e-10, rtol=0)
        client.send([0] * 9)
        client.wait(lambda: np.array_equal(client.effort(), np.zeros(7)))
        start = time.monotonic()
        client.stream(invalid[0], 0.15)
        assert client.statuses[-1] == 'timed_out'
        watchdog_delay = client.status_changed - start
        assert 0.06 < watchdog_delay < 0.15, watchdog_delay
        frozen = client.arm().copy()
        client.stream(hold, 0.15)
        assert client.statuses[-1] == 'timed_out'
        np.testing.assert_array_equal(client.arm(), frozen)
        np.testing.assert_array_equal(client.effort(), np.zeros(7))
        print(json.dumps({
            'home_peak_rad_by_joint': deviations.tolist(), 'hold_state_rate_hz': rate,
            'hold_requested_Nm': gravity.tolist(),
            'signed_pulses_joint_sign_requested_applied_delta_velocity': pulses,
            'accepted_limits_Nm': limits.tolist(), 'rejected_packet_cases': len(invalid),
            'closed_finger_positions_m': list(fingers),
            'watchdog_after_zero_effort_s': watchdog_delay,
            'watchdog_events': sum(a != b and b == 'timed_out'
                                   for a, b in zip(client.statuses, client.statuses[1:])),
        }, indent=2))


def test_legacy(scene, mode):
    with bridge(scene, mode) as (node, process):
        client = Client(node, process)
        home = client.arm().copy()
        target = home.copy()
        target[0] += 0.08
        if mode == 'POSITION':
            client.send([*target, 255, 0])
        else:
            for _ in range(8):
                client.send([0.2, 0, 0, 0, 0, 0, 0, 255, 0])
                deadline = time.monotonic() + 0.05
                client.wait(lambda: time.monotonic() >= deadline)
            client.send([0, 0, 0, 0, 0, 0, 0, 255, 0])
        deadline = time.monotonic() + 1
        client.wait(lambda: time.monotonic() >= deadline)
        actual = client.arm()
        np.testing.assert_allclose(actual, target, atol=0.02, rtol=0)
        assert not client.statuses
        print(json.dumps({'legacy_mode': mode, 'target_rad': target.tolist(),
                          'actual_rad': actual.tolist(),
                          'peak_error_rad': float(np.max(abs(actual - target)))}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene', default=str(DEFAULT_SCENE))
    args = parser.parse_args()
    rclpy.init(args=[])
    try:
        test_torque(args.scene)
        for mode in ('POSITION', 'VELOCITY'):
            test_legacy(args.scene, mode)
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()
