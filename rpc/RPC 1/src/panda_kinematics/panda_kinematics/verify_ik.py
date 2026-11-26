#!/usr/bin/env python3
"""Standalone numerical tests for the RPC 1 Panda IK implementation.

Run inside the sourced ROS 2 Jazzy workspace:
    python3 verify_ik.py
This test requires rclpy, sensor_msgs, std_msgs, and NumPy.
It does not control the robot. Live MuJoCo testing is a separate step.
"""
import numpy as np
from kinematics import (
    Q_MIN, Q_MAX, TARGET_POSE, _pose_error,
    forward_kinematics, inverse_kinematics,
)


def check_target(target, seed, label):
    solution, ok = inverse_kinematics(target, seed)
    assert solution.shape == (7,), f'{label}: wrong output shape'
    assert np.all(solution >= Q_MIN - 1e-12), f'{label}: below joint limits'
    assert np.all(solution <= Q_MAX + 1e-12), f'{label}: above joint limits'
    assert ok, f'{label}: solver did not converge'
    pe, re = _pose_error(target, forward_kinematics(solution))
    p_error = np.linalg.norm(pe)
    r_error = np.linalg.norm(re)
    assert p_error < 0.002, f'{label}: position error {p_error} m'
    assert r_error < 0.01, f'{label}: orientation error {r_error} rad'
    return p_error, r_error


def main():
    home = np.array([0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853])
    p, r = check_target(TARGET_POSE, home, 'assignment target from home')
    print(f'PASS: selected reachable target from home ({p*1000:.3f} mm, {r:.5f} rad)')

    p, r = check_target(TARGET_POSE, np.zeros(7), 'assignment target from zero seed')
    print(f'PASS: selected target from zero seed ({p*1000:.3f} mm, {r:.5f} rad)')

    # Joint 5 = 0 is a wrist-axis singularity mentioned by the RPC PDF.
    singular_q = np.array([0.3, -0.4, 0.7, -1.8, 0.0, 1.4, 0.2])
    p, r = check_target(forward_kinematics(singular_q), singular_q + 0.1,
                        'singular target (joint 5 zero)')
    print(f'PASS: near-singularity target ({p*1000:.3f} mm, {r:.5f} rad)')

    # Check the two full transform components on 50 other reachable poses.
    rng = np.random.default_rng(20261014)
    for i in range(50):
        q = rng.uniform(Q_MIN + 0.1, Q_MAX - 0.1)
        seed = np.clip(q + rng.normal(0.0, 0.25, size=7), Q_MIN, Q_MAX)
        check_target(forward_kinematics(q), seed, f'random reachable {i}')
    print('PASS: 50 generated reachable poses from nearby seeds')

    unreachable = np.eye(4)
    unreachable[:3, 3] = [5.0, 0.0, 1.0]
    invalid_seed = np.zeros(7)  # joint 4 is out of range
    solution, ok = inverse_kinematics(unreachable, invalid_seed)
    assert not ok, 'Unreachable target was incorrectly accepted'
    assert np.all(solution >= Q_MIN) and np.all(solution <= Q_MAX)
    print('PASS: unreachable target rejected, returned vector in joint limits')

    print('All numerical checks passed. Live MuJoCo pose verification is still required.')


if __name__ == '__main__':
    main()
