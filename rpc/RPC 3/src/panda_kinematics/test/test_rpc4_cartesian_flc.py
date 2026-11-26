"""Cartesian velocity inference, redundancy, limits and singularity tests."""

import numpy as np

from panda_kinematics.cartesian_flc import (
    CARTESIAN_FIELDS, JOINT_SPEED, TASK_SPEED, CartesianController, CartesianVelocityFLC,
    damped_inverse, joint_velocity,
)
from panda_kinematics.joint_control import LIMITS
from panda_kinematics.trajectory_node import Q_MIN, Q_START, compute_jacobian, forward_kinematics


def test_three_fuzzy_channels_follow_error_and_remain_bounded():
    flc = CartesianVelocityFLC([1]*3, [1]*3, [.1]*3)
    output = flc([1, -1, 0], [0, 0, 0])
    np.testing.assert_allclose(output, np.array([1, -1, 0])*.1*5/6, atol=1e-14)
    assert np.max(abs(flc([100]*3, [100]*3))) <= .1


def test_cartesian_jacobian_and_null_space():
    jacobian = compute_jacobian(Q_START)
    step = 1e-6
    numerical = [
        (forward_kinematics(Q_START+dq)[:3, 3] - forward_kinematics(Q_START-dq)[:3, 3])
        / (2*step) for dq in np.eye(7)*step
    ]
    np.testing.assert_allclose(jacobian, np.array(numerical).T, atol=1e-8)
    _, null, _, _ = damped_inverse(jacobian)
    np.testing.assert_allclose(jacobian @ null, 0, atol=1e-14)


def test_damping_limits_a_near_singular_inverse_and_handles_zero_rank():
    jacobian = np.zeros((3, 7))
    jacobian[:, :3] = np.diag([1, .2, .00001])
    desired = np.array([0, 0, .1])
    inverse, _, _, damping = damped_inverse(jacobian)
    assert damping > .09
    assert np.linalg.norm(inverse @ desired) < .01
    assert np.linalg.norm(np.linalg.pinv(jacobian) @ desired) > 1000
    np.testing.assert_array_equal(damped_inverse(np.zeros((3, 7)))[0], np.zeros((7, 3)))


def test_braking_and_torque_limits():
    q = Q_START.copy()
    q[0] = Q_MIN[0]
    speeds, _, _, _ = joint_velocity(q, q, np.full(7, -5.), compute_jacobian(q),
                                     np.array([.3, .3, .3]))
    assert speeds[0] >= 0
    assert max(abs(speeds)) <= JOINT_SPEED

    class Dynamics:
        def terms(self, q, dq):
            return np.eye(7)*100, np.zeros(7), np.ones(7)

    profile = {}
    cmd = CartesianController(Dynamics()).compute(Q_START, np.ones(7), q, np.ones(7),
                                                 np.full(7, 50.), profile=profile)
    assert all(key in profile for key in CARTESIAN_FIELDS)
    assert np.linalg.norm([profile[f'cartesian_command_{axis}'] for axis in 'xyz']) <= TASK_SPEED
    np.testing.assert_array_equal(cmd.command, LIMITS)
    assert profile['cartesian_accel_limited'] == 7
    assert profile['cartesian_torque_limited'] == 7
