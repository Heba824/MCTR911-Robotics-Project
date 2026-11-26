#!/usr/bin/env python3
"""Verify RPC2 mathematical functions without requiring a ROS process.

Usage: python3 verify_rpc2.py
Reads the neighbouring trajectory_node.py and executes only its mathematical
section; never launches the ROS2 infrastructure or commands the robot.
"""
from pathlib import Path
import numpy as np


def load_functions():
    source = Path(__file__).with_name('trajectory_node.py').read_text()
    math_code = source.split('#  EDIT ONLY THIS SECTION', 1)[1].split(
        '# INFRASTRUCTURE — DO NOT MODIFY BELOW', 1)[0]
    scope = {'np': np}
    exec(compile(math_code, 'trajectory_node.py (math section)', 'exec'), scope)
    return scope


def check(condition, message):
    if not condition:
        raise AssertionError(message)


def main():
    fn = load_functions()
    J = fn['compute_jacobian']
    Jdot = fn['compute_jacobian_dot']
    fk = fn['forward_kinematics']
    plan = fn['plan_trajectory']
    qmin, qmax = fn['Q_MIN'], fn['Q_MAX']
    rng = np.random.default_rng(2026)
    max_J = 0.0
    max_acc = 0.0

    # Independently differentiate the FK hand position and compare with the
    # geometric cross-product Jacobian. 40 legal configurations.
    for _ in range(40):
        q = rng.uniform(qmin + 0.03, qmax - 0.03)
        j = J(q)
        check(j.shape == (3, 7), 'Jacobian must be 3x7')
        h = 1e-6
        J_num = np.zeros((3, 7))
        for i in range(7):
            e = np.zeros(7)
            e[i] = h
            J_num[:, i] = (fk(q + e)[:3, 3] - fk(q - e)[:3, 3])/(2*h)
        max_J = max(max_J, float(np.max(np.abs(j - J_num))))
        check(np.allclose(j, J_num, atol=5e-7), 'Jacobian vs finite-difference FK')

        dq = rng.uniform(-0.5, 0.5, size=7)
        ddq = rng.uniform(-0.5, 0.5, size=7)
        jd = Jdot(q, dq)
        check(jd.shape == (3, 7), 'Jdot must be 3x7')
        check(np.all(np.isfinite(jd)), 'Jdot must contain finite values')
        # Independent second derivative of hand position along q(t).
        H = 0.001
        p_plus = fk(q + H*dq + 0.5*H*H*ddq)[:3, 3]
        p_minus = fk(q - H*dq + 0.5*H*H*ddq)[:3, 3]
        p0 = fk(q)[:3, 3]
        linear_acc_num = (p_plus - 2*p0 + p_minus)/(H*H)
        linear_acc_model = j @ ddq + jd @ dq
        max_acc = max(max_acc, float(np.max(np.abs(linear_acc_num - linear_acc_model))))
        check(np.allclose(linear_acc_num, linear_acc_model, atol=1e-4),
              'Acceleration formula: pdd = J*ddq + Jdot*dq')

    q_home = np.array([0., 0., 0., -1.57079, 0., 1.57079, -0.7853])
    q_end = np.array([0.12, -0.12, 0.12, -1.68, 0.08, 1.68, -0.68])
    t, q, dq, ddq = plan(q_home, q_end, 4.0, 0.05)
    check(len(t) == 81, '4 seconds at 0.05 => 81 samples')
    check(q.shape == dq.shape == ddq.shape == (81, 7), 'Trajectory arrays need equal (N,7) shapes')
    check(np.allclose(q[0], q_home), 'Start position incorrect')
    check(np.allclose(q[-1], q_end), 'End position incorrect')
    check(np.allclose(dq[[0, -1]], 0), 'Boundary velocities must be zero')
    check(np.allclose(ddq[[0, -1]], 0), 'Boundary accelerations must be zero')
    check(np.all(q >= qmin[None, :] - 1e-12), 'Joint minimum violated')
    check(np.all(q <= qmax[None, :] + 1e-12), 'Joint maximum violated')
    check(np.all(np.isfinite(q)) and np.all(np.isfinite(dq)) and np.all(np.isfinite(ddq)),
          'Non-finite trajectory')
    # Analytical derivatives must agree with centered finite differences.
    dq_fd = (q[2:] - q[:-2])/(t[2:, None]-t[:-2, None])
    ddq_fd = (dq[2:] - dq[:-2])/(t[2:, None]-t[:-2, None])
    check(np.max(np.abs(dq_fd - dq[1:-1])) < 5e-4, 'Quintic first derivative mismatch')
    check(np.max(np.abs(ddq_fd - ddq[1:-1])) < 5e-3, 'Quintic second derivative mismatch')
    check(np.allclose(Jdot(q_home, np.zeros(7)), 0), 'Jdot at dq=0 should be zero')

    # Initial zero vector has invalid joint4; reject, don't silently command it.
    try:
        plan(np.zeros(7), q_end, 4.0, 0.05)
    except ValueError:
        pass
    else:
        raise AssertionError('Invalid all-zero joint start should be rejected')

    print(f'PASS: 40 linear Jacobians match independent FK differences (max {max_J:.2e})')
    print(f'PASS: 40 acceleration checks pdd = J ddq + Jdot dq (max {max_acc:.2e})')
    print('PASS: Jdot shape, finite values and zero-velocity test')
    print('PASS: quintic trajectory shape (81,7), endpoints, and zero boundary velocities/accelerations')
    print('PASS: joint limits, analytic derivatives and invalid-start rejection')
    print('All RPC 2 mathematical checks passed. ROS/MuJoCo tracking not yet tested.')

if __name__ == '__main__':
    main()
