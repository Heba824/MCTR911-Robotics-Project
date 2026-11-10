"""UR5e forward and inverse ACCELERATION kinematics — MCTR911 Milestone 03.

This module extends the team's validated Milestone 02 standard-DH forward
kinematics and Milestone 03 6x6 geometric Jacobian. It neither changes nor
reimplements the team's earlier source files.

All 6D quantities describe MuJoCo's ``attachment_site`` origin and are
expressed in the WORLD frame:
    xddot = [ax, ay, az, alphax, alphay, alphaz]
where a has units m/s^2 and alpha has units rad/s^2. Joint order is the
original JOINT_NAMES order; q, qdot, qddot are radians, rad/s, rad/s^2.

Equations:
    twist = J(q) @ qdot
    twist_dot = J(q) @ qddot + Jdot(q, qdot) @ qdot
    qddot = J_dls_pinv @ (desired_twist_dot - Jdot @ qdot)

The 6D twist derivative is [linear acceleration, angular acceleration], NOT
an Euler-angle acceleration vector. The damping-based inverse is a
least-squares kinematic solution, not a dynamic model or motor controller.

Run from the repository root:
    python 'Milestone 03/kinematics/ur5e_acceleration_kinematics.py'
"""
from __future__ import annotations

import numpy as np

from ur5e_jacobian import (
    DH_CONSTANTS,
    JOINT_NAMES,
    geometric_jacobian,
    standard_dh_transform,
)


def _six_finite(values: np.ndarray, *, name: str) -> np.ndarray:
    result = np.asarray(values, dtype=float)
    if result.shape != (6,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain exactly six finite values, shape (6,)")
    return result


def jacobian_time_derivative(
    joint_angles: np.ndarray, joint_velocities: np.ndarray
) -> np.ndarray:
    """Analytical time derivative dJ(q(t))/dt in the world frame, shape (6, 6).

    Each revolute joint i has axis z_i and origin o_i, both in world frame.
    From the geometric Jacobian:
        Jv_i   = z_i x (p - o_i)
        Jw_i   = z_i
        Jvdot_i = zdot_i x (p - o_i) + z_i x (pdot - odot_i)
        Jwdot_i = zdot_i

    For the earlier joints k < i:
        omega_before_i = sum_k z_k * qdot_k
        zdot_i  = omega_before_i x z_i
        odot_i  = sum_k (z_k x (o_i - o_k)) * qdot_k

    `p` is the origin of the UR5e attachment_site. No extra tool offset or
    implicit base transform is introduced.
    """
    q = _six_finite(joint_angles, name="joint_angles")
    qdot = _six_finite(joint_velocities, name="joint_velocities")

    transform = np.eye(4)
    origins = np.empty((6, 3), dtype=float)
    axes = np.empty((6, 3), dtype=float)
    for i, (theta, (a, alpha, d)) in enumerate(zip(q, DH_CONSTANTS)):
        origins[i] = transform[:3, 3]
        axes[i] = transform[:3, 2]
        transform = transform @ standard_dh_transform(theta, d, a, alpha)

    p = transform[:3, 3]
    p_dot = geometric_jacobian(q)[:3] @ qdot
    jdot = np.zeros((6, 6), dtype=float)

    for i in range(6):
        z = axes[i]
        o = origins[i]
        angular_velocity_before = axes[:i].T @ qdot[:i]
        z_dot = np.cross(angular_velocity_before, z)

        o_dot = np.zeros(3, dtype=float)
        for k in range(i):
            o_dot += np.cross(axes[k], o - origins[k]) * qdot[k]

        jdot[:3, i] = np.cross(z_dot, p - o) + np.cross(z, p_dot - o_dot)
        jdot[3:, i] = z_dot
    return jdot


def finite_difference_jacobian_time_derivative(
    joint_angles: np.ndarray,
    joint_velocities: np.ndarray,
    *,
    time_step: float = 1e-4,
) -> np.ndarray:
    """Independent central-difference dJ/dt check at q +/- h*qdot.

    This check calls the existing geometric_jacobian(), rather than the
    analytical derivative above. It is designed for validation, not runtime.
    """
    q = _six_finite(joint_angles, name="joint_angles")
    qdot = _six_finite(joint_velocities, name="joint_velocities")
    if not np.isfinite(time_step) or time_step <= 0:
        raise ValueError("time_step must be positive and finite")
    return (
        geometric_jacobian(q + time_step * qdot)
        - geometric_jacobian(q - time_step * qdot)
    ) / (2.0 * time_step)


def forward_acceleration(
    joint_angles: np.ndarray,
    joint_velocities: np.ndarray,
    joint_accelerations: np.ndarray,
) -> np.ndarray:
    """Return world-frame [ax, ay, az, alphax, alphay, alphaz]."""
    q = _six_finite(joint_angles, name="joint_angles")
    qdot = _six_finite(joint_velocities, name="joint_velocities")
    qddot = _six_finite(joint_accelerations, name="joint_accelerations")
    return geometric_jacobian(q) @ qddot + jacobian_time_derivative(q, qdot) @ qdot


def inverse_acceleration(
    joint_angles: np.ndarray,
    joint_velocities: np.ndarray,
    desired_acceleration: np.ndarray,
    *,
    damping: float = 1e-5,
    singular_value_cutoff: float = 1e-10,
) -> np.ndarray:
    """Compute a DLS least-squares qddot for the requested 6D acceleration.

    Like inverse_velocity(), this uses SVD and does NOT apply motor torque,
    joint acceleration or collision limits. The desired acceleration might
    be unattainable at singular configurations; always examine the residual
    in forward_acceleration() before commanding a robot.
    """
    q = _six_finite(joint_angles, name="joint_angles")
    qdot = _six_finite(joint_velocities, name="joint_velocities")
    desired = _six_finite(desired_acceleration, name="desired_acceleration")
    if not np.isfinite(damping) or damping < 0:
        raise ValueError("damping must be finite and >= 0")
    if not np.isfinite(singular_value_cutoff) or singular_value_cutoff < 0:
        raise ValueError("singular_value_cutoff must be finite and >= 0")

    j = geometric_jacobian(q)
    rhs = desired - jacobian_time_derivative(q, qdot) @ qdot
    u, singular_values, vt = np.linalg.svd(j, full_matrices=False)
    if damping > 0:
        gains = singular_values / (singular_values**2 + damping**2)
    else:
        gains = np.divide(
            1.0, singular_values, out=np.zeros_like(singular_values),
            where=singular_values > singular_value_cutoff,
        )
    return vt.T @ (gains * (u.T @ rhs))


def main() -> None:
    np.set_printoptions(precision=6, suppress=True)
    q = np.array([0.3, -0.4, 0.2, 0.1, 0.5, 0.6])
    qdot = np.array([0.12, -0.08, 0.15, -0.05, 0.09, 0.07])
    qddot = np.array([0.04, -0.06, 0.03, 0.05, -0.02, 0.07])

    jd_analytic = jacobian_time_derivative(q, qdot)
    jd_numeric = finite_difference_jacobian_time_derivative(q, qdot)
    jdot_error = float(np.max(np.abs(jd_analytic - jd_numeric)))

    accel = forward_acceleration(q, qdot, qddot)
    recovered_qddot = inverse_acceleration(q, qdot, accel)
    recovered_accel = forward_acceleration(q, qdot, recovered_qddot)
    accel_error = float(np.max(np.abs(recovered_accel - accel)))

    print("UR5e acceleration kinematics demo")
    print("Joint order:", ", ".join(JOINT_NAMES))
    print("q [rad]:", q)
    print("qdot [rad/s]:", qdot)
    print("qddot [rad/s^2]:", qddot)
    print("Linear acceleration [m/s^2]:", accel[:3])
    print("Angular acceleration [rad/s^2]:", accel[3:])
    print("Recovered qddot [rad/s^2]:", recovered_qddot)
    print(f"Max analytical-vs-numerical Jdot error: {jdot_error:.3e}")
    print(f"Max reconstructed acceleration error: {accel_error:.3e}")
    if jdot_error >= 5e-6 or accel_error >= 5e-5:
        raise SystemExit("FAIL: acceleration or Jacobian derivative check")
    print("PASS: analytical Jdot and forward/inverse acceleration round-trip")
    print("NOTE: Independent MuJoCo test is in validation/validate_acceleration_mujoco.py")


if __name__ == "__main__":
    main()
