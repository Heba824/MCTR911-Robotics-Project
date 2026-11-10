"""UR5e forward and inverse VELOCITY kinematics for MCTR911 Milestone 03.

All quantities refer to MuJoCo's ``attachment_site`` origin and are expressed
in the world frame, matching the team's Milestone 02 DH convention.

Joint order:
  shoulder_pan_joint, shoulder_lift_joint, elbow_joint,
  wrist_1_joint, wrist_2_joint, wrist_3_joint.

Units:
  q [rad], qdot [rad/s], twist = [vx, vy, vz, wx, wy, wz]
  with v [m/s] and angular velocity omega [rad/s].

The inverse uses damped least-squares (DLS), NOT a direct matrix inverse:
  qdot = V diag(s / (s^2 + lambda^2)) U.T desired_twist
where J = U diag(s) V.T. The solution is least-squares; for an unreachable
instantaneous velocity, or a singular Jacobian, the residual may be nonzero.

Run from repo root:
    python 'Milestone 03/kinematics/ur5e_velocity_kinematics.py'
"""
from __future__ import annotations

import numpy as np

from ur5e_jacobian import JOINT_NAMES, geometric_jacobian


def _six_finite(values: np.ndarray, *, name: str) -> np.ndarray:
    out = np.asarray(values, dtype=float)
    if out.shape != (6,) or not np.all(np.isfinite(out)):
        raise ValueError(f"{name} must contain exactly six finite values, shape (6,)")
    return out


def forward_velocity(joint_angles: np.ndarray, joint_velocities: np.ndarray) -> np.ndarray:
    """Compute the world-frame 6D end-effector twist J(q) @ qdot.

    Returns [vx, vy, vz, wx, wy, wz] where v describes the attachment_site's
    origin and omega describes its angular velocity (both in world axes).
    """
    q = _six_finite(joint_angles, name="joint_angles")
    qdot = _six_finite(joint_velocities, name="joint_velocities")
    return geometric_jacobian(q) @ qdot


def inverse_velocity(
    joint_angles: np.ndarray,
    desired_twist: np.ndarray,
    *,
    damping: float = 1e-5,
    singular_value_cutoff: float = 1e-10,
) -> np.ndarray:
    """Calculate joint rates for a desired world-frame site twist via DLS.

    ``damping`` is in Jacobian singular-value units and must be >= 0.
    For damping=0 the Moore-Penrose pseudoinverse is used, with singular values
    <= ``singular_value_cutoff`` discarded. Damping>0 prevents uncontrolled
    amplification near singularities but trades off exact velocity tracking.

    No joint-speed/position limits are enforced: this is kinematics, not a
    motor controller. Check resulting joint speeds before commanding hardware.
    """
    q = _six_finite(joint_angles, name="joint_angles")
    target = _six_finite(desired_twist, name="desired_twist")
    if not np.isfinite(damping) or damping < 0:
        raise ValueError("damping must be finite and >= 0")
    if not np.isfinite(singular_value_cutoff) or singular_value_cutoff < 0:
        raise ValueError("singular_value_cutoff must be finite and >= 0")

    jac = geometric_jacobian(q)
    u, sigma, vt = np.linalg.svd(jac, full_matrices=False)
    if damping > 0:
        gains = sigma / (sigma**2 + damping**2)
    else:
        gains = np.divide(1.0, sigma, out=np.zeros_like(sigma),
                          where=sigma > singular_value_cutoff)
    return vt.T @ (gains * (u.T @ target))


def jacobian_singular_values(joint_angles: np.ndarray) -> np.ndarray:
    """Return sorted Jacobian singular values for singularity diagnostics."""
    q = _six_finite(joint_angles, name="joint_angles")
    return np.linalg.svd(geometric_jacobian(q), compute_uv=False)


def main() -> None:
    np.set_printoptions(precision=6, suppress=True)
    # An example non-home pose avoids assuming the home pose is nonsingular.
    q = np.array([0.3, -0.4, 0.2, 0.1, 0.5, 0.6])
    qdot = np.array([0.12, -0.08, 0.15, -0.05, 0.09, 0.07])
    twist = forward_velocity(q, qdot)
    recovered_qdot = inverse_velocity(q, twist)
    recovered_twist = forward_velocity(q, recovered_qdot)
    max_twist_error = float(np.max(np.abs(recovered_twist - twist)))
    sv = jacobian_singular_values(q)

    print("UR5e velocity kinematics demo")
    print("Joint order:", ", ".join(JOINT_NAMES))
    print("Joint positions q [rad]:", q)
    print("Joint velocities qdot [rad/s]:", qdot)
    print("Forward site linear velocity [m/s]:", twist[:3])
    print("Forward site angular velocity [rad/s]:", twist[3:])
    print("Inverse recovered joint velocities [rad/s]:", recovered_qdot)
    print("Minimum singular value of J:", f"{sv[-1]:.3e}")
    print("Maximum reconstructed twist error:", f"{max_twist_error:.3e}")
    if max_twist_error >= 5e-5:
        raise SystemExit("FAIL: inverse velocity could not reconstruct the forward twist")
    print("PASS: forward/inverse velocity round-trip")
    print("NOTE: The independent MuJoCo validation is in validation/validate_velocity_mujoco.py")


if __name__ == "__main__":
    main()
