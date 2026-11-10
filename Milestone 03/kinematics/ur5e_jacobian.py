"""Full 6x6 geometric Jacobian for the existing MCTR911 UR5e model.

Convention:
  q = [shoulder_pan, shoulder_lift, elbow, wrist_1, wrist_2, wrist_3]
  J(q) @ qdot = [vx, vy, vz, wx, wy, wz]
  Linear velocity is for MuJoCo's ``attachment_site`` origin.
  Both linear and angular velocities are expressed in the world frame.
  Joint angles are radians; linear dimensions are meters.

This deliberately reuses the *validated* standard-DH implementation from
Milestone 02. No extra tool transform or robot-base offset is introduced.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Existing code is a script directory, not a Python package.
_MILESTONE_2_KINEMATICS = (
    Path(__file__).resolve().parents[2] / "Milestone 02" / "kinematics"
)
if not _MILESTONE_2_KINEMATICS.is_dir():
    raise FileNotFoundError(
        f"Cannot find existing Milestone 02 kinematics: {_MILESTONE_2_KINEMATICS}"
    )
if str(_MILESTONE_2_KINEMATICS) not in sys.path:
    sys.path.insert(0, str(_MILESTONE_2_KINEMATICS))

from ur5e_dh_fk import DH_CONSTANTS, JOINT_NAMES, forward_kinematics, standard_dh_transform


def geometric_jacobian(joint_angles: np.ndarray) -> np.ndarray:
    """Return the UR5e attachment-site geometric Jacobian, shape (6, 6).

    For revolute joint i (using the standard-DH frames):
        J_linear[:, i]  = z_(i-1) x (p_tool - o_(i-1))
        J_angular[:, i] = z_(i-1)

    The rows are [vx, vy, vz, wx, wy, wz], in world coordinates.
    """
    q = np.asarray(joint_angles, dtype=float)
    if q.shape != (6,) or not np.all(np.isfinite(q)):
        raise ValueError("Expected six finite joint angles with shape (6,)")

    transform = np.eye(4)
    joint_origins: list[np.ndarray] = []
    joint_axes: list[np.ndarray] = []

    for theta, (a, alpha, d) in zip(q, DH_CONSTANTS):
        # Joint i rotates around the z-axis of frame i-1.
        joint_origins.append(transform[:3, 3].copy())
        joint_axes.append(transform[:3, 2].copy())
        transform = transform @ standard_dh_transform(theta, d, a, alpha)

    tool_position = transform[:3, 3]
    j_linear = np.column_stack([
        np.cross(z, tool_position - origin)
        for z, origin in zip(joint_axes, joint_origins)
    ])
    j_angular = np.column_stack(joint_axes)
    return np.vstack((j_linear, j_angular))


def finite_difference_jacobian(
    joint_angles: np.ndarray, *, step_rad: float = 1e-6
) -> np.ndarray:
    """Independent numerical FK derivative to cross-check the analytic J.

    Angular columns use Rdot @ R.T so angular velocity is in the WORLD frame.
    This is a validation helper, not the recommended runtime algorithm.
    """
    q = np.asarray(joint_angles, dtype=float)
    if q.shape != (6,) or not np.all(np.isfinite(q)):
        raise ValueError("Expected six finite joint angles with shape (6,)")
    if not np.isfinite(step_rad) or step_rad <= 0:
        raise ValueError("step_rad must be positive and finite")

    rotation = forward_kinematics(q)[:3, :3]
    jacobian = np.zeros((6, 6))
    for i in range(6):
        offset = np.zeros(6)
        offset[i] = step_rad
        plus = forward_kinematics(q + offset)
        minus = forward_kinematics(q - offset)
        jacobian[:3, i] = (plus[:3, 3] - minus[:3, 3]) / (2.0 * step_rad)

        r_dot = (plus[:3, :3] - minus[:3, :3]) / (2.0 * step_rad)
        skew = r_dot @ rotation.T
        skew = 0.5 * (skew - skew.T)
        jacobian[3:, i] = [skew[2, 1], skew[0, 2], skew[1, 0]]
    return jacobian


def main() -> None:
    q = np.array([-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0.0])
    analytic = geometric_jacobian(q)
    numeric = finite_difference_jacobian(q)
    error = np.max(np.abs(analytic - numeric))
    np.set_printoptions(precision=6, suppress=True)
    print("UR5e full 6x6 geometric Jacobian at HOME:")
    print("Joint order:", ", ".join(JOINT_NAMES))
    print(analytic)
    print(f"Max analytic-vs-finite-difference error: {error:.3e}")
    if error >= 5e-6:
        raise SystemExit("FAIL: Jacobian does not match FK numerical derivative")
    print("PASS: Jacobian matches the numerical derivative of Milestone 2 FK")


if __name__ == "__main__":
    main()
