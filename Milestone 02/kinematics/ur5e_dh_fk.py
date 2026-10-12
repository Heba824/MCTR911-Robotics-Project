"""UR5e forward kinematics using standard Denavit-Hartenberg parameters.

Model: Milestone 01/Robot Model/ur5e.xml from the MCTR911 repository.
Frame {0}: MuJoCo world frame, with origin at the robot base.
Frame {6}: MuJoCo `attachment_site` (tool frame).

Usage:
    python3 ur5e_dh_fk.py
    python3 ur5e_dh_fk.py 0.3 0.1 -0.4 0.25 -0.1 0.9

Angles are in radians. Lengths are in meters.
"""

from __future__ import annotations

import argparse
import numpy as np

# (a_i [m], alpha_i [rad], d_i [m]) for STANDARD DH convention.
DH_CONSTANTS = (
    (0.0,     np.pi / 2, 0.163),
    (-0.425,  0.0,       0.0),
    (-0.392,  0.0,       0.0),
    (0.0,     np.pi / 2, 0.134),
    (0.0,    -np.pi / 2, 0.100),
    (0.0,     0.0,       0.100),
)

JOINT_NAMES = (
    "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
)


def standard_dh_transform(theta: float, d: float, a: float, alpha: float) -> np.ndarray:
    """Return Rz(theta) @ Tz(d) @ Tx(a) @ Rx(alpha)."""
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array([
        [ct, -st * ca,  st * sa, a * ct],
        [st,  ct * ca, -ct * sa, a * st],
        [0.,       sa,       ca,      d],
        [0.,      0.0,      0.0,    1.0],
    ], dtype=float)


def forward_kinematics(joint_angles: np.ndarray) -> np.ndarray:
    """Return world-to-attachment_site 4x4 pose for the six MJCF joint angles."""
    q = np.asarray(joint_angles, dtype=float)
    if q.shape != (6,):
        raise ValueError(f"Expected 6 joint angles, got {q.shape}")

    transform = np.eye(4)
    for theta, (a, alpha, d) in zip(q, DH_CONSTANTS):
        transform = transform @ standard_dh_transform(theta, d, a, alpha)
    return transform


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute UR5e attachment-site forward kinematics")
    parser.add_argument("joint_angles", nargs="*", type=float, help="Six joint angles in radians")
    args = parser.parse_args()
    if len(args.joint_angles) not in (0, 6):
        parser.error("Provide either no angles (zero pose) or exactly six angles")
    q = np.array(args.joint_angles if args.joint_angles else [0.0] * 6)
    T = forward_kinematics(q)
    np.set_printoptions(precision=6, suppress=True)
    print("Joint angles [rad]:", q)
    print("T_world_to_attachment_site =")
    print(T)
    print("Tool position [m]:", T[:3, 3])

    if np.allclose(q, 0):
        expected_position = np.array([-0.817, -0.234, 0.063])
        expected_rotation = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
        assert np.allclose(T[:3, 3], expected_position, atol=1e-10)
        assert np.allclose(T[:3, :3], expected_rotation, atol=1e-10)
        print("Zero-pose check: PASS (consistent with the supplied MJCF geometry)")


if __name__ == "__main__":
    main()
