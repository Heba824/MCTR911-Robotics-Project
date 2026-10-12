"""UR5e inverse POSITION kinematics via the Jacobian transpose method.

Uses the same STANDARD DH convention and joint order as ur5e_dh_fk.py.
This is a numerical gradient/Jacobian-transpose method; it is NOT an exact
Newton-Raphson inverse-Jacobian step.

Given target position p_d and current end-effector position p(q):
    e = p_d - p(q)
    q_next = q + eta * J(q).T @ e
where J is the 3x6 position Jacobian derived from the DH chain.

Requires NumPy and ur5e_dh_fk.py in the same directory.
Does not constrain tool orientation, plan collision-free paths, or move MuJoCo.

Run:
    python ur5e_ik_transpose.py
    python ur5e_ik_transpose.py --target -0.133998 0.491999 0.488 \
        --seed -1.3 -1.4 1.4 -1.4 -1.4 -0.2
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import numpy as np
from ur5e_dh_fk import DH_CONSTANTS, forward_kinematics, standard_dh_transform

# Consistent with prior IK implementation, based on approximate MJCF joint bounds.
JOINT_MIN = np.array([-6.2831, -6.2831, -3.1415, -6.2831, -6.2831, -6.2831])
JOINT_MAX = -JOINT_MIN
HOME = np.array([-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0.0])

@dataclass
class IKResult:
    joint_angles: np.ndarray
    success: bool
    iterations: int
    position_error_m: float
    message: str


def position_and_jacobian(q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute end-effector position and 3x6 position Jacobian from standard DH.

    For revolute joint i: J_i = z_(i-1) x (p_tool - o_(i-1)).
    """
    q = np.asarray(q, dtype=float)
    if q.shape != (6,):
        raise ValueError("Expected six joint angles")

    transform = np.eye(4)
    joint_origins, joint_axes = [], []
    for theta, (a, alpha, d) in zip(q, DH_CONSTANTS):
        joint_origins.append(transform[:3, 3].copy())
        joint_axes.append(transform[:3, 2].copy())
        transform = transform @ standard_dh_transform(theta, d, a, alpha)

    position = transform[:3, 3].copy()
    J = np.column_stack([
        np.cross(z, position - o) for z, o in zip(joint_axes, joint_origins)
    ])
    return position, J


def inverse_position_kinematics(
    target_xyz: np.ndarray,
    initial_q: np.ndarray = HOME,
    *,
    gain: float = 1.0,
    tolerance_m: float = 1e-5,
    max_iterations: int = 3000,
    max_step_rad: float = 0.20,
) -> IKResult:
    """Solve for a joint configuration reaching target XYZ (orientation unrestricted).

    Uses q += gain * J.T @ (target - current), with step limiting and
    backtracking if necessary to decrease the Cartesian position error.
    No Jacobian inversion or pseudoinverse is used.
    """
    target = np.asarray(target_xyz, dtype=float)
    q = np.asarray(initial_q, dtype=float).copy()
    if target.shape != (3,) or q.shape != (6,):
        raise ValueError("Expected target (3,) and initial_q (6,)")
    if not np.all(np.isfinite(target)) or not np.all(np.isfinite(q)):
        raise ValueError("Inputs must be finite")
    if np.any(q < JOINT_MIN) or np.any(q > JOINT_MAX):
        raise ValueError("Initial joint angles are outside configured bounds")
    if gain <= 0 or not np.isfinite(gain) or tolerance_m <= 0 or max_iterations < 1 or max_step_rad <= 0:
        raise ValueError("Gain, tolerance, max iterations, step size must be positive")

    for iteration in range(max_iterations + 1):
        current, J = position_and_jacobian(q)
        error_vector = target - current
        error_norm = float(np.linalg.norm(error_vector))
        if error_norm <= tolerance_m:
            return IKResult(q, True, iteration, error_norm, "Converged")
        if iteration == max_iterations:
            break

        # Pure Jacobian transpose correction (no inverse / solve / pseudoinverse).
        delta_q = gain * J.T @ error_vector
        step_norm = float(np.linalg.norm(delta_q))
        if step_norm > max_step_rad:
            delta_q *= max_step_rad / step_norm
        if step_norm < 1e-14:
            return IKResult(q, False, iteration, error_norm,
                            "Stalled; try another initial guess")

        # A short line search only reduces the step size when needed.
        # It does not change the Jacobian-transpose method.
        accepted = False
        for fraction in (1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625):
            candidate = np.clip(q + fraction * delta_q, JOINT_MIN, JOINT_MAX)
            candidate_position = forward_kinematics(candidate)[:3, 3]
            candidate_error = np.linalg.norm(target - candidate_position)
            if candidate_error < error_norm - 1e-14:
                q = candidate
                accepted = True
                break
        if not accepted:
            return IKResult(q, False, iteration, error_norm,
                            "No improving step; try another seed or lower gain")

    final_position = forward_kinematics(q)[:3, 3]
    final_error = float(np.linalg.norm(target - final_position))
    return IKResult(q, False, max_iterations, final_error,
                    "Maximum iterations reached; try another seed")


def show_result(name: str, target_xyz: np.ndarray, seed: np.ndarray, gain: float) -> bool:
    result = inverse_position_kinematics(target_xyz, seed, gain=gain)
    reached = forward_kinematics(result.joint_angles)[:3, 3]
    print(f"\n{name}")
    print("Target position [m]:    ", np.round(target_xyz, 6))
    print("Initial joint angles:   ", np.round(seed, 6))
    print("IK solution angles [rad]:", np.round(result.joint_angles, 6))
    print("FK(IK solution) [m]:    ", np.round(reached, 6))
    print(f"Position error [m]:    {result.position_error_m:.3e}")
    print(f"Iterations: {result.iterations} | {'PASS' if result.success else 'FAIL'} | {result.message}")
    return result.success


def main() -> None:
    parser = argparse.ArgumentParser(description="UR5e Jacobian TRANSPOSE inverse POSITION kinematics")
    parser.add_argument("--target", nargs=3, type=float, metavar=("X", "Y", "Z"),
                        help="Desired XYZ in MuJoCo world frame [m]")
    parser.add_argument("--seed", nargs=6, type=float, metavar=("Q1", "Q2", "Q3", "Q4", "Q5", "Q6"),
                        help="Initial joint angles [rad]; defaults to home")
    parser.add_argument("--gain", type=float, default=1.0,
                        help="Jacobian transpose gain (default: 1.0)")
    args = parser.parse_args()
    np.set_printoptions(precision=6, suppress=True)

    if args.target is not None:
        target = np.asarray(args.target)
        seed = np.asarray(args.seed) if args.seed is not None else HOME.copy()
        if not show_result("USER TARGET", target, seed, args.gain):
            raise SystemExit(1)
    else:
        tests = [
            ("Test A", np.array([-1.1, -0.9, 1.15, -0.85, -1.2, 0.3]), HOME),
            ("Test B", np.array([0.35, -0.6, 0.7, 0.2, 0.65, -0.4]),
             np.array([0.2, -0.3, 0.4, 0.1, 0.3, -0.2])),
            ("Home target", HOME, np.array([-1.3, -1.4, 1.4, -1.4, -1.4, -0.2])),
        ]
        failures = 0
        for name, target_q, seed in tests:
            target = forward_kinematics(target_q)[:3, 3]
            failures += not show_result(name, target, seed, args.gain)
        if failures:
            print(f"\n{failures} IK TEST(S) FAILED")
            raise SystemExit(1)
        print("\nALL 3 IK TESTS PASSED")

if __name__ == "__main__":
    main()
