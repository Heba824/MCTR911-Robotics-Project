"""Full-pose inverse kinematics for the MCTR911 UR5e Milestone 04 task-space path.

This module preserves the project's existing conventions:
- End-effector frame: MuJoCo ``attachment_site``.
- Pose: world-to-attachment_site homogeneous transform.
- Geometric Jacobian rows: [vx, vy, vz, wx, wy, wz], all in the world frame.
- Orientation error: world-frame rotation vector from Log(R_d R(q)^T).

The solver is iterative and uses a damped least-squares pseudoinverse with a
short backtracking line search. It is a kinematic solver only: it does not
perform collision checking, torque control, or time integration in MuJoCo.
"""
from __future__ import annotations

from dataclasses import dataclass
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
M2_DIR = ROOT / 'Milestone 02' / 'kinematics'
M3_DIR = ROOT / 'Milestone 03' / 'kinematics'
for folder in (M2_DIR, M3_DIR):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

from ur5e_dh_fk import JOINT_NAMES, forward_kinematics  # noqa: E402
from ur5e_jacobian import geometric_jacobian  # noqa: E402
from task_space_trajectory import so3_log  # noqa: E402

# Reuse the same broad limits used in the validated inverse-position solver.
JOINT_MIN = np.array([-6.2831, -6.2831, -3.1415, -6.2831, -6.2831, -6.2831], dtype=float)
JOINT_MAX = -JOINT_MIN
HOME = np.array([-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0.0], dtype=float)


@dataclass(frozen=True)
class PoseIKResult:
    joint_angles: np.ndarray
    success: bool
    iterations: int
    position_error_m: float
    orientation_error_rad: float
    weighted_error_norm: float
    min_singular_value: float
    message: str


def _validate_pose(pose: np.ndarray, name: str) -> np.ndarray:
    transform = np.asarray(pose, dtype=float)
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        raise ValueError(f'{name} must be a finite 4x4 homogeneous transform')
    if not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=1e-9):
        raise ValueError(f'{name} must have homogeneous bottom row [0,0,0,1]')
    rotation = transform[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-7):
        raise ValueError(f'{name} rotation must be orthonormal')
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-7):
        raise ValueError(f'{name} rotation must have determinant +1')
    return transform.copy()


def pose_error_world(current_pose_world: np.ndarray, target_pose_world: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return world-frame position and orientation errors.

    Position error:  p_d - p
    Orientation error: Log(R_d R^T), a 3-vector whose direction is the world-frame
    axis and whose norm is the smallest rotation angle [rad].
    """
    current = _validate_pose(current_pose_world, 'current_pose_world')
    target = _validate_pose(target_pose_world, 'target_pose_world')
    position_error = target[:3, 3] - current[:3, 3]
    rotation_error = so3_log(target[:3, :3] @ current[:3, :3].T)
    return position_error, rotation_error


def _weighted_error_vector(position_error: np.ndarray, orientation_error: np.ndarray,
                           position_weight: float, orientation_weight: float) -> np.ndarray:
    return np.concatenate((position_weight * position_error, orientation_weight * orientation_error))


def _weighted_jacobian(jacobian: np.ndarray, position_weight: float, orientation_weight: float) -> np.ndarray:
    weighted = jacobian.copy()
    weighted[:3, :] *= position_weight
    weighted[3:, :] *= orientation_weight
    return weighted


def _damped_least_squares_step(jacobian: np.ndarray, error_vector: np.ndarray, *,
                               damping: float, singular_value_cutoff: float) -> tuple[np.ndarray, float]:
    u, singular_values, vt = np.linalg.svd(jacobian, full_matrices=False)
    if damping > 0.0:
        gains = singular_values / (singular_values**2 + damping**2)
    else:
        gains = np.divide(1.0, singular_values, out=np.zeros_like(singular_values),
                          where=singular_values > singular_value_cutoff)
    delta_q = vt.T @ (gains * (u.T @ error_vector))
    minimum_singular = float(np.min(singular_values)) if singular_values.size else 0.0
    return delta_q, minimum_singular


def full_pose_inverse_kinematics(
    target_pose_world: np.ndarray,
    initial_q: np.ndarray = HOME,
    *,
    position_weight: float = 1.0,
    orientation_weight: float = 1.0,
    tolerance_position_m: float = 1e-5,
    tolerance_orientation_rad: float = 1e-5,
    max_iterations: int = 250,
    max_step_rad: float = 0.15,
    damping: float = 1e-4,
    singular_value_cutoff: float = 1e-10,
) -> PoseIKResult:
    """Solve T(q)=T_d using a DLS step and backtracking line search.

    The cost minimized by the line search is the weighted 6D pose-error norm.
    Joint angles are clipped to the configured UR5e bounds after each candidate step.
    """
    target = _validate_pose(target_pose_world, 'target_pose_world')
    q = np.asarray(initial_q, dtype=float).copy()
    if q.shape != (6,) or not np.all(np.isfinite(q)):
        raise ValueError('initial_q must contain six finite values')
    if np.any(q < JOINT_MIN) or np.any(q > JOINT_MAX):
        raise ValueError('initial_q is outside configured joint limits')
    for value, name in (
        (position_weight, 'position_weight'),
        (orientation_weight, 'orientation_weight'),
        (tolerance_position_m, 'tolerance_position_m'),
        (tolerance_orientation_rad, 'tolerance_orientation_rad'),
        (max_step_rad, 'max_step_rad'),
    ):
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f'{name} must be finite and > 0')
    if max_iterations < 1:
        raise ValueError('max_iterations must be >= 1')
    if not np.isfinite(damping) or damping < 0.0:
        raise ValueError('damping must be finite and >= 0')
    if not np.isfinite(singular_value_cutoff) or singular_value_cutoff < 0.0:
        raise ValueError('singular_value_cutoff must be finite and >= 0')

    best_min_sigma = 0.0
    for iteration in range(max_iterations + 1):
        current_pose = forward_kinematics(q)
        position_error, orientation_error = pose_error_world(current_pose, target)
        pos_norm = float(np.linalg.norm(position_error))
        ori_norm = float(np.linalg.norm(orientation_error))
        weighted_error = _weighted_error_vector(
            position_error, orientation_error, position_weight, orientation_weight
        )
        weighted_norm = float(np.linalg.norm(weighted_error))
        if pos_norm <= tolerance_position_m and ori_norm <= tolerance_orientation_rad:
            min_sigma = float(np.min(np.linalg.svd(geometric_jacobian(q), compute_uv=False)))
            return PoseIKResult(q, True, iteration, pos_norm, ori_norm, weighted_norm,
                                min_sigma, 'Converged')
        if iteration == max_iterations:
            break

        jacobian = geometric_jacobian(q)
        weighted_j = _weighted_jacobian(jacobian, position_weight, orientation_weight)
        delta_q, best_min_sigma = _damped_least_squares_step(
            weighted_j, weighted_error,
            damping=damping,
            singular_value_cutoff=singular_value_cutoff,
        )
        step_norm = float(np.linalg.norm(delta_q))
        if step_norm < 1e-14:
            return PoseIKResult(q, False, iteration, pos_norm, ori_norm, weighted_norm,
                                best_min_sigma, 'Stalled; step norm is near zero')
        if step_norm > max_step_rad:
            delta_q *= max_step_rad / step_norm

        accepted = False
        for fraction in (1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125):
            candidate = np.clip(q + fraction * delta_q, JOINT_MIN, JOINT_MAX)
            candidate_pose = forward_kinematics(candidate)
            p_err_c, r_err_c = pose_error_world(candidate_pose, target)
            weighted_c = _weighted_error_vector(p_err_c, r_err_c, position_weight, orientation_weight)
            if float(np.linalg.norm(weighted_c)) < weighted_norm - 1e-14:
                q = candidate
                accepted = True
                break
        if not accepted:
            return PoseIKResult(q, False, iteration, pos_norm, ori_norm, weighted_norm,
                                best_min_sigma,
                                'No improving DLS step; try another seed or different weights')

    final_pose = forward_kinematics(q)
    final_pos_err, final_ori_err = pose_error_world(final_pose, target)
    final_weighted = _weighted_error_vector(final_pos_err, final_ori_err, position_weight, orientation_weight)
    return PoseIKResult(
        q, False, max_iterations,
        float(np.linalg.norm(final_pos_err)),
        float(np.linalg.norm(final_ori_err)),
        float(np.linalg.norm(final_weighted)),
        best_min_sigma,
        'Maximum iterations reached; target may be unreachable from this seed',
    )


def main() -> None:
    np.set_printoptions(precision=6, suppress=True)
    q_target = np.array([-1.8708, -1.4108, 1.3908, -1.4508, -1.6708, -0.2])
    target_pose = forward_kinematics(q_target)
    result = full_pose_inverse_kinematics(target_pose, HOME)
    print('UR5e full-pose IK demo')
    print('Joint order:', ', '.join(JOINT_NAMES))
    print('Desired joint configuration used to generate target pose [rad]:', q_target)
    print('Recovered joint configuration [rad]:', np.round(result.joint_angles, 6))
    print(f'Position error [m]: {result.position_error_m:.3e}')
    print(f'Orientation error [rad]: {result.orientation_error_rad:.3e}')
    print(f'Min singular value of J [world frame]: {result.min_singular_value:.3e}')
    print(f'Iterations: {result.iterations} | {"PASS" if result.success else "FAIL"} | {result.message}')
    if not result.success:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
