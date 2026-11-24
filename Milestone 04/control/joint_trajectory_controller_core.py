"""Pure validation and analysis helpers for the MCTR911 M4 ROS 2 controller."""
from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

JOINT_COUNT = 6
DEFAULT_JOINT_NAMES = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
)


def reorder_joint_state(
    names: Sequence[str],
    positions: Iterable[float],
    velocities: Iterable[float],
    expected_names: Sequence[str] = DEFAULT_JOINT_NAMES,
) -> tuple[np.ndarray, np.ndarray]:
    """Return position/velocity arrays in expected joint-name order.

    The ROS message must contain exactly the expected six joints. Reordering is
    explicit; silently assuming message order would risk commanding the wrong
    physical/simulated joint.
    """
    expected = tuple(str(name) for name in expected_names)
    received = tuple(str(name) for name in names)
    q = np.asarray(positions, dtype=float)
    qdot = np.asarray(velocities, dtype=float)

    if len(expected) != JOINT_COUNT or len(set(expected)) != JOINT_COUNT:
        raise ValueError("expected_names must contain six unique joint names")
    if len(received) != JOINT_COUNT:
        raise ValueError(f"JointState must contain exactly {JOINT_COUNT} names")
    if len(set(received)) != JOINT_COUNT:
        raise ValueError("JointState contains duplicate joint names")
    if q.shape != (JOINT_COUNT,) or qdot.shape != (JOINT_COUNT,):
        raise ValueError("JointState position and velocity arrays must each contain six values")
    if not np.all(np.isfinite(q)) or not np.all(np.isfinite(qdot)):
        raise ValueError("JointState contains non-finite position or velocity values")

    index = {name: i for i, name in enumerate(received)}
    missing = [name for name in expected if name not in index]
    unexpected = [name for name in received if name not in expected]
    if missing or unexpected:
        raise ValueError(
            f"JointState name mismatch; missing={missing}, unexpected={unexpected}"
        )

    order = np.asarray([index[name] for name in expected], dtype=int)
    return q[order].copy(), qdot[order].copy()


def validate_start_position(
    actual_position_rad: Iterable[float],
    expected_start_rad: Iterable[float],
    tolerance_rad: float,
) -> float:
    """Validate that the current robot pose is close enough to the CSV start.

    Returns the maximum absolute per-joint start error in radians.
    """
    actual = np.asarray(actual_position_rad, dtype=float)
    expected = np.asarray(expected_start_rad, dtype=float)
    tolerance = float(tolerance_rad)

    if actual.shape != (JOINT_COUNT,) or expected.shape != (JOINT_COUNT,):
        raise ValueError("Actual and expected start positions must each contain six values")
    if not np.all(np.isfinite(actual)) or not np.all(np.isfinite(expected)):
        raise ValueError("Start positions must contain only finite values")
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("Start tolerance must be positive and finite")

    maximum_error = float(np.max(np.abs(expected - actual)))
    if maximum_error > tolerance:
        raise ValueError(
            f"Start pose rejected: maximum joint error {maximum_error:.6f} rad "
            f"exceeds tolerance {tolerance:.6f} rad"
        )
    return maximum_error


def validate_start_velocity(
    actual_velocity_rad_s: Iterable[float],
    tolerance_rad_s: float,
) -> float:
    """Require the robot to be nearly stationary before trajectory playback.

    Returns the maximum absolute joint velocity in rad/s.
    """
    velocity = np.asarray(actual_velocity_rad_s, dtype=float)
    tolerance = float(tolerance_rad_s)
    if velocity.shape != (JOINT_COUNT,):
        raise ValueError("Actual start velocity must contain six values")
    if not np.all(np.isfinite(velocity)):
        raise ValueError("Start velocities must contain only finite values")
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("Start velocity tolerance must be positive and finite")

    maximum_velocity = float(np.max(np.abs(velocity)))
    if maximum_velocity > tolerance:
        raise ValueError(
            f"Start rejected: maximum joint velocity {maximum_velocity:.6f} rad/s "
            f"exceeds tolerance {tolerance:.6f} rad/s"
        )
    return maximum_velocity


def tracking_error_metrics(
    desired_positions_rad: Iterable[Iterable[float]],
    actual_positions_rad: Iterable[Iterable[float]],
    joint_names: Sequence[str] = DEFAULT_JOINT_NAMES,
) -> dict:
    """Compute per-joint and aggregate tracking-error statistics.

    Error convention is desired minus actual. Metrics are reported in radians.
    """
    desired = np.asarray(desired_positions_rad, dtype=float)
    actual = np.asarray(actual_positions_rad, dtype=float)
    names = tuple(str(name) for name in joint_names)

    if len(names) != JOINT_COUNT or len(set(names)) != JOINT_COUNT:
        raise ValueError("joint_names must contain six unique names")
    if desired.ndim != 2 or desired.shape[1:] != (JOINT_COUNT,):
        raise ValueError("desired_positions_rad must have shape (N, 6)")
    if actual.shape != desired.shape:
        raise ValueError("actual_positions_rad must have the same shape as desired")
    if desired.shape[0] == 0:
        raise ValueError("At least one tracking sample is required")
    if not np.all(np.isfinite(desired)) or not np.all(np.isfinite(actual)):
        raise ValueError("Tracking positions must contain only finite values")

    error = desired - actual
    absolute = np.abs(error)
    per_joint = {}
    for i, name in enumerate(names):
        per_joint[name] = {
            "rmse_rad": float(np.sqrt(np.mean(error[:, i] ** 2))),
            "mae_rad": float(np.mean(absolute[:, i])),
            "max_abs_error_rad": float(np.max(absolute[:, i])),
            "final_error_rad": float(error[-1, i]),
        }

    return {
        "overall": {
            "rmse_rad": float(np.sqrt(np.mean(error**2))),
            "mae_rad": float(np.mean(absolute)),
            "max_abs_error_rad": float(np.max(absolute)),
        },
        "per_joint": per_joint,
    }
