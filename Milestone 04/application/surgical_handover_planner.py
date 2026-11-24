"""Kinematic planning helpers for the MCTR911 surgical instrument handover.

This module is deliberately independent of ROS 2. It consumes world-frame
MuJoCo poses, calibrates the validated DH model against the live robot pose,
and produces sequential joint-space waypoint targets using the project's
existing full-pose IK and quintic trajectory utilities.

It does not publish commands, move the robot, perform collision checking, or
claim that a planned path is safe for execution.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import sys
from typing import Iterable

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for _folder in (
    PROJECT_ROOT / "Milestone 02" / "kinematics",
    PROJECT_ROOT / "Milestone 03" / "kinematics",
    PROJECT_ROOT / "Milestone 04" / "trajectory",
    PROJECT_ROOT / "Milestone 04" / "control",
):
    if str(_folder) not in sys.path:
        sys.path.insert(0, str(_folder))

from ur5e_dh_fk import forward_kinematics  # noqa: E402
from full_pose_ik import HOME, full_pose_inverse_kinematics  # noqa: E402
from joint_space_trajectory import QuinticJointTrajectory  # noqa: E402


JOINT_NAMES = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
)

DEFAULT_MAX_VELOCITY_RAD_S = 0.50
DEFAULT_MAX_ACCELERATION_RAD_S2 = 0.80
DEFAULT_MIN_SEGMENT_DURATION_S = 1.0


@dataclass(frozen=True)
class PlannedWaypoint:
    name: str
    position_world_m: tuple[float, float, float] | None
    joint_positions_rad: tuple[float, float, float, float, float, float]
    duration_from_previous_s: float
    ik_success: bool
    ik_iterations: int
    ik_position_error_m: float
    ik_orientation_error_rad: float
    ik_message: str


@dataclass(frozen=True)
class SurgicalHandoverPlan:
    instrument: str
    world_from_dh: tuple[tuple[float, ...], ...]
    initial_joint_positions_rad: tuple[float, ...]
    tool_from_instrument: tuple[tuple[float, ...], ...]
    desired_instrument_handover_pose_world: tuple[tuple[float, ...], ...]
    desired_tool_handover_pose_world: tuple[tuple[float, ...], ...]
    tool_orientation_policy: str
    waypoints: tuple[PlannedWaypoint, ...]

    def to_dict(self) -> dict:
        return {
            "instrument": self.instrument,
            "world_from_dh": [list(row) for row in self.world_from_dh],
            "initial_joint_positions_rad": list(self.initial_joint_positions_rad),
            "tool_from_instrument": [list(row) for row in self.tool_from_instrument],
            "desired_instrument_handover_pose_world": [
                list(row) for row in self.desired_instrument_handover_pose_world
            ],
            "desired_tool_handover_pose_world": [
                list(row) for row in self.desired_tool_handover_pose_world
            ],
            "tool_orientation_policy": self.tool_orientation_policy,
            "waypoints": [asdict(item) for item in self.waypoints],
            "planning_only": True,
            "motion_executed": False,
            "collision_check_performed": False,
        }


def _finite_vector(values: Iterable[float], size: int, name: str) -> np.ndarray:
    vector = np.asarray(values, dtype=float)
    if vector.shape != (size,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain exactly {size} finite values")
    return vector.copy()


def validate_transform(transform: np.ndarray, name: str) -> np.ndarray:
    matrix = np.asarray(transform, dtype=float)
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must be a finite 4x4 matrix")
    if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1e-8, rtol=0.0):
        raise ValueError(f"{name} must have homogeneous bottom row [0, 0, 0, 1]")
    rotation = matrix[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5, rtol=0.0):
        raise ValueError(f"{name} rotation is not orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-5, rtol=0.0):
        raise ValueError(f"{name} rotation determinant must be +1")
    return matrix.copy()


def quaternion_xyzw_to_rotation(quaternion_xyzw: Iterable[float]) -> np.ndarray:
    quaternion = _finite_vector(quaternion_xyzw, 4, "quaternion_xyzw")
    norm = float(np.linalg.norm(quaternion))
    if norm < 1e-12:
        raise ValueError("quaternion_xyzw must have non-zero norm")
    x, y, z, w = quaternion / norm
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=float,
    )


def pose_components_to_transform(
    position_xyz: Iterable[float], quaternion_xyzw: Iterable[float]
) -> np.ndarray:
    position = _finite_vector(position_xyz, 3, "position_xyz")
    transform = np.eye(4, dtype=float)
    transform[:3, :3] = quaternion_xyzw_to_rotation(quaternion_xyzw)
    transform[:3, 3] = position
    return validate_transform(transform, "pose transform")


def calibrate_world_from_dh(
    actual_joint_positions_rad: Iterable[float],
    actual_attachment_pose_world: np.ndarray,
) -> np.ndarray:
    """Estimate T_world_DH from simultaneous joint state and MuJoCo tool pose.

    T_world_attachment = T_world_DH @ T_DH_attachment(q), therefore
    T_world_DH = T_world_attachment @ inverse(T_DH_attachment(q)).
    """
    q = _finite_vector(actual_joint_positions_rad, 6, "actual_joint_positions_rad")
    actual_pose = validate_transform(actual_attachment_pose_world, "actual_attachment_pose_world")
    dh_pose = validate_transform(forward_kinematics(q), "DH forward-kinematics pose")
    world_from_dh = actual_pose @ np.linalg.inv(dh_pose)
    return validate_transform(world_from_dh, "calibrated world_from_dh")


def _pose_at(position: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    pose = np.eye(4, dtype=float)
    pose[:3, :3] = rotation
    pose[:3, 3] = _finite_vector(position, 3, "waypoint position")
    return validate_transform(pose, "waypoint pose")


def _segment_duration(
    q_start: np.ndarray, q_goal: np.ndarray,
    max_velocity_rad_s: float, max_acceleration_rad_s2: float,
) -> float:
    duration = QuinticJointTrajectory.minimum_duration_for_limits(
        q_start,
        q_goal,
        max_velocity_rad_s=max_velocity_rad_s,
        max_acceleration_rad_s2=max_acceleration_rad_s2,
    )
    return float(max(DEFAULT_MIN_SEGMENT_DURATION_S, duration))


def build_surgical_handover_plan(
    *,
    instrument: str,
    initial_joint_positions_rad: Iterable[float],
    pickup_pose_world: np.ndarray,
    handover_pose_world: np.ndarray,
    attachment_pose_world: np.ndarray,
    instrument_body_pose_world: np.ndarray,
    max_velocity_rad_s: float = DEFAULT_MAX_VELOCITY_RAD_S,
    max_acceleration_rad_s2: float = DEFAULT_MAX_ACCELERATION_RAD_S2,
    approach_clearance_position_world_m: Iterable[float] = (0.10, 0.20, 0.80),
) -> SurgicalHandoverPlan:
    """Build a tool-site path that carries the instrument body to the receiver.

    ``pickup_pose_world`` is the instrument's pickup-site pose, which the robot
    attachment_site must reach before the bridge accepts a grasp. The bridge
    captures the actual instrument-body transform relative to attachment_site.
    The desired handover pose, by contrast, describes the instrument body itself.
    This function therefore computes the full tool-to-instrument transform and
    compensates the tool target so the instrument body (not the tool site) lands
    at the receiver target.

    An elevated approach-clearance waypoint is inserted between HOME and the
    tray approach. It is a candidate route waypoint and must pass the separate
    MuJoCo collision preflight before any motion is authorized.
    """
    instrument_name = str(instrument).strip().lower()
    if instrument_name not in ("scalpel", "forceps"):
        raise ValueError("instrument must be 'scalpel' or 'forceps'")

    q_start = _finite_vector(initial_joint_positions_rad, 6, "initial_joint_positions_rad")
    pickup = validate_transform(pickup_pose_world, "pickup_pose_world")
    handover_instrument = validate_transform(handover_pose_world, "handover_pose_world")
    attachment = validate_transform(attachment_pose_world, "attachment_pose_world")
    instrument_body = validate_transform(instrument_body_pose_world, "instrument_body_pose_world")

    velocity_limit = float(max_velocity_rad_s)
    acceleration_limit = float(max_acceleration_rad_s2)
    if not np.isfinite(velocity_limit) or velocity_limit <= 0.0:
        raise ValueError("max_velocity_rad_s must be positive and finite")
    if not np.isfinite(acceleration_limit) or acceleration_limit <= 0.0:
        raise ValueError("max_acceleration_rad_s2 must be positive and finite")

    world_from_dh = calibrate_world_from_dh(q_start, attachment)
    dh_from_world = np.linalg.inv(world_from_dh)
    preserved_tool_rotation = attachment[:3, :3].copy()

    # At the pickup waypoint, the attachment_site is aligned with the live
    # instrument pickup site and retains its measured orientation. This is the
    # same relative transform that the MuJoCo bridge captures on attach=true.
    tool_pose_at_pickup = _pose_at(pickup[:3, 3], preserved_tool_rotation)
    tool_from_instrument = validate_transform(
        np.linalg.inv(tool_pose_at_pickup) @ instrument_body,
        "tool_from_instrument",
    )

    # The handover target is the desired instrument-body pose. Solve for the
    # attachment-site pose that makes T_world_tool * T_tool_instrument equal it.
    desired_tool_handover = validate_transform(
        handover_instrument @ np.linalg.inv(tool_from_instrument),
        "desired_tool_handover_pose_world",
    )

    pickup_position = pickup[:3, 3].copy()
    approach_clearance = _finite_vector(
        approach_clearance_position_world_m, 3, "approach_clearance_position_world_m"
    )
    pregrasp_position = pickup_position + np.array([0.0, 0.0, 0.10])
    lift_position = pickup_position + np.array([0.0, 0.0, 0.12])
    handover_tool_position = desired_tool_handover[:3, 3].copy()
    handover_tool_rotation = desired_tool_handover[:3, :3].copy()
    handover_approach_position = handover_tool_position + np.array([0.0, 0.0, 0.10])
    retreat_position = handover_tool_position + np.array([0.0, 0.0, 0.10])
    transfer_clearance_position = 0.5 * (pickup_position + handover_tool_position)
    transfer_clearance_position[2] = max(pickup_position[2], handover_tool_position[2]) + 0.20

    pose_targets = (
        ("approach_clearance", approach_clearance, preserved_tool_rotation),
        ("pregrasp", pregrasp_position, preserved_tool_rotation),
        ("pickup", pickup_position, preserved_tool_rotation),
        ("lift", lift_position, preserved_tool_rotation),
        ("transfer_clearance", transfer_clearance_position, preserved_tool_rotation),
        ("handover_approach", handover_approach_position, handover_tool_rotation),
        ("handover", handover_tool_position, handover_tool_rotation),
        ("retreat", retreat_position, handover_tool_rotation),
    )

    waypoints: list[PlannedWaypoint] = []
    previous_q = q_start.copy()
    for name, position_world, rotation_world in pose_targets:
        target_world = _pose_at(position_world, rotation_world)
        target_dh = validate_transform(dh_from_world @ target_world, f"{name} target in DH frame")
        result = full_pose_inverse_kinematics(
            target_dh,
            initial_q=previous_q,
            position_weight=1.0,
            orientation_weight=1.0,
            tolerance_position_m=2e-4,
            tolerance_orientation_rad=2e-3,
            max_iterations=500,
            max_step_rad=0.15,
            damping=1e-3,
        )
        if not result.success:
            raise RuntimeError(
                f"IK failed for waypoint '{name}': {result.message}; "
                f"position error={result.position_error_m:.6g} m; "
                f"orientation error={result.orientation_error_rad:.6g} rad"
            )
        goal_q = _finite_vector(result.joint_angles, 6, f"{name} joint solution")
        duration = _segment_duration(previous_q, goal_q, velocity_limit, acceleration_limit)
        waypoints.append(
            PlannedWaypoint(
                name=name,
                position_world_m=tuple(float(v) for v in position_world),
                joint_positions_rad=tuple(float(v) for v in goal_q),
                duration_from_previous_s=duration,
                ik_success=bool(result.success),
                ik_iterations=int(result.iterations),
                ik_position_error_m=float(result.position_error_m),
                ik_orientation_error_rad=float(result.orientation_error_rad),
                ik_message=str(result.message),
            )
        )
        previous_q = goal_q

    home_q = _finite_vector(HOME, 6, "HOME")
    home_duration = _segment_duration(previous_q, home_q, velocity_limit, acceleration_limit)
    waypoints.append(
        PlannedWaypoint(
            name="return_home",
            position_world_m=None,
            joint_positions_rad=tuple(float(v) for v in home_q),
            duration_from_previous_s=home_duration,
            ik_success=True,
            ik_iterations=0,
            ik_position_error_m=0.0,
            ik_orientation_error_rad=0.0,
            ik_message="Joint-space HOME target; no IK required",
        )
    )

    return SurgicalHandoverPlan(
        instrument=instrument_name,
        world_from_dh=tuple(tuple(float(v) for v in row) for row in world_from_dh),
        initial_joint_positions_rad=tuple(float(v) for v in q_start),
        tool_from_instrument=tuple(tuple(float(v) for v in row) for row in tool_from_instrument),
        desired_instrument_handover_pose_world=tuple(
            tuple(float(v) for v in row) for row in handover_instrument
        ),
        desired_tool_handover_pose_world=tuple(
            tuple(float(v) for v in row) for row in desired_tool_handover
        ),
        tool_orientation_policy=(
            "Capture the full instrument-body transform relative to attachment_site at pickup; "
            "compensate the handover attachment_site target so the instrument body reaches "
            "handover_target_site. Preserve the measured tool orientation where compatible."
        ),
        waypoints=tuple(waypoints),
    )


def validate_plan_against_actuator_ranges(
    plan: SurgicalHandoverPlan,
    lower_limits_rad: Iterable[float],
    upper_limits_rad: Iterable[float],
) -> None:
    lower = _finite_vector(lower_limits_rad, 6, "lower_limits_rad")
    upper = _finite_vector(upper_limits_rad, 6, "upper_limits_rad")
    if np.any(lower >= upper):
        raise ValueError("Each lower actuator limit must be below its upper limit")
    for waypoint in plan.waypoints:
        q = np.asarray(waypoint.joint_positions_rad, dtype=float)
        invalid = np.flatnonzero((q < lower) | (q > upper))
        if invalid.size:
            details = ", ".join(
                f"{JOINT_NAMES[i]}={q[i]:.5f} rad not in [{lower[i]:.5f}, {upper[i]:.5f}]"
                for i in invalid
            )
            raise ValueError(f"Waypoint '{waypoint.name}' violates actuator limits: {details}")
