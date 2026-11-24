"""Safety gates and trajectory helpers for the MCTR911 surgical handover executor.

This module is deliberately independent of ROS 2. It validates the planner and
collision-preflight artifacts and reconstructs exactly the same sequential
quintic point-to-point segments used by the collision preflight.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAJECTORY_DIR = PROJECT_ROOT / "Milestone 04" / "trajectory"
if str(TRAJECTORY_DIR) not in sys.path:
    sys.path.insert(0, str(TRAJECTORY_DIR))

from joint_space_trajectory import QuinticJointTrajectory  # noqa: E402

JOINT_NAMES = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
)

REQUIRED_WAYPOINTS = (
    "approach_clearance",
    "pregrasp",
    "pickup",
    "lift",
    "transfer_clearance",
    "handover_approach",
    "handover",
    "retreat",
    "return_home",
)


class ExecutionValidationError(RuntimeError):
    """Raised when a plan or preflight artifact is not eligible for execution."""


@dataclass(frozen=True)
class ExecutionSegment:
    name: str
    start_position_rad: np.ndarray
    goal_position_rad: np.ndarray
    duration_s: float

    def trajectory(self) -> QuinticJointTrajectory:
        return QuinticJointTrajectory(
            self.start_position_rad,
            self.goal_position_rad,
            self.duration_s,
        )


def _six(values: Iterable[float], name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.shape != (6,) or not np.all(np.isfinite(array)):
        raise ExecutionValidationError(f"{name} must contain exactly six finite values")
    return array.copy()


def _transform(value: Any, name: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=float)
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise ExecutionValidationError(f"{name} must be a finite 4x4 transform")
    if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1e-8, rtol=0.0):
        raise ExecutionValidationError(f"{name} has an invalid homogeneous bottom row")
    rotation = matrix[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5, rtol=0.0):
        raise ExecutionValidationError(f"{name} rotation is not orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-5, rtol=0.0):
        raise ExecutionValidationError(f"{name} rotation determinant must be +1")
    return matrix.copy()


def load_execution_plan(path: str | Path, expected_instrument: str | None = None) -> dict[str, Any]:
    plan_path = Path(path).expanduser().resolve()
    if not plan_path.is_file():
        raise FileNotFoundError(f"Plan JSON does not exist: {plan_path}")
    try:
        payload = json.loads(plan_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ExecutionValidationError(f"Could not read plan JSON: {exc}") from exc

    instrument = str(payload.get("instrument", "")).strip().lower()
    if instrument not in ("scalpel", "forceps"):
        raise ExecutionValidationError("Plan instrument must be scalpel or forceps")
    if expected_instrument is not None and instrument != expected_instrument:
        raise ExecutionValidationError(
            f"Plan instrument is {instrument!r}, expected {expected_instrument!r}"
        )
    if payload.get("planning_only") is not True:
        raise ExecutionValidationError("Plan is not marked as a planning-only artifact")
    if payload.get("motion_executed") is not False:
        raise ExecutionValidationError("Plan metadata unexpectedly reports executed motion")
    if payload.get("collision_check_performed") is not False:
        raise ExecutionValidationError(
            "Plan metadata is inconsistent: collision checking belongs to the separate preflight"
        )

    initial = _six(payload.get("initial_joint_positions_rad", []), "initial_joint_positions_rad")
    waypoints = payload.get("waypoints")
    if not isinstance(waypoints, list) or len(waypoints) != len(REQUIRED_WAYPOINTS):
        raise ExecutionValidationError(
            f"Plan must contain exactly {len(REQUIRED_WAYPOINTS)} waypoints"
        )
    names = tuple(str(item.get("name", "")) for item in waypoints)
    if names != REQUIRED_WAYPOINTS:
        raise ExecutionValidationError(f"Unexpected waypoint sequence: {names}")

    lower = payload.get("actuator_lower_limits_rad")
    upper = payload.get("actuator_upper_limits_rad")
    if lower is not None or upper is not None:
        lo = _six(lower, "actuator_lower_limits_rad")
        hi = _six(upper, "actuator_upper_limits_rad")
        if np.any(lo >= hi):
            raise ExecutionValidationError("Plan actuator limits are invalid")
    else:
        lo = hi = None

    for item in waypoints:
        if item.get("ik_success") is not True:
            raise ExecutionValidationError(f"Waypoint {item.get('name')} did not pass IK")
        goal = _six(item.get("joint_positions_rad", []), f"{item.get('name')}.joint_positions_rad")
        duration = float(item.get("duration_from_previous_s", 0.0))
        if not math.isfinite(duration) or duration <= 0.0:
            raise ExecutionValidationError(f"Waypoint {item.get('name')} has invalid duration")
        if lo is not None and (np.any(goal < lo - 1e-9) or np.any(goal > hi + 1e-9)):
            raise ExecutionValidationError(f"Waypoint {item.get('name')} violates actuator limits")

    _transform(payload.get("world_from_dh"), "world_from_dh")
    _transform(payload.get("tool_from_instrument"), "tool_from_instrument")
    _transform(
        payload.get("desired_instrument_handover_pose_world"),
        "desired_instrument_handover_pose_world",
    )
    _transform(
        payload.get("desired_tool_handover_pose_world"),
        "desired_tool_handover_pose_world",
    )
    _transform(payload.get("instrument_body_initial_pose_world"), "instrument_body_initial_pose_world")

    payload["_resolved_plan_path"] = str(plan_path)
    payload["_initial_q"] = initial
    return payload


def build_execution_segments(plan: dict[str, Any]) -> tuple[ExecutionSegment, ...]:
    current = _six(plan.get("_initial_q", plan.get("initial_joint_positions_rad", [])), "initial q")
    segments: list[ExecutionSegment] = []
    for waypoint in plan["waypoints"]:
        goal = _six(waypoint["joint_positions_rad"], f"{waypoint['name']}.joint_positions_rad")
        duration = float(waypoint["duration_from_previous_s"])
        segments.append(
            ExecutionSegment(
                name=str(waypoint["name"]),
                start_position_rad=current.copy(),
                goal_position_rad=goal.copy(),
                duration_s=duration,
            )
        )
        current = goal
    return tuple(segments)


def validate_preflight_for_execution(
    preflight: dict[str, Any],
    *,
    plan: dict[str, Any],
    plan_path: str | Path,
    scene_path: str | Path,
    approve_warnings: bool = False,
) -> dict[str, Any]:
    """Fail closed unless the matching native-CCD preflight is clean.

    WARNING_REVIEW_REQUIRED may proceed only when the operator explicitly opts
    in with --approve-warnings. Even then, collisions, joint-limit violations,
    pickup-support violations, or clearance below the hard minimum are fatal.
    """
    instrument = str(plan["instrument"])
    if preflight.get("instrument") != instrument:
        raise ExecutionValidationError("Preflight instrument does not match the plan")

    expected_plan_path = str(Path(plan_path).expanduser().resolve())
    expected_scene_path = str(Path(scene_path).expanduser().resolve())
    if str(Path(str(preflight.get("plan_json", ""))).expanduser().resolve()) != expected_plan_path:
        raise ExecutionValidationError("Preflight was generated for a different plan JSON")
    if str(Path(str(preflight.get("scene_xml", ""))).expanduser().resolve()) != expected_scene_path:
        raise ExecutionValidationError("Preflight was generated for a different scene XML")
    if preflight.get("scene_xml_sha256") != hashlib.sha256(Path(expected_scene_path).read_bytes()).hexdigest():
        raise ExecutionValidationError("Preflight is missing a matching scene XML digest; rerun preflight")
    plan_scene = plan.get("scene_xml")
    if plan_scene and str(Path(str(plan_scene)).expanduser().resolve()) != expected_scene_path:
        raise ExecutionValidationError("Plan scene path does not match the requested scene XML")

    dimensions = preflight.get("model_dimensions", {})
    if (dimensions.get("nq"), dimensions.get("nv"), dimensions.get("nu")) != (6, 6, 6):
        raise ExecutionValidationError("Preflight does not describe the expected 6-DoF/6-actuator model")
    backend = preflight.get("collision_distance_backend", {})
    if backend.get("name") != "MuJoCo native CCD" or backend.get("native_ccd_enabled") is not True:
        raise ExecutionValidationError("Preflight did not confirm MuJoCo native CCD")

    trajectory = preflight.get("trajectory", {})
    if tuple(trajectory.get("segments", ())) != REQUIRED_WAYPOINTS:
        raise ExecutionValidationError("Preflight waypoint sequence does not match the execution plan")
    sample_period = float(trajectory.get("sample_period_s", math.nan))
    if not math.isfinite(sample_period) or sample_period > 0.005 + 1e-12:
        raise ExecutionValidationError("Preflight sampling period must be 5 ms or faster")
    expected_duration = sum(float(item["duration_from_previous_s"]) for item in plan["waypoints"])
    actual_duration = float(trajectory.get("duration_s", math.nan))
    if not math.isfinite(actual_duration) or not math.isclose(
        actual_duration, expected_duration, rel_tol=0.0, abs_tol=1e-6
    ):
        raise ExecutionValidationError("Preflight trajectory duration does not match the plan")
    pickup_index = REQUIRED_WAYPOINTS.index("pickup")
    handover_index = REQUIRED_WAYPOINTS.index("handover")
    expected_attach_time = sum(
        float(item["duration_from_previous_s"]) for item in plan["waypoints"][: pickup_index + 1]
    )
    expected_release_time = sum(
        float(item["duration_from_previous_s"]) for item in plan["waypoints"][: handover_index + 1]
    )
    for field, expected_time in (
        ("pickup_attachment_time_s", expected_attach_time),
        ("handover_release_time_s", expected_release_time),
    ):
        actual_time = float(trajectory.get(field, math.nan))
        if not math.isfinite(actual_time) or not math.isclose(
            actual_time, expected_time, rel_tol=0.0, abs_tol=1e-6
        ):
            raise ExecutionValidationError(f"Preflight {field} does not match the plan")

    thresholds = preflight.get("clearance_thresholds", {})
    minimum_required = float(thresholds.get("minimum_clearance_m", math.nan))
    warning_threshold = float(thresholds.get("warning_clearance_m", math.nan))
    if not math.isfinite(minimum_required) or minimum_required < 0.02 - 1e-9:
        raise ExecutionValidationError("Preflight hard minimum clearance is missing or below 20 mm")
    distance_cap = float(thresholds.get("distance_cap_m", math.nan))
    if not math.isfinite(distance_cap) or distance_cap < warning_threshold:
        raise ExecutionValidationError("Preflight distance cap is below the warning threshold")
    if not math.isfinite(warning_threshold) or warning_threshold <= minimum_required:
        raise ExecutionValidationError("Preflight warning threshold is invalid")

    if int(preflight.get("collision_event_count", -1)) != 0:
        raise ExecutionValidationError("Preflight reports collision events")
    if int(preflight.get("joint_limit_violation_count", -1)) != 0:
        raise ExecutionValidationError("Preflight reports joint-limit violations")
    support = preflight.get("pickup_support_contact", {})
    if support.get("cleared") is not True or support.get("clearance_violation") is not False:
        raise ExecutionValidationError("Preflight pickup-support clearance did not pass")
    if support.get("signed_support_distances_checked_for_penetration") is not True:
        raise ExecutionValidationError("Preflight did not verify signed pickup-support distances; rerun preflight")
    if support.get("penetration_allowed") is not False:
        raise ExecutionValidationError("Preflight pickup-support policy must forbid penetration")
    if support.get("allowed_only_during_pickup_and_lift") is not True:
        raise ExecutionValidationError("Preflight pickup-support allowance has an unexpected phase scope")
    expected_supports = {"tray_base", f"tray_{instrument}_pad", "trolley_top"}
    actual_supports = set(support.get("support_geometries", []))
    if actual_supports != expected_supports:
        raise ExecutionValidationError(
            f"Preflight pickup-support envelope is unexpected: {sorted(actual_supports)}"
        )

    minima = preflight.get("minimum_clearances", {})
    minimum_values: dict[str, float] = {}
    for category in ("robot_environment", "robot_self", "carried_instrument_environment"):
        entry = minima.get(category, {})
        value = entry.get("minimum_distance_m")
        if value is None:
            if entry.get("minimum_distance_is_capped_lower_bound") is True:
                value = entry.get("lower_bound_m")
            else:
                raise ExecutionValidationError(f"Preflight has no minimum clearance for {category}")
        value = float(value)
        if not math.isfinite(value) or value < minimum_required - 1e-9:
            raise ExecutionValidationError(
                f"Preflight {category} clearance {value:.6f} m is below the hard minimum"
            )
        minimum_values[category] = value

    status = str(preflight.get("status", ""))
    warnings = list(preflight.get("closest_pairs_below_warning", []))
    if status == "PASS" and preflight.get("preflight_passed") is not True:
        raise ExecutionValidationError("Preflight PASS status conflicts with preflight_passed metadata")
    if status == "WARNING_REVIEW_REQUIRED" and preflight.get("preflight_passed") is not False:
        raise ExecutionValidationError("Preflight warning status conflicts with preflight_passed metadata")
    if status == "PASS":
        if min(minimum_values.values()) < warning_threshold - 1e-9:
            raise ExecutionValidationError("Preflight status PASS conflicts with its warning threshold")
    elif status == "WARNING_REVIEW_REQUIRED":
        if min(minimum_values.values()) >= warning_threshold - 1e-9:
            raise ExecutionValidationError("Warning status conflicts with all clearances meeting the warning threshold")
        if not warnings:
            raise ExecutionValidationError("Warning status has no review-pair details")
    else:
        raise ExecutionValidationError(f"Preflight status {status!r} is not eligible for execution")

    return {
        "status": status,
        "minimum_required_clearance_m": minimum_required,
        "warning_clearance_m": warning_threshold,
        "distance_cap_m": distance_cap,
        "minimum_clearances_m": minimum_values,
        "review_pairs": warnings,
        "warning_review_approved": bool(status == "WARNING_REVIEW_REQUIRED" and approve_warnings),
    }


def transform_error(actual: np.ndarray, expected: np.ndarray) -> tuple[float, float]:
    """Return translation error [m] and shortest rotation error [rad]."""
    a = _transform(actual, "actual pose")
    e = _transform(expected, "expected pose")
    position_error = float(np.linalg.norm(a[:3, 3] - e[:3, 3]))
    relative = a[:3, :3].T @ e[:3, :3]
    cosine = float(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0))
    orientation_error = float(math.acos(cosine))
    return position_error, orientation_error


def validate_pickup_geometry(
    actual_pickup_site: np.ndarray,
    expected_pickup_site: np.ndarray,
    expected_tool_pickup: np.ndarray,
    *,
    site_position_tolerance_m: float,
    site_orientation_tolerance_rad: float,
    tool_site_position_tolerance_m: float,
    actual_tool_pose: np.ndarray | None = None,
    tool_position_tolerance_m: float | None = None,
    tool_orientation_tolerance_rad: float | None = None,
) -> dict[str, float]:
    """Validate pickup localization without confusing two different frames.

    The instrument pickup-site pose and robot attachment-site pose share the
    planned pickup position, but their orientations do not have to be equal.
    The planner deliberately preserves a reachable robot-tool orientation and
    the bridge captures the resulting full tool-to-instrument transform.
    Therefore this function checks pickup-site drift against its own planned
    pose, checks positional coincidence with the planned tool target, and—at
    grasp time—checks the actual tool against its planned tool pose.
    """
    actual_site = _transform(actual_pickup_site, "actual_pickup_site")
    planned_site = _transform(expected_pickup_site, "expected_pickup_site")
    planned_tool = _transform(expected_tool_pickup, "expected_tool_pickup")

    site_position_error, site_orientation_error = transform_error(actual_site, planned_site)
    if (
        site_position_error > float(site_position_tolerance_m)
        or site_orientation_error > float(site_orientation_tolerance_rad)
    ):
        raise ExecutionValidationError(
            "Instrument pickup site moved since planning: "
            f"position error={site_position_error:.4f} m, "
            f"orientation error={math.degrees(site_orientation_error):.2f} deg"
        )

    tool_site_position_error = float(
        np.linalg.norm(planned_tool[:3, 3] - actual_site[:3, 3])
    )
    if tool_site_position_error > float(tool_site_position_tolerance_m):
        raise ExecutionValidationError(
            "Planned attachment_site target does not align with the instrument pickup position: "
            f"position error={tool_site_position_error:.4f} m"
        )

    # This angular difference is a valid grasp transform component, not a
    # localization failure. The bridge captures it when attach=true is called.
    tool_site_orientation_offset = transform_error(planned_tool, actual_site)[1]

    metrics = {
        "pickup_site_drift_position_m": site_position_error,
        "pickup_site_drift_orientation_rad": site_orientation_error,
        "tool_to_pickup_site_position_error_m": tool_site_position_error,
        "planned_tool_to_pickup_site_orientation_offset_rad": tool_site_orientation_offset,
    }

    if actual_tool_pose is not None:
        actual_tool = _transform(actual_tool_pose, "actual_tool_pose")
        tool_position_error, tool_orientation_error = transform_error(actual_tool, planned_tool)
        if tool_position_tolerance_m is None or tool_orientation_tolerance_rad is None:
            raise ValueError("Tool-pose tolerances are required when actual_tool_pose is supplied")
        if (
            tool_position_error > float(tool_position_tolerance_m)
            or tool_orientation_error > float(tool_orientation_tolerance_rad)
        ):
            raise ExecutionValidationError(
                "Robot attachment_site did not reach its planned pickup pose: "
                f"position error={tool_position_error:.4f} m, "
                f"orientation error={math.degrees(tool_orientation_error):.2f} deg"
            )
        metrics["actual_tool_position_error_m"] = tool_position_error
        metrics["actual_tool_orientation_error_rad"] = tool_orientation_error

    return metrics


def instrument_body_pose_from_pickup_site(
    world_from_pickup_site: np.ndarray,
    instrument_from_pickup_site: np.ndarray,
) -> np.ndarray:
    """Recover instrument-body world pose from its published pickup-site pose."""
    world_from_site = _transform(world_from_pickup_site, "world_from_pickup_site")
    body_from_site = _transform(instrument_from_pickup_site, "instrument_from_pickup_site")
    return _transform(world_from_site @ np.linalg.inv(body_from_site), "world_from_instrument")
