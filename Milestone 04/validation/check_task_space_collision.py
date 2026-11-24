#!/usr/bin/env python3
"""Kinematic collision preflight for the MCTR911 UR5e task-space reference.

The scene's operating-room geometries are intentionally excluded from MuJoCo's
normal contact pipeline (contype=0, conaffinity=0).  This utility therefore
uses mj_geomDistance directly for every UR5e collision-proxy/environment-geom
pair.  That distance query is independent of the normal contact filtering
pipeline.

This is a sampled kinematic geometry sweep, not a dynamic simulation or a
safety-rated collision monitor.  It checks the simplified UR5e collision
proxies in the supplied MJCF scene.  The robot_station body is excluded because
it is the intended mounting support for the robot base.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CONTROL_DIR = ROOT / "Milestone 04" / "control"
KINEMATICS_DIR = ROOT / "Milestone 02" / "kinematics"
for _directory in (CONTROL_DIR, KINEMATICS_DIR):
    if str(_directory) not in sys.path:
        sys.path.insert(0, str(_directory))

from trajectory_csv import JointTrajectoryCSV  # noqa: E402
from ur5e_dh_fk import JOINT_NAMES  # noqa: E402

DEFAULT_SCENE = ROOT / "Milestone 01" / "Robot Model" / "scene_surgical_m3.xml"
DEFAULT_TRAJECTORY = (
    ROOT / "Milestone 04" / "Results" / "Task_Space" / "task_space_joint_reference.csv"
)
DEFAULT_OUTPUT_DIR = ROOT / "Milestone 04" / "Results" / "Collision_Preflight"


@dataclass(frozen=True)
class GeometrySelection:
    robot_geom_ids: tuple[int, ...]
    obstacle_geom_ids: tuple[int, ...]
    robot_geom_names: tuple[str, ...]
    obstacle_geom_names: tuple[str, ...]
    disabled_obstacle_geom_names: tuple[str, ...]
    excluded_body_names: tuple[str, ...]


def _name(mujoco: Any, model: Any, object_type: Any, object_id: int, prefix: str) -> str:
    value = mujoco.mj_id2name(model, object_type, int(object_id))
    return str(value) if value else f"{prefix}_{int(object_id)}"


def _body_is_in_subtree(model: Any, body_id: int, root_body_id: int) -> bool:
    """Return whether body_id is root_body_id or one of its descendants."""
    current = int(body_id)
    while current != 0:
        if current == root_body_id:
            return True
        current = int(model.body_parentid[current])
    return root_body_id == 0


def select_collision_geometries(
    model: Any,
    mujoco: Any,
    *,
    robot_base_body_name: str = "base",
    collision_group: int = 3,
    excluded_body_names: Iterable[str] = ("robot_station",),
) -> GeometrySelection:
    """Select robot collision proxies and environment geoms by model topology.

    Robot visual meshes (group 2) are deliberately not used: the UR5e MJCF
    supplies simplified collision proxies in group 3.  Environment geoms are
    selected regardless of contype/conaffinity so disabled contact masks do
    not hide obstacles from this preflight.
    """
    body_type = mujoco.mjtObj.mjOBJ_BODY
    geom_type = mujoco.mjtObj.mjOBJ_GEOM
    base_body_id = int(mujoco.mj_name2id(model, body_type, robot_base_body_name))
    if base_body_id < 0:
        raise ValueError(
            f"Robot base body '{robot_base_body_name}' was not found in the MuJoCo scene"
        )

    excluded_ids: set[int] = set()
    excluded_names: list[str] = []
    for body_name in excluded_body_names:
        body_id = int(mujoco.mj_name2id(model, body_type, str(body_name)))
        if body_id < 0:
            raise ValueError(f"Required excluded/support body '{body_name}' was not found")
        excluded_ids.add(body_id)
        excluded_names.append(str(body_name))

    robot_body_ids = {
        body_id
        for body_id in range(int(model.nbody))
        if _body_is_in_subtree(model, body_id, base_body_id)
    }

    robot_geom_ids = tuple(
        geom_id
        for geom_id in range(int(model.ngeom))
        if int(model.geom_group[geom_id]) == int(collision_group)
        and int(model.geom_bodyid[geom_id]) in robot_body_ids
    )
    obstacle_geom_ids = tuple(
        geom_id
        for geom_id in range(int(model.ngeom))
        if int(model.geom_bodyid[geom_id]) not in robot_body_ids
        and int(model.geom_bodyid[geom_id]) not in excluded_ids
    )

    if not robot_geom_ids:
        raise ValueError(
            f"No robot collision proxies found in geom group {collision_group}; "
            "check the UR5e MJCF default classes"
        )
    if not obstacle_geom_ids:
        raise ValueError("No environment obstacle geoms remain after exclusions")

    robot_names = tuple(_name(mujoco, model, geom_type, i, "robot_geom") for i in robot_geom_ids)
    obstacle_names = tuple(
        _name(mujoco, model, geom_type, i, "environment_geom") for i in obstacle_geom_ids
    )
    disabled_names = tuple(
        _name(mujoco, model, geom_type, geom_id, "environment_geom")
        for geom_id in obstacle_geom_ids
        if int(model.geom_contype[geom_id]) == 0
        and int(model.geom_conaffinity[geom_id]) == 0
    )

    return GeometrySelection(
        robot_geom_ids=robot_geom_ids,
        obstacle_geom_ids=obstacle_geom_ids,
        robot_geom_names=robot_names,
        obstacle_geom_names=obstacle_names,
        disabled_obstacle_geom_names=disabled_names,
        excluded_body_names=tuple(excluded_names),
    )


def _joint_qpos_addresses(model: Any, mujoco: Any, joint_names: Iterable[str]) -> np.ndarray:
    addresses: list[int] = []
    for joint_name in joint_names:
        joint_id = int(
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, str(joint_name))
        )
        if joint_id < 0:
            raise ValueError(f"Required robot joint '{joint_name}' was not found")
        if int(model.jnt_type[joint_id]) != int(mujoco.mjtJoint.mjJNT_HINGE):
            raise ValueError(f"Joint '{joint_name}' is not a scalar hinge joint")
        addresses.append(int(model.jnt_qposadr[joint_id]))
    if len(set(addresses)) != len(addresses):
        raise ValueError("Joint qpos addresses are not unique")
    return np.asarray(addresses, dtype=int)


def _validate_joint_limits(
    model: Any,
    joint_ids: np.ndarray,
    qpos_addresses: np.ndarray,
    q: np.ndarray,
    *,
    time_s: float,
    tolerance_rad: float = 1e-8,
) -> list[dict[str, float | str]]:
    violations: list[dict[str, float | str]] = []
    for joint_index, (joint_id, qpos_address, joint_name) in enumerate(
        zip(joint_ids, qpos_addresses, JOINT_NAMES)
    ):
        if bool(model.jnt_limited[int(joint_id)]):
            lower, upper = (float(v) for v in model.jnt_range[int(joint_id)])
            value = float(q[joint_index])
            if value < lower - tolerance_rad or value > upper + tolerance_rad:
                violations.append(
                    {
                        "time_s": float(time_s),
                        "joint_name": joint_name,
                        "position_rad": value,
                        "lower_limit_rad": lower,
                        "upper_limit_rad": upper,
                    }
                )
    return violations


def make_sample_times(start_time_s: float, end_time_s: float, sample_period_s: float) -> np.ndarray:
    """Create a uniform sample grid and always include the exact final time."""
    start = float(start_time_s)
    end = float(end_time_s)
    period = float(sample_period_s)
    if not all(math.isfinite(v) for v in (start, end, period)):
        raise ValueError("Sample times and period must be finite")
    if end <= start:
        raise ValueError("Trajectory end time must be greater than start time")
    if period <= 0.0:
        raise ValueError("sample_period_s must be positive")
    count = int(math.floor((end - start) / period))
    times = start + np.arange(count + 1, dtype=float) * period
    if end - float(times[-1]) > 1e-10:
        times = np.append(times, end)
    else:
        times[-1] = end
    return times


def _load_home_keyframe(model: Any, mujoco: Any, qpos_addresses: np.ndarray) -> np.ndarray:
    key_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home"))
    if key_id < 0:
        raise ValueError("The MuJoCo scene must contain the documented 'home' keyframe")
    key_qpos = np.asarray(model.key_qpos, dtype=float)
    if key_qpos.ndim != 2 or key_id >= key_qpos.shape[0]:
        raise ValueError("MuJoCo home keyframe qpos data is malformed")
    return key_qpos[key_id, qpos_addresses].copy()


def run_collision_preflight(
    model: Any,
    data: Any,
    mujoco: Any,
    trajectory: Any,
    *,
    output_dir: str | Path,
    sample_period_s: float = 0.005,
    distance_cap_m: float = 0.05,
    minimum_clearance_m: float = 0.02,
    warning_clearance_m: float = 0.05,
    home_tolerance_rad: float = 0.05,
    robot_base_body_name: str = "base",
    collision_group: int = 3,
    excluded_body_names: Iterable[str] = ("robot_station",),
    scene_xml_path: str | Path = "",
) -> dict[str, Any]:
    """Sweep the interpolated joint reference against scene obstacle geoms."""
    for value, label in (
        (sample_period_s, "sample_period_s"),
        (distance_cap_m, "distance_cap_m"),
        (minimum_clearance_m, "minimum_clearance_m"),
        (warning_clearance_m, "warning_clearance_m"),
        (home_tolerance_rad, "home_tolerance_rad"),
    ):
        if not math.isfinite(float(value)) or float(value) <= 0.0:
            raise ValueError(f"{label} must be positive and finite")
    if minimum_clearance_m >= warning_clearance_m:
        raise ValueError("minimum_clearance_m must be smaller than warning_clearance_m")
    if warning_clearance_m > distance_cap_m:
        raise ValueError("warning_clearance_m cannot exceed distance_cap_m")
    if int(model.nq) != 6 or int(model.nv) != 6:
        raise ValueError(f"Expected UR5e nq=nv=6; got nq={model.nq}, nv={model.nv}")

    joint_ids = np.asarray(
        [int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)) for n in JOINT_NAMES],
        dtype=int,
    )
    if np.any(joint_ids < 0):
        missing = [name for name, jid in zip(JOINT_NAMES, joint_ids) if jid < 0]
        raise ValueError("Missing required UR5e joints: " + ", ".join(missing))
    qpos_addresses = _joint_qpos_addresses(model, mujoco, JOINT_NAMES)
    home_q = _load_home_keyframe(model, mujoco, qpos_addresses)
    start_q = np.asarray(trajectory.evaluate(trajectory.start_time_s).position_rad, dtype=float)
    if start_q.shape != (6,) or not np.all(np.isfinite(start_q)):
        raise ValueError("Trajectory start must contain six finite joint positions")
    home_error = float(np.max(np.abs(start_q - home_q)))
    if home_error > home_tolerance_rad:
        raise ValueError(
            f"Trajectory start does not match MuJoCo HOME: max error {home_error:.6f} rad "
            f"exceeds {home_tolerance_rad:.6f} rad"
        )

    selection = select_collision_geometries(
        model,
        mujoco,
        robot_base_body_name=robot_base_body_name,
        collision_group=collision_group,
        excluded_body_names=excluded_body_names,
    )
    sample_times = make_sample_times(
        trajectory.start_time_s, trajectory.end_time_s, sample_period_s
    )
    fromto = np.zeros(6, dtype=np.float64)
    rows: list[dict[str, Any]] = []
    critical_pairs: dict[tuple[str, str], dict[str, Any]] = {}
    joint_limit_violations: list[dict[str, Any]] = []
    collision_events: list[dict[str, Any]] = []
    warning_sample_count = 0
    minimum_observed = float(distance_cap_m)
    minimum_pair: tuple[str, str] | None = None
    minimum_time = float(sample_times[0])

    for time_s in sample_times:
        point = trajectory.evaluate(float(time_s))
        q = np.asarray(point.position_rad, dtype=float)
        if q.shape != (6,) or not np.all(np.isfinite(q)):
            raise ValueError(f"Invalid six-joint reference at t={time_s:.6f} s")

        limit_violations = _validate_joint_limits(
            model, joint_ids, qpos_addresses, q, time_s=float(time_s)
        )
        joint_limit_violations.extend(limit_violations)

        data.qpos[qpos_addresses] = q
        if hasattr(data, "qvel"):
            data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)

        sample_minimum = float(distance_cap_m)
        sample_pair: tuple[str, str] | None = None
        near_count = 0
        collision_count = 0
        for robot_geom_id, robot_geom_name in zip(
            selection.robot_geom_ids, selection.robot_geom_names
        ):
            for obstacle_geom_id, obstacle_geom_name in zip(
                selection.obstacle_geom_ids, selection.obstacle_geom_names
            ):
                distance = float(
                    mujoco.mj_geomDistance(
                        model,
                        data,
                        int(robot_geom_id),
                        int(obstacle_geom_id),
                        float(distance_cap_m),
                        fromto,
                    )
                )
                if not math.isfinite(distance):
                    raise RuntimeError(
                        f"MuJoCo returned a non-finite distance for "
                        f"{robot_geom_name} vs {obstacle_geom_name} at t={time_s:.6f} s"
                    )
                pair = (robot_geom_name, obstacle_geom_name)
                if distance < sample_minimum:
                    sample_minimum = distance
                    sample_pair = pair
                if distance < warning_clearance_m:
                    near_count += 1
                    previous = critical_pairs.get(pair)
                    if previous is None or distance < float(previous["minimum_distance_m"]):
                        critical_pairs[pair] = {
                            "robot_geom": robot_geom_name,
                            "environment_geom": obstacle_geom_name,
                            "minimum_distance_m": distance,
                            "time_s": float(time_s),
                        }
                if distance <= 0.0:
                    collision_count += 1
                    collision_events.append(
                        {
                            "time_s": float(time_s),
                            "robot_geom": robot_geom_name,
                            "environment_geom": obstacle_geom_name,
                            "signed_distance_m": distance,
                        }
                    )

        if sample_minimum < minimum_observed:
            minimum_observed = sample_minimum
            minimum_pair = sample_pair
            minimum_time = float(time_s)
        if sample_minimum < warning_clearance_m or limit_violations:
            warning_sample_count += 1

        if collision_count:
            sample_status = "COLLISION"
        elif limit_violations:
            sample_status = "JOINT_LIMIT_VIOLATION"
        elif sample_minimum < minimum_clearance_m:
            sample_status = "INSUFFICIENT_CLEARANCE"
        elif sample_minimum < warning_clearance_m:
            sample_status = "WARNING"
        else:
            sample_status = "PASS"

        rows.append(
            {
                "trajectory_time_s": float(time_s),
                "phase": str(point.phase),
                "minimum_distance_reported_m": float(sample_minimum),
                "distance_is_capped_lower_bound": bool(sample_pair is None),
                "closest_robot_geom": sample_pair[0] if sample_pair else "",
                "closest_environment_geom": sample_pair[1] if sample_pair else "",
                "pairs_within_warning_clearance": int(near_count),
                "collision_pairs": int(collision_count),
                "joint_limit_violations": int(len(limit_violations)),
                "sample_status": sample_status,
            }
        )

    any_collision = bool(collision_events)
    any_joint_limit_violation = bool(joint_limit_violations)
    if any_collision:
        status = "FAIL_COLLISION"
    elif any_joint_limit_violation:
        status = "FAIL_JOINT_LIMIT"
    elif minimum_observed < minimum_clearance_m:
        status = "FAIL_CLEARANCE"
    elif minimum_observed < warning_clearance_m:
        status = "WARNING_REVIEW_REQUIRED"
    else:
        status = "PASS"

    capped = minimum_pair is None
    summary: dict[str, Any] = {
        "status": status,
        "safe_to_execute": status == "PASS",
        "scene_xml": str(Path(scene_xml_path).expanduser().resolve()) if scene_xml_path else "",
        "trajectory_csv": str(getattr(trajectory, "source_path", "")),
        "mujoco_version": str(getattr(mujoco, "__version__", "unknown")),
        "trajectory_duration_s": float(trajectory.end_time_s - trajectory.start_time_s),
        "sample_period_s": float(sample_period_s),
        "sample_count": int(sample_times.size),
        "robot_base_body": robot_base_body_name,
        "robot_collision_group": int(collision_group),
        "robot_collision_geom_count": len(selection.robot_geom_ids),
        "environment_geom_count": len(selection.obstacle_geom_ids),
        "environment_geoms_with_disabled_contact_masks": len(
            selection.disabled_obstacle_geom_names
        ),
        "disabled_contact_mask_geoms": list(selection.disabled_obstacle_geom_names),
        "excluded_support_bodies": list(selection.excluded_body_names),
        "robot_collision_geoms": list(selection.robot_geom_names),
        "environment_geoms": list(selection.obstacle_geom_names),
        "geometry_pairs_checked_per_sample": len(selection.robot_geom_ids)
        * len(selection.obstacle_geom_ids),
        "total_geometry_distance_queries": int(
            sample_times.size * len(selection.robot_geom_ids) * len(selection.obstacle_geom_ids)
        ),
        "distance_cap_m": float(distance_cap_m),
        "minimum_required_clearance_m": float(minimum_clearance_m),
        "warning_clearance_m": float(warning_clearance_m),
        "home_keyframe_max_start_error_rad": home_error,
        "minimum_distance_is_capped_lower_bound": capped,
        "minimum_distance_m": None if capped else float(minimum_observed),
        "minimum_clearance_lower_bound_m": float(distance_cap_m)
        if capped
        else float(minimum_observed),
        "minimum_distance_time_s": float(minimum_time),
        "minimum_distance_pair": None
        if minimum_pair is None
        else {
            "robot_geom": minimum_pair[0],
            "environment_geom": minimum_pair[1],
        },
        "samples_below_warning_clearance": int(warning_sample_count),
        "collision_event_count": len(collision_events),
        "collision_events": collision_events[:100],
        "joint_limit_violation_count": len(joint_limit_violations),
        "joint_limit_violations": joint_limit_violations[:100],
        "closest_pairs_below_warning_clearance": sorted(
            critical_pairs.values(), key=lambda item: float(item["minimum_distance_m"])
        )[:25],
        "limitations": [
            "This is a sampled kinematic sweep, not a dynamic simulation of tracking error or actuator lag.",
            "The UR5e is checked using its simplified MJCF collision proxies (geom group 3), not its visual meshes.",
            "This first pass checks robot-to-environment geometry pairs; robot self-collision pairs are not evaluated.",
            "The robot_station body is excluded because it is the intended robot mounting support.",
            "A surgical instrument is not attached to the robot end effector in this scene; held-tool clearance is not checked.",
            "This preflight is not a safety-rated monitor and does not certify hardware operation.",
        ],
    }

    output_path = Path(output_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    csv_path = output_path / "task_space_collision_preflight.csv"
    json_path = output_path / "task_space_collision_preflight.json"
    fields = list(rows[0].keys())
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    summary["csv_report"] = str(csv_path)
    summary["json_report"] = str(json_path)
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a geometry-aware MuJoCo collision preflight for the M4 task-space trajectory."
    )
    parser.add_argument("--scene", type=Path, default=DEFAULT_SCENE)
    parser.add_argument("--trajectory", type=Path, default=DEFAULT_TRAJECTORY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sample-period-s", type=float, default=0.005)
    parser.add_argument("--distance-cap-m", type=float, default=0.05)
    parser.add_argument("--minimum-clearance-m", type=float, default=0.02)
    parser.add_argument("--warning-clearance-m", type=float, default=0.05)
    parser.add_argument("--home-tolerance-rad", type=float, default=0.05)
    parser.add_argument("--robot-base-body", default="base")
    parser.add_argument("--collision-group", type=int, default=3)
    parser.add_argument(
        "--exclude-body",
        action="append",
        dest="exclude_bodies",
        help="Static support body to exclude (repeatable); defaults to robot_station.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    scene_path = args.scene.expanduser().resolve()
    trajectory_path = args.trajectory.expanduser().resolve()
    if not scene_path.is_file():
        raise FileNotFoundError(f"MuJoCo scene not found: {scene_path}")
    if not trajectory_path.is_file():
        raise FileNotFoundError(f"Joint reference CSV not found: {trajectory_path}")

    try:
        import mujoco
    except ImportError as exc:
        raise RuntimeError(
            "MuJoCo Python is required. Activate ~/mctr911_kinematics_venv first."
        ) from exc

    trajectory = JointTrajectoryCSV.load(trajectory_path, JOINT_NAMES)
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    data = mujoco.MjData(model)
    exclude_bodies = args.exclude_bodies or ["robot_station"]
    summary = run_collision_preflight(
        model,
        data,
        mujoco,
        trajectory,
        output_dir=args.output_dir,
        sample_period_s=args.sample_period_s,
        distance_cap_m=args.distance_cap_m,
        minimum_clearance_m=args.minimum_clearance_m,
        warning_clearance_m=args.warning_clearance_m,
        home_tolerance_rad=args.home_tolerance_rad,
        robot_base_body_name=args.robot_base_body,
        collision_group=args.collision_group,
        excluded_body_names=exclude_bodies,
        scene_xml_path=scene_path,
    )

    print("========== TASK-SPACE COLLISION PREFLIGHT ==========")
    print(f"MuJoCo version: {summary['mujoco_version']}")
    print(f"Scene: {scene_path}")
    print(f"Trajectory: {trajectory_path}")
    print(
        f"Robot collision proxies: {summary['robot_collision_geom_count']} | "
        f"Environment geoms: {summary['environment_geom_count']}"
    )
    print(
        "Environment geoms with disabled normal contact masks: "
        f"{summary['environment_geoms_with_disabled_contact_masks']}"
    )
    print(
        f"Samples: {summary['sample_count']} | "
        f"sample period: {summary['sample_period_s']:.4f} s | "
        f"distance queries: {summary['total_geometry_distance_queries']}"
    )
    if summary["minimum_distance_is_capped_lower_bound"]:
        print(
            f"Minimum clearance: >= {summary['minimum_clearance_lower_bound_m']:.3f} m "
            "(no geometry pair came within the distance-query cap)"
        )
    else:
        pair = summary["minimum_distance_pair"]
        print(
            f"Minimum clearance: {summary['minimum_distance_m']:.6f} m at "
            f"t={summary['minimum_distance_time_s']:.3f} s"
        )
        print(
            f"Closest pair: {pair['robot_geom']} <-> {pair['environment_geom']}"
        )
    print(f"Collision events: {summary['collision_event_count']}")
    print(f"Joint-limit violations: {summary['joint_limit_violation_count']}")
    print(f"Status: {summary['status']}")
    print(f"Safe to execute: {'YES' if summary['safe_to_execute'] else 'NO'}")
    print(f"CSV report: {summary['csv_report']}")
    print(f"JSON report: {summary['json_report']}")
    if summary["status"] == "PASS":
        return 0
    if summary["status"] == "WARNING_REVIEW_REQUIRED":
        return 2
    return 3


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"COLLISION PREFLIGHT ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
