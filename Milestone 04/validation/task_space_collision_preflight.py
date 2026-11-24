#!/usr/bin/env python3
"""
Task-space trajectory collision/clearance preflight for the UR5e surgical scene.

This is a geometric clearance audit. It does not change the MuJoCo model,
collision masks, actuators, or the original trajectory. Environment clearance
is computed directly with mujoco.mj_geomDistance, including scene geoms whose
MuJoCo contact masks are disabled.

The trajectory between CSV rows is approximated by piecewise-linear joint
interpolation. The maximum absolute change of any joint between checked
configurations is bounded by --maximum-joint-step-rad.

Self-collision filtering excludes geometry pairs on the same rigid link and
on directly parent-child links. These pairs normally overlap by design around
robot joints. All remaining robot collision-geometry pairs are checked.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

try:
    import mujoco
except ImportError as exc:  # pragma: no cover - depends on local environment
    raise SystemExit(
        "MuJoCo Python bindings are required. Activate the project's "
        "mctr911_kinematics_venv before running this script."
    ) from exc


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[2]

DEFAULT_SCENE = (
    PROJECT_ROOT / "Milestone 01" / "Robot Model" / "scene_surgical_m3.xml"
)
DEFAULT_REFERENCE = (
    PROJECT_ROOT
    / "Milestone 04"
    / "Results"
    / "Task_Space"
    / "task_space_joint_reference.csv"
)
DEFAULT_OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "Milestone 04"
    / "Results"
    / "Validation"
    / "Task_Space_Collision_Preflight"
)

JOINT_NAMES = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
)

# The UR5e model's kinematic chain. "base" is included so its visual geoms
# are not mistakenly treated as environment objects.
ROBOT_BODY_NAMES = (
    "base",
    "shoulder_link",
    "upper_arm_link",
    "forearm_link",
    "wrist_1_link",
    "wrist_2_link",
    "wrist_3_link",
)

ROBOT_LINK_BODY_NAMES = ROBOT_BODY_NAMES[1:]
ROBOT_COLLISION_GROUP = 3

DISTANCE_TOLERANCE_M = 1.0e-9
PENETRATION_TOLERANCE_M = 1.0e-6


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sample the task-space-derived UR5e joint trajectory and evaluate "
            "robot/environment clearance and non-adjacent self-collision."
        )
    )
    parser.add_argument(
        "--scene-xml",
        type=Path,
        default=DEFAULT_SCENE,
        help=f"MuJoCo scene XML (default: {DEFAULT_SCENE})",
    )
    parser.add_argument(
        "--reference-csv",
        type=Path,
        default=DEFAULT_REFERENCE,
        help=f"Task-space joint reference CSV (default: {DEFAULT_REFERENCE})",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=DEFAULT_OUTPUT_DIRECTORY,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIRECTORY})",
    )
    parser.add_argument(
        "--maximum-joint-step-rad",
        type=float,
        default=0.001,
        help=(
            "Maximum absolute change of any joint between checked "
            "configurations, in radians (default: 0.001)."
        ),
    )
    parser.add_argument(
        "--minimum-clearance-m",
        type=float,
        default=0.0,
        help=(
            "Required minimum geometric clearance in metres. Default 0.0 "
            "means touching/overlap fails, while any positive clearance "
            "passes this geometric threshold. This is not a certified safety "
            "margin."
        ),
    )
    parser.add_argument(
        "--distance-cap-m",
        type=float,
        default=100.0,
        help=(
            "Upper distance bound passed to mj_geomDistance (default: 100 m). "
            "The scene is much smaller than this."
        ),
    )
    parser.add_argument(
        "--top-pairs",
        type=int,
        default=15,
        help="Number of closest geometry pairs included per category (default: 15).",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="task_space_collision_preflight",
        help="Prefix for timestamped output files.",
    )
    return parser.parse_args()


def require_file(path: Path, description: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{description} does not exist: {resolved}")
    return resolved


def load_reference(path: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    required_columns = ["time_s", "phase"] + [
        f"q_desired_{name}_rad" for name in JOINT_NAMES
    ]

    with path.open("r", newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        available = set(reader.fieldnames or [])
        missing = [column for column in required_columns if column not in available]
        if missing:
            raise ValueError(
                "Reference CSV is missing required columns: " + ", ".join(missing)
            )

        times: list[float] = []
        positions: list[list[float]] = []
        phases: list[str] = []

        for line_number, row in enumerate(reader, start=2):
            try:
                t = float(row["time_s"])
                q = [float(row[f"q_desired_{name}_rad"]) for name in JOINT_NAMES]
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid numeric trajectory value on CSV line {line_number}"
                ) from exc

            if not math.isfinite(t) or not all(math.isfinite(value) for value in q):
                raise ValueError(
                    f"Non-finite trajectory value on CSV line {line_number}"
                )

            times.append(t)
            positions.append(q)
            phases.append((row.get("phase") or "unspecified").strip() or "unspecified")

    if len(times) < 2:
        raise ValueError("The reference CSV must contain at least two samples.")

    time_array = np.asarray(times, dtype=float)
    q_array = np.asarray(positions, dtype=float)

    if np.any(np.diff(time_array) <= 0.0):
        raise ValueError("Reference time_s values must be strictly increasing.")

    if q_array.shape[1] != len(JOINT_NAMES):
        raise ValueError("Reference must contain exactly six UR5e joint positions.")

    return time_array, q_array, phases


def build_interpolated_samples(
    times: np.ndarray,
    positions: np.ndarray,
    phases: list[str],
    maximum_joint_step_rad: float,
) -> list[dict[str, Any]]:
    """
    Piecewise-linear interpolation with a bound on the largest joint step.

    Each original CSV row is retained as a checked configuration. Additional
    configurations are inserted only when a segment exceeds the requested
    joint-step bound.
    """
    samples: list[dict[str, Any]] = []

    for segment_index in range(len(times) - 1):
        q0 = positions[segment_index]
        q1 = positions[segment_index + 1]
        delta = q1 - q0
        largest_joint_delta = float(np.max(np.abs(delta)))

        subdivisions = max(
            1, int(math.ceil(largest_joint_delta / maximum_joint_step_rad))
        )

        for subdivision in range(subdivisions):
            alpha = subdivision / subdivisions
            q = q0 + alpha * delta
            t = float(
                times[segment_index]
                + alpha * (times[segment_index + 1] - times[segment_index])
            )
            samples.append(
                {
                    "time_s": t,
                    "phase": phases[segment_index],
                    "source_segment_index": segment_index,
                    "interpolation_fraction": float(alpha),
                    "q": q.copy(),
                }
            )

    # Include the final reference endpoint exactly once.
    samples.append(
        {
            "time_s": float(times[-1]),
            "phase": phases[-1],
            "source_segment_index": len(times) - 2,
            "interpolation_fraction": 1.0,
            "q": positions[-1].copy(),
        }
    )

    return samples


def object_name(model: mujoco.MjModel, object_type: Any, object_id: int) -> str | None:
    return mujoco.mj_id2name(model, object_type, int(object_id))


def body_name(model: mujoco.MjModel, body_id: int) -> str:
    return object_name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or f"body_{body_id}"


def geometry_description(model: mujoco.MjModel, geom_id: int) -> dict[str, Any]:
    body_id = int(model.geom_bodyid[geom_id])
    geom_name = object_name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
    return {
        "geom_id": int(geom_id),
        "geom_name": geom_name or f"unnamed_geom_{geom_id}",
        "body_id": body_id,
        "body_name": body_name(model, body_id),
        "label": (
            f"{body_name(model, body_id)}/"
            f"{geom_name or f'unnamed_geom_{geom_id}'}[{geom_id}]"
        ),
    }


def build_geometry_sets(
    model: mujoco.MjModel,
) -> tuple[list[int], list[int], list[tuple[int, int]], list[dict[str, Any]]]:
    robot_body_ids: set[int] = set()
    missing_bodies: list[str] = []

    for name in ROBOT_BODY_NAMES:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if body_id < 0:
            missing_bodies.append(name)
        else:
            robot_body_ids.add(int(body_id))

    if missing_bodies:
        raise ValueError(
            "The MuJoCo model is missing expected UR5e bodies: "
            + ", ".join(missing_bodies)
        )

    robot_link_body_ids = {
        int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name))
        for name in ROBOT_LINK_BODY_NAMES
    }

    # UR5e collision primitives are assigned to geom group 3 in ur5e.xml.
    robot_geoms = [
        geom_id
        for geom_id in range(model.ngeom)
        if int(model.geom_bodyid[geom_id]) in robot_link_body_ids
        and int(model.geom_group[geom_id]) == ROBOT_COLLISION_GROUP
    ]

    if not robot_geoms:
        # Fallback for a model variant that preserves contact masks but changes
        # the geom group assignment.
        robot_geoms = [
            geom_id
            for geom_id in range(model.ngeom)
            if int(model.geom_bodyid[geom_id]) in robot_link_body_ids
            and (
                int(model.geom_contype[geom_id]) != 0
                or int(model.geom_conaffinity[geom_id]) != 0
            )
        ]

    if not robot_geoms:
        raise ValueError(
            "No UR5e collision geometries were found on the six robot links."
        )

    # Everything outside the robot kinematic chain belongs to the static scene.
    # This deliberately includes scene geoms with contype=0/conaffinity=0.
    environment_geoms = [
        geom_id
        for geom_id in range(model.ngeom)
        if int(model.geom_bodyid[geom_id]) not in robot_body_ids
    ]

    environment_pairs = [
        (robot_geom, environment_geom)
        for robot_geom in robot_geoms
        for environment_geom in environment_geoms
    ]

    # Same-body and direct parent-child link pairs are excluded because their
    # collision primitives normally overlap by design around robot joints.
    self_pairs: list[tuple[int, int]] = []
    excluded_self_pairs: list[dict[str, Any]] = []

    for index, geom_a in enumerate(robot_geoms):
        body_a = int(model.geom_bodyid[geom_a])
        parent_a = int(model.body_parentid[body_a])

        for geom_b in robot_geoms[index + 1 :]:
            body_b = int(model.geom_bodyid[geom_b])
            parent_b = int(model.body_parentid[body_b])

            if body_a == body_b:
                reason = "same rigid link"
            elif parent_a == body_b or parent_b == body_a:
                reason = "direct parent-child links"
            else:
                self_pairs.append((geom_a, geom_b))
                continue

            excluded_self_pairs.append(
                {
                    "geom_a": geometry_description(model, geom_a),
                    "geom_b": geometry_description(model, geom_b),
                    "reason": reason,
                }
            )

    return robot_geoms, environment_geoms, self_pairs, excluded_self_pairs


def get_joint_model_data(
    model: mujoco.MjModel,
) -> tuple[list[int], list[int], list[dict[str, Any]]]:
    joint_ids: list[int] = []
    qpos_addresses: list[int] = []
    limits: list[dict[str, Any]] = []

    for name in JOINT_NAMES:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            raise ValueError(f"Joint '{name}' was not found in the MuJoCo model.")

        joint_id = int(joint_id)
        joint_type = int(model.jnt_type[joint_id])
        if joint_type != int(mujoco.mjtJoint.mjJNT_HINGE):
            raise ValueError(f"Expected '{name}' to be a hinge joint.")

        joint_ids.append(joint_id)
        qpos_addresses.append(int(model.jnt_qposadr[joint_id]))

        limited = bool(model.jnt_limited[joint_id])
        if limited:
            lower, upper = [float(v) for v in model.jnt_range[joint_id]]
        else:
            lower, upper = None, None

        limits.append(
            {
                "joint_name": name,
                "joint_id": joint_id,
                "limited": limited,
                "lower_rad": lower,
                "upper_rad": upper,
            }
        )

    return joint_ids, qpos_addresses, limits


def pair_distance(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    geom_a: int,
    geom_b: int,
    distance_cap_m: float,
) -> tuple[float, list[float]]:
    fromto = np.zeros(6, dtype=np.float64)
    distance = float(
        mujoco.mj_geomDistance(
            model,
            data,
            int(geom_a),
            int(geom_b),
            float(distance_cap_m),
            fromto,
        )
    )
    if not math.isfinite(distance):
        raise RuntimeError(
            f"mj_geomDistance returned a non-finite value for "
            f"geometry pair ({geom_a}, {geom_b})."
        )
    return distance, [float(value) for value in fromto]


def update_pair_minimum(
    pair_minima: dict[tuple[int, int], dict[str, Any]],
    key: tuple[int, int],
    kind: str,
    distance: float,
    fromto: list[float],
    sample: dict[str, Any],
    model: mujoco.MjModel,
) -> None:
    current = pair_minima.get(key)
    if current is None or distance < current["minimum_distance_m"]:
        pair_minima[key] = {
            "kind": kind,
            "geom_a": geometry_description(model, key[0]),
            "geom_b": geometry_description(model, key[1]),
            "minimum_distance_m": float(distance),
            "time_s": float(sample["time_s"]),
            "phase": str(sample["phase"]),
            "sample_index": int(sample["sample_index"]),
            "closest_points_world_m": fromto,
        }


def check_joint_limits(
    q: np.ndarray,
    limits: list[dict[str, Any]],
    tolerance_rad: float = 1.0e-6,
) -> list[dict[str, Any]]:
    violations = []
    for value, limit in zip(q, limits):
        if not limit["limited"]:
            continue
        lower = float(limit["lower_rad"])
        upper = float(limit["upper_rad"])
        if value < lower - tolerance_rad or value > upper + tolerance_rad:
            violations.append(
                {
                    "joint_name": limit["joint_name"],
                    "position_rad": float(value),
                    "lower_limit_rad": lower,
                    "upper_limit_rad": upper,
                }
            )
    return violations


def json_safe_distance(value: float | None) -> float | None:
    if value is None or not math.isfinite(value):
        return None
    return float(value)


def run_preflight(args: argparse.Namespace) -> int:
    scene_path = require_file(args.scene_xml, "Scene XML")
    reference_path = require_file(args.reference_csv, "Reference CSV")

    if args.maximum_joint_step_rad <= 0.0:
        raise ValueError("--maximum-joint-step-rad must be greater than zero.")
    if args.minimum_clearance_m < 0.0:
        raise ValueError("--minimum-clearance-m cannot be negative.")
    if args.distance_cap_m <= 0.0:
        raise ValueError("--distance-cap-m must be greater than zero.")
    if args.top_pairs < 1:
        raise ValueError("--top-pairs must be at least 1.")

    times, positions, phases = load_reference(reference_path)
    samples = build_interpolated_samples(
        times, positions, phases, args.maximum_joint_step_rad
    )

    model = mujoco.MjModel.from_xml_path(str(scene_path))
    data = mujoco.MjData(model)

    home_key_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_KEY, "home"
    )
    if home_key_id >= 0:
        mujoco.mj_resetDataKeyframe(model, data, int(home_key_id))
        home_keyframe_used = "home"
    else:
        mujoco.mj_resetData(model, data)
        home_keyframe_used = None

    robot_geoms, environment_geoms, self_pairs, excluded_self_pairs = (
        build_geometry_sets(model)
    )
    _, qpos_addresses, joint_limits = get_joint_model_data(model)

    environment_pair_count = len(robot_geoms) * len(environment_geoms)
    if environment_pair_count == 0:
        raise ValueError("No robot-to-environment geometry pairs are available.")

    environment_pair_minima: dict[tuple[int, int], dict[str, Any]] = {}
    self_pair_minima: dict[tuple[int, int], dict[str, Any]] = {}

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    args.output_directory.mkdir(parents=True, exist_ok=True)
    output_csv = (
        args.output_directory
        / f"{args.output_prefix}_{timestamp}_samples.csv"
    )
    output_json = (
        args.output_directory
        / f"{args.output_prefix}_{timestamp}_metrics.json"
    )

    csv_fields = [
        "sample_index",
        "source_segment_index",
        "interpolation_fraction",
        "time_s",
        "phase",
        *[f"q_{name}_rad" for name in JOINT_NAMES],
        "minimum_environment_clearance_m",
        "closest_environment_robot_geometry",
        "closest_environment_scene_geometry",
        "minimum_self_clearance_m",
        "closest_self_geometry_a",
        "closest_self_geometry_b",
        "minimum_overall_clearance_m",
        "environment_pairs_at_or_below_threshold",
        "self_pairs_at_or_below_threshold",
        "environment_penetrating_pairs",
        "self_penetrating_pairs",
        "joint_limit_violation_count",
        "clearance_ok",
        "joint_limits_ok",
    ]

    minimum_environment_clearance = math.inf
    minimum_self_clearance = math.inf
    minimum_overall_clearance = math.inf
    first_clearance_violation: dict[str, Any] | None = None
    first_joint_limit_violation: dict[str, Any] | None = None
    clearance_violation_events = 0
    penetration_events = 0
    unique_violating_pairs: set[tuple[str, int, int]] = set()
    joint_limit_violation_events = 0
    max_observed_joint_step = 0.0

    with output_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=csv_fields)
        writer.writeheader()

        previous_q: np.ndarray | None = None

        for sample_index, sample in enumerate(samples):
            q = np.asarray(sample["q"], dtype=float)
            sample["sample_index"] = sample_index

            if previous_q is not None:
                max_observed_joint_step = max(
                    max_observed_joint_step,
                    float(np.max(np.abs(q - previous_q))),
                )
            previous_q = q.copy()

            # Set the six hinge-joint positions and evaluate forward kinematics.
            for address, value in zip(qpos_addresses, q):
                data.qpos[address] = float(value)
            data.qvel[:] = 0.0
            mujoco.mj_forward(model, data)

            joint_violations = check_joint_limits(q, joint_limits)
            joint_limit_violation_events += len(joint_violations)
            if joint_violations and first_joint_limit_violation is None:
                first_joint_limit_violation = {
                    "sample_index": sample_index,
                    "time_s": float(sample["time_s"]),
                    "phase": str(sample["phase"]),
                    "violations": joint_violations,
                }

            env_min = math.inf
            env_min_pair: tuple[int, int] | None = None
            env_below_threshold = 0
            env_penetrations = 0

            for geom_a, geom_b in (
                (robot_geom, environment_geom)
                for robot_geom in robot_geoms
                for environment_geom in environment_geoms
            ):
                distance, fromto = pair_distance(
                    model, data, geom_a, geom_b, args.distance_cap_m
                )
                key = (geom_a, geom_b)
                update_pair_minimum(
                    environment_pair_minima,
                    key,
                    "robot_environment",
                    distance,
                    fromto,
                    sample,
                    model,
                )

                if distance < env_min:
                    env_min = distance
                    env_min_pair = key

                if distance <= args.minimum_clearance_m + DISTANCE_TOLERANCE_M:
                    env_below_threshold += 1
                    clearance_violation_events += 1
                    unique_violating_pairs.add(("environment", geom_a, geom_b))
                    if first_clearance_violation is None:
                        first_clearance_violation = {
                            "sample_index": sample_index,
                            "time_s": float(sample["time_s"]),
                            "phase": str(sample["phase"]),
                            "pair_type": "robot_environment",
                            "distance_m": float(distance),
                            "geom_a": geometry_description(model, geom_a),
                            "geom_b": geometry_description(model, geom_b),
                        }

                if distance < -PENETRATION_TOLERANCE_M:
                    env_penetrations += 1
                    penetration_events += 1

            self_min = math.inf
            self_min_pair: tuple[int, int] | None = None
            self_below_threshold = 0
            self_penetrations = 0

            for geom_a, geom_b in self_pairs:
                distance, fromto = pair_distance(
                    model, data, geom_a, geom_b, args.distance_cap_m
                )
                key = (geom_a, geom_b)
                update_pair_minimum(
                    self_pair_minima,
                    key,
                    "robot_self",
                    distance,
                    fromto,
                    sample,
                    model,
                )

                if distance < self_min:
                    self_min = distance
                    self_min_pair = key

                if distance <= args.minimum_clearance_m + DISTANCE_TOLERANCE_M:
                    self_below_threshold += 1
                    clearance_violation_events += 1
                    unique_violating_pairs.add(("self", geom_a, geom_b))
                    if first_clearance_violation is None:
                        first_clearance_violation = {
                            "sample_index": sample_index,
                            "time_s": float(sample["time_s"]),
                            "phase": str(sample["phase"]),
                            "pair_type": "robot_self",
                            "distance_m": float(distance),
                            "geom_a": geometry_description(model, geom_a),
                            "geom_b": geometry_description(model, geom_b),
                        }

                if distance < -PENETRATION_TOLERANCE_M:
                    self_penetrations += 1
                    penetration_events += 1

            minimum_environment_clearance = min(
                minimum_environment_clearance, env_min
            )
            if math.isfinite(self_min):
                minimum_self_clearance = min(minimum_self_clearance, self_min)

            sample_overall = min(env_min, self_min)
            minimum_overall_clearance = min(
                minimum_overall_clearance, sample_overall
            )

            env_pair_labels = ("", "")
            if env_min_pair is not None:
                env_pair_labels = (
                    geometry_description(model, env_min_pair[0])["label"],
                    geometry_description(model, env_min_pair[1])["label"],
                )

            self_pair_labels = ("", "")
            if self_min_pair is not None:
                self_pair_labels = (
                    geometry_description(model, self_min_pair[0])["label"],
                    geometry_description(model, self_min_pair[1])["label"],
                )

            clearance_ok = (
                env_below_threshold == 0 and self_below_threshold == 0
            )
            joint_limits_ok = len(joint_violations) == 0

            output_row: dict[str, Any] = {
                "sample_index": sample_index,
                "source_segment_index": sample["source_segment_index"],
                "interpolation_fraction": sample["interpolation_fraction"],
                "time_s": sample["time_s"],
                "phase": sample["phase"],
                "minimum_environment_clearance_m": env_min,
                "closest_environment_robot_geometry": env_pair_labels[0],
                "closest_environment_scene_geometry": env_pair_labels[1],
                "minimum_self_clearance_m": (
                    self_min if math.isfinite(self_min) else ""
                ),
                "closest_self_geometry_a": self_pair_labels[0],
                "closest_self_geometry_b": self_pair_labels[1],
                "minimum_overall_clearance_m": sample_overall,
                "environment_pairs_at_or_below_threshold": env_below_threshold,
                "self_pairs_at_or_below_threshold": self_below_threshold,
                "environment_penetrating_pairs": env_penetrations,
                "self_penetrating_pairs": self_penetrations,
                "joint_limit_violation_count": len(joint_violations),
                "clearance_ok": int(clearance_ok),
                "joint_limits_ok": int(joint_limits_ok),
            }

            for joint_name, value in zip(JOINT_NAMES, q):
                output_row[f"q_{joint_name}_rad"] = float(value)

            writer.writerow(output_row)

    all_checks_passed = (
        clearance_violation_events == 0 and joint_limit_violation_events == 0
    )
    status = "PASS" if all_checks_passed else "FAIL"

    def closest_pairs(
        pair_minima: dict[tuple[int, int], dict[str, Any]], count: int
    ) -> list[dict[str, Any]]:
        ordered = sorted(
            pair_minima.values(),
            key=lambda item: item["minimum_distance_m"],
        )
        return ordered[:count]

    scene_collision_masks = {
        "environment_geoms_with_contype_zero": int(
            sum(
                int(model.geom_contype[g]) == 0
                for g in environment_geoms
            )
        ),
        "environment_geoms_with_conaffinity_zero": int(
            sum(
                int(model.geom_conaffinity[g]) == 0
                for g in environment_geoms
            )
        ),
    }

    summary = {
        "status": status,
        "status_meaning": (
            "PASS means no sampled geometry pair was at or below the requested "
            "clearance threshold and no sampled joint position violated its "
            "MuJoCo joint limits. It is not a certified safety result."
        ),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "mujoco_version": str(mujoco.__version__),
        "scene_xml": str(scene_path),
        "reference_csv": str(reference_path),
        "home_keyframe_used": home_keyframe_used,
        "trajectory": {
            "reference_row_count": int(len(times)),
            "sampled_configuration_count": int(len(samples)),
            "start_time_s": float(times[0]),
            "end_time_s": float(times[-1]),
            "duration_s": float(times[-1] - times[0]),
            "sampling_method": "piecewise-linear interpolation of CSV joint positions",
            "maximum_joint_step_requested_rad": float(
                args.maximum_joint_step_rad
            ),
            "maximum_joint_step_observed_rad": float(
                max_observed_joint_step
            ),
        },
        "geometry_counts": {
            "model_total_geometries": int(model.ngeom),
            "robot_collision_geometries": int(len(robot_geoms)),
            "environment_geometries": int(len(environment_geoms)),
            "robot_environment_pairs_per_configuration": int(
                environment_pair_count
            ),
            "robot_self_pairs_checked_per_configuration": int(len(self_pairs)),
            "robot_self_pairs_excluded": int(len(excluded_self_pairs)),
        },
        "scene_collision_masks": scene_collision_masks,
        "clearance_requirement": {
            "minimum_clearance_m": float(args.minimum_clearance_m),
            "distance_tolerance_m": DISTANCE_TOLERANCE_M,
            "penetration_tolerance_m": PENETRATION_TOLERANCE_M,
        },
        "minimum_clearances": {
            "robot_environment_m": json_safe_distance(
                minimum_environment_clearance
            ),
            "robot_self_m": json_safe_distance(minimum_self_clearance),
            "overall_m": json_safe_distance(minimum_overall_clearance),
        },
        "events": {
            "clearance_violation_pair_sample_events": int(
                clearance_violation_events
            ),
            "unique_geometry_pairs_violating_clearance": int(
                len(unique_violating_pairs)
            ),
            "penetrating_pair_sample_events": int(penetration_events),
            "joint_limit_violation_events": int(joint_limit_violation_events),
        },
        "first_clearance_violation": first_clearance_violation,
        "first_joint_limit_violation": first_joint_limit_violation,
        "joint_limits": joint_limits,
        "closest_robot_environment_pairs": closest_pairs(
            environment_pair_minima, args.top_pairs
        ),
        "closest_robot_self_pairs": closest_pairs(
            self_pair_minima, args.top_pairs
        ),
        "excluded_self_collision_pairs": excluded_self_pairs,
        "method_limitations": [
            (
                "The path is checked at discrete configurations generated by "
                "piecewise-linear interpolation; this is not a mathematical "
                "continuous-collision-detection guarantee."
            ),
            (
                "Same-link and direct parent-child robot geometry pairs are "
                "excluded from self-collision checks because their primitives "
                "normally overlap by design around joints."
            ),
            (
                "The scene environment geoms have disabled MuJoCo contact masks. "
                "This script evaluates their geometric distances directly and "
                "does not enable physical contact response."
            ),
            (
                "The result is a simulation geometry preflight, not a clinical, "
                "mechanical, or functional safety certification."
            ),
        ],
        "output_files": {
            "sample_csv": str(output_csv.resolve()),
            "metrics_json": str(output_json.resolve()),
        },
    }

    with output_json.open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, allow_nan=False)

    print("\n========== TASK-SPACE COLLISION PREFLIGHT ==========")
    print(f"MuJoCo version:                 {mujoco.__version__}")
    print(f"Reference CSV rows:             {len(times)}")
    print(f"Trajectory duration:            {times[-1] - times[0]:.3f} s")
    print(f"Checked configurations:         {len(samples)}")
    print(f"Maximum joint step observed:    {max_observed_joint_step:.6f} rad")
    print()
    print(f"Robot collision geometries:     {len(robot_geoms)}")
    print(f"Environment geometries:         {len(environment_geoms)}")
    print(f"Robot/environment pairs:        {environment_pair_count}")
    print(f"Non-adjacent self pairs:         {len(self_pairs)}")
    print()
    print(
        "Minimum environment clearance: "
        f"{minimum_environment_clearance:.6f} m"
    )
    if math.isfinite(minimum_self_clearance):
        print(f"Minimum self clearance:         {minimum_self_clearance:.6f} m")
    else:
        print("Minimum self clearance:         N/A")
    print(f"Minimum overall clearance:      {minimum_overall_clearance:.6f} m")
    print(
        "Required clearance threshold:  "
        f"{args.minimum_clearance_m:.6f} m"
    )
    print()
    print(f"Clearance violation events:     {clearance_violation_events}")
    print(f"Penetrating pair events:        {penetration_events}")
    print(f"Joint-limit violation events:   {joint_limit_violation_events}")
    print(f"\nRESULT: {status}")
    print(f"\nSample report: {output_csv.resolve()}")
    print(f"Metrics report: {output_json.resolve()}")
    print("====================================================\n")

    if first_clearance_violation is not None:
        print("First clearance violation:")
        print(json.dumps(first_clearance_violation, indent=2))
    if first_joint_limit_violation is not None:
        print("First joint-limit violation:")
        print(json.dumps(first_joint_limit_violation, indent=2))

    return 0 if status == "PASS" else 2


def main() -> int:
    args = parse_arguments()
    try:
        return run_preflight(args)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
