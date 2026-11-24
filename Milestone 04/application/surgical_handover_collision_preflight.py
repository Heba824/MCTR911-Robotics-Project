#!/usr/bin/env python3
"""Sampled MuJoCo collision preflight for a planned surgical instrument handover.

This checker consumes the JSON produced by plan_surgical_handover.py and
reconstructs each point-to-point segment with the project's quintic joint
trajectory generator. It checks:
  * UR5e collision proxies against scene obstacles;
  * non-adjacent UR5e self-collision pairs;
  * the selected instrument against scene obstacles while it is rigidly
    carried by the MuJoCo attachment_site;
  * the robot retreat after the instrument is released at the handover pose;
  * MuJoCo joint limits.

The instrument attachment is a kinematic mocap approximation, matching the
simulation bridge. It is not a physical gripper or contact-dynamics model.
Native MuJoCo CCD is used for pairwise geometry-distance queries. The
trajectory itself is still checked at discrete samples, so this is not
continuous trajectory collision detection or a safety certification.
"""
from __future__ import annotations

import argparse
import hashlib
import csv
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TRAJECTORY_DIR = ROOT / "Milestone 04" / "trajectory"
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
ROBOT_BASE_BODY = "base"
ROBOT_COLLISION_GROUP = 3
SUPPORT_BODY_NAMES = ("robot_station",)
ATTACHMENT_SITE_NAME = "attachment_site"
HANDOVER_BODY_NAME = "surgeon_handover_target"
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

DEFAULT_SCENE = ROOT / "Milestone 01" / "Robot Model" / "scene_surgical_handover.xml"
DEFAULT_PLAN_DIR = ROOT / "Milestone 04" / "Results" / "Surgical_Handover" / "Planning"
DEFAULT_OUTPUT_DIR = ROOT / "Milestone 04" / "Results" / "Surgical_Handover" / "Collision_Preflight"


@dataclass(frozen=True)
class TrajectorySample:
    time_s: float
    phase: str
    q: np.ndarray


def _six(values: Iterable[float], name: str) -> np.ndarray:
    result = np.asarray(values, dtype=float)
    if result.shape != (6,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain exactly six finite values")
    return result.copy()


def load_plan(path: str | Path, expected_instrument: str) -> dict[str, Any]:
    plan_path = Path(path).expanduser().resolve()
    if not plan_path.is_file():
        raise FileNotFoundError(f"Plan JSON does not exist: {plan_path}")
    payload = json.loads(plan_path.read_text(encoding="utf-8"))
    if payload.get("instrument") != expected_instrument:
        raise ValueError(
            f"Plan instrument is {payload.get('instrument')!r}, expected {expected_instrument!r}"
        )
    initial = _six(payload.get("initial_joint_positions_rad", []), "initial_joint_positions_rad")
    waypoints = payload.get("waypoints")
    if not isinstance(waypoints, list) or len(waypoints) != len(REQUIRED_WAYPOINTS):
        raise ValueError(f"Plan must contain exactly {len(REQUIRED_WAYPOINTS)} waypoints")
    names = tuple(str(item.get("name", "")) for item in waypoints)
    if names != REQUIRED_WAYPOINTS:
        raise ValueError(f"Unexpected waypoint sequence: {names}")
    for item in waypoints:
        _six(item.get("joint_positions_rad", []), f"{item['name']}.joint_positions_rad")
        duration = float(item.get("duration_from_previous_s", 0.0))
        if not math.isfinite(duration) or duration <= 0.0:
            raise ValueError(f"Waypoint {item['name']} has an invalid segment duration")
        if not bool(item.get("ik_success", False)):
            raise ValueError(f"Waypoint {item['name']} is not marked as IK-successful")
    payload["_resolved_plan_path"] = str(plan_path)
    payload["_initial_q"] = initial
    return payload


def build_quintic_samples(plan: dict[str, Any], sample_period_s: float) -> tuple[list[TrajectorySample], float, float]:
    """Reconstruct the plan as sequential quintic segments sampled at dt.

    Returns samples, the time at which pickup is completed (attachment), and
    the time at which the handover waypoint is completed (release).
    """
    dt = float(sample_period_s)
    if not math.isfinite(dt) or dt <= 0.0:
        raise ValueError("sample_period_s must be positive and finite")
    current_q = _six(plan["_initial_q"], "initial_joint_positions_rad")
    global_start = 0.0
    samples: list[TrajectorySample] = []
    attach_time: float | None = None
    release_time: float | None = None

    for index, waypoint in enumerate(plan["waypoints"]):
        goal_q = _six(waypoint["joint_positions_rad"], f"{waypoint['name']}.joint_positions_rad")
        duration = float(waypoint["duration_from_previous_s"])
        trajectory = QuinticJointTrajectory(current_q, goal_q, duration)
        count = int(math.floor(duration / dt))
        local_times = (dt * np.arange(count + 1, dtype=float)).tolist()
        if duration - local_times[-1] > 1e-10:
            local_times.append(duration)
        else:
            local_times[-1] = duration
        if index > 0:
            local_times = local_times[1:]
        for local_time in local_times:
            point = trajectory.evaluate(float(local_time))
            samples.append(
                TrajectorySample(
                    time_s=float(global_start + local_time),
                    phase=str(waypoint["name"]),
                    q=np.asarray(point.position_rad, dtype=float).copy(),
                )
            )
        global_start += duration
        current_q = goal_q
        if waypoint["name"] == "pickup":
            attach_time = global_start
        if waypoint["name"] == "handover":
            release_time = global_start

    if attach_time is None or release_time is None or release_time <= attach_time:
        raise ValueError("Could not determine valid pickup attachment and handover release times")
    if len(samples) < 2:
        raise ValueError("The reconstructed trajectory has too few samples")
    return samples, float(attach_time), float(release_time)


def rotation_matrix_to_quaternion_wxyz(rotation: np.ndarray) -> np.ndarray:
    """Convert a proper 3x3 rotation matrix to MuJoCo's wxyz quaternion order."""
    r = np.asarray(rotation, dtype=float)
    if r.shape != (3, 3) or not np.all(np.isfinite(r)):
        raise ValueError("rotation must be a finite 3x3 matrix")
    if not np.allclose(r.T @ r, np.eye(3), atol=1e-5, rtol=0.0) or not np.isclose(
        np.linalg.det(r), 1.0, atol=1e-5, rtol=0.0
    ):
        raise ValueError("rotation must be orthonormal with determinant +1")
    trace = float(np.trace(r))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (r[2, 1] - r[1, 2]) / s
        y = (r[0, 2] - r[2, 0]) / s
        z = (r[1, 0] - r[0, 1]) / s
    elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
        s = math.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2.0
        w = (r[2, 1] - r[1, 2]) / s
        x = 0.25 * s
        y = (r[0, 1] + r[1, 0]) / s
        z = (r[0, 2] + r[2, 0]) / s
    elif r[1, 1] > r[2, 2]:
        s = math.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2.0
        w = (r[0, 2] - r[2, 0]) / s
        x = (r[0, 1] + r[1, 0]) / s
        y = 0.25 * s
        z = (r[1, 2] + r[2, 1]) / s
    else:
        s = math.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2.0
        w = (r[1, 0] - r[0, 1]) / s
        x = (r[0, 2] + r[2, 0]) / s
        y = (r[1, 2] + r[2, 1]) / s
        z = 0.25 * s
    q = np.asarray([w, x, y, z], dtype=float)
    return q / np.linalg.norm(q)


def compute_attachment_relative_pose(
    tool_position: np.ndarray,
    tool_rotation: np.ndarray,
    instrument_position: np.ndarray,
    instrument_rotation: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return instrument pose expressed in the tool frame."""
    p_tool = np.asarray(tool_position, dtype=float)
    r_tool = np.asarray(tool_rotation, dtype=float)
    p_inst = np.asarray(instrument_position, dtype=float)
    r_inst = np.asarray(instrument_rotation, dtype=float)
    if p_tool.shape != (3,) or p_inst.shape != (3,) or not np.all(np.isfinite(p_tool)) or not np.all(np.isfinite(p_inst)):
        raise ValueError("tool and instrument positions must be finite 3-vectors")
    for value, label in ((r_tool, "tool_rotation"), (r_inst, "instrument_rotation")):
        if value.shape != (3, 3) or not np.all(np.isfinite(value)):
            raise ValueError(f"{label} must be a finite 3x3 matrix")
        if not np.allclose(value.T @ value, np.eye(3), atol=1e-5, rtol=0.0):
            raise ValueError(f"{label} must be orthonormal")
    relative_position = r_tool.T @ (p_inst - p_tool)
    relative_rotation = r_tool.T @ r_inst
    return relative_position, relative_rotation


def _body_id(mujoco: Any, model: Any, name: str) -> int:
    value = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name))
    if value < 0:
        raise ValueError(f"Required MuJoCo body '{name}' was not found")
    return value


def _is_descendant(model: Any, body_id: int, root_id: int) -> bool:
    current = int(body_id)
    while current != 0:
        if current == root_id:
            return True
        current = int(model.body_parentid[current])
    return root_id == 0


def _geom_name(mujoco: Any, model: Any, geom_id: int) -> str:
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(geom_id))
    return str(name) if name else f"geom_{geom_id}"


def _body_name(mujoco: Any, model: Any, body_id: int) -> str:
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(body_id))
    return str(name) if name else f"body_{body_id}"


def _select_geometries(mujoco: Any, model: Any, instrument: str) -> dict[str, Any]:
    base_id = _body_id(mujoco, model, ROBOT_BASE_BODY)
    support_ids = {_body_id(mujoco, model, name) for name in SUPPORT_BODY_NAMES}
    robot_body_ids = {
        body_id for body_id in range(int(model.nbody)) if _is_descendant(model, body_id, base_id)
    }
    robot_geoms = [
        geom_id for geom_id in range(int(model.ngeom))
        if int(model.geom_bodyid[geom_id]) in robot_body_ids
        and int(model.geom_group[geom_id]) == ROBOT_COLLISION_GROUP
    ]
    if not robot_geoms:
        raise ValueError(f"No UR5e collision proxies found in geom group {ROBOT_COLLISION_GROUP}")

    instrument_body_id = _body_id(mujoco, model, instrument)
    instrument_geoms = [
        geom_id for geom_id in range(int(model.ngeom))
        if int(model.geom_bodyid[geom_id]) == instrument_body_id
    ]
    if not instrument_geoms:
        raise ValueError(f"No geometries were found on instrument body '{instrument}'")
    mocap_id = int(model.body_mocapid[instrument_body_id])
    if mocap_id < 0:
        raise ValueError(f"Instrument body '{instrument}' must be a MuJoCo mocap body")

    all_environment_geoms = [
        geom_id for geom_id in range(int(model.ngeom))
        if int(model.geom_bodyid[geom_id]) not in robot_body_ids
        and int(model.geom_bodyid[geom_id]) not in support_ids
    ]
    static_environment_geoms = [g for g in all_environment_geoms if g not in instrument_geoms]

    # Exclude same-link and direct parent-child pairs, whose proxies overlap by design.
    self_pairs: list[tuple[int, int]] = []
    for i, geom_a in enumerate(robot_geoms):
        body_a = int(model.geom_bodyid[geom_a])
        parent_a = int(model.body_parentid[body_a])
        for geom_b in robot_geoms[i + 1:]:
            body_b = int(model.geom_bodyid[geom_b])
            parent_b = int(model.body_parentid[body_b])
            if body_a == body_b or parent_a == body_b or parent_b == body_a:
                continue
            self_pairs.append((geom_a, geom_b))

    handover_body_id = _body_id(mujoco, model, HANDOVER_BODY_NAME)
    receiver_geoms = [g for g in static_environment_geoms if int(model.geom_bodyid[g]) == handover_body_id]

    tray_body_id = _body_id(mujoco, model, "sterile_tray")
    trolley_body_id = _body_id(mujoco, model, "tool_trolley")
    tray_base_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "tray_base"))
    pad_name = "tray_scalpel_pad" if instrument == "scalpel" else "tray_forceps_pad"
    pickup_pad_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, pad_name))
    trolley_top_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "trolley_top"))
    if tray_base_id < 0 or pickup_pad_id < 0 or trolley_top_id < 0:
        raise ValueError("Required tray/trolley pickup-assembly geometry was not found")
    if (
        int(model.geom_bodyid[tray_base_id]) != tray_body_id
        or int(model.geom_bodyid[pickup_pad_id]) != tray_body_id
        or int(model.geom_bodyid[trolley_top_id]) != trolley_body_id
    ):
        raise ValueError(
            "Pickup support envelope must contain tray_base and the selected pad "
            "on sterile_tray, plus trolley_top on tool_trolley"
        )
    # The sterile tray rests directly on trolley_top. The instrument is picked
    # from the tray while the tray is carried by this trolley, so the top is a
    # secondary pickup-assembly support surface. Its distance is still measured
    # and must reach the full configured clearance before the allowance ends.
    # Only this one trolley geometry is included; the rim, shelf, legs and all
    # other trolley geometry remain ordinary collision-checking obstacles.
    pickup_support_geoms = [tray_base_id, pickup_pad_id, trolley_top_id]

    return {
        "robot_body_ids": robot_body_ids,
        "robot_geoms": robot_geoms,
        "all_environment_geoms": all_environment_geoms,
        "static_environment_geoms": static_environment_geoms,
        "instrument_body_id": instrument_body_id,
        "instrument_geoms": instrument_geoms,
        "instrument_mocap_id": mocap_id,
        "self_pairs": self_pairs,
        "handover_body_id": handover_body_id,
        "receiver_geoms": receiver_geoms,
        "pickup_support_geoms": pickup_support_geoms,
        "pickup_support_geom_names": [
            _geom_name(mujoco, model, geom_id) for geom_id in pickup_support_geoms
        ],
    }


def _joint_data(mujoco: Any, model: Any) -> tuple[list[int], list[int], list[tuple[float, float]]]:
    joint_ids: list[int] = []
    qpos_addresses: list[int] = []
    limits: list[tuple[float, float]] = []
    for name in JOINT_NAMES:
        joint_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name))
        if joint_id < 0 or int(model.jnt_type[joint_id]) != int(mujoco.mjtJoint.mjJNT_HINGE):
            raise ValueError(f"Expected scalar hinge joint '{name}' in the handover scene")
        joint_ids.append(joint_id)
        qpos_addresses.append(int(model.jnt_qposadr[joint_id]))
        if not bool(model.jnt_limited[joint_id]):
            raise ValueError(f"Joint '{name}' must have finite MuJoCo position limits")
        limits.append(tuple(float(x) for x in model.jnt_range[joint_id]))
    return joint_ids, qpos_addresses, limits


def _enable_native_ccd(mujoco: Any, model: Any) -> None:
    """Enable MuJoCo's native convex-collider distance pipeline explicitly.

    MuJoCo 3.2.7 exposes this through mjtEnableBit.mjENBL_NATIVECCD.
    Fail closed rather than silently reverting to the legacy distance backend.
    """
    flag = getattr(mujoco.mjtEnableBit, "mjENBL_NATIVECCD", None)
    if flag is None:
        raise RuntimeError(
            "Native MuJoCo CCD is unavailable. Activate the project venv and "
            "install a MuJoCo version exposing "
            "mjtEnableBit.mjENBL_NATIVECCD (tested with 3.2.7)."
        )
    model.opt.enableflags |= int(flag)
    if not (model.opt.enableflags & int(flag)):
        raise RuntimeError("Failed to enable MuJoCo native CCD")


def _distance(mujoco: Any, model: Any, data: Any, geom_a: int, geom_b: int, cap: float) -> float:
    fromto = np.zeros(6, dtype=np.float64)
    value = float(mujoco.mj_geomDistance(model, data, int(geom_a), int(geom_b), float(cap), fromto))
    if not math.isfinite(value):
        raise RuntimeError(f"Non-finite geometry distance for pair {geom_a}, {geom_b}")
    return value


def _update_instrument_mocap(
    mujoco: Any, model: Any, data: Any, selection: dict[str, Any],
    relative_position: np.ndarray, relative_rotation: np.ndarray,
) -> None:
    site_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, ATTACHMENT_SITE_NAME))
    if site_id < 0:
        raise ValueError(f"Required attachment site '{ATTACHMENT_SITE_NAME}' was not found")
    tool_position = np.asarray(data.site_xpos[site_id], dtype=float)
    tool_rotation = np.asarray(data.site_xmat[site_id], dtype=float).reshape(3, 3)
    instrument_position = tool_position + tool_rotation @ relative_position
    instrument_rotation = tool_rotation @ relative_rotation
    mocap_id = int(selection["instrument_mocap_id"])
    data.mocap_pos[mocap_id] = instrument_position
    data.mocap_quat[mocap_id] = rotation_matrix_to_quaternion_wxyz(instrument_rotation)
    mujoco.mj_forward(model, data)


def _record_pair(
    mujoco: Any, model: Any, data: Any, pairs: Iterable[tuple[int, int]],
    cap: float, warning_clearance_m: float, category: str, time_s: float, phase: str,
    *, allow_touch: bool = False,
) -> tuple[float, str, str, int, int, list[dict[str, Any]]]:
    minimum = float(cap)
    closest_a = ""
    closest_b = ""
    within_warning = 0
    collisions = 0
    events: list[dict[str, Any]] = []
    for geom_a, geom_b in pairs:
        distance = _distance(mujoco, model, data, geom_a, geom_b, cap)
        name_a = _geom_name(mujoco, model, geom_a)
        name_b = _geom_name(mujoco, model, geom_b)
        if distance < minimum:
            minimum = distance
            closest_a, closest_b = name_a, name_b
        if distance < warning_clearance_m:
            within_warning += 1
        if distance < 0.0 or (distance == 0.0 and not allow_touch):
            collisions += 1
            events.append({
                "category": category,
                "time_s": float(time_s),
                "phase": phase,
                "geom_a": name_a,
                "geom_b": name_b,
                "signed_distance_m": float(distance),
            })
    return minimum, closest_a, closest_b, within_warning, collisions, events



def pickup_support_contact_allowed(phase: str, instrument_attached: bool, contact_cleared: bool) -> bool:
    """Allow a monitored pickup-assembly near-support envelope during pickup/lift.

    This is not a collision exemption: support distances remain measured, any
    penetration still fails, and the allowance ends only after all selected
    support clearances reach the configured minimum threshold.
    """
    return bool(instrument_attached and not contact_cleared and phase in ("pickup", "lift"))


def pickup_support_pairs(selection: dict[str, Any]) -> list[tuple[int, int]]:
    return [
        (instrument, support)
        for instrument in selection["instrument_geoms"]
        for support in selection["pickup_support_geoms"]
    ]


def carried_environment_pairs(
    model: Any, selection: dict[str, Any], phase: str, allow_near_support: bool,
) -> list[tuple[int, int]]:
    obstacles = selection["static_environment_geoms"]
    if phase in ("handover", "retreat", "return_home"):
        obstacles = [g for g in obstacles if int(model.geom_bodyid[g]) != selection["handover_body_id"]]
    if allow_near_support:
        obstacles = [g for g in obstacles if g not in selection["pickup_support_geoms"]]
    return [(a, b) for a in selection["instrument_geoms"] for b in obstacles]


def run_preflight(
    plan: dict[str, Any], scene_path: Path, output_dir: Path, *,
    sample_period_s: float, distance_cap_m: float,
    minimum_clearance_m: float, warning_clearance_m: float,
) -> dict[str, Any]:
    try:
        import mujoco
    except ImportError as exc:
        raise RuntimeError(
            "Activate mctr911_kinematics_venv with MuJoCo Python 3.2.7 "
            "or another version exposing native CCD"
        ) from exc

    model = mujoco.MjModel.from_xml_path(str(scene_path))
    _enable_native_ccd(mujoco, model)
    if (int(model.nq), int(model.nv), int(model.nu)) != (6, 6, 6):
        raise ValueError(f"Expected nq=6, nv=6, nu=6; got {model.nq}, {model.nv}, {model.nu}")
    data = mujoco.MjData(model)
    key_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home"))
    if key_id < 0:
        raise ValueError("The handover scene must contain the 'home' keyframe")
    mujoco.mj_resetDataKeyframe(model, data, key_id)
    mujoco.mj_forward(model, data)

    selection = _select_geometries(mujoco, model, plan["instrument"])
    if "instrument_body_initial_pose_world" in plan:
        body_id = selection["instrument_body_id"]
        scene_body_pose = np.eye(4)
        scene_body_pose[:3, :3] = data.xmat[body_id].reshape(3, 3)
        scene_body_pose[:3, 3] = data.xpos[body_id]
        if not np.allclose(plan["instrument_body_initial_pose_world"], scene_body_pose, atol=1e-8, rtol=0):
            raise ValueError("Planned instrument body pose differs from the scene; regenerate the plan")
    joint_ids, qpos_addresses, joint_limits = _joint_data(mujoco, model)
    samples, attach_time, release_time = build_quintic_samples(plan, sample_period_s)
    initial_q = _six(plan["_initial_q"], "initial_joint_positions_rad")
    home_q = np.asarray(data.qpos[qpos_addresses], dtype=float).copy()
    home_error = float(np.max(np.abs(initial_q - home_q)))
    if home_error > 0.05:
        raise ValueError(f"Plan start differs from scene HOME by {home_error:.6f} rad")

    site_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, ATTACHMENT_SITE_NAME))
    if site_id < 0:
        raise ValueError(f"Required attachment site '{ATTACHMENT_SITE_NAME}' was not found")

    output_rows: list[dict[str, Any]] = []
    collision_events: list[dict[str, Any]] = []
    joint_limit_events: list[dict[str, Any]] = []
    closest_pairs: dict[tuple[str, str, str], dict[str, Any]] = {}
    minimum_by_category = {"robot_environment": float(distance_cap_m), "robot_self": float(distance_cap_m), "carried_instrument_environment": float(distance_cap_m)}
    minimum_pair_by_category: dict[str, tuple[str, str] | None] = {key: None for key in minimum_by_category}
    minimum_time_by_category = {key: 0.0 for key in minimum_by_category}

    relative_position: np.ndarray | None = None
    relative_rotation: np.ndarray | None = None
    released_instrument_position: np.ndarray | None = None
    released_instrument_rotation: np.ndarray | None = None
    attachment_captured = False
    instrument_attached = False
    instrument_released = False
    pickup_support_contact_cleared = False
    pickup_support_contact_clear_time_s: float | None = None
    pickup_support_contact_allowed_sample_count = 0
    pickup_support_contact_clearance_violation = False

    robot_geom_ids = selection["robot_geoms"]
    static_env = selection["static_environment_geoms"]
    selected_geoms = selection["instrument_geoms"]
    self_pairs = selection["self_pairs"]
    all_env_with_selected = selection["all_environment_geoms"]
    support_pairs = pickup_support_pairs(selection)

    for sample_index, sample in enumerate(samples):
        q = _six(sample.q, f"sample[{sample_index}].q")
        data.qpos[qpos_addresses] = q
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)

        if not attachment_captured and sample.time_s >= attach_time - 1e-10:
            instrument_body_id = selection["instrument_body_id"]
            relative_position, relative_rotation = compute_attachment_relative_pose(
                np.asarray(data.site_xpos[site_id], dtype=float),
                np.asarray(data.site_xmat[site_id], dtype=float).reshape(3, 3),
                np.asarray(data.xpos[instrument_body_id], dtype=float),
                np.asarray(data.xmat[instrument_body_id], dtype=float).reshape(3, 3),
            )
            attachment_captured = True
            instrument_attached = True

        if instrument_attached and sample.time_s <= release_time + 1e-10:
            assert relative_position is not None and relative_rotation is not None
            _update_instrument_mocap(
                mujoco, model, data, selection, relative_position, relative_rotation
            )
        elif instrument_attached and sample.time_s > release_time + 1e-10:
            # The preceding sample at the handover endpoint left the instrument
            # at its released pose. Preserve that world pose during robot retreat.
            instrument_body_id = selection["instrument_body_id"]
            released_instrument_position = np.asarray(data.xpos[instrument_body_id], dtype=float).copy()
            released_instrument_rotation = np.asarray(data.xmat[instrument_body_id], dtype=float).reshape(3, 3).copy()
            mocap_id = int(selection["instrument_mocap_id"])
            data.mocap_pos[mocap_id] = released_instrument_position
            data.mocap_quat[mocap_id] = rotation_matrix_to_quaternion_wxyz(released_instrument_rotation)
            mujoco.mj_forward(model, data)
            instrument_attached = False
            instrument_released = True

        # Before pickup, the selected instrument is a stationary obstacle.
        # While carried, omit its intentional robot/tool attachment pair.
        # After release, it is again a stationary obstacle for the retreat.
        if sample.time_s < attach_time - 1e-10:
            robot_obstacles = all_env_with_selected
        elif sample.time_s <= release_time + 1e-10:
            robot_obstacles = static_env
        else:
            robot_obstacles = all_env_with_selected

        robot_env_pairs = [(a, b) for a in robot_geom_ids for b in robot_obstacles]
        robot_self_pairs = self_pairs
        env_min, env_a, env_b, env_near, _, env_events = _record_pair(
            mujoco, model, data, robot_env_pairs, distance_cap_m, warning_clearance_m,
            "robot_environment", sample.time_s, sample.phase,
        )
        self_min, self_a, self_b, self_near, _, self_events = _record_pair(
            mujoco, model, data, robot_self_pairs, distance_cap_m, warning_clearance_m,
            "robot_self", sample.time_s, sample.phase,
        )

        carried_min = float(distance_cap_m)
        carried_a = carried_b = ""
        carried_near = 0
        carried_events: list[dict[str, Any]] = []
        pickup_support_clearance_m: float | None = None
        allow_pickup_support_contact = False
        support_events: list[dict[str, Any]] = []
        if not instrument_released and not pickup_support_contact_cleared:
            (pickup_support_clearance_m, _, _, _, _, support_events) = _record_pair(
                mujoco, model, data, support_pairs, distance_cap_m, warning_clearance_m,
                "pickup_support", sample.time_s, sample.phase, allow_touch=True,
            )
            if instrument_attached:
                if not pickup_support_contact_allowed(sample.phase, True, False):
                    pickup_support_contact_clearance_violation = True
                elif pickup_support_clearance_m >= minimum_clearance_m:
                    pickup_support_contact_cleared = True
                    pickup_support_contact_clear_time_s = float(sample.time_s)
                else:
                    allow_pickup_support_contact = True
                    pickup_support_contact_allowed_sample_count += 1

        if instrument_attached or instrument_released:
            carried_pairs = carried_environment_pairs(
                model, selection, sample.phase, allow_pickup_support_contact
            )
            (carried_min, carried_a, carried_b, carried_near,
             _, carried_events) = _record_pair(
                mujoco, model, data, carried_pairs, distance_cap_m, warning_clearance_m,
                "carried_instrument_environment", sample.time_s, sample.phase,
            )

        for category, value, a, b in (
            ("robot_environment", env_min, env_a, env_b),
            ("robot_self", self_min, self_a, self_b),
            ("carried_instrument_environment", carried_min, carried_a, carried_b),
        ):
            if value < minimum_by_category[category]:
                minimum_by_category[category] = value
                minimum_pair_by_category[category] = (a, b) if a and b else None
                minimum_time_by_category[category] = float(sample.time_s)
            if a and b and value < warning_clearance_m:
                key = (category, a, b)
                prior = closest_pairs.get(key)
                if prior is None or value < float(prior["minimum_distance_m"]):
                    closest_pairs[key] = {
                        "category": category, "geom_a": a, "geom_b": b,
                        "minimum_distance_m": float(value), "time_s": float(sample.time_s),
                        "phase": sample.phase,
                    }

        all_events = env_events + self_events + support_events + carried_events
        collision_events.extend(all_events)
        joint_violations = []
        for joint_index, (joint_id, name, bounds) in enumerate(zip(joint_ids, JOINT_NAMES, joint_limits)):
            lo, hi = bounds
            if q[joint_index] < lo - 1e-8 or q[joint_index] > hi + 1e-8:
                joint_violations.append({
                    "time_s": float(sample.time_s), "phase": sample.phase,
                    "joint_name": name, "position_rad": float(q[joint_index]),
                    "lower_limit_rad": lo, "upper_limit_rad": hi,
                })
        joint_limit_events.extend(joint_violations)

        category_minima = (env_min, self_min, carried_min)
        sample_minimum = min(category_minima)
        sample_collisions = len(all_events)
        if sample_collisions:
            sample_status = "COLLISION"
        elif pickup_support_contact_clearance_violation:
            sample_status = "PICKUP_SUPPORT_NOT_CLEARED"
        elif joint_violations:
            sample_status = "JOINT_LIMIT_VIOLATION"
        elif sample_minimum < minimum_clearance_m:
            sample_status = "INSUFFICIENT_CLEARANCE"
        elif sample_minimum < warning_clearance_m:
            sample_status = "WARNING"
        else:
            sample_status = "PASS"
        output_rows.append({
            "sample_index": sample_index,
            "time_s": float(sample.time_s),
            "phase": sample.phase,
            "minimum_robot_environment_distance_m": float(env_min),
            "closest_robot_environment_pair": f"{env_a} <-> {env_b}" if env_a else "",
            "minimum_robot_self_distance_m": float(self_min),
            "closest_robot_self_pair": f"{self_a} <-> {self_b}" if self_a else "",
            "minimum_carried_instrument_environment_distance_m": float(carried_min),
            "closest_carried_instrument_pair": f"{carried_a} <-> {carried_b}" if carried_a else "",
            "instrument_attached_to_robot": int(instrument_attached),
            "instrument_released_at_handover": int(instrument_released),
            "pickup_support_contact_allowed": int(allow_pickup_support_contact),
            "pickup_support_clearance_m": (
                "" if pickup_support_clearance_m is None else float(pickup_support_clearance_m)
            ),
            "pickup_support_clearance_violation": int(pickup_support_contact_clearance_violation),
            "robot_environment_pairs_within_warning": int(env_near),
            "robot_self_pairs_within_warning": int(self_near),
            "instrument_environment_pairs_within_warning": int(carried_near),
            "collision_pair_count": int(sample_collisions),
            "joint_limit_violation_count": len(joint_violations),
            "sample_status": sample_status,
        })

    any_collision = bool(collision_events)
    any_joint_limit = bool(joint_limit_events)
    global_min = min(minimum_by_category.values())
    if any_collision:
        status = "FAIL_COLLISION"
    elif pickup_support_contact_clearance_violation:
        status = "FAIL_PICKUP_SUPPORT_NOT_CLEARED"
    elif any_joint_limit:
        status = "FAIL_JOINT_LIMIT"
    elif global_min < minimum_clearance_m:
        status = "FAIL_CLEARANCE"
    elif global_min < warning_clearance_m:
        status = "WARNING_REVIEW_REQUIRED"
    else:
        status = "PASS"

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"surgical_handover_collision_{plan['instrument']}_{timestamp}.csv"
    json_path = output_dir / f"surgical_handover_collision_{plan['instrument']}_{timestamp}.json"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(output_rows[0].keys()))
        writer.writeheader()
        writer.writerows(output_rows)

    def category_summary(name: str) -> dict[str, Any]:
        pair = minimum_pair_by_category[name]
        return {
            "minimum_distance_m": None if pair is None else float(minimum_by_category[name]),
            "minimum_distance_is_capped_lower_bound": pair is None,
            "lower_bound_m": float(distance_cap_m) if pair is None else float(minimum_by_category[name]),
            "time_s": float(minimum_time_by_category[name]),
            "closest_pair": None if pair is None else {"geom_a": pair[0], "geom_b": pair[1]},
        }

    summary = {
        "status": status,
        "preflight_passed": status == "PASS",
        "instrument": plan["instrument"],
        "plan_json": plan["_resolved_plan_path"],
        "scene_xml": str(scene_path.resolve()),
        "scene_xml_sha256": hashlib.sha256(scene_path.read_bytes()).hexdigest(),
        "mujoco_version": str(mujoco.__version__),
        "collision_distance_backend": {
            "name": "MuJoCo native CCD",
            "native_ccd_enabled": True,
            "enable_flag": "mjtEnableBit.mjENBL_NATIVECCD",
        },
        "model_dimensions": {"nq": int(model.nq), "nv": int(model.nv), "nu": int(model.nu)},
        "home_start_error_max_rad": home_error,
        "trajectory": {
            "sample_count": len(samples), "sample_period_s": float(sample_period_s),
            "duration_s": float(samples[-1].time_s),
            "pickup_attachment_time_s": attach_time,
            "handover_release_time_s": release_time,
            "segments": REQUIRED_WAYPOINTS,
            "interpolation": "project QuinticJointTrajectory point-to-point segments",
        },
        "geometry_counts": {
            "robot_collision_geoms": len(robot_geom_ids),
            "static_scene_geoms_excluding_selected_instrument": len(static_env),
            "selected_instrument_geoms": len(selected_geoms),
            "robot_self_pairs_checked": len(self_pairs),
            "robot_environment_pairs_per_sample_before_pickup_and_after_release": len(robot_geom_ids) * len(all_env_with_selected),
            "robot_environment_pairs_per_sample_while_attached": len(robot_geom_ids) * len(static_env),
            "carried_instrument_environment_pairs_per_sample": len(selected_geoms) * len(static_env),
        },
        "clearance_thresholds": {
            "distance_cap_m": float(distance_cap_m),
            "minimum_clearance_m": float(minimum_clearance_m),
            "warning_clearance_m": float(warning_clearance_m),
        },
        "minimum_clearances": {name: category_summary(name) for name in minimum_by_category},
        "collision_event_count": len(collision_events),
        "collision_events": collision_events[:100],
        "joint_limit_violation_count": len(joint_limit_events),
        "joint_limit_violations": joint_limit_events[:100],
        "closest_pairs_below_warning": sorted(
            closest_pairs.values(), key=lambda item: float(item["minimum_distance_m"])
        )[:50],
        "pickup_support_contact": {
            "policy_type": "temporary_pickup_assembly_near_support_allowance",
            "support_geometries": selection["pickup_support_geom_names"],
            "allowed_only_during_pickup_and_lift": True,
            "support_distances_measured_each_sample": True,
            "penetration_allowed": False,
            "signed_support_distances_checked_for_penetration": True,
            "required_clearance_to_end_allowance_m": float(minimum_clearance_m),
            "cleared": bool(pickup_support_contact_cleared),
            "clear_time_s": pickup_support_contact_clear_time_s,
            "allowed_sample_count": int(pickup_support_contact_allowed_sample_count),
            "clearance_violation": bool(pickup_support_contact_clearance_violation),
        },
        "intentional_contact_policy": {
            "robot_instrument_pair": "excluded while instrument is attached because this is the modeled rigid grasp",
            "instrument_pickup_assembly_support_pair": "tray_base, selected instrument pad, and trolley_top form a temporary monitored pickup-assembly near-support envelope; their distances are measured each sample, penetration is never allowed, and the allowance ends only when every selected support pair reaches the configured minimum clearance during pickup/lift; all other tray/trolley geometry remains checked",
            "instrument_receiver_pair": "excluded during handover, retreat and return_home; the simulated hand is the intended receiver",
            "robot_receiver_pair": "never excluded; UR5e collision proxies remain checked against surgeon hand geometries",
        },
        "limitations": [
            "Sampled kinematic geometry sweep; not continuous collision detection.",
            "Does not simulate actuator lag, tracking error, forces, friction, or physical gripper contact.",
            "UR5e uses simplified collision proxies in MuJoCo geom group 3, not visual meshes.",
            "Same-link and direct parent-child self-collision pairs are excluded because their proxies overlap by design.",
            "The instrument is carried with a mocap rigid-transform approximation matching the bridge; the robot-instrument attachment pair is treated as intentional contact.",
            "Instrument contact with the surgeon receiver body is excluded during handover and after release; robot-to-receiver clearance remains checked.",
            "The result is a simulation preflight, not a safety certification or authorization for real hardware.",
        ],
        "csv_report": str(csv_path.resolve()),
        "json_report": str(json_path.resolve()),
    }
    json_path.write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    return summary


def _latest_plan(plan_dir: Path, instrument: str) -> Path:
    matches = list(plan_dir.glob(f"surgical_handover_plan_{instrument}_*.json"))
    if not matches:
        raise FileNotFoundError(f"No {instrument} plan JSON found in {plan_dir}")
    return max(matches, key=lambda path: path.stat().st_mtime)


def main() -> int:
    parser = argparse.ArgumentParser(description="Collision preflight for a planned UR5e surgical handover")
    parser.add_argument("--instrument", choices=("scalpel", "forceps"), required=True)
    parser.add_argument("--plan-json", type=Path, default=None, help="Specific planner JSON; default is newest plan for instrument")
    parser.add_argument("--plan-dir", type=Path, default=DEFAULT_PLAN_DIR)
    parser.add_argument("--scene-xml", type=Path, default=DEFAULT_SCENE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sample-period-s", type=float, default=0.005)
    parser.add_argument("--distance-cap-m", type=float, default=0.05)
    parser.add_argument("--minimum-clearance-m", type=float, default=0.02)
    parser.add_argument("--warning-clearance-m", type=float, default=0.05)
    args = parser.parse_args()
    if args.minimum_clearance_m < 0.0 or args.warning_clearance_m <= args.minimum_clearance_m:
        raise ValueError("Require 0 <= minimum clearance < warning clearance")
    if args.distance_cap_m < args.warning_clearance_m:
        raise ValueError("distance cap must be >= warning clearance")
    scene_path = args.scene_xml.expanduser().resolve()
    if not scene_path.is_file():
        raise FileNotFoundError(f"MuJoCo scene not found: {scene_path}")
    plan_path = args.plan_json.expanduser().resolve() if args.plan_json else _latest_plan(args.plan_dir.expanduser().resolve(), args.instrument)
    plan = load_plan(plan_path, args.instrument)
    summary = run_preflight(
        plan, scene_path, args.output_dir.expanduser().resolve(),
        sample_period_s=args.sample_period_s,
        distance_cap_m=args.distance_cap_m,
        minimum_clearance_m=args.minimum_clearance_m,
        warning_clearance_m=args.warning_clearance_m,
    )
    print("========== SURGICAL HANDOVER COLLISION PREFLIGHT ==========")
    print(f"Instrument: {summary['instrument']}")
    print(f"Scene: {summary['scene_xml']}")
    print(f"Plan: {summary['plan_json']}")
    print(f"Samples: {summary['trajectory']['sample_count']} at {summary['trajectory']['sample_period_s']:.4f} s")
    print(f"Duration: {summary['trajectory']['duration_s']:.3f} s")
    print(
        "Distance backend: "
        f"{summary['collision_distance_backend']['name']} "
        f"(enabled={summary['collision_distance_backend']['native_ccd_enabled']})"
    )
    print("Geometry counts:")
    for name, value in summary["geometry_counts"].items():
        print(f"  {name}: {value}")
    print("Minimum clearances:")
    for name, value in summary["minimum_clearances"].items():
        if value["minimum_distance_is_capped_lower_bound"]:
            print(f"  {name}: >= {value['lower_bound_m']:.3f} m")
        else:
            pair = value["closest_pair"]
            print(f"  {name}: {value['minimum_distance_m']:.6f} m at {value['time_s']:.3f} s ({pair['geom_a']} <-> {pair['geom_b']})")
    print(f"Collision events: {summary['collision_event_count']}")
    print(f"Joint-limit violations: {summary['joint_limit_violation_count']}")
    pickup_contact = summary["pickup_support_contact"]
    print("Pickup support-contact handling:")
    print(f"  support geometries: {', '.join(pickup_contact['support_geometries'])}")
    print(f"  allowed samples: {pickup_contact['allowed_sample_count']}")
    print(f"  cleared: {pickup_contact['cleared']}")
    print(f"  clear time: {pickup_contact['clear_time_s']}")
    print(f"  clearance violation: {pickup_contact['clearance_violation']}")
    print(f"RESULT: {summary['status']}")
    print(f"Preflight passed: {'YES' if summary['preflight_passed'] else 'NO'}")
    print(f"CSV: {summary['csv_report']}")
    print(f"JSON: {summary['json_report']}")
    print("No ROS joint commands were published; no robot motion was executed.")
    return 0 if summary["preflight_passed"] else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"SURGICAL HANDOVER PREFLIGHT ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
