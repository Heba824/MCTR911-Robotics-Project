#!/usr/bin/env python3
"""Plan a no-motion surgical handover from live poses or the scene HOME keyframe."""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
import sys

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parents[1]
CONTROL_DIR = PROJECT_ROOT / "Milestone 04" / "control"
if str(CONTROL_DIR) not in sys.path:
    sys.path.insert(0, str(CONTROL_DIR))

from joint_trajectory_controller_core import DEFAULT_JOINT_NAMES, reorder_joint_state  # noqa: E402
from surgical_handover_planner import (  # noqa: E402
    HOME,
    build_surgical_handover_plan,
    pose_components_to_transform,
    validate_plan_against_actuator_ranges,
)


class LivePoseCollector(Node):
    def __init__(self, instrument: str) -> None:
        super().__init__("m4_surgical_handover_planner_dry_run")
        self.instrument = instrument
        self.joint_state = None
        self.joint_state_received_at = None
        self.poses = {}
        self.pose_received_at = {}

        # The MuJoCo bridge publishes PoseStamped localization with BEST_EFFORT
        # reliability. A RELIABLE subscription is incompatible with that
        # publisher in ROS 2, so all pose subscriptions must request BEST_EFFORT.
        # JointState remains RELIABLE and therefore keeps the default QoS.
        pose_qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )

        self.create_subscription(JointState, "/joint_state", self._joint_state_callback, 10)
        selected_topic = f"/mujoco/{instrument}_pickup_pose"
        self.create_subscription(
            PoseStamped,
            selected_topic,
            lambda message: self._pose_callback("pickup", message),
            pose_qos,
        )
        self.create_subscription(
            PoseStamped,
            "/mujoco/handover_target_pose",
            lambda message: self._pose_callback("handover", message),
            pose_qos,
        )
        self.create_subscription(
            PoseStamped,
            "/mujoco/attachment_site_pose",
            lambda message: self._pose_callback("attachment", message),
            pose_qos,
        )

    def _joint_state_callback(self, message: JointState) -> None:
        try:
            q, qdot = reorder_joint_state(
                message.name,
                message.position,
                message.velocity,
                DEFAULT_JOINT_NAMES,
            )
        except ValueError as exc:
            self.get_logger().warning(f"Ignoring invalid JointState: {exc}")
            return
        self.joint_state = (q, qdot)
        self.joint_state_received_at = time.monotonic()

    def _pose_callback(self, key: str, message: PoseStamped) -> None:
        if message.header.frame_id and message.header.frame_id != "world":
            self.get_logger().warning(
                f"Ignoring {key} pose in frame '{message.header.frame_id}'; expected 'world'."
            )
            return
        pose = message.pose
        try:
            transform = pose_components_to_transform(
                [pose.position.x, pose.position.y, pose.position.z],
                [
                    pose.orientation.x,
                    pose.orientation.y,
                    pose.orientation.z,
                    pose.orientation.w,
                ],
            )
        except ValueError as exc:
            self.get_logger().warning(f"Ignoring invalid {key} pose: {exc}")
            return
        self.poses[key] = transform
        self.pose_received_at[key] = time.monotonic()

    def wait_for_data(self, timeout_s: float) -> None:
        required = {"pickup", "handover", "attachment"}
        deadline = time.monotonic() + timeout_s
        while rclpy.ok() and time.monotonic() < deadline:
            if self.joint_state is not None and required.issubset(self.poses):
                ages = [time.monotonic() - self.pose_received_at[key] for key in required]
                joint_age = time.monotonic() - self.joint_state_received_at
                if max(ages) <= 0.5 and joint_age <= 0.15:
                    return
            rclpy.spin_once(self, timeout_sec=0.05)
        missing = []
        if self.joint_state is None:
            missing.append("/joint_state")
        for key, topic in (
            ("pickup", f"/mujoco/{self.instrument}_pickup_pose"),
            ("handover", "/mujoco/handover_target_pose"),
            ("attachment", "/mujoco/attachment_site_pose"),
        ):
            if key not in self.poses:
                missing.append(topic)
        raise TimeoutError(
            "Timed out waiting for fresh localization data: " + ", ".join(missing)
        )


def _read_actuator_limits(scene_xml: Path):
    try:
        import mujoco
    except ImportError as exc:
        raise RuntimeError(
            "The active mctr911_kinematics_venv must provide MuJoCo Python 3.2.0."
        ) from exc

    model = mujoco.MjModel.from_xml_path(str(scene_xml))
    if model.nq != 6 or model.nv != 6 or model.nu != 6:
        raise ValueError(
            f"Expected the six-joint/six-actuator UR5e model; "
            f"got nq={model.nq}, nv={model.nv}, nu={model.nu}."
        )

    actuator_names = (
        "shoulder_pan",
        "shoulder_lift",
        "elbow",
        "wrist_1",
        "wrist_2",
        "wrist_3",
    )
    lower = []
    upper = []
    for joint_name, actuator_name in zip(DEFAULT_JOINT_NAMES, actuator_names):
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)
        if joint_id < 0 or actuator_id < 0:
            raise ValueError(f"Could not resolve '{joint_name}' / '{actuator_name}' in the scene.")
        if int(model.actuator_trnid[actuator_id, 0]) != joint_id:
            raise ValueError(f"Actuator '{actuator_name}' is not attached to '{joint_name}'.")
        if not bool(model.actuator_ctrllimited[actuator_id]):
            raise ValueError(f"Actuator '{actuator_name}' has no finite control range.")
        lo, hi = model.actuator_ctrlrange[actuator_id]
        lower.append(float(lo))
        upper.append(float(hi))
    return model, np.asarray(lower), np.asarray(upper)


def _read_instrument_body_pose_world(model, instrument: str) -> np.ndarray:
    """Read the instrument mocap body's initial world pose from the scene XML."""
    import mujoco

    body_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, instrument))
    if body_id < 0:
        raise ValueError(f"Instrument body '{instrument}' was not found in the handover scene.")
    if int(model.body_mocapid[body_id]) < 0:
        raise ValueError(f"Instrument body '{instrument}' must be a mocap body.")

    data = mujoco.MjData(model)
    key_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home"))
    if key_id < 0:
        raise ValueError("The handover scene must contain the 'home' keyframe.")
    mujoco.mj_resetDataKeyframe(model, data, key_id)
    mujoco.mj_forward(model, data)

    transform = np.eye(4, dtype=float)
    transform[:3, :3] = np.asarray(data.xmat[body_id], dtype=float).reshape(3, 3)
    transform[:3, 3] = np.asarray(data.xpos[body_id], dtype=float)
    return transform


def _offline_home_state(model, instrument: str):
    import mujoco

    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    mujoco.mj_forward(model, data)
    poses = {}
    for key, name in (
        ("pickup", f"{instrument}_pickup_site"),
        ("handover", "handover_target_site"),
        ("attachment", "attachment_site"),
    ):
        site = data.site(name)
        pose = np.eye(4)
        pose[:3, :3] = site.xmat.reshape(3, 3)
        pose[:3, 3] = site.xpos
        poses[key] = pose
    return data.qpos.copy(), data.qvel.copy(), poses


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plan surgical instrument handover waypoints from live MuJoCo poses; does not move the robot."
    )
    parser.add_argument("--instrument", choices=("scalpel", "forceps"), required=True)
    parser.add_argument(
        "--scene-xml",
        default=str(
            PROJECT_ROOT / "Milestone 01" / "Robot Model" / "scene_surgical_handover.xml"
        ),
    )
    parser.add_argument("--offline", action="store_true", help="Use the scene HOME keyframe and sites without starting ROS")
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--max-home-error", type=float, default=0.08)
    parser.add_argument("--max-joint-speed", type=float, default=0.05)
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "Milestone 04" / "Results" / "Surgical_Handover" / "Planning"),
    )
    args = parser.parse_args()

    scene_xml = Path(args.scene_xml).expanduser().resolve()
    if not scene_xml.is_file():
        raise FileNotFoundError(f"Handover scene not found: {scene_xml}")

    model, lower, upper = _read_actuator_limits(scene_xml)
    instrument_body_pose = _read_instrument_body_pose_world(model, args.instrument)
    if args.offline:
        q, qdot, poses = _offline_home_state(model, args.instrument)
        pose_ages = {}
    else:
        rclpy.init()
        node = LivePoseCollector(args.instrument)
        try:
            print("Waiting for fresh joint state and MuJoCo pose topics ...")
            node.wait_for_data(args.timeout)
            q, qdot = node.joint_state
            poses = node.poses.copy()
            pose_ages = {
                f"{key}_pose_age_s": time.monotonic() - node.pose_received_at[key]
                for key in poses
            }
        finally:
            node.destroy_node()
            rclpy.shutdown()

    max_home_error = float(np.max(np.abs(q - HOME)))
    max_joint_speed = float(np.max(np.abs(qdot)))
    if max_home_error > args.max_home_error:
        raise RuntimeError(
            f"Planning preflight requires the robot near HOME: maximum joint error "
            f"{max_home_error:.5f} rad exceeds {args.max_home_error:.5f} rad."
        )
    if max_joint_speed > args.max_joint_speed:
        raise RuntimeError(
            f"Robot must be stationary: maximum joint speed {max_joint_speed:.5f} rad/s "
            f"exceeds {args.max_joint_speed:.5f} rad/s."
        )

    plan = build_surgical_handover_plan(
        instrument=args.instrument,
        initial_joint_positions_rad=q,
        pickup_pose_world=poses["pickup"],
        handover_pose_world=poses["handover"],
        attachment_pose_world=poses["attachment"],
        instrument_body_pose_world=instrument_body_pose,
    )
    validate_plan_against_actuator_ranges(plan, lower, upper)

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_path = output_dir / f"surgical_handover_plan_{args.instrument}_{timestamp}.json"
    payload = plan.to_dict()
    payload.update(
        {
            "scene_xml": str(scene_xml),
            "model_nq": int(model.nq),
            "model_nv": int(model.nv),
            "model_nu": int(model.nu),
            "maximum_initial_home_error_rad": max_home_error,
            "maximum_initial_joint_speed_rad_s": max_joint_speed,
            "actuator_lower_limits_rad": lower.tolist(),
            "actuator_upper_limits_rad": upper.tolist(),
            "instrument_body_initial_pose_world": instrument_body_pose.tolist(),
            "desired_tool_handover_pose_world": [list(row) for row in plan.desired_tool_handover_pose_world],
            "planning_source": "scene HOME keyframe" if args.offline else "live ROS poses",
            **pose_ages,
        }
    )
    output_path.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")

    print("\nSURGICAL HANDOVER IK PLANNING: PASS")
    print(f"Instrument: {args.instrument}")
    print(f"MuJoCo model: nq={model.nq}, nv={model.nv}, nu={model.nu}")
    print(f"Maximum HOME joint error: {max_home_error:.6f} rad")
    print(f"Maximum joint speed: {max_joint_speed:.6f} rad/s")
    print("Instrument-body handover target [m]:", np.round(poses["handover"][:3, 3], 6).tolist())
    print("Compensated attachment-site target [m]:", np.round(np.asarray(plan.desired_tool_handover_pose_world, dtype=float)[:3, 3], 6).tolist())
    print("\nWaypoint plan:")
    print(f"{'Waypoint':<22} {'Duration [s]':>13} {'IK pos err [m]':>16} {'IK ori err [rad]':>18}")
    for waypoint in plan.waypoints:
        print(
            f"{waypoint.name:<22} {waypoint.duration_from_previous_s:>13.3f} "
            f"{waypoint.ik_position_error_m:>16.6e} "
            f"{waypoint.ik_orientation_error_rad:>18.6e}"
        )
    print(f"\nPlan JSON: {output_path}")
    print("No joint commands were published. No motion was executed.")
    print("Collision checking is not included in this planning-only step.")
    print("The approach_clearance waypoint is a candidate route and is NOT collision-certified.")


if __name__ == "__main__":
    main()
