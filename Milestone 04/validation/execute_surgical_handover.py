#!/usr/bin/env python3
"""Execute a preflighted UR5e surgical instrument handover in MuJoCo via ROS 2.

The node is motion-inert unless the CLI includes --execute. It reuses the
project's quintic trajectory and motor-level controller, verifies live
localization at pickup and handover, and uses the MuJoCo bridge's SetBool
attachment services. It never automatically releases an instrument after an
execution fault.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parents[1]
CONTROL_DIR = PROJECT_ROOT / "Milestone 04" / "control"
TRAJECTORY_DIR = PROJECT_ROOT / "Milestone 04" / "trajectory"
KINEMATICS_DIR = PROJECT_ROOT / "Milestone 02" / "kinematics"
for _path in (CONTROL_DIR, TRAJECTORY_DIR, KINEMATICS_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

if not (KINEMATICS_DIR / "ur5e_dh_fk.py").is_file():
    raise FileNotFoundError(
        "Required forward-kinematics module was not found: "
        f"{KINEMATICS_DIR / 'ur5e_dh_fk.py'}"
    )

from joint_space_trajectory import QuinticJointTrajectory  # noqa: E402
from joint_trajectory_controller_core import (  # noqa: E402
    DEFAULT_JOINT_NAMES,
    reorder_joint_state,
    tracking_error_metrics,
)
from motor_level_controller import (  # noqa: E402
    compute_inverse_dynamics_feedforward,
    compute_position_servo_command,
    read_position_actuator_gains,
)
from ur5e_dh_fk import forward_kinematics  # noqa: E402
from surgical_handover_planner import pose_components_to_transform  # noqa: E402
from surgical_handover_execution_core import (  # noqa: E402
    ExecutionValidationError,
    build_execution_segments,
    instrument_body_pose_from_pickup_site,
    load_execution_plan,
    transform_error,
    validate_pickup_geometry,
    validate_preflight_for_execution,
)
import surgical_handover_collision_preflight as collision_preflight  # noqa: E402

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
from std_srvs.srv import SetBool


DEFAULT_SCENE = PROJECT_ROOT / "Milestone 01" / "Robot Model" / "scene_surgical_handover.xml"
DEFAULT_OUTPUT = PROJECT_ROOT / "Milestone 04" / "Results" / "Surgical_Handover" / "Execution"


class SurgicalHandoverExecutor(Node):
    def __init__(self, *, instrument: str, scene_path: Path, strategy: str, args: argparse.Namespace):
        super().__init__("m4_surgical_handover_executor")
        import mujoco

        self.instrument = instrument
        self.scene_path = scene_path.resolve()
        self.strategy = strategy
        self.args = args
        self.mujoco = mujoco
        self.joint_names = tuple(DEFAULT_JOINT_NAMES)
        self.latest_q: np.ndarray | None = None
        self.latest_qdot: np.ndarray | None = None
        self.last_joint_state_monotonic: float | None = None
        self.poses: dict[str, np.ndarray] = {}
        self.pose_received_at: dict[str, float] = {}
        self.pose_stamp_ns: dict[str, int] = {}
        self.rows: list[dict[str, Any]] = []
        self.instrument_attached = False
        self.instrument_released = False
        self.release_pickup_site_pose: np.ndarray | None = None
        self.release_instrument_body_pose: np.ndarray | None = None
        self.release_monotonic: float = 0.0
        self.last_collision_sample: dict[str, Any] | None = None
        self.tool_from_instrument_actual: np.ndarray | None = None
        self.run_started_monotonic: float | None = None
        self.run_started_utc: str | None = None
        self.run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        self.max_feedback_age_seen_s = 0.0
        self.max_tracking_error_seen_rad = 0.0
        self.tracking_error_streak = 0

        self._initialize_mujoco_model()

        command_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        pose_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.command_publisher = self.create_publisher(
            Float64MultiArray, "/joint_commands", command_qos
        )
        self.create_subscription(JointState, "/joint_state", self._joint_state_callback, command_qos)
        self.create_subscription(
            PoseStamped,
            f"/mujoco/{instrument}_pickup_pose",
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
        self.attachment_client = self.create_client(
            SetBool, f"/mujoco/attach_{instrument}"
        )

        self.get_logger().info(
            f"Surgical handover executor ready for {instrument}; strategy={strategy}; "
            f"control rate={args.control_rate_hz:.1f} Hz. No motion is commanded at startup."
        )

    def _initialize_mujoco_model(self) -> None:
        mujoco = self.mujoco
        model = mujoco.MjModel.from_xml_path(str(self.scene_path))
        if (model.nq, model.nv, model.nu) != (6, 6, 6):
            raise ValueError(
                f"Expected nq=6, nv=6, nu=6; got nq={model.nq}, nv={model.nv}, nu={model.nu}"
            )
        key_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home"))
        if key_id < 0:
            raise ValueError("The handover scene must contain the 'home' keyframe")
        actuator_names = (
            "shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3"
        )
        joint_ids: list[int] = []
        actuator_ids: list[int] = []
        for joint_name, actuator_name in zip(self.joint_names, actuator_names):
            joint_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name))
            actuator_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name))
            if joint_id < 0 or actuator_id < 0:
                raise ValueError(f"Could not resolve '{joint_name}' / '{actuator_name}' in scene")
            if int(model.actuator_trnid[actuator_id, 0]) != joint_id:
                raise ValueError(f"Actuator '{actuator_name}' is not attached to '{joint_name}'")
            joint_ids.append(joint_id)
            actuator_ids.append(actuator_id)

        self.model = model
        self.qpos_indices = np.asarray([model.jnt_qposadr[j] for j in joint_ids], dtype=int)
        self.dof_indices = np.asarray([model.jnt_dofadr[j] for j in joint_ids], dtype=int)
        self.kp, self.kd = read_position_actuator_gains(model, actuator_ids, expected_count=6)
        if not np.all(np.asarray(model.actuator_ctrllimited)[actuator_ids]):
            raise ValueError("All six position actuators must have finite control limits")
        self.control_range = np.asarray(model.actuator_ctrlrange[actuator_ids], dtype=float).copy()
        self.actual_data = mujoco.MjData(model)
        self.reference_data = mujoco.MjData(model)
        mujoco.mj_resetDataKeyframe(model, self.actual_data, key_id)
        mujoco.mj_resetDataKeyframe(model, self.reference_data, key_id)

        # Use a separate MuJoCo model for collision queries so enabling native
        # CCD cannot alter the model used by inverse-dynamics feedforward.
        collision_model = mujoco.MjModel.from_xml_path(str(self.scene_path))
        if (collision_model.nq, collision_model.nv, collision_model.nu) != (6, 6, 6):
            raise ValueError("Collision-monitor model dimensions do not match the UR5e model")
        collision_key_id = int(
            mujoco.mj_name2id(collision_model, mujoco.mjtObj.mjOBJ_KEY, "home")
        )
        if collision_key_id < 0:
            raise ValueError("Collision-monitor scene is missing the 'home' keyframe")
        collision_preflight._enable_native_ccd(mujoco, collision_model)
        self.collision_model = collision_model
        self.collision_data = mujoco.MjData(collision_model)
        mujoco.mj_resetDataKeyframe(collision_model, self.collision_data, collision_key_id)
        self.collision_selection = collision_preflight._select_geometries(
            mujoco, collision_model, self.instrument
        )
        self.collision_joint_ids, self.collision_qpos_addresses, self.collision_joint_limits = (
            collision_preflight._joint_data(mujoco, collision_model)
        )
        self.collision_minimum_clearance_m = 0.02
        self.collision_distance_cap_m = 0.05
        self.pickup_support_contact_cleared = False
        self.online_minimum_clearances_m = {
            "robot_environment": self.collision_distance_cap_m,
            "robot_self": self.collision_distance_cap_m,
            "carried_instrument_environment": self.collision_distance_cap_m,
        }

        body_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, self.instrument))
        site_id = int(
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"{self.instrument}_pickup_site")
        )
        if body_id < 0 or site_id < 0:
            raise ValueError(f"Could not resolve {self.instrument} body and pickup site")
        if int(model.site_bodyid[site_id]) != body_id:
            raise ValueError("The selected pickup site is not attached to the selected instrument body")
        self.instrument_from_pickup_site = np.eye(4, dtype=float)
        self.instrument_from_pickup_site[:3, :3] = self._quaternion_wxyz_to_rotation(
            np.asarray(model.site_quat[site_id], dtype=float)
        )
        self.instrument_from_pickup_site[:3, 3] = np.asarray(model.site_pos[site_id], dtype=float)

    @staticmethod
    def _quaternion_wxyz_to_rotation(quaternion: np.ndarray) -> np.ndarray:
        q = np.asarray(quaternion, dtype=float)
        if q.shape != (4,) or not np.all(np.isfinite(q)):
            raise ValueError("MuJoCo site quaternion must contain four finite values")
        norm = float(np.linalg.norm(q))
        if norm < 1e-12:
            raise ValueError("MuJoCo site quaternion has zero norm")
        w, x, y, z = q / norm
        return np.array(
            [
                [1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)],
                [2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)],
                [2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)],
            ],
            dtype=float,
        )

    def _joint_state_callback(self, message: JointState) -> None:
        try:
            q, qdot = reorder_joint_state(
                message.name, message.position, message.velocity, self.joint_names
            )
        except ValueError as exc:
            self.get_logger().warning(f"Ignoring invalid JointState: {exc}")
            return
        self.latest_q = q
        self.latest_qdot = qdot
        self.last_joint_state_monotonic = time.monotonic()

    def _pose_callback(self, key: str, message: PoseStamped) -> None:
        if message.header.frame_id and message.header.frame_id != "world":
            self.get_logger().warning(
                f"Ignoring {key} pose in frame {message.header.frame_id!r}; expected 'world'"
            )
            return
        pose = message.pose
        try:
            transform = pose_components_to_transform(
                [pose.position.x, pose.position.y, pose.position.z],
                [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w],
            )
        except ValueError as exc:
            self.get_logger().warning(f"Ignoring invalid {key} pose: {exc}")
            return
        self.poses[key] = transform
        self.pose_received_at[key] = time.monotonic()
        self.pose_stamp_ns[key] = message.header.stamp.sec * 10**9 + message.header.stamp.nanosec

    def _spin_once(self, timeout_s: float = 0.0) -> None:
        rclpy.spin_once(self, timeout_sec=max(0.0, float(timeout_s)))

    def wait_for_fresh_data(self, timeout_s: float = 10.0) -> None:
        required = ("pickup", "handover", "attachment")
        deadline = time.monotonic() + timeout_s
        while rclpy.ok() and time.monotonic() < deadline:
            self._spin_once(0.05)
            if self.latest_q is None or self.latest_qdot is None:
                continue
            if self.last_joint_state_monotonic is None:
                continue
            if any(key not in self.poses for key in required):
                continue
            now = time.monotonic()
            pose_ages = [now - self.pose_received_at[key] for key in required]
            joint_age = now - self.last_joint_state_monotonic
            if max(pose_ages) <= 0.5 and joint_age <= self.args.feedback_timeout_s:
                return
        missing = [key for key in required if key not in self.poses]
        if self.latest_q is None:
            missing.append("/joint_state")
        raise TimeoutError("Timed out waiting for fresh ROS feedback: " + ", ".join(missing))

    def _feedback_age(self) -> float:
        if self.last_joint_state_monotonic is None:
            return math.inf
        return max(0.0, time.monotonic() - self.last_joint_state_monotonic)

    def _require_fresh_feedback(self) -> tuple[np.ndarray, np.ndarray]:
        age = self._feedback_age()
        if age > self.args.feedback_timeout_s:
            raise RuntimeError(
                f"Joint-state feedback is stale ({age:.4f}s > {self.args.feedback_timeout_s:.4f}s)"
            )
        self.max_feedback_age_seen_s = max(self.max_feedback_age_seen_s, age)
        if self.latest_q is None or self.latest_qdot is None:
            raise RuntimeError("No valid /joint_state feedback has been received")
        return self.latest_q.copy(), self.latest_qdot.copy()

    def _require_fresh_pose(self, key: str, maximum_age_s: float = 0.5) -> np.ndarray:
        if key not in self.poses:
            raise RuntimeError(f"No {key} localization pose has been received")
        age = time.monotonic() - self.pose_received_at[key]
        if age > maximum_age_s:
            raise RuntimeError(f"{key} localization pose is stale ({age:.3f}s)")
        return self.poses[key].copy()

    def validate_live_start(self, plan: dict[str, Any]) -> None:
        q, qdot = self._require_fresh_feedback()
        start_q = np.asarray(plan["_initial_q"], dtype=float)
        start_error = float(np.max(np.abs(q - start_q)))
        start_speed = float(np.max(np.abs(qdot)))
        if start_error > self.args.start_tolerance_rad:
            raise RuntimeError(
                f"Start pose rejected: max joint error {start_error:.5f} rad exceeds "
                f"{self.args.start_tolerance_rad:.5f} rad; regenerate the plan from the current pose"
            )
        if start_speed > self.args.start_velocity_tolerance_rad_s:
            raise RuntimeError(
                f"Robot is not stationary: max joint speed {start_speed:.5f} rad/s"
            )

        world_from_dh = np.asarray(plan["world_from_dh"], dtype=float)
        expected_attachment = world_from_dh @ forward_kinematics(start_q)
        actual_attachment = self._require_fresh_pose("attachment")
        attachment_position_error, attachment_orientation_error = transform_error(
            actual_attachment, expected_attachment
        )
        if (
            attachment_position_error > self.args.pose_position_tolerance_m
            or attachment_orientation_error > self.args.pose_orientation_tolerance_rad
        ):
            raise RuntimeError(
                "Live attachment_site pose no longer matches the planner calibration: "
                f"position error={attachment_position_error:.4f} m, "
                f"orientation error={math.degrees(attachment_orientation_error):.2f} deg"
            )

        pickup_waypoint = next(item for item in plan["waypoints"] if item["name"] == "pickup")
        expected_pickup_tool = world_from_dh @ forward_kinematics(
            np.asarray(pickup_waypoint["joint_positions_rad"], dtype=float)
        )
        actual_pickup_site = self._require_fresh_pose("pickup")
        expected_instrument_body = np.asarray(
            plan["instrument_body_initial_pose_world"], dtype=float
        )
        expected_pickup_site = expected_instrument_body @ self.instrument_from_pickup_site
        pickup_metrics = validate_pickup_geometry(
            actual_pickup_site,
            expected_pickup_site,
            expected_pickup_tool,
            site_position_tolerance_m=self.args.pose_position_tolerance_m,
            site_orientation_tolerance_rad=self.args.pose_orientation_tolerance_rad,
            tool_site_position_tolerance_m=self.args.pickup_alignment_tolerance_m,
        )
        self._set_collision_instrument_pose(
            instrument_body_pose_from_pickup_site(actual_pickup_site, self.instrument_from_pickup_site)
        )
        self._check_pickup_support()
        pickup_position_error = pickup_metrics["tool_to_pickup_site_position_error_m"]
        pickup_orientation_offset = pickup_metrics[
            "planned_tool_to_pickup_site_orientation_offset_rad"
        ]

        expected_handover = np.asarray(
            plan["desired_instrument_handover_pose_world"], dtype=float
        )
        actual_handover = self._require_fresh_pose("handover")
        handover_position_error, handover_orientation_error = transform_error(
            actual_handover, expected_handover
        )
        if (
            handover_position_error > self.args.pose_position_tolerance_m
            or handover_orientation_error > self.args.pose_orientation_tolerance_rad
        ):
            raise RuntimeError(
                "The surgeon handover target has moved since planning: "
                f"position error={handover_position_error:.4f} m, "
                f"orientation error={math.degrees(handover_orientation_error):.2f} deg; replan"
            )

        self.get_logger().info(
            "Live start checks passed: "
            f"max joint error={start_error:.6f} rad, max speed={start_speed:.6f} rad/s, "
            f"pickup position alignment={pickup_position_error*1000:.2f} mm, "
            f"planned tool/site orientation offset={math.degrees(pickup_orientation_offset):.2f} deg, "
            f"handover target error={handover_position_error*1000:.2f} mm."
        )

    def _compute_command(
        self,
        q_ref: np.ndarray,
        qdot_ref: np.ndarray,
        qddot_ref: np.ndarray,
        q_actual: np.ndarray,
        qdot_actual: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.strategy == "baseline":
            return q_ref.copy(), np.zeros(6, dtype=float), np.zeros(6, dtype=bool)

        self.actual_data.qpos[:] = self.model.qpos0
        self.actual_data.qvel[:] = 0.0
        self.actual_data.qpos[self.qpos_indices] = q_actual
        self.actual_data.qvel[self.dof_indices] = qdot_actual
        tau_ff = compute_inverse_dynamics_feedforward(
            self.model,
            self.actual_data,
            self.reference_data,
            self.qpos_indices,
            self.dof_indices,
            q_ref,
            qdot_ref,
            qddot_ref,
        )
        servo = compute_position_servo_command(
            q_ref,
            qdot_ref,
            self.kp,
            self.kd,
            feedforward_torque_nm=tau_ff,
            control_range_rad=self.control_range,
        )
        return servo.command_rad, servo.feedforward_torque_nm, servo.clipped

    def _publish_position(self, position: np.ndarray) -> None:
        values = np.asarray(position, dtype=float)
        if values.shape != (6,) or not np.all(np.isfinite(values)):
            raise ValueError("Refusing to publish an invalid six-joint command")
        message = Float64MultiArray()
        message.data = [float(value) for value in values]
        self.command_publisher.publish(message)

    def _publish_hold(self) -> None:
        if self.latest_q is None:
            return
        q, qdot = self.latest_q.copy(), self.latest_qdot.copy() if self.latest_qdot is not None else np.zeros(6)
        try:
            command, _, clipped = self._compute_command(
                q, np.zeros(6), np.zeros(6), q, qdot
            )
            if np.any(clipped):
                command = q
            self._publish_position(command)
        except Exception as exc:
            self.get_logger().error(f"Could not calculate model-based hold; publishing measured pose: {exc}")
            self._publish_position(q)

    def _record(
        self,
        phase: str,
        elapsed_s: float,
        q_ref: np.ndarray,
        qdot_ref: np.ndarray,
        qddot_ref: np.ndarray,
        q_actual: np.ndarray,
        qdot_actual: np.ndarray,
        command: np.ndarray,
        tau_ff: np.ndarray,
        clipped: np.ndarray,
    ) -> None:
        row: dict[str, Any] = {
            "elapsed_s": float(elapsed_s),
            "phase": phase,
            "instrument_attached": int(self.instrument_attached),
            "instrument_released": int(self.instrument_released),
        }
        for i, name in enumerate(self.joint_names):
            row[f"q_desired_{name}_rad"] = float(q_ref[i])
            row[f"q_actual_{name}_rad"] = float(q_actual[i])
            row[f"error_{name}_rad"] = float(q_ref[i] - q_actual[i])
            row[f"qdot_desired_{name}_rad_s"] = float(qdot_ref[i])
            row[f"qdot_actual_{name}_rad_s"] = float(qdot_actual[i])
            row[f"qddot_desired_{name}_rad_s2"] = float(qddot_ref[i])
            row[f"command_{name}_rad"] = float(command[i])
            row[f"tau_ff_{name}_Nm"] = float(tau_ff[i])
            row[f"command_clipped_{name}"] = int(bool(clipped[i]))
        self.rows.append(row)

    def _online_collision_check(
        self, phase: str, q_actual: np.ndarray, qdot_actual: np.ndarray
    ) -> None:
        """Check the measured MuJoCo state with the same native-CCD geometry sets.

        This is a sampled online guard, not continuous collision detection. It
        prevents the executor from continuing after measured clearance falls
        below the preflight hard minimum.
        """
        mujoco = self.mujoco
        model = self.collision_model
        data = self.collision_data
        selection = self.collision_selection
        phase_name = phase.removesuffix("_settle")

        data.qpos[self.collision_qpos_addresses] = q_actual
        data.qvel[:] = 0.0
        data.qvel[self.dof_indices] = qdot_actual
        mujoco.mj_forward(model, data)

        if self.instrument_attached:
            if self.tool_from_instrument_actual is None:
                raise RuntimeError("Online instrument transform is unavailable while attached")
            relative = self.tool_from_instrument_actual
            collision_preflight._update_instrument_mocap(
                mujoco, model, data, selection, relative[:3, 3], relative[:3, :3],
            )
        elif self.instrument_released:
            if self.release_instrument_body_pose is None:
                raise RuntimeError("Released instrument pose was not retained")
            self._set_collision_instrument_pose(self.release_instrument_body_pose)

        self.last_collision_sample = {
            "phase": phase,
            "q_rad": q_actual.tolist(),
            "qdot_rad_s": qdot_actual.tolist(),
            "instrument_body_position_world_m": data.mocap_pos[selection["instrument_mocap_id"]].tolist(),
            "instrument_body_quaternion_world_wxyz": data.mocap_quat[selection["instrument_mocap_id"]].tolist(),
        }

        def checked_minimum(pairs: list[tuple[int, int]], category: str) -> float:
            minimum = self.collision_distance_cap_m
            for geom_a, geom_b in pairs:
                distance = collision_preflight._distance(
                    mujoco, model, data, geom_a, geom_b, self.collision_distance_cap_m
                )
                minimum = min(minimum, distance)
                if distance <= 0.0:
                    name_a = collision_preflight._geom_name(mujoco, model, geom_a)
                    name_b = collision_preflight._geom_name(mujoco, model, geom_b)
                    raise RuntimeError(
                        f"Online collision detected in {category}: {name_a} <-> {name_b} "
                        f"at {distance*1000:.2f} mm"
                    )
                if distance < self.collision_minimum_clearance_m:
                    name_a = collision_preflight._geom_name(mujoco, model, geom_a)
                    name_b = collision_preflight._geom_name(mujoco, model, geom_b)
                    raise RuntimeError(
                        f"Online clearance violation in {category}: {name_a} <-> {name_b} "
                        f"at {distance*1000:.2f} mm; hard minimum is "
                        f"{self.collision_minimum_clearance_m*1000:.1f} mm"
                    )
            self.online_minimum_clearances_m[category] = min(
                self.online_minimum_clearances_m[category], minimum
            )
            return minimum

        if self.instrument_attached:
            robot_obstacles = selection["static_environment_geoms"]
        else:
            robot_obstacles = selection["all_environment_geoms"]
        robot_pairs = [
            (geom_a, geom_b)
            for geom_a in selection["robot_geoms"]
            for geom_b in robot_obstacles
        ]
        checked_minimum(robot_pairs, "robot_environment")
        checked_minimum(selection["self_pairs"], "robot_self")

        for index, (joint_id, limits) in enumerate(
            zip(self.collision_joint_ids, self.collision_joint_limits)
        ):
            lower, upper = limits
            if q_actual[index] < lower - 1e-8 or q_actual[index] > upper + 1e-8:
                raise RuntimeError(
                    f"Online joint-limit violation for {self.joint_names[index]}: "
                    f"{q_actual[index]:.6f} rad not in [{lower:.6f}, {upper:.6f}]"
                )

        allow_near_support = False
        if not self.instrument_released and not self.pickup_support_contact_cleared:
            minimum = self._check_pickup_support()
            if self.instrument_attached:
                if not collision_preflight.pickup_support_contact_allowed(phase_name, True, False):
                    raise RuntimeError("Instrument did not clear the pickup support envelope by the end of lift")
                if minimum >= self.collision_minimum_clearance_m:
                    self.pickup_support_contact_cleared = True
                    self.get_logger().info(f"Online pickup-support envelope cleared at {minimum*1000:.2f} mm.")
                else:
                    allow_near_support = True

        if self.instrument_attached or self.instrument_released:
            pairs = collision_preflight.carried_environment_pairs(
                model, selection, phase_name, allow_near_support
            )
            checked_minimum(pairs, "carried_instrument_environment")

    def _set_collision_instrument_pose(self, pose: np.ndarray) -> None:
        mocap_id = self.collision_selection["instrument_mocap_id"]
        self.collision_data.mocap_pos[mocap_id] = pose[:3, 3]
        self.collision_data.mocap_quat[mocap_id] = collision_preflight.rotation_matrix_to_quaternion_wxyz(
            pose[:3, :3]
        )
        self.mujoco.mj_forward(self.collision_model, self.collision_data)

    def _check_pickup_support(self) -> float:
        minimum, _, _, _, _, penetrations = collision_preflight._record_pair(
            self.mujoco, self.collision_model, self.collision_data,
            collision_preflight.pickup_support_pairs(self.collision_selection),
            self.collision_distance_cap_m, self.collision_minimum_clearance_m,
            "pickup_support", 0.0, "", allow_touch=True,
        )
        if penetrations:
            pair = min(penetrations, key=lambda event: event["signed_distance_m"])
            raise RuntimeError(
                f"Online pickup-support penetration: {pair['geom_a']} <-> {pair['geom_b']} "
                f"at {pair['signed_distance_m']*1000:.6f} mm"
            )
        return minimum

    def _control_sample(self, phase: str, elapsed_s: float, reference: Any) -> None:
        q_actual, qdot_actual = self._require_fresh_feedback()
        self._online_collision_check(phase, q_actual, qdot_actual)
        error = reference.position_rad - q_actual
        max_error = float(np.max(np.abs(error)))
        self.max_tracking_error_seen_rad = max(self.max_tracking_error_seen_rad, max_error)
        if max_error > self.args.max_tracking_error_rad:
            self.tracking_error_streak += 1
        else:
            self.tracking_error_streak = 0
        if self.tracking_error_streak >= self.args.tracking_error_consecutive_samples:
            raise RuntimeError(
                f"Tracking error exceeded {self.args.max_tracking_error_rad:.3f} rad for "
                f"{self.args.tracking_error_consecutive_samples} consecutive samples"
            )

        command, tau_ff, clipped = self._compute_command(
            reference.position_rad,
            reference.velocity_rad_s,
            reference.acceleration_rad_s2,
            q_actual,
            qdot_actual,
        )
        if np.any(clipped):
            joints = [self.joint_names[i] for i in np.flatnonzero(clipped)]
            raise RuntimeError("Actuator command saturation detected; refusing clipped command for " + ", ".join(joints))
        self._publish_position(command)
        self._record(
            phase,
            elapsed_s,
            reference.position_rad,
            reference.velocity_rad_s,
            reference.acceleration_rad_s2,
            q_actual,
            qdot_actual,
            command,
            tau_ff,
            clipped,
        )

    def _run_segment(self, segment: Any) -> None:
        trajectory = segment.trajectory()
        period = 1.0 / self.args.control_rate_hz
        start = time.monotonic()
        next_tick = start
        self.tracking_error_streak = 0
        self.get_logger().info(
            f"Executing {segment.name}: duration={segment.duration_s:.3f}s, "
            f"max joint displacement={np.max(np.abs(segment.goal_position_rad-segment.start_position_rad)):.3f} rad."
        )

        while rclpy.ok():
            self._spin_once(0.0)
            now = time.monotonic()
            elapsed = min(max(0.0, now - start), segment.duration_s)
            reference = trajectory.evaluate(elapsed)
            global_elapsed = now - (self.run_started_monotonic or start)
            self._control_sample(segment.name, global_elapsed, reference)
            if elapsed >= segment.duration_s:
                break
            next_tick += period
            time.sleep(max(0.0, next_tick - time.monotonic()))

        # Continue commanding the final waypoint until measured position and
        # velocity settle. The next segment never starts on a moving robot.
        settle_deadline = time.monotonic() + self.args.goal_settle_timeout_s
        settled_samples = 0
        final_reference = trajectory.evaluate(segment.duration_s)
        while rclpy.ok() and time.monotonic() < settle_deadline:
            self._spin_once(0.0)
            now = time.monotonic()
            global_elapsed = now - (self.run_started_monotonic or start)
            self._control_sample(segment.name + "_settle", global_elapsed, final_reference)
            q, qdot = self._require_fresh_feedback()
            position_error = float(np.max(np.abs(segment.goal_position_rad - q)))
            speed = float(np.max(np.abs(qdot)))
            if (
                position_error <= self.args.goal_position_tolerance_rad
                and speed <= self.args.goal_velocity_tolerance_rad_s
            ):
                settled_samples += 1
                if settled_samples >= self.args.goal_settle_samples:
                    self.get_logger().info(
                        f"{segment.name} settled: max position error={position_error:.5f} rad, "
                        f"max speed={speed:.5f} rad/s."
                    )
                    return
            else:
                settled_samples = 0
            time.sleep(period)
        raise TimeoutError(
            f"Waypoint {segment.name} did not settle within {self.args.goal_settle_timeout_s:.1f}s"
        )

    def _call_attachment_service(self, attach: bool) -> None:
        deadline = time.monotonic() + self.args.service_timeout_s
        while rclpy.ok() and not self.attachment_client.service_is_ready():
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Service /mujoco/attach_{self.instrument} is unavailable")
            self._spin_once(0.05)
        request = SetBool.Request()
        request.data = bool(attach)
        future = self.attachment_client.call_async(request)
        while rclpy.ok() and not future.done():
            if time.monotonic() >= deadline:
                raise TimeoutError("Instrument attachment service call timed out")
            self._spin_once(0.05)
        if not future.done():
            raise RuntimeError("ROS shut down before attachment service completed")
        response = future.result()
        if response is None or not response.success:
            message = "no response" if response is None else response.message
            raise RuntimeError(f"Instrument {'attach' if attach else 'release'} rejected: {message}")
        self.get_logger().info(response.message)

    def _attach_at_pickup(self, plan: dict[str, Any]) -> None:
        attachment = self._require_fresh_pose("attachment")
        pickup = self._require_fresh_pose("pickup")
        world_from_dh = np.asarray(plan["world_from_dh"], dtype=float)
        pickup_waypoint = next(item for item in plan["waypoints"] if item["name"] == "pickup")
        expected_tool_pickup = world_from_dh @ forward_kinematics(
            np.asarray(pickup_waypoint["joint_positions_rad"], dtype=float)
        )
        expected_instrument_body = np.asarray(
            plan["instrument_body_initial_pose_world"], dtype=float
        )
        expected_pickup_site = expected_instrument_body @ self.instrument_from_pickup_site
        pickup_metrics = validate_pickup_geometry(
            pickup,
            expected_pickup_site,
            expected_tool_pickup,
            site_position_tolerance_m=self.args.pose_position_tolerance_m,
            site_orientation_tolerance_rad=self.args.pose_orientation_tolerance_rad,
            tool_site_position_tolerance_m=self.args.pickup_alignment_tolerance_m,
            actual_tool_pose=attachment,
            tool_position_tolerance_m=self.args.pose_position_tolerance_m,
            tool_orientation_tolerance_rad=self.args.pose_orientation_tolerance_rad,
        )

        instrument_body_pose = instrument_body_pose_from_pickup_site(
            pickup, self.instrument_from_pickup_site
        )
        measured_grasp = np.linalg.inv(attachment) @ instrument_body_pose
        planned_tool_from_instrument = np.asarray(plan["tool_from_instrument"], dtype=float)
        relative_position_error, relative_orientation_error = transform_error(
            measured_grasp, planned_tool_from_instrument
        )
        if (
            relative_position_error > self.args.pose_position_tolerance_m
            or relative_orientation_error > self.args.pose_orientation_tolerance_rad
        ):
            raise RuntimeError(
                "Measured grasp transform does not match the planned tool-to-instrument transform: "
                f"position error={relative_position_error*1000:.2f} mm, "
                f"orientation error={math.degrees(relative_orientation_error):.2f} deg"
            )

        self._set_collision_instrument_pose(instrument_body_pose)
        self._check_pickup_support()
        self._call_attachment_service(True)
        self.instrument_attached = True
        # The bridge stamps both sites while holding the same physics-state lock.
        # Once attached their relative pose is the transform the service captured.
        acknowledged_ns = self.get_clock().now().nanoseconds
        deadline = time.monotonic() + self.args.service_timeout_s
        while rclpy.ok() and time.monotonic() < deadline:
            self._spin_once(0.02)
            stamp = self.pose_stamp_ns.get("attachment", 0)
            if stamp > acknowledged_ns and stamp == self.pose_stamp_ns.get("pickup"):
                site = self._require_fresh_pose("attachment")
                body = instrument_body_pose_from_pickup_site(
                    self._require_fresh_pose("pickup"), self.instrument_from_pickup_site
                )
                self.tool_from_instrument_actual = np.linalg.inv(site) @ body
                break
        if self.tool_from_instrument_actual is None:
            raise RuntimeError("No synchronized post-attachment pose pair from MuJoCo")
        relative_position_error, relative_orientation_error = transform_error(
            self.tool_from_instrument_actual, planned_tool_from_instrument
        )
        if (
            relative_position_error > self.args.pose_position_tolerance_m
            or relative_orientation_error > self.args.pose_orientation_tolerance_rad
        ):
            raise RuntimeError("Bridge-captured grasp transform is outside the planned pose tolerances")
        orientation_offset = pickup_metrics[
            "planned_tool_to_pickup_site_orientation_offset_rad"
        ]
        self.get_logger().info(
            f"{self.instrument} attachment confirmed at pickup; relative transform captured by MuJoCo bridge. "
            f"Planned tool/site orientation offset={math.degrees(orientation_offset):.2f} deg; "
            f"grasp-transform error={relative_position_error*1000:.2f} mm / "
            f"{math.degrees(relative_orientation_error):.2f} deg."
        )

    def _release_at_handover(self, plan: dict[str, Any]) -> None:
        attachment = self._require_fresh_pose("attachment")
        expected_tool = np.asarray(plan["desired_tool_handover_pose_world"], dtype=float)
        tool_position_error, tool_orientation_error = transform_error(attachment, expected_tool)
        if (
            tool_position_error > self.args.handover_position_tolerance_m
            or tool_orientation_error > self.args.pose_orientation_tolerance_rad
        ):
            raise RuntimeError(
                "Handover tool pose rejected: "
                f"position error={tool_position_error*1000:.2f} mm, "
                f"orientation error={math.degrees(tool_orientation_error):.2f} deg"
            )

        pickup_site_pose = self._require_fresh_pose("pickup")
        instrument_body_pose = instrument_body_pose_from_pickup_site(
            pickup_site_pose, self.instrument_from_pickup_site
        )
        expected_instrument = np.asarray(
            plan["desired_instrument_handover_pose_world"], dtype=float
        )
        body_position_error, body_orientation_error = transform_error(
            instrument_body_pose, expected_instrument
        )
        if (
            body_position_error > self.args.handover_position_tolerance_m
            or body_orientation_error > self.args.pose_orientation_tolerance_rad
        ):
            raise RuntimeError(
                "Instrument body is not at the surgeon's handover target: "
                f"position error={body_position_error*1000:.2f} mm, "
                f"orientation error={math.degrees(body_orientation_error):.2f} deg"
            )

        self._call_attachment_service(False)
        self.instrument_attached = False
        self.instrument_released = True
        self.release_pickup_site_pose = pickup_site_pose.copy()
        self.release_instrument_body_pose = instrument_body_pose.copy()
        self.release_monotonic = time.monotonic()
        self.get_logger().info(
            f"{self.instrument} release verified at handover target; instrument remains at the receiver."
        )

    def _verify_release_stability(self) -> None:
        if self.release_pickup_site_pose is None:
            raise RuntimeError("No release pose was recorded")
        # Require a localization message received after the release service
        # returned; a cached pre-release sample is not sufficient evidence.
        release_monotonic = self.release_monotonic
        deadline = time.monotonic() + 1.5
        while rclpy.ok() and time.monotonic() < deadline:
            self._spin_once(0.05)
            received_at = self.pose_received_at.get("pickup", 0.0)
            if received_at > release_monotonic and time.monotonic() - received_at <= 0.25:
                break
        else:
            raise TimeoutError("No fresh post-release instrument pose was received")
        current = self._require_fresh_pose("pickup")
        position_error, orientation_error = transform_error(
            current, self.release_pickup_site_pose
        )
        if position_error > self.args.release_stability_tolerance_m or orientation_error > 0.02:
            raise RuntimeError(
                "Released instrument moved during robot retreat: "
                f"position change={position_error*1000:.2f} mm, "
                f"orientation change={math.degrees(orientation_error):.2f} deg"
            )
        self.get_logger().info("Released instrument remained stationary during robot retreat.")

    def _save_results(self, status: str, reason: str, plan_path: Path, preflight_path: Path, preflight_review: dict[str, Any]) -> tuple[Path, Path]:
        self.args.output_dir.mkdir(parents=True, exist_ok=True)
        stem = f"surgical_handover_execution_{self.instrument}_{self.run_id}"
        csv_path = self.args.output_dir / f"{stem}.csv"
        json_path = self.args.output_dir / f"{stem}_metrics.json"

        fields = ["elapsed_s", "phase", "instrument_attached", "instrument_released"]
        for name in self.joint_names:
            fields.extend(
                [
                    f"q_desired_{name}_rad",
                    f"q_actual_{name}_rad",
                    f"error_{name}_rad",
                    f"qdot_desired_{name}_rad_s",
                    f"qdot_actual_{name}_rad_s",
                    f"qddot_desired_{name}_rad_s2",
                    f"command_{name}_rad",
                    f"tau_ff_{name}_Nm",
                    f"command_clipped_{name}",
                ]
            )
        with csv_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(self.rows)

        metrics: dict[str, Any] = {}
        if self.rows:
            desired = np.asarray(
                [[row[f"q_desired_{name}_rad"] for name in self.joint_names] for row in self.rows],
                dtype=float,
            )
            actual = np.asarray(
                [[row[f"q_actual_{name}_rad"] for name in self.joint_names] for row in self.rows],
                dtype=float,
            )
            metrics = tracking_error_metrics(desired, actual, self.joint_names)
        summary = {
            "run_id": self.run_id,
            "started_at_utc": self.run_started_utc,
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": status,
            "reason": reason,
            "instrument": self.instrument,
            "control_strategy": self.strategy,
            "scene_xml": str(self.scene_path),
            "plan_json": str(plan_path),
            "collision_preflight_json": str(preflight_path),
            "collision_preflight_status": preflight_review["status"],
            "warning_review_approved": preflight_review["warning_review_approved"],
            "minimum_clearances_m": preflight_review["minimum_clearances_m"],
            "sample_count": len(self.rows),
            "configured_control_rate_hz": self.args.control_rate_hz,
            "maximum_feedback_age_observed_s": self.max_feedback_age_seen_s,
            "maximum_joint_tracking_error_observed_rad": self.max_tracking_error_seen_rad,
            "online_measured_state_minimum_clearances_m": self.online_minimum_clearances_m,
            "last_collision_sample": self.last_collision_sample,
            "site_from_instrument_actual": (
                None if self.tool_from_instrument_actual is None else self.tool_from_instrument_actual.tolist()
            ),
            "instrument_attached_at_end": self.instrument_attached,
            "instrument_released": self.instrument_released,
            "units": {"joint_position": "rad", "joint_velocity": "rad/s", "feedforward_torque": "N*m"},
            **metrics,
            "csv_report": str(csv_path.resolve()),
        }
        json_path.write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
        self.get_logger().info(f"Execution CSV saved: {csv_path}")
        self.get_logger().info(f"Execution metrics saved: {json_path}")
        return csv_path, json_path

    def execute(self, plan: dict[str, Any], segments: tuple[Any, ...], plan_path: Path, preflight_path: Path, preflight_review: dict[str, Any]) -> None:
        status = "ABORTED"
        reason = "Execution did not complete"
        self.collision_minimum_clearance_m = preflight_review["minimum_required_clearance_m"]
        self.collision_distance_cap_m = preflight_review["distance_cap_m"]
        self.online_minimum_clearances_m = dict.fromkeys(
            self.online_minimum_clearances_m, self.collision_distance_cap_m
        )
        try:
            self.wait_for_fresh_data(self.args.startup_timeout_s)
            self.validate_live_start(plan)
            self.run_started_monotonic = time.monotonic()
            self.run_started_utc = datetime.now(timezone.utc).isoformat()
            self.get_logger().warning(
                "SIMULATION EXECUTION STARTED. This command controls the MuJoCo UR5e simulation only."
            )

            for segment in segments:
                self._run_segment(segment)
                if segment.name == "pickup":
                    self._attach_at_pickup(plan)
                elif segment.name == "lift":
                    if not self.pickup_support_contact_cleared:
                        raise RuntimeError("Instrument did not achieve the required pickup-support clearance during lift")
                elif segment.name == "handover":
                    self._release_at_handover(plan)
                elif segment.name == "retreat":
                    self._verify_release_stability()

            q, qdot = self._require_fresh_feedback()
            home = np.asarray(segments[-1].goal_position_rad, dtype=float)
            home_error = float(np.max(np.abs(q - home)))
            if home_error > self.args.goal_position_tolerance_rad:
                raise RuntimeError(f"Final HOME error is {home_error:.5f} rad")
            if self.instrument_attached or not self.instrument_released:
                raise RuntimeError("Execution ended without a confirmed instrument release")
            self._publish_hold()
            status = "COMPLETED"
            reason = "All nine waypoints completed; instrument released and held at handover; robot returned HOME."
            self.get_logger().info(reason)
        except Exception as exc:
            reason = str(exc)
            self.get_logger().error(f"Surgical handover aborted: {reason}")
            self._publish_hold()
            if self.instrument_attached:
                self.get_logger().error(
                    "The instrument remains attached to the robot. No automatic release was attempted. "
                    "Keep the simulation stopped and inspect the scene before any recovery action."
                )
        finally:
            self._save_results(status, reason, plan_path, preflight_path, preflight_review)
        if status != "COMPLETED":
            raise RuntimeError(reason)


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Preflight JSON does not exist: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Preflight JSON root must be an object")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Execute a validated surgical instrument handover in the MuJoCo/ROS 2 simulation."
    )
    parser.add_argument("--instrument", choices=("scalpel", "forceps"), required=True)
    parser.add_argument("--plan-json", type=Path, required=True)
    parser.add_argument("--preflight-json", type=Path, required=True)
    parser.add_argument("--scene-xml", type=Path, default=DEFAULT_SCENE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--strategy", choices=("baseline", "inverse_dynamics"), default="inverse_dynamics")
    parser.add_argument("--control-rate-hz", type=float, default=100.0)
    parser.add_argument("--feedback-timeout-s", type=float, default=0.10)
    parser.add_argument("--startup-timeout-s", type=float, default=10.0)
    parser.add_argument("--service-timeout-s", type=float, default=5.0)
    parser.add_argument("--start-tolerance-rad", type=float, default=0.05)
    parser.add_argument("--start-velocity-tolerance-rad-s", type=float, default=0.05)
    parser.add_argument("--max-tracking-error-rad", type=float, default=0.04)
    parser.add_argument("--tracking-error-consecutive-samples", type=int, default=10)
    parser.add_argument("--goal-position-tolerance-rad", type=float, default=0.02)
    parser.add_argument("--goal-velocity-tolerance-rad-s", type=float, default=0.05)
    parser.add_argument("--goal-settle-samples", type=int, default=8)
    parser.add_argument("--goal-settle-timeout-s", type=float, default=5.0)
    parser.add_argument("--pose-position-tolerance-m", type=float, default=0.02)
    parser.add_argument("--pose-orientation-tolerance-rad", type=float, default=0.10)
    parser.add_argument("--pickup-alignment-tolerance-m", type=float, default=0.02)
    parser.add_argument("--handover-position-tolerance-m", type=float, default=0.02)
    parser.add_argument("--release-stability-tolerance-m", type=float, default=0.003)
    parser.add_argument(
        "--approve-warnings",
        action="store_true",
        help="Explicitly approve review-only clearances below the preflight warning threshold; hard failures remain blocked.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually publish joint commands. Without this flag, only plan/preflight files are validated.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.plan_json = args.plan_json.expanduser().resolve()
    args.preflight_json = args.preflight_json.expanduser().resolve()
    args.scene_xml = args.scene_xml.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()

    if not math.isfinite(args.control_rate_hz) or args.control_rate_hz <= 0.0:
        raise ValueError("control-rate-hz must be positive and finite")
    if not math.isfinite(args.feedback_timeout_s) or args.feedback_timeout_s <= 0.0:
        raise ValueError("feedback-timeout-s must be positive and finite")
    if args.goal_settle_samples < 1 or args.tracking_error_consecutive_samples < 1:
        raise ValueError("Sample-count parameters must be positive")
    if not args.scene_xml.is_file():
        raise FileNotFoundError(f"Scene XML does not exist: {args.scene_xml}")

    if args.preflight_json.stat().st_mtime + 1e-6 < args.plan_json.stat().st_mtime:
        raise ExecutionValidationError(
            "The collision-preflight file is older than the plan JSON; rerun preflight for this plan."
        )
    plan = load_execution_plan(args.plan_json, args.instrument)
    preflight = _load_json(args.preflight_json)
    review = validate_preflight_for_execution(
        preflight,
        plan=plan,
        plan_path=args.plan_json,
        scene_path=args.scene_xml,
        approve_warnings=args.approve_warnings,
    )
    segments = build_execution_segments(plan)

    print("========== SURGICAL HANDOVER EXECUTION GATE ==========")
    print(f"Instrument: {args.instrument}")
    print(f"Plan: {args.plan_json}")
    print(f"Preflight: {args.preflight_json}")
    print(f"Preflight status: {review['status']}")
    print(f"Distance backend: MuJoCo native CCD")
    print("Minimum clearances:")
    for name, value in review["minimum_clearances_m"].items():
        print(f"  {name}: {value*1000:.3f} mm")
    if review["status"] == "WARNING_REVIEW_REQUIRED":
        print("Review-only pairs below warning threshold:")
        for pair in review["review_pairs"]:
            print(
                f"  {pair['minimum_distance_m']*1000:.3f} mm | {pair['phase']} | "
                f"{pair['geom_a']} <-> {pair['geom_b']}"
            )
        print(f"Explicit warning approval supplied: {review['warning_review_approved']}")
    print(f"Trajectory segments: {len(segments)}")
    print(f"Total planned duration: {sum(s.duration_s for s in segments):.3f} s")

    if not args.execute:
        print("DRY RUN: plan and preflight artifacts are consistent. No ROS commands were published.")
        print("Add --execute only when the MuJoCo scene is running and the operator is ready.")
        return 0

    if review["status"] == "WARNING_REVIEW_REQUIRED" and not args.approve_warnings:
        raise ExecutionValidationError("Warning review approval is required before execution")

    rclpy.init()
    node = None
    try:
        node = SurgicalHandoverExecutor(
            instrument=args.instrument,
            scene_path=args.scene_xml,
            strategy=args.strategy,
            args=args,
        )
        node.execute(plan, segments, args.plan_json, args.preflight_json, review)
        return 0
    except KeyboardInterrupt:
        if node is not None:
            node._publish_hold()
        print("Interrupted. A hold command was sent at the latest measured joint pose.")
        return 130
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
