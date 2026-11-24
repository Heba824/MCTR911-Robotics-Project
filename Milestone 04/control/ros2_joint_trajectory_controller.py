#!/usr/bin/env python3
"""ROS 2 closed-loop joint-space trajectory runner for MCTR911 Milestone 04.

The node starts in IDLE and never moves the robot automatically. An operator
must call /m4_joint_trajectory/start after verifying the MuJoCo scene and HOME
pose. /m4_joint_trajectory/stop requests a controlled hold at the latest
measured position.

Strategies:
  baseline          Publish q_ref to the existing POSITION actuators.
  inverse_dynamics  Add MuJoCo inverse-dynamics feedforward to the same
                    position-actuator PD feedback law derived in M4 Stage 4A.

This is a simulation controller, not a safety-rated or hardware controller.
"""
from __future__ import annotations

import csv
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
from std_srvs.srv import Trigger

from joint_trajectory_controller_core import (
    DEFAULT_JOINT_NAMES,
    reorder_joint_state,
    tracking_error_metrics,
    validate_start_position,
    validate_start_velocity,
)
from trajectory_csv import JointTrajectoryCSV

try:
    from motor_level_controller import (
        compute_inverse_dynamics_feedforward,
        compute_position_servo_command,
        read_position_actuator_gains,
    )
except ImportError:
    compute_inverse_dynamics_feedforward = None
    compute_position_servo_command = None
    read_position_actuator_gains = None


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REFERENCE = (
    PROJECT_ROOT
    / "Milestone 04"
    / "Results"
    / "Joint_Space"
    / "joint_space_reference.csv"
)
DEFAULT_SCENE = (
    PROJECT_ROOT
    / "Milestone 01"
    / "Robot Model"
    / "scene_surgical_m3.xml"
)


class JointSpaceTrajectoryController(Node):
    def __init__(self) -> None:
        super().__init__("m4_joint_space_trajectory_controller")

        self.declare_parameter("trajectory_csv", str(DEFAULT_REFERENCE))
        self.declare_parameter("joint_names", list(DEFAULT_JOINT_NAMES))
        self.declare_parameter("joint_state_topic", "/joint_state")
        self.declare_parameter("joint_command_topic", "/joint_commands")
        self.declare_parameter("control_rate_hz", 100.0)
        self.declare_parameter("feedback_timeout_s", 0.10)
        self.declare_parameter("start_tolerance_rad", 0.05)
        self.declare_parameter("start_velocity_tolerance_rad_s", 0.05)
        self.declare_parameter("max_tracking_error_rad", 0.25)
        self.declare_parameter("tracking_error_consecutive_samples", 10)
        self.declare_parameter("control_strategy", "baseline")
        self.declare_parameter("scene_xml", str(DEFAULT_SCENE))
        self.declare_parameter(
            "output_directory",
            str(PROJECT_ROOT / "Milestone 04" / "Results" / "Tracking" / "Joint_Space"),
        )
        self.declare_parameter("abort_on_command_saturation", True)

        self.joint_names = tuple(self.get_parameter("joint_names").value)
        if len(self.joint_names) != 6 or len(set(self.joint_names)) != 6:
            raise ValueError("joint_names must contain the six unique UR5e joint names")
        if self.joint_names != DEFAULT_JOINT_NAMES:
            raise ValueError(
                "This M4 reference and motor model currently require the documented "
                "UR5e joint order: " + ", ".join(DEFAULT_JOINT_NAMES)
            )

        self.reference_path = Path(
            self.get_parameter("trajectory_csv").value
        ).expanduser().resolve()
        self.trajectory = JointTrajectoryCSV.load(
            self.reference_path, self.joint_names
        )

        self.joint_state_topic = str(self.get_parameter("joint_state_topic").value)
        self.joint_command_topic = str(self.get_parameter("joint_command_topic").value)
        self.control_rate_hz = float(self.get_parameter("control_rate_hz").value)
        self.feedback_timeout_s = float(self.get_parameter("feedback_timeout_s").value)
        self.start_tolerance_rad = float(self.get_parameter("start_tolerance_rad").value)
        self.start_velocity_tolerance_rad_s = float(
            self.get_parameter("start_velocity_tolerance_rad_s").value
        )
        self.max_tracking_error_rad = float(
            self.get_parameter("max_tracking_error_rad").value
        )
        self.tracking_error_consecutive_samples = int(
            self.get_parameter("tracking_error_consecutive_samples").value
        )
        self.control_strategy = str(self.get_parameter("control_strategy").value).lower()
        self.output_directory = Path(
            self.get_parameter("output_directory").value
        ).expanduser().resolve()
        self.abort_on_command_saturation = bool(
            self.get_parameter("abort_on_command_saturation").value
        )

        if not np.isfinite(self.control_rate_hz) or self.control_rate_hz <= 0.0:
            raise ValueError("control_rate_hz must be positive and finite")
        if not np.isfinite(self.feedback_timeout_s) or self.feedback_timeout_s <= 0.0:
            raise ValueError("feedback_timeout_s must be positive and finite")
        if not np.isfinite(self.start_tolerance_rad) or self.start_tolerance_rad <= 0.0:
            raise ValueError("start_tolerance_rad must be positive and finite")
        if (
            not np.isfinite(self.start_velocity_tolerance_rad_s)
            or self.start_velocity_tolerance_rad_s <= 0.0
        ):
            raise ValueError("start_velocity_tolerance_rad_s must be positive and finite")
        if (
            not np.isfinite(self.max_tracking_error_rad)
            or self.max_tracking_error_rad <= 0.0
        ):
            raise ValueError("max_tracking_error_rad must be positive and finite")
        if self.tracking_error_consecutive_samples < 1:
            raise ValueError("tracking_error_consecutive_samples must be at least one")
        if self.control_strategy not in ("baseline", "inverse_dynamics"):
            raise ValueError("control_strategy must be baseline or inverse_dynamics")

        self._mujoco = None
        self._model = None
        self._actual_data = None
        self._reference_data = None
        self._qpos_indices = None
        self._dof_indices = None
        self._kp = None
        self._kd = None
        self._control_range = None

        if self.control_strategy == "inverse_dynamics":
            self._initialize_inverse_dynamics()

        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        self._publisher = self.create_publisher(
            Float64MultiArray, self.joint_command_topic, qos
        )
        self._subscription = self.create_subscription(
            JointState, self.joint_state_topic, self._joint_state_callback, qos
        )
        self._start_service = self.create_service(
            Trigger, "/m4_joint_trajectory/start", self._start_callback
        )
        self._stop_service = self.create_service(
            Trigger, "/m4_joint_trajectory/stop", self._stop_callback
        )

        self._state = "IDLE"
        self._latest_position: Optional[np.ndarray] = None
        self._latest_velocity: Optional[np.ndarray] = None
        self._last_feedback_monotonic: Optional[float] = None
        self._run_start_monotonic: Optional[float] = None
        self._run_start_wall_iso: Optional[str] = None
        self._run_id: Optional[str] = None
        self._rows: list[dict] = []
        self._start_error_max_rad: Optional[float] = None
        self._tracking_error_streak = 0
        self._max_feedback_age_seen_s = 0.0
        self._last_output_csv: Optional[Path] = None
        self._last_output_json: Optional[Path] = None

        timer_period = 1.0 / self.control_rate_hz
        self._timer = self.create_timer(timer_period, self._control_timer_callback)

        self.get_logger().info(
            "Joint-space controller ready in IDLE. "
            f"Reference: {self.reference_path}; samples={self.trajectory.sample_count}; "
            f"duration={self.trajectory.end_time_s:.3f}s; "
            f"strategy={self.control_strategy}; rate={self.control_rate_hz:.1f}Hz."
        )
        self.get_logger().info(
            "No motion is commanded at startup. Verify the MuJoCo HOME pose, then "
            "call /m4_joint_trajectory/start. Use /m4_joint_trajectory/stop to hold."
        )

    def _initialize_inverse_dynamics(self) -> None:
        try:
            import mujoco
        except ImportError as exc:
            raise RuntimeError(
                "inverse_dynamics strategy requires the Python 'mujoco' package "
                "in the active ROS 2 Python environment"
            ) from exc

        scene_path = Path(self.get_parameter("scene_xml").value).expanduser().resolve()
        if not scene_path.is_file():
            raise FileNotFoundError(f"MuJoCo scene XML not found: {scene_path}")
        if (
            compute_inverse_dynamics_feedforward is None
            or compute_position_servo_command is None
            or read_position_actuator_gains is None
        ):
            raise RuntimeError("Could not import the existing M4 motor-level controller")

        model = mujoco.MjModel.from_xml_path(str(scene_path))
        if model.nq != 6 or model.nv != 6 or model.nu != 6:
            raise ValueError(
                f"Expected a six-joint/six-actuator UR5e model; got "
                f"nq={model.nq}, nv={model.nv}, nu={model.nu}"
            )

        joint_ids = []
        actuator_ids = []
        actuator_names = (
            "shoulder_pan",
            "shoulder_lift",
            "elbow",
            "wrist_1",
            "wrist_2",
            "wrist_3",
        )
        for joint_name, actuator_name in zip(self.joint_names, actuator_names):
            joint_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_JOINT, joint_name
            )
            actuator_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name
            )
            if joint_id < 0 or actuator_id < 0:
                raise ValueError(
                    f"Could not resolve joint '{joint_name}' or actuator '{actuator_name}'"
                )
            if int(model.actuator_trnid[actuator_id, 0]) != joint_id:
                raise ValueError(
                    f"Actuator '{actuator_name}' is not attached to '{joint_name}'"
                )
            joint_ids.append(joint_id)
            actuator_ids.append(actuator_id)

        self._qpos_indices = np.asarray(
            [model.jnt_qposadr[jid] for jid in joint_ids], dtype=int
        )
        self._dof_indices = np.asarray(
            [model.jnt_dofadr[jid] for jid in joint_ids], dtype=int
        )
        self._kp, self._kd = read_position_actuator_gains(
            model, actuator_ids, expected_count=6
        )
        if np.all(np.asarray(model.actuator_ctrllimited)[actuator_ids]):
            self._control_range = np.asarray(
                model.actuator_ctrlrange[actuator_ids], dtype=float
            ).copy()

        self._mujoco = mujoco
        self._model = model
        self._actual_data = mujoco.MjData(model)
        self._reference_data = mujoco.MjData(model)

        if self._control_range is not None:
            q = self.trajectory.positions_rad
            lower = self._control_range[:, 0][None, :]
            upper = self._control_range[:, 1][None, :]
            if np.any(q < lower) or np.any(q > upper):
                raise ValueError(
                    "The reference contains joint positions outside the MuJoCo "
                    "actuator control ranges"
                )

    def _joint_state_callback(self, message: JointState) -> None:
        try:
            q, qdot = reorder_joint_state(
                message.name,
                message.position,
                message.velocity,
                self.joint_names,
            )
        except ValueError as exc:
            self.get_logger().warning(f"Ignoring invalid JointState: {exc}")
            return

        self._latest_position = q
        self._latest_velocity = qdot
        self._last_feedback_monotonic = time.monotonic()

    def _feedback_age(self) -> Optional[float]:
        if self._last_feedback_monotonic is None:
            return None
        return max(0.0, time.monotonic() - self._last_feedback_monotonic)

    def _start_callback(self, _request: Trigger.Request, response: Trigger.Response):
        if self._state == "RUNNING":
            response.success = False
            response.message = "A trajectory is already running."
            return response

        age = self._feedback_age()
        if (
            self._latest_position is None
            or self._latest_velocity is None
            or age is None
            or age > self.feedback_timeout_s
        ):
            response.success = False
            response.message = (
                "Fresh /joint_state feedback is required before starting; "
                f"feedback age={age}."
            )
            return response

        try:
            start_error = validate_start_position(
                self._latest_position,
                self.trajectory.start_position_rad,
                self.start_tolerance_rad,
            )
            start_velocity = validate_start_velocity(
                self._latest_velocity,
                self.start_velocity_tolerance_rad_s,
            )
        except ValueError as exc:
            response.success = False
            response.message = str(exc)
            return response

        self._run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self._run_start_wall_iso = datetime.now(timezone.utc).isoformat()
        self._run_start_monotonic = time.monotonic()
        self._rows = []
        self._start_error_max_rad = start_error
        self._tracking_error_streak = 0
        self._max_feedback_age_seen_s = 0.0
        self._state = "RUNNING"

        self.get_logger().info(
            f"Trajectory started: run_id={self._run_id}, "
            f"maximum initial joint error={start_error:.6f} rad; "
            f"maximum initial joint speed={start_velocity:.6f} rad/s."
        )
        response.success = True
        response.message = (
            f"Started {self.reference_path.name} using {self.control_strategy}; "
            f"max initial error={start_error:.6f} rad; "
            f"max initial speed={start_velocity:.6f} rad/s."
        )
        return response

    def _stop_callback(self, _request: Trigger.Request, response: Trigger.Response):
        if self._state != "RUNNING":
            response.success = False
            response.message = f"No active trajectory (state={self._state})."
            return response

        self._finish_run("STOPPED", "Operator requested stop.")
        response.success = True
        response.message = "Trajectory stopped; hold command sent at latest measured pose."
        return response

    def _compute_command(self, reference):
        if self.control_strategy == "baseline":
            return (
                reference.position_rad.copy(),
                np.zeros(6, dtype=float),
                np.zeros(6, dtype=bool),
            )

        self._actual_data.qpos[:] = 0.0
        self._actual_data.qvel[:] = 0.0
        self._actual_data.qpos[self._qpos_indices] = self._latest_position
        self._actual_data.qvel[self._dof_indices] = self._latest_velocity

        tau_ff = compute_inverse_dynamics_feedforward(
            self._model,
            self._actual_data,
            self._reference_data,
            self._qpos_indices,
            self._dof_indices,
            reference.position_rad,
            reference.velocity_rad_s,
            reference.acceleration_rad_s2,
        )
        servo = compute_position_servo_command(
            reference.position_rad,
            reference.velocity_rad_s,
            self._kp,
            self._kd,
            feedforward_torque_nm=tau_ff,
            control_range_rad=self._control_range,
        )
        return servo.command_rad, servo.feedforward_torque_nm, servo.clipped

    def _publish_position(self, position_rad: np.ndarray) -> None:
        message = Float64MultiArray()
        message.data = [float(value) for value in position_rad]
        self._publisher.publish(message)

    def _control_timer_callback(self) -> None:
        if self._state != "RUNNING" or self._run_start_monotonic is None:
            return

        age = self._feedback_age()
        if age is None or age > self.feedback_timeout_s:
            self._finish_run(
                "ABORTED",
                f"Joint-state feedback became stale (age={age}; "
                f"timeout={self.feedback_timeout_s:.3f}s).",
            )
            return

        self._max_feedback_age_seen_s = max(self._max_feedback_age_seen_s, age)
        elapsed = max(0.0, time.monotonic() - self._run_start_monotonic)
        trajectory_time = min(elapsed, self.trajectory.end_time_s)
        reference = self.trajectory.evaluate(trajectory_time)
        actual_q = self._latest_position.copy()
        actual_qdot = self._latest_velocity.copy()
        tracking_error = reference.position_rad - actual_q

        if np.max(np.abs(tracking_error)) > self.max_tracking_error_rad:
            self._tracking_error_streak += 1
        else:
            self._tracking_error_streak = 0

        if self._tracking_error_streak >= self.tracking_error_consecutive_samples:
            self._append_sample(
                elapsed,
                reference,
                actual_q,
                actual_qdot,
                actual_q,
                np.zeros(6),
                np.zeros(6, dtype=bool),
            )
            self._finish_run(
                "ABORTED",
                "Tracking error exceeded "
                f"{self.max_tracking_error_rad:.3f} rad for "
                f"{self.tracking_error_consecutive_samples} consecutive samples.",
            )
            return

        try:
            command, tau_ff, clipped = self._compute_command(reference)
        except Exception as exc:
            self._finish_run("ABORTED", f"Control calculation failed: {exc}")
            return

        if np.any(clipped) and self.abort_on_command_saturation:
            self._append_sample(
                elapsed, reference, actual_q, actual_qdot, command, tau_ff, clipped
            )
            self._finish_run(
                "ABORTED",
                "Position command reached an actuator control limit; no clipped "
                "trajectory command was published.",
            )
            return

        self._publish_position(command)
        self._append_sample(
            elapsed, reference, actual_q, actual_qdot, command, tau_ff, clipped
        )

        if elapsed >= self.trajectory.end_time_s:
            self._finish_run(
                "COMPLETED",
                "Reference duration reached; final goal position is held.",
                hold_position=self.trajectory.goal_position_rad,
            )

    def _append_sample(
        self,
        elapsed_s: float,
        reference,
        actual_q: np.ndarray,
        actual_qdot: np.ndarray,
        command: np.ndarray,
        tau_ff: np.ndarray,
        clipped: np.ndarray,
    ) -> None:
        row = {
            "wall_time_s": float(elapsed_s),
            "trajectory_time_s": float(reference.time_s),
            "phase": str(reference.phase),
        }
        for i, name in enumerate(self.joint_names):
            row[f"q_desired_{name}_rad"] = float(reference.position_rad[i])
            row[f"q_actual_{name}_rad"] = float(actual_q[i])
            row[f"error_{name}_rad"] = float(reference.position_rad[i] - actual_q[i])
            row[f"qdot_desired_{name}_rad_s"] = float(reference.velocity_rad_s[i])
            row[f"qdot_actual_{name}_rad_s"] = float(actual_qdot[i])
            row[f"command_{name}_rad"] = float(command[i])
            row[f"tau_ff_{name}_Nm"] = float(tau_ff[i])
            row[f"command_clipped_{name}"] = int(bool(clipped[i]))
        self._rows.append(row)

    def _publish_hold(self, hold_position: Optional[np.ndarray] = None) -> None:
        if hold_position is None:
            hold_position = self._latest_position
        if hold_position is not None:
            try:
                self._publish_position(np.asarray(hold_position, dtype=float))
            except Exception as exc:
                self.get_logger().error(f"Could not publish hold command: {exc}")

    def _finish_run(
        self,
        status: str,
        reason: str,
        *,
        hold_position: Optional[np.ndarray] = None,
    ) -> None:
        if self._state != "RUNNING":
            return
        self._state = status
        self._publish_hold(hold_position)
        self._save_run_results(status, reason)
        self.get_logger().info(f"Trajectory {status}: {reason}")

    def _save_run_results(self, status: str, reason: str) -> None:
        if not self._rows or not self._run_id:
            self.get_logger().warning("No tracking samples were recorded; no result file saved.")
            return

        self.output_directory.mkdir(parents=True, exist_ok=True)
        stem = f"joint_space_tracking_{self.control_strategy}_{self._run_id}"
        csv_path = self.output_directory / f"{stem}.csv"
        json_path = self.output_directory / f"{stem}_metrics.json"

        fieldnames = ["wall_time_s", "trajectory_time_s", "phase"]
        for name in self.joint_names:
            fieldnames.extend(
                (
                    f"q_desired_{name}_rad",
                    f"q_actual_{name}_rad",
                    f"error_{name}_rad",
                    f"qdot_desired_{name}_rad_s",
                    f"qdot_actual_{name}_rad_s",
                    f"command_{name}_rad",
                    f"tau_ff_{name}_Nm",
                    f"command_clipped_{name}",
                )
            )

        with csv_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self._rows)

        desired = np.asarray(
            [
                [row[f"q_desired_{name}_rad"] for name in self.joint_names]
                for row in self._rows
            ],
            dtype=float,
        )
        actual = np.asarray(
            [
                [row[f"q_actual_{name}_rad"] for name in self.joint_names]
                for row in self._rows
            ],
            dtype=float,
        )
        metrics = tracking_error_metrics(desired, actual, self.joint_names)
        wall_times = np.asarray([row["wall_time_s"] for row in self._rows])
        observed_rate = (
            float((len(wall_times) - 1) / (wall_times[-1] - wall_times[0]))
            if len(wall_times) > 1 and wall_times[-1] > wall_times[0]
            else None
        )
        saturation_count = int(
            sum(
                row[f"command_clipped_{name}"]
                for row in self._rows
                for name in self.joint_names
            )
        )
        summary = {
            "run_id": self._run_id,
            "started_at_utc": self._run_start_wall_iso,
            "status": status,
            "reason": reason,
            "control_strategy": self.control_strategy,
            "reference_csv": str(self.reference_path),
            "sample_count": len(self._rows),
            "trajectory_duration_s": self.trajectory.end_time_s,
            "configured_control_rate_hz": self.control_rate_hz,
            "observed_controller_rate_hz": observed_rate,
            "feedback_timeout_s": self.feedback_timeout_s,
            "maximum_feedback_age_observed_s": self._max_feedback_age_seen_s,
            "maximum_initial_joint_error_rad": self._start_error_max_rad,
            "command_saturation_count": saturation_count,
            **metrics,
            "tracking_error_sign_convention": "desired_position - actual_position",
            "units": {"position": "rad", "velocity": "rad/s", "feedforward_torque": "N*m"},
        }
        with json_path.open("w", encoding="utf-8") as stream:
            json.dump(summary, stream, indent=2, allow_nan=False)

        self._last_output_csv = csv_path
        self._last_output_json = json_path
        self.get_logger().info(f"Tracking data saved: {csv_path}")
        self.get_logger().info(f"Tracking metrics saved: {json_path}")

    def destroy_node(self):
        if getattr(self, "_state", None) == "RUNNING":
            self._finish_run("STOPPED", "Controller node is shutting down.")
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = None
    try:
        node = JointSpaceTrajectoryController()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
