#!/usr/bin/env python3
"""Reproduce the Milestone 04 evaluation from the archived experiment files."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

MILESTONE = Path(__file__).resolve().parents[1]
ROOT = MILESTONE.parent
RESULTS = MILESTONE / "Results"
TRACKING = RESULTS / "Tracking"
HANDOVER = RESULTS / "Surgical_Handover"
OUTPUT = RESULTS / "Final_Evaluation"
for folder in ("application", "control"):
    sys.path.insert(0, str(MILESTONE / folder))

from surgical_handover_planner import JOINT_NAMES, forward_kinematics, validate_transform
from surgical_handover_execution_core import REQUIRED_WAYPOINTS, transform_error
from trajectory_csv import JointTrajectoryCSV

RUNS = {
    "Joint-space reference": (
        TRACKING / "Joint_Space/joint_space_tracking_baseline_20261123_172028_062702.csv",
        TRACKING / "Joint_Space/joint_space_tracking_inverse_dynamics_20261123_194152_897742.csv",
    ),
    "Task-space-derived reference": (
        TRACKING / "Task_Space/Baseline/joint_space_tracking_baseline_20261123_212819_869677.csv",
        TRACKING / "Task_Space/Inverse_Dynamics/joint_space_tracking_inverse_dynamics_20261123_231450_053239.csv",
    ),
}
CARTESIAN = (
    TRACKING / "Task_Space/Baseline/cartesian_tracking_baseline.csv",
    TRACKING / "Task_Space/Inverse_Dynamics/task_space_tracking_inverse_dynamics_20261123_231450_053239.csv",
)
EXECUTIONS = (
    "scalpel_20261124_114950_878632", "forceps_20261124_115713_181134",
    "scalpel_20261124_111756_178765", "scalpel_20261124_113401_483815",
)
DISTANCES = ("robot_environment", "robot_self", "carried_instrument_environment")
CART_METRICS = {
    "position": ("m", "Position"),
    "orientation": ("rad", "Orientation"),
    "linear_velocity": ("m_s", "Linear velocity"),
    "angular_velocity": ("rad_s", "Angular velocity"),
}
LABELS = ("Shoulder pan", "Shoulder lift", "Elbow", "Wrist 1", "Wrist 2", "Wrist 3")
STRATEGIES = ("Baseline PD", "Inverse dynamics + PD")
COLORS = ("#176783", "#AF531B")


class Evidence:
    def __init__(self):
        self.sources = {}

    def record(self, path):
        path = Path(path)
        name = path.relative_to(ROOT).as_posix()
        self.sources[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        return name

    def json(self, path):
        self.record(path)
        def reject_constant(value):
            raise ValueError(f"{path}: invalid JSON number {value}")
        return json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)

    def csv(self, path, columns, *, allow_empty=False):
        self.record(path)
        with path.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream, strict=True)
            header = reader.fieldnames or []
            if len(header) != len(set(header)) or not set(columns) <= set(header):
                raise ValueError(f"{path}: invalid CSV header or missing {sorted(set(columns) - set(header))}")
            rows = list(reader)
        if (not allow_empty and not rows) or any(None in row or None in row.values() for row in rows):
            raise ValueError(f"{path}: empty CSV or inconsistent row width")
        return rows


def local_artifact(value):
    # Archived manifests use absolute paths from the recording machine.
    parts = Path(value).parts
    return ROOT.joinpath(*parts[parts.index("Milestone 04"):])


def check(condition, context):
    if not condition:
        raise ValueError(context)


def close(actual, expected, context, *, atol=1e-10):
    check(np.allclose(actual, expected, rtol=0, atol=atol), context)


def numbers(rows, columns, path):
    try:
        values = np.array([[float(row[column]) for column in columns] for row in rows])
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{path}: invalid numeric values in {columns}") from exc
    check(np.isfinite(values).all(), f"{path}: nonfinite CSV values in {columns}")
    return values


def rmse(values, axis=None):
    return np.sqrt(np.mean(np.square(values), axis=axis))


def joint_data(evidence, path, meta, *, handover=False):
    time_key = "elapsed_s" if handover else "wall_time_s"
    patterns = {
        "q": "q_desired_{}_rad", "actual": "q_actual_{}_rad", "error": "error_{}_rad",
        "qd": "qdot_desired_{}_rad_s", "actual_d": "qdot_actual_{}_rad_s",
        "command": "command_{}_rad", "tau": "tau_ff_{}_Nm", "clipped": "command_clipped_{}",
    }
    columns = {key: [pattern.format(joint) for joint in JOINT_NAMES] for key, pattern in patterns.items()}
    other = ["instrument_attached", "instrument_released"] if handover else ["trajectory_time_s"]
    rows = evidence.csv(path, [time_key, "phase", *other, *sum(columns.values(), [])],
                        allow_empty=meta["status"] == "ABORTED")
    check(len(rows) == meta["sample_count"], f"{path}: sample count disagrees with metrics")
    units = {"joint_position": "rad", "joint_velocity": "rad/s"} if handover else {
        "position": "rad", "velocity": "rad/s"}
    check(meta["units"] == {**units, "feedforward_torque": "N*m"}, f"{path}: unexpected joint units")
    if not rows:
        return None
    data = {key: numbers(rows, names, path) for key, names in columns.items()}
    data.update({key: numbers(rows, [key], path)[:, 0] for key in (time_key, *other)})
    data["time"] = data[time_key]
    data["phase"] = np.array([row["phase"] for row in rows])
    check(np.all(np.diff(data["time"]) > 0) and np.all(data["time"] >= 0), f"{path}: invalid timestamps")
    check(all(row["phase"] for row in rows), f"{path}: blank phase")
    for key in ("clipped", *[field for field in other if field.startswith("instrument_")]):
        check(np.isin(data[key], (0, 1)).all(), f"{path}: {key} must be binary")
    close(data["error"], data["q"] - data["actual"], f"{path}: position error sign/value mismatch")
    data["velocity_error"] = data["qd"] - data["actual_d"]
    close(rmse(data["error"]), meta["overall"]["rmse_rad"], f"{path}: RMSE disagrees with metrics")
    return data


def joint_stats(data):
    error, speed = data["error"], data["velocity_error"]
    return dict(
        samples=len(error), position_rmse_rad=float(rmse(error)), position_peak_rad=float(np.abs(error).max()),
        velocity_rmse_rad_s=float(rmse(speed)), velocity_peak_rad_s=float(np.abs(speed).max()),
        peak_position_command_rad=float(np.abs(data["command"]).max()),
        peak_feedforward_torque_Nm=float(np.abs(data["tau"]).max()),
        clipped_samples=int(data["clipped"].any(axis=1).sum()),
        clipped_joint_commands=int(data["clipped"].sum()),
        final_max_joint_error_rad=float(np.abs(error[-1]).max()),
        final_max_joint_speed_rad_s=float(np.abs(data["actual_d"][-1]).max()),
    )


def joint_rows(data, source, **labels):
    return [
        dict(**labels, joint=joint, position_rmse_rad=float(rmse(data["error"][:, i])),
             position_peak_rad=float(np.abs(data["error"][:, i]).max()),
             velocity_rmse_rad_s=float(rmse(data["velocity_error"][:, i])),
             velocity_peak_rad_s=float(np.abs(data["velocity_error"][:, i]).max()),
             peak_position_command_rad=float(np.abs(data["command"][:, i]).max()),
             peak_feedforward_torque_Nm=float(np.abs(data["tau"][:, i]).max()),
             clipped_joint_commands=int(data["clipped"][:, i].sum()), source_csv=source)
        for i, joint in enumerate(JOINT_NAMES)
    ]


def tracking(evidence, tables):
    groups = {}
    for group, paths in RUNS.items():
        group_data = []
        reference_paths = []
        conditions = []
        for label, expected, path in zip(STRATEGIES, ("baseline", "inverse_dynamics"), paths):
            metric_path = path.with_name(f"{path.stem}_metrics.json")
            meta = evidence.json(metric_path)
            check(meta["status"] == "COMPLETED" and meta["control_strategy"] == expected, str(metric_path))
            reference_path = local_artifact(meta["reference_csv"])
            reference_name = evidence.record(reference_path)
            reference_paths.append(reference_path)
            conditions.append([meta[key] for key in ("configured_control_rate_hz", "feedback_timeout_s",
                                                      "trajectory_duration_s", "maximum_initial_joint_error_rad")])
            data = joint_data(evidence, path, meta)
            ref = JointTrajectoryCSV.load(reference_path, JOINT_NAMES)
            t = data["trajectory_time_s"]
            check(np.all(np.diff(t) >= 0) and t.min() >= 0 and t.max() <= ref.end_time_s + 1e-9, str(path))
            evaluated = [ref.evaluate(value) for value in t]
            close(data["q"], [point.position_rad for point in evaluated], f"{path}: reference position mismatch")
            close(data["qd"], [point.velocity_rad_s for point in evaluated], f"{path}: reference velocity mismatch")
            source = evidence.record(path)
            stats = joint_stats(data)
            check(stats["clipped_samples"] == meta["command_saturation_count"], f"{path}: clipping mismatch")
            hold = np.array(["hold" in phase for phase in data["phase"]])
            stats.update(final_hold_samples=int(hold.sum()),
                         final_hold_position_rmse_rad=float(rmse(data["error"][hold])) if hold.any() else None)
            tables["controller_runs"].append(dict(
                experiment=group, controller=label, **stats, nominal_duration_s=ref.end_time_s,
                logged_span_s=float(np.ptp(data["time"])), configured_rate_hz=meta["configured_control_rate_hz"],
                observed_rate_hz=meta["observed_controller_rate_hz"],
                maximum_feedback_age_s=meta["maximum_feedback_age_observed_s"],
                source_csv=source, metrics_json=evidence.record(metric_path), reference_csv=reference_name,
            ))
            tables["joint_metrics"].extend(joint_rows(data, source, experiment=group, condition=label))
            group_data.append(data)
        check(reference_paths[0] == reference_paths[1], f"{group}: distinct reference files")
        close(*conditions, f"{group}: recorded run settings differ", atol=1e-8)
        for field in ("actual", "actual_d"):
            close(group_data[0][field][0], group_data[1][field][0],
                  f"{group}: recorded initial feedback differs", atol=1e-8)
        groups[group] = group_data
    return groups


def cartesian(evidence, tables, joint_group):
    runs = []
    for label, path, joint, joint_path in zip(STRATEGIES, CARTESIAN, joint_group, list(RUNS.values())[1]):
        meta_path = path.with_name(f"{path.stem}_metrics.json")
        meta = evidence.json(meta_path)
        check(local_artifact(meta["tracking_csv"]) == joint_path, f"{path}: wrong feedback provenance")
        reference = local_artifact(meta["task_space_reference_csv"])
        check(reference == RESULTS / "Task_Space/task_space_reference.csv", f"{path}: wrong task reference")
        check(meta["coordinate_frame"] == "world" and meta["end_effector_frame"] == "attachment_site",
              f"{path}: wrong Cartesian frame")
        columns = {key: f"{key}_error_{unit}" for key, (unit, _) in CART_METRICS.items()}
        angular_fields = [f"w{axis}_{kind}_rad_s" for kind in ("desired", "actual", "error") for axis in "xyz"]
        position_fields = [f"{axis}_{kind}_m" for kind in ("desired", "actual") for axis in "xyz"]
        data = evidence.csv(path, ["wall_time_s", "trajectory_time_s", "phase", *columns.values(),
                                   *angular_fields, *position_fields])
        check(len(data) == meta["sample_count"], f"{path}: sample count mismatch")
        t = numbers(data, ["trajectory_time_s"], path)[:, 0]
        close(t, joint["trajectory_time_s"], f"{path}: reference time mismatch")
        close(numbers(data, ["wall_time_s"], path)[:, 0], joint["time"], f"{path}: wall time mismatch")
        phase = np.array([row["phase"] for row in data])
        check(np.array_equal(phase, joint["phase"]), f"{path}: phase mismatch")
        for field, suffix, values in (("q_actual", "rad", joint["actual"]),
                                       ("qdot_actual", "rad_s", joint["actual_d"])):
            close(numbers(data, [f"{field}_{name}_{suffix}" for name in JOINT_NAMES], path),
                  values, f"{path}: feedback mismatch")
        errors = {key: numbers(data, [column], path)[:, 0] for key, column in columns.items()}
        desired_w, actual_w, angular = np.hsplit(numbers(data, angular_fields, path), 3)
        desired_p, actual_p = np.hsplit(numbers(data, position_fields, path), 2)
        close(desired_w - actual_w, angular, f"{path}: angular velocity residual mismatch")
        close(np.linalg.norm(angular, axis=1), errors["angular_velocity"], f"{path}: angular norm mismatch")
        close(np.linalg.norm(desired_p - actual_p, axis=1), errors["position"], f"{path}: position norm mismatch")
        movement = meta["reference_model_validation"]["movement_duration_s"]
        for window, mask in {
            "overall": np.ones(len(t), dtype=bool), "movement": t < movement, "hold": t >= movement,
            "first_0.25_s": t < 0.25, "movement_after_0.25_s": (t >= 0.25) & (t < movement),
        }.items():
            for key, (unit, _) in CART_METRICS.items():
                check(meta["units"][key] == unit.replace("_", "/"), f"{path}: wrong {key} units")
                values = errors[key][mask]
                check(len(values) and np.all(values >= 0), f"{path}: invalid {key} norm/window")
                if window in ("overall", "movement", "hold"):
                    close(rmse(values), meta[window][key][f"rmse_norm_{unit}"], f"{path}: Cartesian RMSE mismatch")
                peak = np.flatnonzero(mask)[np.argmax(values)]
                record = dict(
                    controller=label, window=window, quantity=key, unit=meta["units"][key],
                    samples=len(values), rmse=float(rmse(values)), peak=float(values.max()),
                    peak_trajectory_time_s=float(t[peak]),
                    angular_mean_square_fraction=float(np.square(values).sum() / np.square(errors[key]).sum())
                    if key == "angular_velocity" else None,
                    source_csv=evidence.record(path), metrics_json=evidence.record(meta_path),
                    reference_csv=evidence.record(reference), feedback_csv=evidence.record(joint_path),
                )
                if key == "angular_velocity":
                    for i, axis in enumerate("xyz"):
                        record[f"angular_peak_desired_w{axis}_rad_s"] = float(desired_w[peak, i])
                        record[f"angular_peak_actual_w{axis}_rad_s"] = float(actual_w[peak, i])
                tables["cartesian_metrics"].append(record)
        runs.append((t, errors, desired_p, actual_p, movement))
    return runs


def handovers(evidence, tables):
    successful = {}
    for run in EXECUTIONS:
        path = HANDOVER / "Execution" / f"surgical_handover_execution_{run}.csv"
        meta_path = path.with_name(f"{path.stem}_metrics.json")
        meta = evidence.json(meta_path)
        instrument, run_id = run.split("_", 1)
        check((meta["instrument"], meta["run_id"]) == (instrument, run_id), f"{path}: run mismatch")
        check(meta["status"] in ("COMPLETED", "ABORTED"), f"{path}: unknown status")
        check(local_artifact(meta["csv_report"]) == path, f"{path}: report reference mismatch")
        plan_path = local_artifact(meta["plan_json"])
        preflight_path = local_artifact(meta["collision_preflight_json"])
        plan, report = evidence.json(plan_path), evidence.json(preflight_path)
        check(plan["instrument"] == report["instrument"] == instrument, f"{path}: instrument mismatch")
        check(local_artifact(report["plan_json"]) == plan_path, f"{path}: preflight plan mismatch")
        check(meta["collision_preflight_status"] == report["status"], f"{path}: preflight status mismatch")
        check(report["collision_distance_backend"]["native_ccd_enabled"] is True, f"{path}: CCD disabled")
        waypoints = {entry["name"]: entry for entry in plan["waypoints"]}
        check(tuple(waypoints) == tuple(REQUIRED_WAYPOINTS) and len(plan["waypoints"]) == len(waypoints)
              and all(entry["ik_success"] is True for entry in waypoints.values()), f"{path}: invalid waypoint plan")
        durations = np.array([entry["duration_from_previous_s"] for entry in waypoints.values()], dtype=float)
        check(np.isfinite(durations).all() and np.all(durations > 0), f"{path}: invalid durations")
        close(durations.sum(), report["trajectory"]["duration_s"], f"{path}: plan duration mismatch")
        data = joint_data(evidence, path, meta, handover=True)
        finished = datetime.fromisoformat(meta["finished_at_utc"])
        started = datetime.fromisoformat(meta["started_at_utc"]) if meta["started_at_utc"] else None
        wall_duration = (finished - started).total_seconds() if started is not None else None
        done = meta["status"] == "COMPLETED"
        source = evidence.record(path)
        plan_source, report_source = evidence.record(plan_path), evidence.record(preflight_path)
        phases = (list(dict.fromkeys(data["phase"])) if data is not None else [])
        legal_phases = [phase for name in waypoints for phase in (name, f"{name}_settle")]
        check(phases == legal_phases[:len(phases)], f"{path}: unexpected phase sequence")
        if done:
            check(phases == legal_phases and data is not None, f"{path}: missing completed phase")
            check(np.any(data["instrument_attached"]) and data["instrument_released"][-1] == 1
                  and meta["instrument_released"] is True and meta["instrument_attached_at_end"] is False,
                  f"{path}: incomplete attachment/release")
            check(wall_duration >= data["time"][-1], f"{path}: wall time before last recorded sample")
        row = dict(
            instrument=instrument, run_id=run_id, status=meta["status"], reason=meta["reason"],
            waypoints_completed=len(waypoints) if done else None,
            waypoints_with_recorded_settle_rows=sum(phase.endswith("_settle") for phase in phases),
            nominal_duration_s=float(durations.sum()), recorded_wall_duration_s=wall_duration,
            first_logged_elapsed_s=float(data["time"][0]) if data is not None else None,
            last_logged_elapsed_s=float(data["time"][-1]) if data is not None else None,
            strategy=meta["control_strategy"], configured_rate_hz=meta["configured_control_rate_hz"],
            maximum_feedback_age_s=meta["maximum_feedback_age_observed_s"],
            attachment_confirmed=bool(meta["instrument_attached_at_end"] or
                                      (data is not None and np.any(data["instrument_attached"]))),
            attached_at_end=meta["instrument_attached_at_end"], released=meta["instrument_released"],
            release_stability_check_passed=True if done else None,
            online_monitor_outcome="completed without abort" if done else "execution aborted",
            preflight_status=report["status"], warning_approved=meta["warning_review_approved"],
            live_pickup_position_error_m=None, live_handover_position_error_m=None,
            measured_release_drift_m=None, online_support_transition_clearance_m=None,
            source_csv=source, metrics_json=evidence.record(meta_path), plan_json=plan_source,
            preflight_json=report_source,
        )
        if data is not None:
            row.update(joint_stats(data))
            close(row["position_peak_rad"], meta["overall"]["max_abs_error_rad"], f"{path}: peak mismatch")
        if done:
            actual_grasp = validate_transform(np.asarray(meta["site_from_instrument_actual"]), "actual grasp")
            planned_grasp = validate_transform(np.asarray(plan["tool_from_instrument"]), "planned grasp")
            grasp_p, grasp_r = transform_error(actual_grasp, planned_grasp)
            calibration = validate_transform(np.asarray(plan["world_from_dh"]), "world from DH")
            pickup_i = np.flatnonzero(data["phase"] == "pickup_settle")[-1]
            handover_i = np.flatnonzero(data["phase"] == "handover_settle")[-1]
            pickup_tool = calibration @ forward_kinematics(data["actual"][pickup_i])
            pickup_target = np.asarray(waypoints["pickup"]["position_world_m"], dtype=float)
            body = calibration @ forward_kinematics(data["actual"][handover_i]) @ actual_grasp
            body_p, body_r = transform_error(body, np.asarray(plan["desired_instrument_handover_pose_world"]))
            home_error = data["actual"][-1] - np.asarray(plan["initial_joint_positions_rad"])
            row.update(
                grasp_translation_error_m=grasp_p, grasp_orientation_error_rad=grasp_r,
                fk_pickup_tool_to_planned_site_m=float(np.linalg.norm(pickup_tool[:3, 3] - pickup_target)),
                fk_handover_body_position_error_m=body_p, fk_handover_body_orientation_error_rad=body_r,
                home_max_abs_joint_error_rad=float(np.abs(home_error).max()), home_joint_rmse_rad=float(rmse(home_error)),
            )
            successful[instrument] = data
        tables["handover_runs"].append(row)
        thresholds = report["clearance_thresholds"]
        for category in DISTANCES:
            entry = report["minimum_clearances"][category]
            offline_m = entry["minimum_distance_m"]
            close(offline_m, meta["minimum_clearances_m"][category], f"{path}: preflight clearance mismatch")
            online_m = meta.get("online_measured_state_minimum_clearances_m", {}).get(category)
            lower_bound = online_m is not None and online_m >= thresholds["distance_cap_m"]
            tables["collision_clearances"].append(dict(
                instrument=instrument, run_id=run_id, execution_status=meta["status"], category=category,
                preflight_status=report["status"], offline_minimum_m=offline_m,
                online_minimum_m=online_m if data is not None and not lower_bound else None,
                online_cap_or_unupdated_m=online_m if data is not None and lower_bound else None,
                hard_minimum_m=thresholds["minimum_clearance_m"], warning_threshold_m=thresholds["warning_clearance_m"],
                offline_collision_events=report["collision_event_count"],
                offline_joint_limit_violations=report["joint_limit_violation_count"],
                offline_support_clear_time_s=report["pickup_support_contact"]["clear_time_s"],
                offline_support_cleared=report["pickup_support_contact"]["cleared"],
                warning_approved=meta["warning_review_approved"], preflight_json=report_source,
                execution_metrics_json=evidence.record(meta_path),
            ))
        if data is None:
            continue
        tables["joint_metrics"].extend(joint_rows(data, source, experiment="Surgical handover", condition=run_id))
        for name, waypoint in waypoints.items():
            motion = data["phase"] == name
            settle = data["phase"] == f"{name}_settle"
            if not motion.any():
                continue
            last_settle = np.flatnonzero(settle)[-1] if settle.any() else None
            end_motion = np.flatnonzero(motion)[-1]
            close(data["q"][end_motion], waypoint["joint_positions_rad"], f"{path}: {name} endpoint mismatch")
            tables["handover_stages"].append(dict(
                instrument=instrument, run_id=run_id, status=meta["status"], waypoint=name,
                planned_motion_s=waypoint["duration_from_previous_s"], motion_samples=int(motion.sum()),
                settle_samples=int(settle.sum()), motion_logged_span_s=float(np.ptp(data["time"][motion])),
                settle_logged_span_s=float(np.ptp(data["time"][settle])) if settle.any() else None,
                first_motion_elapsed_s=float(data["time"][motion][0]),
                last_settle_elapsed_s=float(data["time"][last_settle]) if settle.any() else None,
                motion_position_rmse_rad=float(rmse(data["error"][motion])),
                motion_position_peak_rad=float(np.abs(data["error"][motion]).max()),
                motion_velocity_rmse_rad_s=float(rmse(data["velocity_error"][motion])),
                end_motion_max_abs_joint_error_rad=float(np.abs(data["error"][end_motion]).max()),
                settle_position_rmse_rad=float(rmse(data["error"][settle])) if settle.any() else None,
                final_settle_max_abs_joint_error_rad=float(np.abs(data["error"][last_settle]).max()) if settle.any() else None,
                final_settle_max_abs_joint_speed_rad_s=float(np.abs(data["actual_d"][last_settle]).max()) if settle.any() else None,
                source_csv=source, plan_json=plan_source,
            ))
    return successful


def save_figure(fig, directory, name):
    fig.savefig(directory / f"{name}.png", dpi=220, facecolor="white")
    plt.close(fig)


def joint_figure(data, instrument, directory):
    fig, axes = plt.subplots(6, 3, figsize=(13, 13), sharex=True, layout="constrained")
    for i, label in enumerate(LABELS):
        for ax, desired, actual in (
            (axes[i, 0], data["q"][:, i], data["actual"][:, i]),
            (axes[i, 2], data["qd"][:, i], data["actual_d"][:, i]),
        ):
            ax.plot(data["time"], actual, color=COLORS[0], label="Actual", linewidth=1.5)
            ax.plot(data["time"], desired, "--", color=COLORS[1], label="Desired", linewidth=1.1)
        axes[i, 1].plot(data["time"], 1000 * data["error"][:, i], color=COLORS[0], linewidth=1.1)
        axes[i, 0].set_ylabel(label)
    for ax, title in zip(axes[0], ("Joint position (rad)", "Desired − actual (mrad)", "Joint velocity (rad/s)")):
        ax.set_title(title)
    for ax in axes[-1]:
        ax.set_xlabel("Execution elapsed time (s)")
    lines, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(lines, labels, loc="outside upper right", ncols=2)
    fig.suptitle(f"{instrument.capitalize()} joint tracking", fontsize=16)
    save_figure(fig, directory, f"{instrument}_joint_tracking")


def figures(directory, tables, groups, task, successful):
    directory.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "axes.axisbelow": True, "grid.alpha": 0.24,
                         "font.size": 10, "figure.dpi": 120})
    for instrument, data in successful.items():
        joint_figure(data, instrument, directory)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), sharey=True, layout="constrained")
    x = np.arange(len(JOINT_NAMES))
    bars = [(group, list(zip(STRATEGIES, data))) for group, data in groups.items()]
    bars.append(("Handover runs", [(key.capitalize(), value) for key, value in successful.items()]))
    for ax, (title, datasets) in zip(axes, bars):
        for index, (label, data) in enumerate(datasets):
            ax.bar(x + (index - 0.5) * .36, 1000 * rmse(data["error"], axis=0),
                   width=.36, color=COLORS[index], label=label)
        ax.set(title=title, xticks=x, xticklabels=("J1", "J2", "J3", "J4", "J5", "J6"), xlabel="UR5e joint")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Joint position RMSE (mrad)")
    save_figure(fig, directory, "per_joint_rmse")

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True, layout="constrained")
    for ax, (quantity, (unit, title)) in zip(axes.flat, CART_METRICS.items()):
        for color, label, (t, errors, _, _, _) in zip(COLORS, STRATEGIES, task):
            ax.plot(t, 1000 * errors[quantity], color=color, label=label, linewidth=1.3)
        ax.axvline(task[0][-1], color="0.45", linestyle=":", linewidth=1)
        ax.set(title=title, ylabel=f"Error norm ({'mrad' if unit.startswith('rad') else 'mm'}"
                                   f"{'/s' if unit.endswith('_s') else ''})")
    axes[0, 0].legend(fontsize=8)
    for ax in axes[-1]:
        ax.set_xlabel("Reference time (s)")
    save_figure(fig, directory, "cartesian_tracking_errors")

    fig, axes = plt.subplots(3, 2, figsize=(11, 7), sharex=True, sharey="row", layout="constrained")
    for col, (label, (t, _, desired, actual, movement)) in enumerate(zip(STRATEGIES, task)):
        axes[0, col].set_title(label)
        for row, axis in enumerate("XYZ"):
            ax = axes[row, col]
            ax.plot(t, actual[:, row], color=COLORS[0], label="Actual (FK)")
            ax.plot(t, desired[:, row], "--", color=COLORS[1], label="Desired")
            ax.axvline(movement, color="0.45", linestyle=":", linewidth=1)
            if col == 0:
                ax.set_ylabel(f"World {axis} (m)")
        axes[-1, col].set_xlabel("Reference time (s)")
    axes[0, 1].legend(fontsize=8)
    save_figure(fig, directory, "cartesian_position_tracking")

    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True, layout="constrained")
    for ax, instrument in zip(axes, successful):
        rows = [row for row in tables["handover_stages"]
                if row["instrument"] == instrument and row["status"] == "COMPLETED"]
        y = np.arange(len(rows))
        ax.barh(y - .17, [row["planned_motion_s"] for row in rows], height=.26,
                color=COLORS[0], label="Planned motion")
        ax.barh(y + .17, [row["motion_logged_span_s"] for row in rows], height=.26,
                color=COLORS[1], label="Logged motion span")
        ax.barh(y + .17, [row["settle_logged_span_s"] for row in rows], height=.26,
                left=[row["motion_logged_span_s"] for row in rows], color="#847F75", label="Logged settle span")
        ax.set(title=instrument.capitalize(), xlabel="Duration (s)", yticks=y,
               yticklabels=[row["waypoint"].replace("_", " ").capitalize() for row in rows])
    axes[0].invert_yaxis()
    axes[0].legend(fontsize=8)
    save_figure(fig, directory, "handover_stage_durations")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True, layout="constrained")
    for index, (ax, instrument) in enumerate(zip(axes, successful)):
        rows = [row for row in tables["collision_clearances"]
                if row["instrument"] == instrument and row["execution_status"] == "COMPLETED"]
        x = np.arange(len(DISTANCES))
        for offset, field, label, color in (
            (-.17, "offline_minimum_m", "Offline preflight", COLORS[0]),
            (.17, "online_minimum_m", "Online monitor", COLORS[1]),
        ):
            ax.bar(x + offset, [1000 * row[field] for row in rows], width=.32, label=label, color=color)
        for field, style, label in (("hard_minimum_m", "--", "Hard minimum"),
                                    ("warning_threshold_m", ":", "Review threshold")):
            ax.axhline(1000 * rows[0][field], linestyle=style, color=".2", label=label)
        ax.set(title=instrument.capitalize(), xticks=x,
               xticklabels=("Robot–scene", "Robot self", "Instrument–scene"), ylim=(0, 56))
        if index == 1:
            ax.legend(fontsize=8, ncols=2, loc="lower center")
    axes[0].set_ylabel("Minimum signed clearance (mm)")
    save_figure(fig, directory, "collision_clearances")


def write_outputs(output, tables, evidence):
    for name, rows in tables.items():
        columns = list(dict.fromkeys(key for row in rows for key in row))
        with (output / f"{name}.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, columns)
            writer.writeheader()
            writer.writerows(rows)
    payload = {
        "methods": {
            "joint_rmse": "Square root of the unweighted mean squared error across samples and joints.",
            "cartesian_rmse": "Square root of the unweighted mean of squared vector error norms.",
            "logged_span_s": "Timestamp of last sample minus timestamp of first sample in the selected window.",
            "missing": "JSON null or omitted field, CSV blank. Includes terminal-only quantities absent from archived artifacts.",
            "online_cap_or_unupdated_m": "A capped value can be an initialized accumulator. No minimum is inferred from it.",
        },
        "tables": tables,
        "source_sha256": dict(sorted(evidence.sources.items())),
    }
    (output / "summary.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT, help="Dedicated generated-artifact directory")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    protected = (TRACKING, HANDOVER, RESULTS / "Task_Space", RESULTS / "Joint_Space",
                 RESULTS / "Collision_Preflight", RESULTS / "Validation")
    if (ROOT.is_relative_to(output) or output in (RESULTS, MILESTONE)
            or any(output.is_relative_to(path) for path in protected)):
        parser.error("Use a dedicated evaluation output directory outside the raw experiment directories")
    evidence = Evidence()
    tables = {name: [] for name in ("controller_runs", "joint_metrics", "cartesian_metrics", "handover_runs",
                                     "handover_stages", "collision_clearances")}
    groups = tracking(evidence, tables)
    task = cartesian(evidence, tables, groups["Task-space-derived reference"])
    successful = handovers(evidence, tables)
    output.mkdir(parents=True, exist_ok=True)
    figures(output / "figures", tables, groups, task, successful)
    write_outputs(output, tables, evidence)
    print(f"Wrote {len(tables)} tables, summary.json and figures to {output}")


if __name__ == "__main__":
    main()
