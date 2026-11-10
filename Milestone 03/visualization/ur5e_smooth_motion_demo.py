"""MCTR911 Milestone 03: smooth UR5e start-accelerate-decelerate-stop demo.

The robot is driven by the EXISTING MuJoCo joint-position actuators. A quintic
joint-space reference moves the arm from HOME to a nearby, conservative target:

    s(t) = 10*tau^3 - 15*tau^4 + 6*tau^5,  tau = t / T
    q_ref = q_start + (q_goal - q_start) * s(t)

The analytical derivatives s_dot and s_ddot are used by a computed-torque
feedforward term for the existing position actuators. The actuators remain
position servos; no velocity or torque actuator is added. Their built-in
proportional/derivative behavior is retained, while inverse-dynamics
feedforward compensates gravity, velocity-dependent bias forces, passive
forces, and the desired reference acceleration.

During physics stepping, the script reads actual q, qdot and instantaneous qacc
from MuJoCo, and separately records qacc immediately after mj_step.  After the
run it differentiates the sampled qvel trace to estimate trajectory acceleration.
Both acceleration definitions are retained and compared with the quintic reference.

All Cartesian quantities refer to attachment_site and are expressed in world
coordinates. This is a simulation/kinematics validation, not a hardware test.

Run from the repository root:
  python "Milestone 03/visualization/ur5e_smooth_motion_demo.py" --headless
  python "Milestone 03/visualization/ur5e_smooth_motion_demo.py" --camera overview
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
KIN_DIR = ROOT / "Milestone 03" / "kinematics"
FK_DIR = ROOT / "Milestone 02" / "kinematics"
for folder in (KIN_DIR, FK_DIR):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

from ur5e_dh_fk import forward_kinematics  # noqa: E402
from ur5e_jacobian import JOINT_NAMES  # noqa: E402
from ur5e_velocity_kinematics import forward_velocity  # noqa: E402
from ur5e_acceleration_kinematics import forward_acceleration  # noqa: E402

SITE_NAME = "attachment_site"
HOME = np.array([-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0.0], dtype=float)
# A conservative, coordinated 6-joint offset. It moves the tool about 0.19 m
# from the home pose toward the handover area, without changing the robot model.
JOINT_DELTA = np.array([-0.30, 0.16, -0.18, 0.12, -0.10, -0.20], dtype=float)

POSITION_TOL_M = 1e-5
VELOCITY_TOL = 1e-4
ACCELERATION_TOL = 1e-3
JDOT_PROBE_S = 1e-4


def quintic_reference(
    t: float, move_duration: float, q_start: np.ndarray, q_delta: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Return q_ref, qdot_ref, qddot_ref and scalar path parameter s(t).

    The quintic time scaling has zero velocity and acceleration at both ends.
    After move_duration, the reference remains at q_goal with zero derivatives.
    """
    if t <= 0.0:
        tau = 0.0
    else:
        tau = min(t / move_duration, 1.0)

    s = 10.0 * tau**3 - 15.0 * tau**4 + 6.0 * tau**5
    ds_dtau = 30.0 * tau**2 - 60.0 * tau**3 + 30.0 * tau**4
    d2s_dtau2 = 60.0 * tau - 180.0 * tau**2 + 120.0 * tau**3

    if t >= move_duration:
        s = 1.0
        s_dot = 0.0
        s_ddot = 0.0
    else:
        s_dot = ds_dtau / move_duration
        s_ddot = d2s_dtau2 / (move_duration**2)

    q_ref = q_start + q_delta * s
    qdot_ref = q_delta * s_dot
    qddot_ref = q_delta * s_ddot
    return q_ref, qdot_ref, qddot_ref, float(s)


def robot_indices(model: mujoco.MjModel) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Resolve each named hinge joint, its qpos/dof indices and its actuator."""
    qpos_idx: list[int] = []
    dof_idx: list[int] = []
    actuator_idx: list[int] = []
    joint_ids: list[int] = []

    for name in JOINT_NAMES:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise RuntimeError(f"Scene is missing required UR5e joint: {name}")
        if int(model.jnt_type[jid]) != int(mujoco.mjtJoint.mjJNT_HINGE):
            raise RuntimeError(f"Expected a hinge joint for {name}")

        matches = [
            i for i in range(model.nu)
            if int(model.actuator_trnid[i, 0]) == jid
            and int(model.actuator_trntype[i]) == int(mujoco.mjtTrn.mjTRN_JOINT)
        ]
        if len(matches) != 1:
            raise RuntimeError(f"Expected exactly one joint actuator for {name}; found {len(matches)}")

        joint_ids.append(jid)
        qpos_idx.append(int(model.jnt_qposadr[jid]))
        dof_idx.append(int(model.jnt_dofadr[jid]))
        actuator_idx.append(matches[0])

    return (
        np.asarray(qpos_idx, dtype=int),
        np.asarray(dof_idx, dtype=int),
        np.asarray(actuator_idx, dtype=int),
        np.asarray(joint_ids, dtype=int),
    )


def actuator_pd_parameters(
    model: mujoco.MjModel, actuator_ids: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Read and validate the UR5e position actuators' built-in PD parameters.

    For the supplied model, each actuator implements:
        tau = Kp * (ctrl - q) - Kd * qdot
    The controller below adds feedforward through a small position-command
    offset. Unit gear ratios are required so that ctrl and q are both radians.
    """
    gain = np.asarray(model.actuator_gainprm[actuator_ids, 0], dtype=float)
    bias = np.asarray(model.actuator_biasprm[actuator_ids, :], dtype=float)
    kp = -bias[:, 1]
    kd = -bias[:, 2]
    gear = np.asarray(model.actuator_gear[actuator_ids, 0], dtype=float)

    if np.any(gain <= 0.0) or np.any(kp <= 0.0) or np.any(kd < 0.0):
        raise RuntimeError("Position actuators must have positive Kp and non-negative Kd")
    if not np.allclose(gain, kp, rtol=1e-8, atol=1e-10):
        raise RuntimeError(
            "Unsupported actuator configuration: fixed gain must match -biasprm[1]"
        )
    if not np.allclose(gear, 1.0, rtol=0.0, atol=1e-10):
        raise RuntimeError(
            "This controller expects unit gear ratios for the six UR5e joint actuators"
        )
    return kp, kd


def compute_position_control(
    model: mujoco.MjModel,
    actual_data: mujoco.MjData,
    reference_data: mujoco.MjData,
    qpos_idx: np.ndarray,
    dof_idx: np.ndarray,
    actuator_idx: np.ndarray,
    q_ref: np.ndarray,
    qdot_ref: np.ndarray,
    qddot_ref: np.ndarray,
    kp: np.ndarray,
    kd: np.ndarray,
    use_dynamics_feedforward: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build clipped position commands with inverse-dynamics feedforward.

    MuJoCo's affine position actuators produce:
        tau_act = Kp * (ctrl - q) - Kd * qdot

    The desired generalized actuator torque at the reference state is:
        tau_required = M(q_ref) qddot_ref + qfrc_bias - qfrc_passive

    Since the actuator itself contributes -Kd*qdot, the feedforward offset
    also includes +Kd*qdot_ref. At perfect tracking, the actuator then supplies
    tau_required. The remaining feedback term is Kp*(q_ref-q) - Kd*(qdot-qdot_ref).
    """
    if use_dynamics_feedforward:
        reference_data.qpos[:] = actual_data.qpos
        reference_data.qvel[:] = actual_data.qvel
        reference_data.qpos[qpos_idx] = q_ref
        reference_data.qvel[dof_idx] = qdot_ref
        mujoco.mj_forward(model, reference_data)

        mass_matrix = np.zeros((model.nv, model.nv), dtype=float)
        mujoco.mj_fullM(model, mass_matrix, reference_data.qM)

        desired_qacc = np.zeros(model.nv, dtype=float)
        desired_qacc[dof_idx] = qddot_ref
        required_generalized_force = (
            mass_matrix @ desired_qacc
            + reference_data.qfrc_bias
            - reference_data.qfrc_passive
        )
        required_torque = required_generalized_force[dof_idx].copy()
        actuator_feedforward_torque = required_torque + kd * qdot_ref
    else:
        required_torque = np.zeros(6, dtype=float)
        actuator_feedforward_torque = np.zeros(6, dtype=float)

    raw_ctrl = q_ref + actuator_feedforward_torque / kp
    ctrl = raw_ctrl.copy()
    clipped = np.zeros(6, dtype=bool)

    for i, aid in enumerate(actuator_idx):
        if bool(model.actuator_ctrllimited[aid]):
            lower, upper = model.actuator_ctrlrange[aid]
            ctrl[i] = np.clip(ctrl[i], lower, upper)
            clipped[i] = not np.isclose(ctrl[i], raw_ctrl[i], rtol=0.0, atol=1e-12)

    return ctrl, required_torque, actuator_feedforward_torque, raw_ctrl, clipped


def validate_reference_limits(
    model: mujoco.MjModel,
    joint_ids: np.ndarray,
    actuator_ids: np.ndarray,
    q_start: np.ndarray,
    q_goal: np.ndarray,
) -> None:
    """Fail early if either endpoint exceeds a model or actuator limit."""
    for i, (jid, aid, name) in enumerate(zip(joint_ids, actuator_ids, JOINT_NAMES)):
        for label, q_value in (("start", q_start[i]), ("goal", q_goal[i])):
            if bool(model.jnt_limited[jid]):
                lower, upper = model.jnt_range[jid]
                if not (lower <= q_value <= upper):
                    raise ValueError(
                        f"{name} {label} angle {q_value:.4f} rad exceeds joint range "
                        f"[{lower:.4f}, {upper:.4f}] rad"
                    )
            if bool(model.actuator_ctrllimited[aid]):
                lower, upper = model.actuator_ctrlrange[aid]
                if not (lower <= q_value <= upper):
                    raise ValueError(
                        f"{name} {label} target {q_value:.4f} rad exceeds actuator range "
                        f"[{lower:.4f}, {upper:.4f}] rad"
                    )


def site_jacobian(
    model: mujoco.MjModel, data: mujoco.MjData, site_id: int, dof_idx: np.ndarray
) -> np.ndarray:
    jacp = np.zeros((3, model.nv), dtype=float)
    jacr = np.zeros((3, model.nv), dtype=float)
    mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
    return np.vstack((jacp[:, dof_idx], jacr[:, dof_idx]))


def mujoco_jdot(
    model: mujoco.MjModel,
    probe: mujoco.MjData,
    original: mujoco.MjData,
    site_id: int,
    qpos_idx: np.ndarray,
    dof_idx: np.ndarray,
    qdot: np.ndarray,
) -> np.ndarray:
    """Estimate dJ/dt using central differences in a separate MjData object."""
    probe.qpos[:] = original.qpos
    probe.qvel[:] = original.qvel

    probe.qpos[qpos_idx] = original.qpos[qpos_idx] + JDOT_PROBE_S * qdot
    mujoco.mj_forward(model, probe)
    jac_plus = site_jacobian(model, probe, site_id, dof_idx)

    probe.qpos[qpos_idx] = original.qpos[qpos_idx] - JDOT_PROBE_S * qdot
    mujoco.mj_forward(model, probe)
    jac_minus = site_jacobian(model, probe, site_id, dof_idx)

    return (jac_plus - jac_minus) / (2.0 * JDOT_PROBE_S)


def evaluate_sample(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    probe: mujoco.MjData,
    site_id: int,
    qpos_idx: np.ndarray,
    dof_idx: np.ndarray,
    q_ref: np.ndarray,
    qdot_ref: np.ndarray,
    qddot_ref: np.ndarray,
    path_s: float,
    move_duration: float,
) -> dict[str, Any]:
    """Compare Python kinematics with MuJoCo at the same actual state."""
    q = data.qpos[qpos_idx].copy()
    qdot = data.qvel[dof_idx].copy()
    qddot = data.qacc[dof_idx].copy()

    position_python = forward_kinematics(q)[:3, 3]
    position_mujoco = data.site_xpos[site_id].copy()

    twist_python = forward_velocity(q, qdot)
    acceleration_python = forward_acceleration(q, qdot, qddot)

    jac_mujoco = site_jacobian(model, data, site_id, dof_idx)
    twist_mujoco = jac_mujoco @ qdot
    jdot_mujoco = mujoco_jdot(model, probe, data, site_id, qpos_idx, dof_idx, qdot)
    acceleration_mujoco = jac_mujoco @ qddot + jdot_mujoco @ qdot

    twist_reference = forward_velocity(q_ref, qdot_ref)
    acceleration_reference = forward_acceleration(q_ref, qdot_ref, qddot_ref)

    t = float(data.time)
    if t < 0.5 * move_duration:
        phase = "accelerating"
    elif t < move_duration:
        phase = "decelerating"
    else:
        phase = "hold"

    row: dict[str, Any] = {
        "time_s": t,
        "phase": phase,
        "path_parameter_s": path_s,
        "move_duration_s": move_duration,
    }

    for i in range(6):
        row[f"target_q{i+1}_rad"] = float(q_ref[i])
        row[f"actual_q{i+1}_rad"] = float(q[i])
        row[f"target_qdot{i+1}_rad_s"] = float(qdot_ref[i])
        row[f"actual_qdot{i+1}_rad_s"] = float(qdot[i])
        row[f"target_qddot{i+1}_rad_s2"] = float(qddot_ref[i])
        row[f"actual_qddot{i+1}_rad_s2"] = float(qddot[i])

    for i, axis in enumerate("xyz"):
        row[f"python_position_{axis}_m"] = float(position_python[i])
        row[f"mujoco_position_{axis}_m"] = float(position_mujoco[i])

    twist_names = ("vx", "vy", "vz", "wx", "wy", "wz")
    accel_names = ("ax", "ay", "az", "alphax", "alphay", "alphaz")
    for i, name in enumerate(twist_names):
        row[f"python_{name}"] = float(twist_python[i])
        row[f"mujoco_{name}"] = float(twist_mujoco[i])
        row[f"reference_{name}"] = float(twist_reference[i])
    for i, name in enumerate(accel_names):
        row[f"python_{name}"] = float(acceleration_python[i])
        row[f"mujoco_{name}"] = float(acceleration_mujoco[i])
        row[f"reference_{name}"] = float(acceleration_reference[i])

    row["position_error_m"] = float(np.linalg.norm(position_python - position_mujoco))
    row["linear_velocity_error_m_s"] = float(
        np.linalg.norm(twist_python[:3] - twist_mujoco[:3])
    )
    row["angular_velocity_error_rad_s"] = float(
        np.linalg.norm(twist_python[3:] - twist_mujoco[3:])
    )
    row["linear_acceleration_error_m_s2"] = float(
        np.linalg.norm(acceleration_python[:3] - acceleration_mujoco[:3])
    )
    row["angular_acceleration_error_rad_s2"] = float(
        np.linalg.norm(acceleration_python[3:] - acceleration_mujoco[3:])
    )
    row["joint_position_tracking_error_norm_rad"] = float(np.linalg.norm(q - q_ref))
    row["joint_velocity_tracking_error_norm_rad_s"] = float(np.linalg.norm(qdot - qdot_ref))
    row["joint_acceleration_tracking_error_norm_rad_s2"] = float(np.linalg.norm(qddot - qddot_ref))
    row["ee_linear_velocity_tracking_error_m_s"] = float(
        np.linalg.norm(twist_python[:3] - twist_reference[:3])
    )
    row["ee_angular_velocity_tracking_error_rad_s"] = float(
        np.linalg.norm(twist_python[3:] - twist_reference[3:])
    )
    row["ee_linear_acceleration_tracking_error_m_s2"] = float(
        np.linalg.norm(acceleration_python[:3] - acceleration_reference[:3])
    )
    row["ee_angular_acceleration_tracking_error_rad_s2"] = float(
        np.linalg.norm(acceleration_python[3:] - acceleration_reference[3:])
    )

    row["target_joint_speed_norm_rad_s"] = float(np.linalg.norm(qdot_ref))
    row["actual_joint_speed_norm_rad_s"] = float(np.linalg.norm(qdot))
    row["target_joint_acceleration_norm_rad_s2"] = float(np.linalg.norm(qddot_ref))
    row["actual_joint_acceleration_norm_rad_s2"] = float(np.linalg.norm(qddot))
    row["target_ee_linear_speed_m_s"] = float(np.linalg.norm(twist_reference[:3]))
    row["actual_ee_linear_speed_m_s"] = float(np.linalg.norm(twist_python[:3]))
    row["target_ee_angular_speed_rad_s"] = float(np.linalg.norm(twist_reference[3:]))
    row["actual_ee_angular_speed_rad_s"] = float(np.linalg.norm(twist_python[3:]))
    row["target_ee_linear_acceleration_m_s2"] = float(
        np.linalg.norm(acceleration_reference[:3])
    )
    row["actual_ee_linear_acceleration_m_s2"] = float(
        np.linalg.norm(acceleration_python[:3])
    )
    row["target_ee_angular_acceleration_rad_s2"] = float(
        np.linalg.norm(acceleration_reference[3:])
    )
    row["actual_ee_angular_acceleration_rad_s2"] = float(
        np.linalg.norm(acceleration_python[3:])
    )
    return row


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError("No simulation samples were recorded; CSV was not written")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def add_sampled_acceleration_diagnostics(rows: list[dict[str, Any]]) -> None:
    """Add acceleration estimates derived from the sampled MuJoCo qvel trace.

    The existing ``actual_qddot`` columns are the instantaneous forward-dynamics
    qacc values recomputed by mj_forward at each recorded state.  The new
    ``fd_actual_qddot`` columns are a numerical time derivative of the sampled
    actual qvel trace.  They are intentionally kept separate: with an implicit
    integrator, the acceleration reported by a forward-dynamics evaluation and
    the finite-difference acceleration of the integrated velocity trace need
    not be identical sample by sample.

    The function also derives an end-effector acceleration estimate from the
    finite-difference joint acceleration using the existing analytical
    Jacobian/Jdot implementation.  It does not modify the controller or model.
    """
    if len(rows) < 3:
        raise RuntimeError("At least three samples are required for acceleration differentiation")

    times = np.asarray([float(row["time_s"]) for row in rows], dtype=float)
    if not np.all(np.isfinite(times)) or np.any(np.diff(times) <= 0.0):
        raise RuntimeError("CSV sample times must be finite and strictly increasing")

    q = np.column_stack([
        np.asarray([float(row[f"actual_q{i}_rad"]) for row in rows], dtype=float)
        for i in range(1, 7)
    ])
    qdot = np.column_stack([
        np.asarray([float(row[f"actual_qdot{i}_rad_s"]) for row in rows], dtype=float)
        for i in range(1, 7)
    ])
    qddot_forward = np.column_stack([
        np.asarray([float(row[f"actual_qddot{i}_rad_s2"]) for row in rows], dtype=float)
        for i in range(1, 7)
    ])
    qddot_reference = np.column_stack([
        np.asarray([float(row[f"target_qddot{i}_rad_s2"]) for row in rows], dtype=float)
        for i in range(1, 7)
    ])

    # Nonuniform-time gradient is supported, although the recommended test
    # samples once per 0.002 s physics step.
    qddot_fd = np.gradient(qdot, times, axis=0, edge_order=2)
    interval_dt = np.diff(times)
    interval_qddot_fd = np.diff(qdot, axis=0) / interval_dt[:, None]

    accel_names = ("ax", "ay", "az", "alphax", "alphay", "alphaz")
    for k, row in enumerate(rows):
        for j in range(6):
            row[f"fd_actual_qddot{j+1}_rad_s2"] = float(qddot_fd[k, j])
            row[f"interval_fd_qddot{j+1}_rad_s2"] = (
                float(interval_qddot_fd[k-1, j]) if k > 0 else float("nan")
            )
            row[f"mj_step_qacc_j{j+1}_rad_s2"] = float(
                row.get(f"mj_step_qacc_j{j+1}_rad_s2", float("nan"))
            )

        row["fd_actual_joint_acceleration_norm_rad_s2"] = float(np.linalg.norm(qddot_fd[k]))
        row["fd_joint_acceleration_tracking_error_norm_rad_s2"] = float(
            np.linalg.norm(qddot_fd[k] - qddot_reference[k])
        )
        row["qacc_vs_fd_joint_acceleration_error_norm_rad_s2"] = float(
            np.linalg.norm(qddot_forward[k] - qddot_fd[k])
        )

        # For the interval diagnostic, compare the qacc copied immediately
        # after mj_step with the average qvel change over that same step.
        if k > 0:
            step_qacc = np.asarray([
                float(row.get(f"mj_step_qacc_j{j+1}_rad_s2", float("nan")))
                for j in range(6)
            ])
            if np.all(np.isfinite(step_qacc)):
                row["mj_step_qacc_vs_interval_fd_error_norm_rad_s2"] = float(
                    np.linalg.norm(step_qacc - interval_qddot_fd[k-1])
                )
            else:
                row["mj_step_qacc_vs_interval_fd_error_norm_rad_s2"] = float("nan")
        else:
            row["mj_step_qacc_vs_interval_fd_error_norm_rad_s2"] = float("nan")

        # Compute Cartesian acceleration using actual q/qdot and the sampled
        # finite-difference qddot, then compare with the reference Cartesian
        # acceleration already stored by evaluate_sample().
        fd_twist_dot = forward_acceleration(q[k], qdot[k], qddot_fd[k])
        reference_twist_dot = np.asarray(
            [float(row[f"reference_{name}"]) for name in accel_names], dtype=float
        )
        for j, name in enumerate(accel_names):
            row[f"fd_python_{name}"] = float(fd_twist_dot[j])

        row["fd_actual_ee_linear_acceleration_m_s2"] = float(
            np.linalg.norm(fd_twist_dot[:3])
        )
        row["fd_actual_ee_angular_acceleration_rad_s2"] = float(
            np.linalg.norm(fd_twist_dot[3:])
        )
        row["fd_ee_linear_acceleration_tracking_error_m_s2"] = float(
            np.linalg.norm(fd_twist_dot[:3] - reference_twist_dot[:3])
        )
        row["fd_ee_angular_acceleration_tracking_error_rad_s2"] = float(
            np.linalg.norm(fd_twist_dot[3:] - reference_twist_dot[3:])
        )
        row["fd_vs_qacc_ee_linear_acceleration_error_m_s2"] = float(
            np.linalg.norm(fd_twist_dot[:3] - np.asarray(
                [float(row[f"python_{name}"]) for name in accel_names[:3]], dtype=float
            ))
        )
        row["fd_vs_qacc_ee_angular_acceleration_error_rad_s2"] = float(
            np.linalg.norm(fd_twist_dot[3:] - np.asarray(
                [float(row[f"python_{name}"]) for name in accel_names[3:]], dtype=float
            ))
        )


def summarize(rows: list[dict[str, Any]], move_duration: float, total_duration: float,
              physics_timestep: float) -> bool:
    """Print kinematic agreement, tracking diagnostics, and the measured speed profile."""
    metrics = (
        ("position_error_m", POSITION_TOL_M, "m"),
        ("linear_velocity_error_m_s", VELOCITY_TOL, "m/s"),
        ("angular_velocity_error_rad_s", VELOCITY_TOL, "rad/s"),
        ("linear_acceleration_error_m_s2", ACCELERATION_TOL, "m/s^2"),
        ("angular_acceleration_error_rad_s2", ACCELERATION_TOL, "m/s^2"),
    )
    print("\nKINEMATIC AGREEMENT: Python equations vs MuJoCo at the same actual state")
    kinematics_pass = True
    for key, tolerance, unit in metrics:
        maximum = max(float(row[key]) for row in rows)
        passed = np.isfinite(maximum) and maximum <= tolerance
        kinematics_pass &= passed
        print(f"{key:40s} max={maximum:.3e} {unit:6s} tol={tolerance:.1e} {'PASS' if passed else 'FAIL'}")

    print("\nACTUATOR POSITION / VELOCITY TRACKING: actual state vs quintic reference")
    tracking_metrics = (
        ("joint_position_tracking_error_norm_rad", "rad"),
        ("joint_velocity_tracking_error_norm_rad_s", "rad/s"),
        ("ee_linear_velocity_tracking_error_m_s", "m/s"),
        ("ee_angular_velocity_tracking_error_rad_s", "rad/s"),
    )
    for key, unit in tracking_metrics:
        values = np.asarray([float(row[key]) for row in rows], dtype=float)
        rms = float(np.sqrt(np.mean(values**2)))
        maximum = float(np.max(np.abs(values)))
        print(f"{key:46s} RMS={rms:.4e} {unit:7s} max={maximum:.4e} {unit}")

    def print_error_summary(label: str, key: str, unit: str, *, interior: bool = True) -> None:
        values = np.asarray([float(row[key]) for row in rows], dtype=float)
        if interior and values.size > 4:
            values = values[2:-2]
        values = values[np.isfinite(values)]
        if values.size == 0:
            print(f"{label}: unavailable")
            return
        rms = float(np.sqrt(np.mean(values**2)))
        maximum = float(np.max(np.abs(values)))
        print(f"{label:58s} RMS={rms:.4e} {unit:7s} max={maximum:.4e} {unit}")

    print("\nACCELERATION DEFINITIONS — KEPT SEPARATE")
    print("Instantaneous qacc: data.qacc after mj_forward at the recorded state.")
    print("Sampled qddot: numerical derivative of the recorded actual qvel trace.")
    print_error_summary(
        "Instantaneous MuJoCo qacc vs reference qddot",
        "joint_acceleration_tracking_error_norm_rad_s2", "rad/s^2"
    )
    print_error_summary(
        "Finite-difference qvel vs reference qddot",
        "fd_joint_acceleration_tracking_error_norm_rad_s2", "rad/s^2"
    )
    print_error_summary(
        "Instantaneous qacc vs finite-difference qddot",
        "qacc_vs_fd_joint_acceleration_error_norm_rad_s2", "rad/s^2"
    )
    print_error_summary(
        "FD-derived EE linear acceleration vs reference",
        "fd_ee_linear_acceleration_tracking_error_m_s2", "m/s^2"
    )
    print_error_summary(
        "FD-derived EE angular acceleration vs reference",
        "fd_ee_angular_acceleration_tracking_error_rad_s2", "rad/s^2"
    )

    times_for_step = np.asarray([float(row["time_s"]) for row in rows])
    if len(rows) > 2:
        sample_dt = float(np.median(np.diff(times_for_step)))
        if abs(sample_dt - physics_timestep) <= max(1e-8, physics_timestep * 1e-3):
            print("\nINTEGRATOR-STEP DIAGNOSTIC (one CSV sample per physics step)")
            print_error_summary(
                "mj_step qacc vs same-step average qvel derivative",
                "mj_step_qacc_vs_interval_fd_error_norm_rad_s2", "rad/s^2"
            )
        else:
            print(
                "\nINTEGRATOR-STEP DIAGNOSTIC skipped: CSV interval "
                f"({sample_dt:.6f} s) is not one physics step ({physics_timestep:.6f} s)."
            )

    ctrl_clip_keys = [f"position_control_clipped_j{i}" for i in range(1, 7)]
    force_sat_keys = [f"actuator_force_saturated_j{i}" for i in range(1, 7)]
    def csv_bool(value: Any) -> bool:
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "y"}
        return bool(value)

    ctrl_clip_count = sum(csv_bool(row[key]) for row in rows for key in ctrl_clip_keys)
    force_sat_count = sum(csv_bool(row[key]) for row in rows for key in force_sat_keys)
    total_actuator_samples = len(rows) * 6
    print("\nACTUATOR LIMIT CHECK")
    print(f"Position-control clipping events: {ctrl_clip_count}/{total_actuator_samples}")
    print(f"Actuator force saturation events: {force_sat_count}/{total_actuator_samples}")

    times = np.asarray([float(row["time_s"]) for row in rows])

    if len(times) >= 2:
        sample_steps = np.diff(times)
        print(
            "Recorded CSV sample interval: "
            f"median={np.median(sample_steps):.6f} s, "
            f"min={np.min(sample_steps):.6f} s, max={np.max(sample_steps):.6f} s"
        )

    speeds = np.asarray([float(row["actual_joint_speed_norm_rad_s"]) for row in rows])
    moving_mask = times <= move_duration + 1e-9
    if not np.any(moving_mask):
        print("MOTION PROFILE: WARN — no samples inside movement interval")
        return kinematics_pass

    moving_indices = np.flatnonzero(moving_mask)
    peak_local = int(np.argmax(speeds[moving_mask]))
    peak_index = int(moving_indices[peak_local])
    peak_speed = float(speeds[peak_index])
    peak_time = float(times[peak_index])

    start_mask = times <= min(0.25, 0.08 * move_duration)
    end_window = min(0.5, max(0.1, total_duration - move_duration))
    end_mask = times >= total_duration - end_window
    start_speed = float(np.mean(speeds[start_mask])) if np.any(start_mask) else float(speeds[0])
    end_speed = float(np.mean(speeds[end_mask])) if np.any(end_mask) else float(speeds[-1])

    start_ratio = start_speed / max(peak_speed, 1e-12)
    end_ratio = end_speed / max(peak_speed, 1e-12)
    peak_fraction = peak_time / move_duration
    motion_pass = (
        peak_speed > 1e-3
        and 0.15 <= peak_fraction <= 0.85
        and start_ratio < 0.35
        and end_ratio < 0.35
    )

    initial_acceleration = float(rows[0]["actual_joint_acceleration_norm_rad_s2"])
    initial_target_acceleration = float(rows[0]["target_joint_acceleration_norm_rad_s2"])
    print("\nMEASURED MOTION PROFILE (joint-speed norm from MuJoCo)")
    print(f"Initial speed average : {start_speed:.5f} rad/s ({start_ratio:.1%} of peak)")
    print(f"Peak speed            : {peak_speed:.5f} rad/s at t={peak_time:.2f} s")
    print(f"Final hold speed avg  : {end_speed:.5f} rad/s ({end_ratio:.1%} of peak)")
    print(f"Initial acceleration  : {initial_acceleration:.5f} rad/s^2")
    print(f"Initial reference acc.: {initial_target_acceleration:.5f} rad/s^2")
    print("Motion profile check  :", "PASS — starts slowly, peaks mid-motion, settles at end"
          if motion_pass else "WARN — inspect speed plot and actuator tracking")
    print("\nNote: kinematic equation agreement and actuator tracking are separate checks.")
    return kinematics_pass


def set_viewer_camera(viewer: Any, model: mujoco.MjModel, camera_name: str) -> None:
    if camera_name == "free":
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        return
    camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
    if camera_id < 0:
        raise RuntimeError(f"Scene does not contain camera '{camera_name}'")
    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
    viewer.cam.fixedcamid = camera_id


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scene", type=Path,
        default=ROOT / "Milestone 01" / "Robot Model" / "scene_surgical_m3.xml",
        help="MuJoCo scene (defaults to the updated Milestone 3 surgical environment)",
    )
    parser.add_argument("--move-duration", type=float, default=6.0,
                        help="Duration of the smooth point-to-point movement in seconds")
    parser.add_argument("--hold-duration", type=float, default=2.0,
                        help="Time to hold the final pose so deceleration/stop is visible")
    parser.add_argument("--motion-scale", type=float, default=1.0,
                        help="Scale the conservative six-joint offset (0.5 to 1.25)")
    parser.add_argument("--sample-interval", type=float, default=0.02,
                        help="CSV sampling interval in simulated seconds")
    parser.add_argument("--csv", type=Path,
                        default=ROOT / "Milestone 03" / "Results" / "Smooth_Motion" / "smooth_motion.csv")
    parser.add_argument(
        "--no-dynamics-feedforward",
        action="store_true",
        help="Disable inverse-dynamics feedforward (for baseline comparison only)",
    )
    parser.add_argument("--headless", action="store_true", help="Run without opening the viewer")
    parser.add_argument("--camera", choices=("overview", "instrument_tray", "handover_closeup", "free"),
                        default="overview", help="Initial viewer camera; use free for manual mouse control")
    args = parser.parse_args()

    if not (2.0 <= args.move_duration <= 20.0):
        parser.error("--move-duration must be between 2 and 20 seconds")
    if not (0.0 <= args.hold_duration <= 10.0):
        parser.error("--hold-duration must be between 0 and 10 seconds")
    if not (0.5 <= args.motion_scale <= 1.25):
        parser.error("--motion-scale must be between 0.5 and 1.25")
    if not (0.001 <= args.sample_interval <= 0.5):
        parser.error("--sample-interval must be between 0.001 and 0.5 seconds")

    scene = args.scene.expanduser().resolve()
    if not scene.is_file():
        parser.error(f"Scene not found: {scene}")

    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    probe = mujoco.MjData(model)
    reference_data = mujoco.MjData(model)
    qpos_idx, dof_idx, actuator_idx, joint_ids = robot_indices(model)
    kp, kd = actuator_pd_parameters(model, actuator_idx)
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, SITE_NAME)
    if site_id < 0:
        raise RuntimeError(f"Scene does not contain required site '{SITE_NAME}'")

    q_start = HOME.copy()
    q_delta = JOINT_DELTA * args.motion_scale
    q_goal = q_start + q_delta
    validate_reference_limits(model, joint_ids, actuator_idx, q_start, q_goal)

    home_key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    if home_key_id >= 0:
        mujoco.mj_resetDataKeyframe(model, data, home_key_id)
    else:
        mujoco.mj_resetData(model, data)
    # Initialization only: the motion loop never writes qpos directly.
    data.qpos[qpos_idx] = q_start
    data.qvel[dof_idx] = 0.0

    use_dynamics_feedforward = not args.no_dynamics_feedforward
    total_duration = args.move_duration + args.hold_duration
    csv_path = args.csv.expanduser().resolve()
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    last_sample_time = -float("inf")
    last_sync_time = -float("inf")
    sample_count = 0

    def apply_reference_control(t: float):
        q_ref, qdot_ref, qddot_ref, path_s = quintic_reference(
            t, args.move_duration, q_start, q_delta
        )
        ctrl, required_torque, ff_torque, raw_ctrl, ctrl_clipped = compute_position_control(
            model, data, reference_data, qpos_idx, dof_idx, actuator_idx,
            q_ref, qdot_ref, qddot_ref, kp, kd, use_dynamics_feedforward,
        )
        data.ctrl[actuator_idx] = ctrl
        return (
            q_ref, qdot_ref, qddot_ref, path_s, ctrl,
            required_torque, ff_torque, raw_ctrl, ctrl_clipped,
        )

    def capture_current_state(
        step_qacc: np.ndarray | None = None,
        step_qacc_source_time: float | None = None,
    ) -> None:
        nonlocal last_sample_time, sample_count
        (
            q_ref, qdot_ref, qddot_ref, path_s, ctrl,
            required_torque, ff_torque, raw_ctrl, ctrl_clipped,
        ) = apply_reference_control(float(data.time))

        # Recompute forward dynamics at the actual state with the compensated
        # position command, so qacc and actuator_force describe this state.
        mujoco.mj_forward(model, data)
        row = evaluate_sample(
            model, data, probe, site_id, qpos_idx, dof_idx,
            q_ref, qdot_ref, qddot_ref, path_s, args.move_duration,
        )

        # Preserve the qacc value produced by mj_step before the subsequent
        # mj_forward call above recomputes instantaneous forward dynamics with
        # the newly updated controller command.  It is a separate diagnostic.
        row["mj_step_qacc_source_time_s"] = (
            float(step_qacc_source_time) if step_qacc_source_time is not None else float("nan")
        )
        for i in range(6):
            row[f"mj_step_qacc_j{i+1}_rad_s2"] = (
                float(step_qacc[i]) if step_qacc is not None else float("nan")
            )

        for i, aid in enumerate(actuator_idx):
            force = float(data.actuator_force[aid])
            gear = float(model.actuator_gear[aid, 0])
            torque = force * gear
            if bool(model.actuator_forcelimited[aid]):
                force_range = model.actuator_forcerange[aid]
                force_limit = float(max(abs(force_range[0]), abs(force_range[1])))
                force_saturated = abs(force) >= force_limit - max(1e-6, 1e-4 * force_limit)
            else:
                force_limit = float("nan")
                force_saturated = False

            row[f"required_inverse_dynamics_torque_j{i+1}_Nm"] = float(required_torque[i])
            row[f"feedforward_command_torque_j{i+1}_Nm"] = float(ff_torque[i])
            row[f"feedforward_position_offset_j{i+1}_rad"] = float(raw_ctrl[i] - q_ref[i])
            row[f"position_actuator_target_j{i+1}_rad"] = float(ctrl[i])
            row[f"position_control_clipped_j{i+1}"] = bool(ctrl_clipped[i])
            row[f"actual_actuator_torque_j{i+1}_Nm"] = torque
            row[f"actuator_torque_limit_j{i+1}_Nm"] = force_limit
            row[f"actuator_force_saturated_j{i+1}"] = bool(force_saturated)

        row["dynamics_feedforward_enabled"] = bool(use_dynamics_feedforward)
        rows.append(row)
        last_sample_time = float(data.time)
        sample_count += 1

    def advance(viewer: Any | None = None) -> None:
        nonlocal last_sync_time
        capture_current_state()  # record t=0 exactly, with feedforward active
        dt = float(model.opt.timestep)
        last_step_qacc: np.ndarray | None = None
        last_step_qacc_source_time: float | None = None
        while (data.time < total_duration - 0.5 * dt
               and (viewer is None or viewer.is_running())):
            step_started = time.perf_counter()
            apply_reference_control(float(data.time))
            step_source_time = float(data.time)
            mujoco.mj_step(model, data)
            # Copy immediately: capture_current_state() calls mj_forward at the
            # new time and therefore overwrites data.qacc with a fresh forward
            # dynamics evaluation using the next reference/control command.
            last_step_qacc = data.qacc[dof_idx].copy()
            last_step_qacc_source_time = step_source_time

            if data.time - last_sample_time + 1e-10 >= args.sample_interval:
                capture_current_state(last_step_qacc, last_step_qacc_source_time)

            if viewer is not None and data.time - last_sync_time >= 1.0 / 30.0:
                viewer.sync()
                last_sync_time = float(data.time)

            if viewer is not None:
                remaining = dt - (time.perf_counter() - step_started)
                if remaining > 0.0:
                    time.sleep(remaining)

        # Capture the final settled state, even if it falls between sample ticks.
        if not rows or data.time > last_sample_time + 1e-10:
            capture_current_state(last_step_qacc, last_step_qacc_source_time)
        if viewer is not None and viewer.is_running():
            viewer.sync()

    print("UR5e smooth point-to-point motion demonstration")
    print(f"MuJoCo: {mujoco.__version__} | Scene: {scene.name}")
    print("Joint order:", ", ".join(JOINT_NAMES))
    print("Input: six coordinated POSITION references sent to existing position actuators")
    print("Dynamics feedforward:", "ON (inverse dynamics)" if use_dynamics_feedforward else "OFF (baseline)")
    print("Actuator Kp:", np.array2string(kp, precision=1), "| Kd:", np.array2string(kd, precision=1))
    print("Feedforward is applied as a position-command offset; actuator model is unchanged")
    print("Acceleration logging: instantaneous mj_forward qacc + pre-mj_forward mj_step qacc + finite-difference qvel")
    print(f"Physics timestep: {model.opt.timestep:.6f} s | Requested CSV sample interval: {args.sample_interval:.6f} s")
    print(f"Movement: {args.move_duration:.2f} s | Final hold: {args.hold_duration:.2f} s")
    print("Start q [rad]:", np.array2string(q_start, precision=4))
    print("Goal  q [rad]:", np.array2string(q_goal, precision=4))
    print("Viewer:", "OFF (headless)" if args.headless else f"ON (camera: {args.camera})")
    print("CSV:", csv_path)

    try:
        if args.headless:
            advance()
        else:
            from mujoco import viewer as mujoco_viewer
            with mujoco_viewer.launch_passive(model, data, show_left_ui=False) as viewer:
                set_viewer_camera(viewer, model, args.camera)
                advance(viewer)
    except KeyboardInterrupt:
        print("\nInterrupted by user; saving collected samples.")

    add_sampled_acceleration_diagnostics(rows)
    write_csv(csv_path, rows)
    print(f"\nSaved {len(rows)} samples to {csv_path}")
    kinematics_pass = summarize(rows, args.move_duration, total_duration, float(model.opt.timestep))
    print("\nRESULT:", "PASS — KINEMATIC AGREEMENT" if kinematics_pass else "FAIL — KINEMATIC ERROR ABOVE TOLERANCE")
    if not kinematics_pass:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
