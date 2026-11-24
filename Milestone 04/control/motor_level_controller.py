"""Motor-level control equations for the MCTR911 UR5e Milestone 04.

The existing UR5e MJCF uses affine MuJoCo position actuators. For the supplied
model, each actuator follows:

    tau_act = Kp * (ctrl - q) - Kd * qdot

The M4 controller retains those actuators and adds inverse-dynamics feedforward:

    tau_ff = M(q_ref) qddot_ref + qfrc_bias(q_ref, qdot_ref)
             - qfrc_passive(q_ref, qdot_ref)

    ctrl = q_ref + (tau_ff + Kd*qdot_ref) / Kp

Substituting the command into the actuator law gives:

    tau_act = tau_ff + Kp*(q_ref - q) + Kd*(qdot_ref - qdot)

Thus the existing position servo supplies PD tracking feedback while the
feedforward compensates the modelled generalized force at the reference state.

This module does not add a new MuJoCo actuator and does not command physical
hardware. The ROS 2 node added in a later stage will publish ``ctrl`` through
the existing /joint_commands topic in POSITION mode.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

JOINT_COUNT = 6


def _six(values: Iterable[float], name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.shape != (JOINT_COUNT,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain exactly six finite values")
    return array.copy()


def _six_positive(values: Iterable[float], name: str, *, allow_zero: bool = False) -> np.ndarray:
    array = _six(values, name)
    if np.any(array < 0.0) or (not allow_zero and np.any(array <= 0.0)):
        relation = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be {relation}")
    return array


@dataclass(frozen=True)
class PositionServoCommand:
    """Position-actuator command and saturation diagnostics."""
    command_rad: np.ndarray
    unclipped_command_rad: np.ndarray
    feedforward_torque_nm: np.ndarray
    clipped: np.ndarray


def position_actuator_torque(
    command_rad: Iterable[float],
    actual_position_rad: Iterable[float],
    actual_velocity_rad_s: Iterable[float],
    kp_nm_per_rad: Iterable[float],
    kd_nm_s_per_rad: Iterable[float],
) -> np.ndarray:
    """Evaluate the supplied model's affine position-actuator torque law."""
    command = _six(command_rad, "command_rad")
    q = _six(actual_position_rad, "actual_position_rad")
    qdot = _six(actual_velocity_rad_s, "actual_velocity_rad_s")
    kp = _six_positive(kp_nm_per_rad, "kp_nm_per_rad")
    kd = _six_positive(kd_nm_s_per_rad, "kd_nm_s_per_rad", allow_zero=True)
    return kp * (command - q) - kd * qdot


def compute_position_servo_command(
    q_ref_rad: Iterable[float],
    qdot_ref_rad_s: Iterable[float],
    kp_nm_per_rad: Iterable[float],
    kd_nm_s_per_rad: Iterable[float],
    *,
    feedforward_torque_nm: Iterable[float] | None = None,
    control_range_rad: Iterable[Iterable[float]] | None = None,
) -> PositionServoCommand:
    """Map reference state and feedforward torque to position actuator commands.

    If ``feedforward_torque_nm`` is None, the baseline command is simply
    ``q_ref``. If supplied, the command includes both inverse-dynamics torque
    and the ``Kd*qdot_ref`` compensation required by the actuator's built-in
    velocity damping.

    ``control_range_rad`` must be a 6x2 array of [lower, upper] limits. The
    returned ``clipped`` vector identifies any command that hit a limit.
    """
    q_ref = _six(q_ref_rad, "q_ref_rad")
    qdot_ref = _six(qdot_ref_rad_s, "qdot_ref_rad_s")
    kp = _six_positive(kp_nm_per_rad, "kp_nm_per_rad")
    kd = _six_positive(kd_nm_s_per_rad, "kd_nm_s_per_rad", allow_zero=True)

    if feedforward_torque_nm is None:
        tau_ff = np.zeros(JOINT_COUNT, dtype=float)
        raw = q_ref.copy()
    else:
        tau_ff = _six(feedforward_torque_nm, "feedforward_torque_nm")
        raw = q_ref + (tau_ff + kd * qdot_ref) / kp

    command = raw.copy()
    clipped = np.zeros(JOINT_COUNT, dtype=bool)

    if control_range_rad is not None:
        limits = np.asarray(control_range_rad, dtype=float)
        if limits.shape != (JOINT_COUNT, 2) or not np.all(np.isfinite(limits)):
            raise ValueError("control_range_rad must be a finite 6x2 array")
        if np.any(limits[:, 0] >= limits[:, 1]):
            raise ValueError("Each control lower limit must be below its upper limit")
        command = np.clip(raw, limits[:, 0], limits[:, 1])
        clipped = ~np.isclose(command, raw, rtol=0.0, atol=1e-12)

    return PositionServoCommand(
        command_rad=command,
        unclipped_command_rad=raw,
        feedforward_torque_nm=tau_ff,
        clipped=clipped,
    )


def expected_tracking_torque(
    q_ref_rad: Iterable[float],
    qdot_ref_rad_s: Iterable[float],
    actual_position_rad: Iterable[float],
    actual_velocity_rad_s: Iterable[float],
    kp_nm_per_rad: Iterable[float],
    kd_nm_s_per_rad: Iterable[float],
    feedforward_torque_nm: Iterable[float],
) -> np.ndarray:
    """Equivalent closed-loop torque expression for the feedforward command."""
    q_ref = _six(q_ref_rad, "q_ref_rad")
    qdot_ref = _six(qdot_ref_rad_s, "qdot_ref_rad_s")
    q = _six(actual_position_rad, "actual_position_rad")
    qdot = _six(actual_velocity_rad_s, "actual_velocity_rad_s")
    kp = _six_positive(kp_nm_per_rad, "kp_nm_per_rad")
    kd = _six_positive(kd_nm_s_per_rad, "kd_nm_s_per_rad", allow_zero=True)
    tau_ff = _six(feedforward_torque_nm, "feedforward_torque_nm")
    return tau_ff + kp * (q_ref - q) + kd * (qdot_ref - qdot)


def read_position_actuator_gains(
    model,
    actuator_ids: Iterable[int],
    *,
    expected_count: int = JOINT_COUNT,
) -> tuple[np.ndarray, np.ndarray]:
    """Read Kp/Kd from MuJoCo actuator parameters and verify the affine PD form.

    This intentionally mirrors the checks already used in the Milestone 03
    smooth-motion demonstration. Unit gear ratios are required because the
    controller interprets ``ctrl`` and ``q`` in radians.
    """
    ids = np.asarray(list(actuator_ids), dtype=int)
    if ids.shape != (expected_count,) or np.any(ids < 0) or np.any(ids >= model.nu):
        raise ValueError(f"actuator_ids must contain {expected_count} valid actuator indices")

    gain = np.asarray(model.actuator_gainprm[ids, 0], dtype=float)
    bias = np.asarray(model.actuator_biasprm[ids, :], dtype=float)
    kp = -bias[:, 1]
    kd = -bias[:, 2]
    gear = np.asarray(model.actuator_gear[ids, 0], dtype=float)

    if np.any(gain <= 0.0) or np.any(kp <= 0.0) or np.any(kd < 0.0):
        raise ValueError("Expected positive Kp and non-negative Kd position actuators")
    if not np.allclose(gain, kp, rtol=1e-8, atol=1e-10):
        raise ValueError("Unsupported actuator: fixed gain must equal -biasprm[:, 1]")
    if not np.allclose(gear, 1.0, rtol=0.0, atol=1e-10):
        raise ValueError("This controller requires unit gear ratios")
    return kp, kd


def compute_inverse_dynamics_feedforward(
    model,
    actual_data,
    reference_data,
    qpos_indices: Iterable[int],
    dof_indices: Iterable[int],
    q_ref_rad: Iterable[float],
    qdot_ref_rad_s: Iterable[float],
    qddot_ref_rad_s2: Iterable[float],
) -> np.ndarray:
    """Compute reference-state inverse-dynamics feedforward with MuJoCo.

    The same model-based calculation is already used in the M3 smooth-motion
    experiment. ``actual_data`` supplies the complete scene state as a template;
    the six UR5e coordinates are replaced by the reference q and qdot before
    calling ``mj_forward``. The returned vector is ordered by the six supplied
    UR5e DOF indices.
    """
    try:
        import mujoco
    except ImportError as exc:  # pragma: no cover - depends on user's ROS/MuJoCo env
        raise RuntimeError("MuJoCo Python is required for inverse-dynamics feedforward") from exc

    qpos_idx = np.asarray(list(qpos_indices), dtype=int)
    dof_idx = np.asarray(list(dof_indices), dtype=int)
    if qpos_idx.shape != (JOINT_COUNT,) or dof_idx.shape != (JOINT_COUNT,):
        raise ValueError("qpos_indices and dof_indices must each contain six indices")
    if np.any(qpos_idx < 0) or np.any(qpos_idx >= model.nq):
        raise ValueError("qpos_indices contain an invalid model index")
    if np.any(dof_idx < 0) or np.any(dof_idx >= model.nv):
        raise ValueError("dof_indices contain an invalid model index")

    q_ref = _six(q_ref_rad, "q_ref_rad")
    qdot_ref = _six(qdot_ref_rad_s, "qdot_ref_rad_s")
    qddot_ref = _six(qddot_ref_rad_s2, "qddot_ref_rad_s2")

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
    return np.asarray(required_generalized_force[dof_idx], dtype=float).copy()
