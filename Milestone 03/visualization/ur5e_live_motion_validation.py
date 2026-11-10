"""MCTR911 Milestone 03: PHYSICS-STEPPED UR5e motion and kinematics comparison.

The script sends smooth joint-angle references to the EXISTING MuJoCo position
actuators, advances actual MuJoCo physics with mj_step, and reads q, qdot,
qacc from the simulated robot. It does NOT directly write qpos in the motion
loop, so the visible robot follows the controls rather than a teleported pose.

In world coordinates at the same UR5e attachment_site origin:
  Python:  p = FK(q), twist = J(q) qdot,
           twist_dot = J(q) qacc + Jdot(q, qdot) qdot
  MuJoCo:  p = site_xpos, twist = J_mj qdot,
           twist_dot = J_mj qacc + (dJ_mj/dt) qdot

The numerical MuJoCo Jdot is evaluated using +/- h*qdot probes in separate
MjData. It never modifies the visible simulation state. This checks the
kinematic relationship under physically simulated motion, not a motor-control
tracking certification or a real-hardware test.

Run from repository root:
  python 'Milestone 03/visualization/ur5e_live_motion_validation.py'
  python 'Milestone 03/visualization/ur5e_live_motion_validation.py' --headless --duration 8
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from datetime import datetime
from pathlib import Path

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
HOME = np.array([-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0.0])
# Small, smooth motions around home; avoid large movements near work surfaces.
AMPLITUDE = np.array([0.18, 0.08, 0.11, 0.085, 0.09, 0.16])
FREQUENCY_HZ = np.array([0.14, 0.17, 0.13, 0.19, 0.16, 0.15])
PHASE = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
POSITION_TOL = 1e-5       # m (kinematic model agreement)
VELOCITY_TOL = 1e-4       # m/s or rad/s
ACCELERATION_TOL = 1e-3   # m/s^2 or rad/s^2; numerical Jdot on simulated data
JDOT_PROBE_S = 1e-4


def command_pose(t: float) -> np.ndarray:
    """Smooth joint position reference; motion begins from HOME at zero speed."""
    ramp = 1.0 - np.exp(-(max(t, 0.0) / 1.0) ** 2)
    return HOME + ramp * AMPLITUDE * np.sin(2.0 * np.pi * FREQUENCY_HZ * t + PHASE)


def robot_indices(model: mujoco.MjModel) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Resolve joints and their position actuators by name, never by guessed order."""
    qpos_idx, qvel_idx, actuator_idx = [], [], []
    for name in JOINT_NAMES:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            raise RuntimeError(f"Missing joint '{name}' in the MuJoCo scene")
        if model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_HINGE:
            raise RuntimeError(f"Expected single-DOF hinge joint '{name}'")
        matches = [i for i in range(model.nu)
                   if int(model.actuator_trnid[i, 0]) == joint_id
                   and int(model.actuator_trntype[i]) == int(mujoco.mjtTrn.mjTRN_JOINT)]
        if len(matches) != 1:
            raise RuntimeError(f"Expected one position actuator for {name}, found {len(matches)}")
        qpos_idx.append(int(model.jnt_qposadr[joint_id]))
        qvel_idx.append(int(model.jnt_dofadr[joint_id]))
        actuator_idx.append(matches[0])
    return (np.asarray(qpos_idx, dtype=int), np.asarray(qvel_idx, dtype=int),
            np.asarray(actuator_idx, dtype=int))


def site_jacobian(model: mujoco.MjModel, data: mujoco.MjData,
                  site_id: int, dof_idx: np.ndarray) -> np.ndarray:
    jacp = np.zeros((3, model.nv), dtype=float)
    jacr = np.zeros((3, model.nv), dtype=float)
    mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
    return np.vstack((jacp[:, dof_idx], jacr[:, dof_idx]))


def mujoco_jdot(model: mujoco.MjModel, probe: mujoco.MjData,
                original: mujoco.MjData, site_id: int,
                qpos_idx: np.ndarray, dof_idx: np.ndarray, qdot: np.ndarray) -> np.ndarray:
    """Finite-difference MuJoCo Jdot at the CURRENT actual simulated q/qdot."""
    # Clone all general positions so this remains usable if the scene gains props.
    probe.qpos[:] = original.qpos
    probe.qvel[:] = original.qvel
    probe.qpos[qpos_idx] = original.qpos[qpos_idx] + JDOT_PROBE_S * qdot
    mujoco.mj_forward(model, probe)
    plus = site_jacobian(model, probe, site_id, dof_idx)
    probe.qpos[qpos_idx] = original.qpos[qpos_idx] - JDOT_PROBE_S * qdot
    mujoco.mj_forward(model, probe)
    minus = site_jacobian(model, probe, site_id, dof_idx)
    return (plus - minus) / (2.0 * JDOT_PROBE_S)


def evaluate(model: mujoco.MjModel, data: mujoco.MjData,
             probe: mujoco.MjData, site_id: int,
             qpos_idx: np.ndarray, dof_idx: np.ndarray,
             targets: np.ndarray) -> dict[str, float]:
    q = data.qpos[qpos_idx].copy()
    qdot = data.qvel[dof_idx].copy()
    qddot = data.qacc[dof_idx].copy()
    p_fk = forward_kinematics(q)[:3, 3]
    p_mj = data.site_xpos[site_id].copy()
    twist_py = forward_velocity(q, qdot)
    accel_py = forward_acceleration(q, qdot, qddot)
    j_mj = site_jacobian(model, data, site_id, dof_idx)
    twist_mj = j_mj @ qdot
    accel_mj = j_mj @ qddot + mujoco_jdot(model, probe, data, site_id,
                                          qpos_idx, dof_idx, qdot) @ qdot
    out: dict[str, float] = {"time_s": float(data.time)}
    for i in range(6):
        out[f"target_q{i+1}_rad"] = float(targets[i])
        out[f"q{i+1}_rad"] = float(q[i])
        out[f"qdot{i+1}_rad_s"] = float(qdot[i])
        out[f"qddot{i+1}_rad_s2"] = float(qddot[i])
    for i, axis in enumerate("xyz"):
        out[f"fk_{axis}_m"] = float(p_fk[i])
        out[f"mujoco_{axis}_m"] = float(p_mj[i])
    names = ["vx", "vy", "vz", "wx", "wy", "wz"]
    a_names = ["ax", "ay", "az", "alphax", "alphay", "alphaz"]
    for i, name in enumerate(names):
        out[f"python_{name}"] = float(twist_py[i])
        out[f"mujoco_{name}"] = float(twist_mj[i])
    for i, name in enumerate(a_names):
        out[f"python_{name}"] = float(accel_py[i])
        out[f"mujoco_{name}"] = float(accel_mj[i])
    out["position_error_m"] = float(np.linalg.norm(p_fk - p_mj))
    out["linear_velocity_error_m_s"] = float(np.linalg.norm(twist_py[:3] - twist_mj[:3]))
    out["angular_velocity_error_rad_s"] = float(np.linalg.norm(twist_py[3:] - twist_mj[3:]))
    out["linear_acceleration_error_m_s2"] = float(np.linalg.norm(accel_py[:3] - accel_mj[:3]))
    out["angular_acceleration_error_rad_s2"] = float(np.linalg.norm(accel_py[3:] - accel_mj[3:]))
    out["joint_tracking_error_rad"] = float(np.linalg.norm(q - targets))
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scene", type=Path, default=ROOT / "Milestone 01" / "Robot Model" / "scene_handover.xml")
    p.add_argument("--duration", type=float, default=16.0, help="Simulated duration, seconds")
    p.add_argument("--headless", action="store_true", help="Run physics without opening the MuJoCo viewer")
    p.add_argument("--no-markers", action="store_true", help="Backward-compatible flag; custom overlays are disabled in this stable version")
    p.add_argument("--csv", type=Path, default=ROOT / "Milestone 03" / "Results" / "live_motion.csv")
    p.add_argument("--sample-interval", type=float, default=0.05, help="CSV interval in simulated seconds")
    args = p.parse_args()
    if not (0.1 <= args.duration <= 120 and 0.01 <= args.sample_interval <= 2.0):
        p.error("Duration must be 0.1-120 s and sample-interval must be 0.01-2 s")
    scene = args.scene.expanduser().resolve()
    if not scene.exists():
        p.error(f"Scene not found: {scene}")
    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    probe = mujoco.MjData(model)
    qpos_idx, dof_idx, actuators = robot_indices(model)
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, SITE_NAME)
    if site_id < 0:
        raise RuntimeError(f"Scene has no site '{SITE_NAME}'")
    # Use the current robot's explicit home keyframe or same values in fallback.
    home_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    if home_id >= 0:
        mujoco.mj_resetDataKeyframe(model, data, home_id)
    else:
        mujoco.mj_resetData(model, data)
    data.qpos[qpos_idx] = HOME
    data.qvel[dof_idx] = 0
    data.ctrl[actuators] = HOME
    mujoco.mj_forward(model, data)
    print("UR5e LIVE PHYSICS motion demo (no manual qpos changes during stepping)")
    print("MuJoCo:", mujoco.__version__, "| scene:", scene.name)
    print("Actual joint states q, qdot, qacc -> Python vs MuJoCo world-frame site kinematics")
    print("CSV output:", args.csv.resolve())
    print("Viewer:", "OFF (headless)" if args.headless else "ON (close window to finish early)")
    if not args.headless:
        print("Custom viewer overlays: OFF (stable mode; numerical data still recorded)")
    print("\n  t(s)  FK err(m)  Vel lin err  Vel ang err  Acc lin err  Acc ang err  Joint target err")
    csv_path = args.csv.expanduser().resolve()
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, float]] = []
    last_sample = -float("inf")
    last_print = -float("inf")
    next_draw = -float("inf")

    def advance(viewer_handle=None) -> None:
        nonlocal last_sample, last_print, next_draw
        dt = float(model.opt.timestep)
        wall_start = time.perf_counter()
        # Run until the requested simulated duration (or window closes).
        while data.time < args.duration and (viewer_handle is None or viewer_handle.is_running()):
            targets = command_pose(float(data.time))
            data.ctrl[actuators] = targets
            mujoco.mj_step(model, data)
            if data.time - last_sample + 1e-9 >= args.sample_interval:
                # mj_step has advanced state; refresh accelerations at that state.
                mujoco.mj_forward(model, data)
                sample = evaluate(model, data, probe, site_id, qpos_idx, dof_idx, targets)
                records.append(sample)
                last_sample = float(data.time)
                if data.time - last_print >= 0.9:
                    print(f"{data.time:6.2f}  {sample['position_error_m']:9.2e}"
                          f"   {sample['linear_velocity_error_m_s']:9.2e}"
                          f"   {sample['angular_velocity_error_rad_s']:9.2e}"
                          f"   {sample['linear_acceleration_error_m_s2']:9.2e}"
                          f"   {sample['angular_acceleration_error_rad_s2']:9.2e}"
                          f"      {sample['joint_tracking_error_rad']:9.3f}", flush=True)
                    last_print = float(data.time)
            if viewer_handle is not None and data.time - next_draw >= 1.0 / 30.0:
                # MuJoCo 3.2.0 on the user's WSL viewer crashed when adding
                # custom geometry via viewer.user_scn. Avoid those native
                # calls entirely; the base viewer and validation both work.
                # All motion/velocity/acceleration data remain in the CSV.
                viewer_handle.sync()
                next_draw = float(data.time)
            if not args.headless:
                # Keep physics roughly synchronized with real time for recording.
                remaining = float(data.time) - (time.perf_counter() - wall_start)
                if remaining > 0:
                    time.sleep(min(remaining, dt))

    try:
        if args.headless:
            advance()
        else:
            from mujoco import viewer as mujoco_viewer
            with mujoco_viewer.launch_passive(model, data, show_left_ui=False) as viewer:
                advance(viewer)
    except KeyboardInterrupt:
        print("\nInterrupted: saving available results.")

    if not records:
        raise SystemExit("No samples collected; no CSV produced.")
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    metrics = [
        ("position_error_m", POSITION_TOL),
        ("linear_velocity_error_m_s", VELOCITY_TOL),
        ("angular_velocity_error_rad_s", VELOCITY_TOL),
        ("linear_acceleration_error_m_s2", ACCELERATION_TOL),
        ("angular_acceleration_error_rad_s2", ACCELERATION_TOL),
    ]
    print(f"\nSaved {len(records)} actual-motion samples to {csv_path}")
    overall_pass = True
    for key, tol in metrics:
        finite = [sample[key] for sample in records]
        maximum = max(finite)
        passed = np.isfinite(maximum) and maximum < tol
        overall_pass &= passed
        print(f"{key:40} max={maximum:.3e} tol={tol:.1e} {'PASS' if passed else 'FAIL'}")
    print("Joint-target tracking is logged separately; do not confuse it with kinematic agreement.")
    print("RESULT:", "PASS — LIVE KINEMATIC AGREEMENT" if overall_pass else "CHECK — See CSV and diagnostics")
    if not overall_pass:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
