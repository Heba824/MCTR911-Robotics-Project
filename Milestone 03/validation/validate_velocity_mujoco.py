"""Independent MuJoCo check of UR5e forward/inverse velocity kinematics.

For each configuration and imposed qdot:
  1. Calculate the 6D site twist from our Python J(q) @ qdot.
  2. Calculate MuJoCo's own site Jacobian via mj_jacSite and its J @ qvel.
  3. Independently differentiate MuJoCo site poses at q +/- dt*qdot,
     including world-frame angular velocity from Rdot @ R.T.
  4. Ask inverse_velocity to reproduce MuJoCo's twist, and compare the
     reconstructed velocity via MuJoCo's Jacobian (not our Python Jacobian).

This is a state-level kinematic validation. It does not claim the robot's
actuators track commanded velocities or that dynamic acceleration is tested.
No GUI, ROS 2 node or simulation stepping is required.

Usage from repository root:
    python 'Milestone 03/validation/validate_velocity_mujoco.py' \\
      --scene 'Milestone 01/Robot Model/scene_handover.xml'
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mujoco
import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
_KINEMATICS_DIR = _ROOT / "Milestone 03" / "kinematics"
if str(_KINEMATICS_DIR) not in sys.path:
    sys.path.insert(0, str(_KINEMATICS_DIR))

from ur5e_velocity_kinematics import (  # noqa: E402
    JOINT_NAMES,
    forward_velocity,
    inverse_velocity,
    jacobian_singular_values,
)

SITE_NAME = "attachment_site"
FORWARD_TOLERANCE = 5e-6  # m/s or rad/s; maximum per-component difference
INVERSE_TOLERANCE = 5e-5  # m/s or rad/s; maximum reconstructed twist difference
DT = 1e-5               # s, for independent central differences of pose

# q [rad]; qdot [rad/s], in exact JOINT_NAMES order.
TEST_CASES = (
    ("zero", (0., 0., 0., 0., 0., 0.), (0.10, -0.05, 0.12, -0.06, 0.09, 0.03)),
    ("home", (-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0.),
     (0.08, 0.11, -0.07, 0.09, -0.12, 0.05)),
    ("configuration_A", (0.3, -0.4, 0.2, 0.1, 0.5, 0.6),
     (0.12, -0.08, 0.15, -0.05, 0.09, 0.07)),
    ("configuration_B", (-0.6, 0.2, -0.7, 0.4, -0.2, -0.5),
     (-0.11, 0.06, -0.09, 0.12, -0.05, 0.08)),
    ("configuration_C", (1.0, -0.8, 1.1, -0.6, 0.5, -0.2),
     (0.07, -0.12, 0.09, 0.04, 0.11, -0.06)),
)


def _site_pose(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    site_id: int,
    qpos_addresses: list[int],
    q: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute site position and rotation from MuJoCo for a supplied pose."""
    for address, angle in zip(qpos_addresses, q):
        data.qpos[address] = float(angle)
    mujoco.mj_forward(model, data)
    return (data.site_xpos[site_id].copy(),
            data.site_xmat[site_id].reshape(3, 3).copy())


def _mujoco_finite_difference_twist(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    site_id: int,
    qpos_addresses: list[int],
    q: np.ndarray,
    qdot: np.ndarray,
) -> np.ndarray:
    """Differentiate MuJoCo FK (not the Python DH FK or Jacobian)."""
    _, r = _site_pose(model, data, site_id, qpos_addresses, q)
    p_plus, r_plus = _site_pose(model, data, site_id, qpos_addresses, q + DT * qdot)
    p_minus, r_minus = _site_pose(model, data, site_id, qpos_addresses, q - DT * qdot)
    linear = (p_plus - p_minus) / (2 * DT)
    r_dot = (r_plus - r_minus) / (2 * DT)
    omega_skew = r_dot @ r.T
    omega_skew = 0.5 * (omega_skew - omega_skew.T)
    angular = np.array([omega_skew[2, 1], omega_skew[0, 2], omega_skew[1, 0]])
    return np.r_[linear, angular]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scene", type=Path,
        default=_ROOT / "Milestone 01" / "Robot Model" / "scene_handover.xml",
    )
    args = parser.parse_args()
    scene = args.scene.expanduser().resolve()
    if not scene.is_file():
        parser.error(f"Scene does not exist: {scene}")

    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, SITE_NAME)
    if site_id < 0:
        raise RuntimeError(f"Missing MuJoCo site: {SITE_NAME}")

    qpos_addresses: list[int] = []
    qvel_addresses: list[int] = []
    for joint_name in JOINT_NAMES:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if jid < 0:
            raise RuntimeError(f"Missing MuJoCo joint: {joint_name}")
        if model.jnt_type[jid] != mujoco.mjtJoint.mjJNT_HINGE:
            raise RuntimeError(f"Expected hinge joint: {joint_name}")
        qpos_addresses.append(int(model.jnt_qposadr[jid]))
        qvel_addresses.append(int(model.jnt_dofadr[jid]))

    print("MuJoCo:", mujoco.__version__)
    print("Scene:", scene)
    print("Site:", SITE_NAME)
    print("World-frame twist order: [vx, vy, vz, wx, wy, wz]")
    print(f"Tolerance forward: {FORWARD_TOLERANCE:.1e} m/s or rad/s")
    print(f"Tolerance inverse: {INVERSE_TOLERANCE:.1e} m/s or rad/s")
    print(f"{'Configuration':<18} {'MJ J*qdot err':>15} {'MJ pose FD err':>16} {'Inverse err':>14} {'Result':>8}")
    print("-" * 83)
    failures = 0
    for name, q_values, qdot_values in TEST_CASES:
        q = np.asarray(q_values, dtype=float)
        qdot = np.asarray(qdot_values, dtype=float)
        mujoco.mj_resetData(model, data)
        for address, angle in zip(qpos_addresses, q):
            data.qpos[address] = float(angle)
        for address, vel in zip(qvel_addresses, qdot):
            data.qvel[address] = float(vel)
        mujoco.mj_forward(model, data)

        mj_linear_jac = np.zeros((3, model.nv))
        mj_angular_jac = np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, data, mj_linear_jac, mj_angular_jac, site_id)
        mj_jac = np.vstack((mj_linear_jac[:, qvel_addresses],
                            mj_angular_jac[:, qvel_addresses]))
        mj_twist = mj_jac @ qdot
        py_twist = forward_velocity(q, qdot)
        fd_twist = _mujoco_finite_difference_twist(
            model, data, site_id, qpos_addresses, q, qdot
        )

        # Inverse is checked through the simulator's Jacobian, not our J.
        qdot_inverse = inverse_velocity(q, mj_twist, damping=1e-5)
        mj_reconstructed = mj_jac @ qdot_inverse
        err_mujoco = float(np.max(np.abs(py_twist - mj_twist)))
        err_fd = float(np.max(np.abs(py_twist - fd_twist)))
        err_inverse = float(np.max(np.abs(mj_reconstructed - mj_twist)))
        passed = (err_mujoco < FORWARD_TOLERANCE
                  and err_fd < FORWARD_TOLERANCE
                  and err_inverse < INVERSE_TOLERANCE)
        failures += not passed
        print(f"{name:<18} {err_mujoco:>15.3e} {err_fd:>16.3e} {err_inverse:>14.3e} {'PASS' if passed else 'FAIL':>8}")
        if not passed:
            sv = jacobian_singular_values(q)
            print(f"  Min singular value: {sv[-1]:.3e}")
            print("  Python twist:          ", np.array2string(py_twist, precision=7))
            print("  MuJoCo Jacobian twist: ", np.array2string(mj_twist, precision=7))
            print("  MuJoCo pose FD twist:  ", np.array2string(fd_twist, precision=7))
            print("  Reconstructed twist:   ", np.array2string(mj_reconstructed, precision=7))

    if failures:
        print(f"\nFAILED: {failures} of {len(TEST_CASES)} velocity test configurations")
        raise SystemExit(1)
    print("\nALL 5 MUJOCO VELOCITY TESTS PASSED")
    print("NOTE: These validate kinematic velocities at prescribed joint states, not motor tracking.")


if __name__ == "__main__":
    main()
