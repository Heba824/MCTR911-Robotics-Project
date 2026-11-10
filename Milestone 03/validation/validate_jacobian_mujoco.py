"""Validate the Milestone 3 full geometric Jacobian against MuJoCo.

At five fixed joint configurations, independently compare:
  1) Analytical standard-DH Jacobian from ur5e_jacobian.py
  2) Numerical finite differences of the Milestone 2 forward kinematics
  3) MuJoCo's own ``mj_jacSite`` Jacobian at ``attachment_site``

This is a KINEMATIC comparison only: it does not step dynamics, apply motor
commands, or claim to validate joint acceleration and tracking yet.

Run from repository root:
    python "Milestone 03/validation/validate_jacobian_mujoco.py" \
      --scene "Milestone 01/Robot Model/scene_handover.xml"
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mujoco
import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
_KINEMATICS = _ROOT / "Milestone 03" / "kinematics"
sys.path.insert(0, str(_KINEMATICS))
from ur5e_jacobian import geometric_jacobian, finite_difference_jacobian, JOINT_NAMES

SITE_NAME = "attachment_site"
TOLERANCE = 5e-6  # max absolute Jacobian-entry difference
TEST_POSES = (
    ("zero", (0., 0., 0., 0., 0., 0.)),
    ("home", (-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0.)),
    ("configuration_A", (0.3, -0.4, 0.2, 0.1, 0.5, 0.6)),
    ("configuration_B", (-0.6, 0.2, -0.7, 0.4, -0.2, -0.5)),
    ("configuration_C", (1.0, -0.8, 1.1, -0.6, 0.5, -0.2)),
)


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

    qpos_addresses = []
    qvel_addresses = []
    for name in JOINT_NAMES:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            raise RuntimeError(f"Missing MuJoCo joint: {name}")
        if model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_HINGE:
            raise RuntimeError(f"Expected hinge joint for {name}")
        qpos_addresses.append(int(model.jnt_qposadr[joint_id]))
        qvel_addresses.append(int(model.jnt_dofadr[joint_id]))

    print("MuJoCo:", mujoco.__version__)
    print("Scene:", scene)
    print("Site:", SITE_NAME)
    print(f"Max entry-wise error tolerance: {TOLERANCE:.1e}")
    print(f"{'Configuration':<18} {'vs MuJoCo':>14} {'vs FK finite diff':>20} {'Result':>8}")
    print("-" * 66)

    failures = 0
    for name, angles in TEST_POSES:
        q = np.asarray(angles, dtype=float)
        mujoco.mj_resetData(model, data)
        for address, value in zip(qpos_addresses, q):
            data.qpos[address] = float(value)
        mujoco.mj_forward(model, data)

        # Each MuJoCo Jacobian array has 3 rows and model.nv columns.
        mj_linear = np.zeros((3, model.nv))
        mj_angular = np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, data, mj_linear, mj_angular, site_id)
        mj_jac = np.vstack((mj_linear[:, qvel_addresses], mj_angular[:, qvel_addresses]))

        analytic = geometric_jacobian(q)
        numeric = finite_difference_jacobian(q)
        mujoco_error = float(np.max(np.abs(analytic - mj_jac)))
        numeric_error = float(np.max(np.abs(analytic - numeric)))
        passed = mujoco_error < TOLERANCE and numeric_error < TOLERANCE
        failures += not passed
        print(f"{name:<18} {mujoco_error:>14.3e} {numeric_error:>20.3e} {'PASS' if passed else 'FAIL':>8}")

    if failures:
        print(f"\nFAILED: {failures} of {len(TEST_POSES)} Jacobian configurations")
        raise SystemExit(1)
    print("\nALL 5 MUJOCO JACOBIAN TESTS PASSED")


if __name__ == "__main__":
    main()
