"""Validate UR5e Jacobian-transpose inverse POSITION kinematics in MuJoCo.

Use from `Milestone 02/kinematics` with the original Milestone 01 model:

    source ~/mctr911_kinematics_venv/bin/activate
    python validate_ik_mujoco.py

The target XYZ of each test is obtained independently from MuJoCo at a
reference joint configuration. The Jacobian-transpose IK solves for angles,
which are applied back to MuJoCo and validated against that same target.

This checks static POSITION kinematics using mj_forward(), NOT simulated
motor motion, trajectory tracking, joint safety, or tool orientation.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
import numpy as np

from ur5e_dh_fk import JOINT_NAMES
from ur5e_ik_transpose import HOME, inverse_position_kinematics

SITE_NAME = "attachment_site"
POSITION_TOLERANCE_M = 1e-5

TESTS = (
    ("Test A", np.array([-1.1, -0.9, 1.15, -0.85, -1.2, 0.3]), HOME),
    ("Test B", np.array([0.35, -0.6, 0.7, 0.2, 0.65, -0.4]),
     np.array([0.2, -0.3, 0.4, 0.1, 0.3, -0.2])),
    ("Home target", HOME, np.array([-1.3, -1.4, 1.4, -1.4, -1.4, -0.2])),
)


def site_position(model: mujoco.MjModel, data: mujoco.MjData,
                  site_id: int, qpos_addresses: list[int],
                  q: np.ndarray) -> np.ndarray:
    """Get true MJCF site position at q, without running dynamics."""
    mujoco.mj_resetData(model, data)
    for address, value in zip(qpos_addresses, q):
        data.qpos[address] = float(value)
    mujoco.mj_forward(model, data)
    return np.asarray(data.site_xpos[site_id], dtype=float).copy()


def main() -> None:
    default_scene = (Path(__file__).resolve().parents[2] /
                     "Milestone 01" / "Robot Model" / "scene.xml")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, default=default_scene,
                        help="Path to original scene.xml")
    parser.add_argument("--gain", type=float, default=1.0,
                        help="Positive Jacobian-transpose gain")
    args = parser.parse_args()
    scene = args.scene.expanduser().resolve()
    if not scene.is_file():
        parser.error(f"Cannot find scene: {scene}")

    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, SITE_NAME)
    if site_id < 0:
        raise RuntimeError(f"Site {SITE_NAME!r} not found in MJCF")

    addresses = []
    for joint_name in JOINT_NAMES:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if jid < 0:
            raise RuntimeError(f"Joint {joint_name!r} not found in MJCF")
        if model.jnt_type[jid] != mujoco.mjtJoint.mjJNT_HINGE:
            raise RuntimeError(f"Joint {joint_name!r} is not revolute")
        addresses.append(int(model.jnt_qposadr[jid]))

    np.set_printoptions(precision=6, suppress=True)
    print("MuJoCo:", mujoco.__version__)
    print("Scene:", scene)
    print("Target site:", SITE_NAME)
    print("Method: Jacobian transpose (position only)")
    failures = 0

    for name, reference_q, seed in TESTS:
        # Target obtained from MuJoCo, not the DH function.
        target = site_position(model, data, site_id, addresses, reference_q)
        ik = inverse_position_kinematics(target, seed, gain=args.gain)
        # Independently verify the IK solution using MuJoCo geometry.
        actual = site_position(model, data, site_id, addresses, ik.joint_angles)
        err = float(np.linalg.norm(actual - target))
        passed = ik.success and err <= POSITION_TOLERANCE_M
        failures += not passed

        print(f"\n{name}")
        print("MuJoCo target XYZ [m]:", target)
        print("IK solution q [rad]:   ", np.round(ik.joint_angles, 6))
        print("MuJoCo reached XYZ [m]:", actual)
        print(f"MuJoCo XYZ error [m]:  {err:.3e}")
        print(f"Iterations: {ik.iterations} | {'PASS' if passed else 'FAIL'} | {ik.message}")

    if failures:
        print(f"\n{failures} of {len(TESTS)} MuJoCo IK tests FAILED")
        raise SystemExit(1)
    print(f"\nALL {len(TESTS)} MUJOCO IK TESTS PASSED")


if __name__ == "__main__":
    main()
