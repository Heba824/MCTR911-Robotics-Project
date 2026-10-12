"""Validate the UR5e DH forward kinematics against the original MuJoCo MJCF.

Run in the MCTR911 Milestone 02/kinematics directory:
    python validate_kinematics.py

By default, loads Milestone 01/Robot Model/scene.xml in the same repository.
Use --scene to supply a different scene.xml if the folders have moved.

The test sets MuJoCo joint positions and calls mj_forward(). It does not step
physics, drive the motors, or require a MuJoCo graphics window or ROS 2.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
import numpy as np

from ur5e_dh_fk import JOINT_NAMES, forward_kinematics

SITE_NAME = "attachment_site"
POSITION_TOLERANCE_M = 1e-5
ORIENTATION_TOLERANCE_RAD = 1e-4

# Six angles in the exact order defined in JOINT_NAMES. All values in radians.
TEST_POSES = (
    ("zero",             (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)),
    ("home",             (-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0.0)),
    ("configuration_A",  (0.3, -0.4, 0.2, 0.1, 0.5, 0.6)),
    ("configuration_B",  (-0.6, 0.2, -0.7, 0.4, -0.2, -0.5)),
    ("configuration_C",  (1.0, -0.8, 1.1, -0.6, 0.5, -0.2)),
)


def orientation_error_rad(r_dh: np.ndarray, r_mj: np.ndarray) -> float:
    """Geodesic angular error between rotation matrices, without SciPy."""
    r_rel = r_mj.T @ r_dh
    cosine = float(np.clip((np.trace(r_rel) - 1.0) / 2.0, -1.0, 1.0))
    skew_vee = np.array([
        r_rel[2, 1] - r_rel[1, 2],
        r_rel[0, 2] - r_rel[2, 0],
        r_rel[1, 0] - r_rel[0, 1],
    ])
    sine = float(np.linalg.norm(skew_vee) / 2.0)
    return float(np.arctan2(sine, cosine))


def main() -> None:
    default_scene = (
        Path(__file__).resolve().parents[2]
        / "Milestone 01" / "Robot Model" / "scene.xml"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, default=default_scene,
                        help="Path to the MuJoCo scene.xml (default: sibling Milestone 01 scene)")
    args = parser.parse_args()
    scene_path = args.scene.expanduser().resolve()
    if not scene_path.is_file():
        parser.error(f"Scene file not found: {scene_path}")

    print("MuJoCo version:", mujoco.__version__)
    print("Scene:", scene_path)
    print("Testing site:", SITE_NAME)
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    data = mujoco.MjData(model)

    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, SITE_NAME)
    if site_id == -1:
        raise RuntimeError(f"MJCF model has no site named {SITE_NAME!r}")

    qpos_addresses = []
    for name in JOINT_NAMES:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id == -1:
            raise RuntimeError(f"MJCF model has no joint named {name!r}")
        if model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_HINGE:
            raise RuntimeError(f"Expected a hinge joint for {name!r}")
        qpos_addresses.append(int(model.jnt_qposadr[joint_id]))

    np.set_printoptions(precision=6, suppress=True)
    print("\n{:<18} {:>18} {:>21}  {}".format(
        "Configuration", "Position error (m)", "Orientation error (deg)", "Result"
    ))
    print("-" * 73)

    failures = 0
    for name, angles in TEST_POSES:
        q = np.asarray(angles, dtype=float)
        mujoco.mj_resetData(model, data)
        for qpos_address, angle in zip(qpos_addresses, q):
            data.qpos[qpos_address] = angle
        # Important: update kinematics without advancing simulation dynamics.
        mujoco.mj_forward(model, data)

        dh_transform = forward_kinematics(q)
        mj_position = data.site_xpos[site_id].copy()
        mj_rotation = data.site_xmat[site_id].reshape(3, 3).copy()

        pos_error = float(np.linalg.norm(dh_transform[:3, 3] - mj_position))
        rot_error = orientation_error_rad(dh_transform[:3, :3], mj_rotation)
        passed = pos_error < POSITION_TOLERANCE_M and rot_error < ORIENTATION_TOLERANCE_RAD
        failures += not passed
        print(f"{name:<18} {pos_error:>18.3e} {np.degrees(rot_error):>21.3e}  {'PASS' if passed else 'FAIL'}")
        if name in ("zero", "home"):
            print("  DH position:    ", dh_transform[:3, 3])
            print("  MuJoCo position:", mj_position)

    print("\n" + ("ALL TESTS PASSED" if failures == 0 else f"FAILED: {failures} test(s)"))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
