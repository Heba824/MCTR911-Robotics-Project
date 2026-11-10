"""MuJoCo validation of UR5e forward/inverse ACCELERATION kinematics.

Checks five configurations at prescribed q, qdot, and qddot using three
independent comparisons:
  1. Our analytic Jdot vs finite differences of the MuJoCo mj_jacSite.
  2. Our analytic acceleration vs MuJoCo J @ qddot + Jdot_fd @ qdot.
  3. Our acceleration vs second differences of MuJoCo SITE poses along
       q(t) = q0 + qdot*t + 0.5*qddot*t^2.
     The rotation second derivative gives world angular acceleration via
       skew(alpha) = skew_part(Rddot @ R0.T).
  4. Inverse-acceleration reconstruction using the MuJoCo Jacobians.

These are kinematic state tests; they do NOT simulate motor/PD tracking and
are not a validation of inertial dynamics or actuator torque.

Usage from repository root (with the user's existing WSL venv):
    python 'Milestone 03/validation/validate_acceleration_mujoco.py' \\
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

from ur5e_acceleration_kinematics import (  # noqa: E402
    JOINT_NAMES,
    forward_acceleration,
    inverse_acceleration,
    jacobian_time_derivative,
)

SITE_NAME = "attachment_site"
JDOT_TOLERANCE = 5e-5              # per matrix entry, m/(rad*s) or 1/s
FORWARD_TOLERANCE = 5e-6           # m/s^2 or rad/s^2
POSE_FD_TOLERANCE = 5e-5           # m/s^2 or rad/s^2
INVERSE_TOLERANCE = 5e-5           # m/s^2 or rad/s^2
JDOT_DT = 2e-4                    # s for central difference of MuJoCo Jacobian
POSE_DT = 1e-3                    # s for central difference of MuJoCo site pose

# Every row is (case name, q [rad], qdot [rad/s], qddot [rad/s^2]).
TEST_CASES = (
    ("zero", (0., 0., 0., 0., 0., 0.),
     (0.10, -0.05, 0.12, -0.06, 0.09, 0.03),
     (0.05, -0.04, 0.02, 0.03, 0.06, -0.01)),
    ("home", (-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0.),
     (0.08, 0.11, -0.07, 0.09, -0.12, 0.05),
     (-0.02, 0.08, -0.04, 0.02, 0.03, 0.01)),
    ("configuration_A", (0.3, -0.4, 0.2, 0.1, 0.5, 0.6),
     (0.12, -0.08, 0.15, -0.05, 0.09, 0.07),
     (0.04, -0.06, 0.03, 0.05, -0.02, 0.07)),
    ("configuration_B", (-0.6, 0.2, -0.7, 0.4, -0.2, -0.5),
     (-0.11, 0.06, -0.09, 0.12, -0.05, 0.08),
     (0.03, -0.08, 0.06, 0.04, -0.05, 0.02)),
    ("configuration_C", (1.0, -0.8, 1.1, -0.6, 0.5, -0.2),
     (0.07, -0.12, 0.09, 0.04, 0.11, -0.06),
     (-0.04, 0.05, 0.08, -0.02, 0.06, -0.03)),
)


def _set_pose(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    qpos_addresses: list[int],
    q: np.ndarray,
) -> None:
    for address, angle in zip(qpos_addresses, q):
        data.qpos[address] = float(angle)
    mujoco.mj_forward(model, data)


def _site_jacobian(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    site_id: int,
    qpos_addresses: list[int],
    qvel_addresses: list[int],
    q: np.ndarray,
) -> np.ndarray:
    _set_pose(model, data, qpos_addresses, q)
    jac_linear = np.zeros((3, model.nv), dtype=float)
    jac_angular = np.zeros((3, model.nv), dtype=float)
    mujoco.mj_jacSite(model, data, jac_linear, jac_angular, site_id)
    return np.vstack((jac_linear[:, qvel_addresses], jac_angular[:, qvel_addresses]))


def _site_pose(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    site_id: int,
    qpos_addresses: list[int],
    q: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    _set_pose(model, data, qpos_addresses, q)
    p = data.site_xpos[site_id].copy()
    r = data.site_xmat[site_id].reshape(3, 3).copy()
    return p, r


def _site_acceleration_via_pose(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    site_id: int,
    qpos_addresses: list[int],
    q: np.ndarray,
    qdot: np.ndarray,
    qddot: np.ndarray,
) -> np.ndarray:
    """Independent second time difference of MuJoCo position AND rotation.

    A rotation matrix satisfies:
        Rddot R.T = skew(alpha) + skew(omega)^2.
    Since skew(omega)^2 is symmetric, antisymmetrizing Rddot R.T yields
    angular acceleration alpha in the world frame.
    """
    h = POSE_DT
    p0, r0 = _site_pose(model, data, site_id, qpos_addresses, q)
    p_plus, r_plus = _site_pose(
        model, data, site_id, qpos_addresses, q + h * qdot + 0.5 * h**2 * qddot
    )
    p_minus, r_minus = _site_pose(
        model, data, site_id, qpos_addresses, q - h * qdot + 0.5 * h**2 * qddot
    )
    a = (p_plus - 2.0 * p0 + p_minus) / h**2
    r_ddot = (r_plus - 2.0 * r0 + r_minus) / h**2
    alpha_matrix = r_ddot @ r0.T
    alpha_skew = 0.5 * (alpha_matrix - alpha_matrix.T)
    alpha = np.array([alpha_skew[2, 1], alpha_skew[0, 2], alpha_skew[1, 0]])
    return np.r_[a, alpha]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scene", type=Path,
        default=_ROOT / "Milestone 01" / "Robot Model" / "scene_handover.xml",
    )
    args = parser.parse_args()
    scene = args.scene.expanduser().resolve()
    if not scene.is_file():
        parser.error(f"Scene file does not exist: {scene}")

    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, SITE_NAME)
    if site_id < 0:
        raise RuntimeError(f"MuJoCo site '{SITE_NAME}' is missing")

    qpos_addresses: list[int] = []
    qvel_addresses: list[int] = []
    for joint_name in JOINT_NAMES:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id < 0:
            raise RuntimeError(f"MuJoCo joint '{joint_name}' is missing")
        if model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_HINGE:
            raise RuntimeError(f"Expected hinge joint: {joint_name}")
        qpos_addresses.append(int(model.jnt_qposadr[joint_id]))
        qvel_addresses.append(int(model.jnt_dofadr[joint_id]))

    print("MuJoCo:", mujoco.__version__)
    print("Scene:", scene)
    print("Site:", SITE_NAME)
    print("World-frame acceleration order: [ax, ay, az, alphax, alphay, alphaz]")
    print(f"Tolerance Jdot:          {JDOT_TOLERANCE:.1e} per entry")
    print(f"Tolerance forward:       {FORWARD_TOLERANCE:.1e} m/s^2 or rad/s^2")
    print(f"Tolerance pose FD:       {POSE_FD_TOLERANCE:.1e} m/s^2 or rad/s^2")
    print(f"Tolerance inverse:       {INVERSE_TOLERANCE:.1e} m/s^2 or rad/s^2")
    print(f"{'Configuration':<18} {'Jdot err':>11} {'MJ accel err':>15} {'Pose FD err':>15} {'Inverse err':>14} {'Result':>8}")
    print("-" * 90)

    failures = 0
    for name, q_values, qdot_values, qddot_values in TEST_CASES:
        q = np.asarray(q_values, dtype=float)
        qdot = np.asarray(qdot_values, dtype=float)
        qddot = np.asarray(qddot_values, dtype=float)
        mujoco.mj_resetData(model, data)

        mj_j = _site_jacobian(
            model, data, site_id, qpos_addresses, qvel_addresses, q
        )
        mj_j_plus = _site_jacobian(
            model, data, site_id, qpos_addresses, qvel_addresses, q + JDOT_DT * qdot
        )
        mj_j_minus = _site_jacobian(
            model, data, site_id, qpos_addresses, qvel_addresses, q - JDOT_DT * qdot
        )
        mj_jdot_fd = (mj_j_plus - mj_j_minus) / (2.0 * JDOT_DT)
        mj_accel = mj_j @ qddot + mj_jdot_fd @ qdot

        analytic_jdot = jacobian_time_derivative(q, qdot)
        py_accel = forward_acceleration(q, qdot, qddot)
        pose_accel = _site_acceleration_via_pose(
            model, data, site_id, qpos_addresses, q, qdot, qddot
        )

        recovered_qddot = inverse_acceleration(q, qdot, mj_accel)
        reconstructed_mj_accel = mj_j @ recovered_qddot + mj_jdot_fd @ qdot

        err_jdot = float(np.max(np.abs(analytic_jdot - mj_jdot_fd)))
        err_forward = float(np.max(np.abs(py_accel - mj_accel)))
        err_pose = float(np.max(np.abs(py_accel - pose_accel)))
        err_inverse = float(np.max(np.abs(reconstructed_mj_accel - mj_accel)))
        passed = (
            err_jdot < JDOT_TOLERANCE
            and err_forward < FORWARD_TOLERANCE
            and err_pose < POSE_FD_TOLERANCE
            and err_inverse < INVERSE_TOLERANCE
        )
        failures += not passed
        print(
            f"{name:<18} {err_jdot:>11.3e} {err_forward:>15.3e} "
            f"{err_pose:>15.3e} {err_inverse:>14.3e} {'PASS' if passed else 'FAIL':>8}"
        )
        if not passed:
            print("  Python acceleration: ", np.array2string(py_accel, precision=8))
            print("  MuJoCo acceleration: ", np.array2string(mj_accel, precision=8))
            print("  Site pose FD accel:  ", np.array2string(pose_accel, precision=8))
            print("  Inverse reconstruction:", np.array2string(reconstructed_mj_accel, precision=8))

    if failures:
        print(f"\nFAILED: {failures} of {len(TEST_CASES)} acceleration test configurations")
        raise SystemExit(1)
    print("\nALL 5 MUJOCO ACCELERATION TESTS PASSED")
    print("NOTE: Kinematic state/pose validation, not actuator acceleration tracking.")


if __name__ == "__main__":
    main()
