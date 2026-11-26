#!/usr/bin/env python3
"""Validate the template's FK against independent XML-derived transforms.

Runs using just Python 3 + NumPy, without importing ROS 2.
Example: python3 verify_fk.py

This checks equivalence to the exact fixed transforms documented in the
assignment's pinned MuJoCo Menagerie panda.xml. It is NOT a live MuJoCo test.
"""
import ast
from pathlib import Path
import numpy as np


def load_math_from_template():
    """Load only math functions/constants, skipping ROS 2 imports/node."""
    path = Path(__file__).with_name("kinematics.py")
    module = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names = {"DH_PARAMS", "T_L7_HAND", "Q_MIN", "Q_MAX", "TARGET_POSE", "_hand_angle"}
    funcs = {"dh_transform", "forward_kinematics"}
    selected = []
    for node in module.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id in names
            for target in node.targets
        ):
            selected.append(node)
        if isinstance(node, ast.FunctionDef) and node.name in funcs:
            selected.append(node)
    clean_module = ast.Module(body=selected, type_ignores=[])
    scope = {"np": np}
    exec(compile(clean_module, str(path), "exec"), scope)
    return scope


def translation(x, y, z):
    T = np.eye(4)
    T[:3, 3] = [x, y, z]
    return T


def rot_x(angle):
    c, s = np.cos(angle), np.sin(angle)
    T = np.eye(4)
    T[:3, :3] = [[1, 0, 0], [0, c, -s], [0, s, c]]
    return T


def rot_z(angle):
    c, s = np.cos(angle), np.sin(angle)
    T = np.eye(4)
    T[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
    return T


def xml_reference_fk(q):
    """Chain MuJoCo body poses and revolute joints, with NO DH matrices."""
    fixed_body_transforms = [
        translation(0, 0, 0.333),
        rot_x(-np.pi / 2),
        translation(0, -0.316, 0) @ rot_x(np.pi / 2),
        translation(0.0825, 0, 0) @ rot_x(np.pi / 2),
        translation(-0.0825, 0.384, 0) @ rot_x(-np.pi / 2),
        rot_x(np.pi / 2),
        translation(0.088, 0, 0) @ rot_x(np.pi / 2),
    ]
    T = np.eye(4)
    for body_pose, qi in zip(fixed_body_transforms, q):
        T = T @ body_pose @ rot_z(qi)
    return T @ translation(0, 0, 0.107) @ rot_z(-np.pi / 4)


def rotation_angle_error(R1, R2):
    R_diff = R1 @ R2.T
    return np.arccos(np.clip((np.trace(R_diff) - 1) / 2, -1, 1))


def main():
    math = load_math_from_template()
    fk = math["forward_kinematics"]
    lower, upper = math["Q_MIN"], math["Q_MAX"]
    q_home = np.array([0, 0, 0, -1.57079, 0, 1.57079, -0.7853])

    random = np.random.default_rng(1234)
    samples = [q_home, np.zeros(7)]
    samples += [random.uniform(lower, upper) for _ in range(100)]

    max_pos_err = 0.0
    max_rot_err = 0.0
    for i, q in enumerate(samples):
        computed = fk(q)
        expected = xml_reference_fk(q)
        pos_err = np.linalg.norm(computed[:3, 3] - expected[:3, 3])
        rot_err = rotation_angle_error(computed[:3, :3], expected[:3, :3])
        max_pos_err = max(max_pos_err, pos_err)
        max_rot_err = max(max_rot_err, rot_err)
        np.testing.assert_allclose(computed, expected, atol=1e-10, rtol=0,
                                   err_msg=f"Mismatch at sample {i}")

    print(f"PASS: {len(samples)} FK poses match XML-derived transforms")
    print(f"Max position discrepancy: {max_pos_err:.3e} m")
    print(f"Max rotation discrepancy: {max_rot_err:.3e} rad")
    print("Home-configuration hand pose:")
    print(np.round(fk(q_home), 6))
    print("\nNext: run verify_ik.py and compare the pose with live MuJoCo.")


if __name__ == "__main__":
    main()
