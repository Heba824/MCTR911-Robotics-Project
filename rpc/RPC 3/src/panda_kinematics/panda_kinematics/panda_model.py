"""Static inertial and geometric parameters from the compiled Panda MJCF."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np


DEFAULT_SCENE = Path.home() / 'mujoco_menagerie/franka_emika_panda/scene.xml'
ARM_NAMES = tuple(f'joint{i}' for i in range(1, 8))
FINGER_NAMES = ('finger_joint1', 'finger_joint2')


def named_id(model, kind, name):
    """Resolve a required object by name."""
    import mujoco
    index = mujoco.mj_name2id(model, kind, name)
    if index < 0:
        raise ValueError(f'Panda model is missing {name}')
    return index


def rotation(quaternion):
    """Convert a compiled MJCF wxyz quaternion to a rotation matrix."""
    w, x, y, z = quaternion
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
        [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)],
    ])


@dataclass(frozen=True)
class Body:
    """Rigid body parameters expressed in its parent and local frames."""

    parent: int
    offset: np.ndarray
    rotation: np.ndarray
    mass: float
    com: np.ndarray
    inertia: np.ndarray
    arm_joint: int | None
    axis: np.ndarray | None
    anchor: np.ndarray | None
    reference: float


@dataclass(frozen=True)
class PandaModel:
    """Parameters for seven arm coordinates with fixed home finger opening."""

    bodies: tuple[Body, ...]
    armature: np.ndarray
    gravity: np.ndarray

    @classmethod
    def from_scene(cls, scene=DEFAULT_SCENE):
        """Compile MJCF once to resolve defaults and principal inertia frames."""
        import mujoco

        model = mujoco.MjModel.from_xml_path(str(Path(scene).expanduser()))
        joint_kind = mujoco.mjtObj.mjOBJ_JOINT
        arm = [named_id(model, joint_kind, name) for name in ARM_NAMES]
        fingers = [named_id(model, joint_kind, name) for name in FINGER_NAMES]
        home = named_id(model, mujoco.mjtObj.mjOBJ_KEY, 'home')
        names = [f'link{i}' for i in range(8)] + [
            'hand', 'left_finger', 'right_finger',
        ]
        ids = [named_id(model, mujoco.mjtObj.mjOBJ_BODY, name) for name in names]
        by_body = {bid: i for i, bid in enumerate(ids)}
        bodies = []
        for i, bid in enumerate(ids):
            joint = arm[i - 1] if 1 <= i <= 7 else None
            finger = fingers[i - 9] if i >= 9 else None
            attached = joint if joint is not None else finger
            expected_count = int(attached is not None)
            if (model.body_jntnum[bid] != expected_count
                    or (expected_count and model.body_jntadr[bid] != attached)):
                raise ValueError(f'Unexpected joint layout on {names[i]}')

            parent = int(model.body_parentid[bid])
            if parent != (0 if i == 0 else ids[i - 1] if i <= 8 else ids[8]):
                raise ValueError(f'Unexpected parent of {names[i]}')
            offset = model.body_pos[bid].copy()
            body_rotation = rotation(model.body_quat[bid])
            axis = anchor = None
            reference = 0.0
            if joint is not None:
                if model.jnt_type[joint] != mujoco.mjtJoint.mjJNT_HINGE:
                    raise ValueError('Expected a Panda arm hinge')
                axis = model.jnt_axis[joint].copy()
                anchor = model.jnt_pos[joint].copy()
                reference = float(model.qpos0[model.jnt_qposadr[joint]])
            if finger is not None:
                if model.jnt_type[finger] != mujoco.mjtJoint.mjJNT_SLIDE:
                    raise ValueError('Expected a Panda finger slide')
                address = model.jnt_qposadr[finger]
                slide = model.key_qpos[home, address] - model.qpos0[address]
                offset += body_rotation @ model.jnt_axis[finger] * slide

            inertial_rotation = rotation(model.body_iquat[bid])
            inertia = (inertial_rotation * model.body_inertia[bid]) @ inertial_rotation.T
            bodies.append(Body(
                by_body.get(parent, -1), offset, body_rotation,
                float(model.body_mass[bid]), model.body_ipos[bid].copy(),
                inertia, i - 1 if joint is not None else None, axis, anchor, reference,
            ))
        return cls(tuple(bodies), model.dof_armature[model.jnt_dofadr[arm]].copy(),
                   model.opt.gravity.copy())
