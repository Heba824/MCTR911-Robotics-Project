"""Compare the independent Lagrangian calculation with MuJoCo inverse dynamics."""

import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

from .dynamics import PandaDynamics
from .panda_model import ARM_NAMES, DEFAULT_SCENE, FINGER_NAMES, PandaModel, named_id


THRESHOLDS = {'mass': 1e-7, 'bias': 1e-3, 'inverse': 1e-3}


def _statistics(errors):
    axes = (0, 2) if errors.ndim == 3 else 0
    return {
        'maximum': float(errors.max()),
        'mean': float(errors.mean()),
        'joint_maximum': errors.max(axis=axes).tolist(),
        'joint_mean': errors.mean(axis=axes).tolist(),
    }


def validate(scene=DEFAULT_SCENE, seed=911, samples=200):
    """Return seeded absolute error statistics and acceptance outcomes."""
    if samples < 1:
        raise ValueError('samples must be positive')
    scene = Path(scene).expanduser().resolve()
    dynamics = PandaDynamics(PandaModel.from_scene(scene))
    model = mujoco.MjModel.from_xml_path(str(scene))
    disable = mujoco.mjtDisableBit
    model.opt.disableflags |= int(disable.mjDSBL_CONSTRAINT | disable.mjDSBL_ACTUATION)
    kind = mujoco.mjtObj.mjOBJ_JOINT
    arm = [named_id(model, kind, name) for name in ARM_NAMES]
    fingers = [named_id(model, kind, name) for name in FINGER_NAMES]
    q_index = model.jnt_qposadr[arm]
    v_index = model.jnt_dofadr[arm]
    finger_q = model.jnt_qposadr[fingers]
    home = named_id(model, mujoco.mjtObj.mjOBJ_KEY, 'home')
    lower, upper = model.jnt_range[arm].T
    data = mujoco.MjData(model)
    data.qpos[finger_q] = model.key_qpos[home, finger_q]
    full_mass = np.empty((model.nv, model.nv))
    rng = np.random.default_rng(seed)
    errors = {metric: [] for metric in THRESHOLDS}
    for _ in range(samples):
        q = rng.uniform(lower + 0.05, upper - 0.05)
        dq = rng.uniform(-1.0, 1.0, 7)
        ddq = rng.uniform(-2.0, 2.0, 7)
        data.qpos[q_index], data.qvel[v_index], data.qacc[v_index] = q, dq, ddq
        mujoco.mj_inverse(model, data)
        mujoco.mj_fullM(model, full_mass, data.qM)
        mass, coriolis, gravity = dynamics.terms(q, dq)
        bias = coriolis + gravity
        # Inverse dynamics subtracts passive forces from the inertial and bias terms.
        reference = (data.qfrc_inverse + data.qfrc_passive)[v_index]
        errors['mass'].append(abs(mass - full_mass[np.ix_(v_index, v_index)]))
        errors['bias'].append(abs(bias - data.qfrc_bias[v_index]))
        errors['inverse'].append(abs(mass @ ddq + bias - reference))

    results = {metric: _statistics(np.asarray(values)) for metric, values in errors.items()}
    for metric, stats in results.items():
        stats['threshold'] = THRESHOLDS[metric]
        stats['passed'] = stats['maximum'] <= stats['threshold']
    return {
        'seed': seed, 'samples': samples, 'scene': str(scene),
        'scene_sha256': hashlib.sha256(scene.read_bytes()).hexdigest(),
        'panda_xml_sha256': hashlib.sha256((scene.parent / 'panda.xml').read_bytes()).hexdigest(),
        'mujoco_version': mujoco.__version__, 'numpy_version': np.__version__,
        'results': results,
    }


def main():
    """Print the validation report and signal acceptance with the exit code."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene', type=Path, default=DEFAULT_SCENE)
    parser.add_argument('--seed', type=int, default=911)
    parser.add_argument('--samples', type=int, default=200)
    args = parser.parse_args()
    report = validate(args.scene, args.seed, args.samples)
    print(json.dumps(report, indent=2))
    return int(not all(stats['passed'] for stats in report['results'].values()))


if __name__ == '__main__':
    raise SystemExit(main())
