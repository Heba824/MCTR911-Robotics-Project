"""Numerical verification of the Panda Lagrangian dynamics."""

import os

import numpy as np
import pytest

from panda_kinematics.dynamics import PandaDynamics
from panda_kinematics.panda_model import DEFAULT_SCENE, PandaModel
from panda_kinematics.validate_dynamics import validate


SCENE = os.environ.get('RPC3_SCENE', DEFAULT_SCENE)
POSES = [
    np.array([0, 0, 0, -1.57079, 0, 1.57079, -0.7853]),
    np.array([0.4, -0.2, 0.7, -1.1, 0.2, 2.5, 0.6]),
]


@pytest.fixture(scope='module')
def dynamics():
    return PandaDynamics(PandaModel.from_scene(SCENE))


@pytest.mark.parametrize('q', POSES)
def test_com_jacobians_and_gravity(dynamics, q):
    step = 1e-6
    offsets = step * np.eye(7)
    plus = [dynamics.com_kinematics(q + offset) for offset in offsets]
    minus = [dynamics.com_kinematics(q - offset) for offset in offsets]
    for idx, (_, orientation, jv, jw) in enumerate(dynamics.com_kinematics(q)):
        numeric_jv, numeric_jw = [], []
        for high, low in zip(plus, minus):
            numeric_jv.append((high[idx][0] - low[idx][0]) / (2 * step))
            skew = ((high[idx][1] - low[idx][1]) / (2 * step)) @ orientation.T
            numeric_jw.append([skew[2, 1], skew[0, 2], skew[1, 0]])
        np.testing.assert_allclose(jv, np.asarray(numeric_jv).T, atol=1e-8)
        np.testing.assert_allclose(jw, np.asarray(numeric_jw).T, atol=1e-8)
    gradient = [(dynamics.potential_energy(q + offset)
                 - dynamics.potential_energy(q - offset)) / (2 * step)
                for offset in offsets]
    np.testing.assert_allclose(dynamics.gravity(q), gradient, atol=1e-7)


@pytest.mark.parametrize('q', POSES)
def test_mass_and_coriolis(dynamics, q):
    mass, coriolis, gravity = dynamics.terms(q, np.zeros(7))
    assert mass.shape == (7, 7)
    assert gravity.shape == (7,)
    np.testing.assert_allclose(mass, mass.T, atol=1e-13, rtol=0)
    assert np.all(np.linalg.eigvalsh(mass) > 0)
    np.testing.assert_array_equal(coriolis, np.zeros(7))

    dq = np.linspace(-0.5, 0.7, 7)
    _, coriolis, _ = dynamics.terms(q, dq)
    step = 1e-6
    mdot = (dynamics.mass_matrix(q + step*dq)
            - dynamics.mass_matrix(q - step*dq)) / (2*step)
    np.testing.assert_allclose(dq @ coriolis, 0.5*dq @ mdot @ dq, atol=1e-8)


def test_invalid_state(dynamics):
    with pytest.raises(ValueError, match='q must contain seven finite values'):
        dynamics.mass_matrix(np.zeros(6))
    with pytest.raises(ValueError, match='dq must contain seven finite values'):
        dynamics.terms(POSES[0], np.full(7, np.nan))


def test_inverse_dynamics_against_mujoco():
    report = validate(SCENE, seed=911, samples=200)
    assert report['samples'] == 200
    for metric, stats in report['results'].items():
        assert stats['passed'], (metric, stats)
