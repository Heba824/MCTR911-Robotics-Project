"""Tests of membership, centroid and Stage 3 torque composition."""

import numpy as np
import pytest

from panda_kinematics.fuzzy import CENTERS, JointFLC, memberships
from panda_kinematics.joint_control import JointController, LIMITS


def test_membership_boundaries():
    np.testing.assert_array_equal(memberships(CENTERS), np.eye(5))
    np.testing.assert_array_equal(memberships([-5, 5]), np.eye(5)[[0, 4]])
    np.testing.assert_allclose(memberships([0.25]), [[0, 0, 0.5, 0.5, 0]])


def test_rules_centroid_and_saturation():
    fuzzy = JointFLC([1]*7, [1]*7, np.arange(1, 8))
    one, zero = np.ones(7), np.zeros(7)
    positive = fuzzy(one, zero)
    assert positive.shape == (7,)
    np.testing.assert_allclose(positive, np.arange(1, 8)*5/6, atol=4e-5)
    np.testing.assert_allclose(fuzzy(-one, zero), -positive, atol=1e-14)
    np.testing.assert_allclose(fuzzy(one, -one), zero, atol=1e-14)
    np.testing.assert_allclose(fuzzy(100*one, 100*one), positive)
    assert np.all(abs(positive) <= np.arange(1, 8))
    with pytest.raises(ValueError):
        fuzzy(np.full(7, np.nan), zero)


def test_torque_modes_and_limits():
    class Dynamics:
        def terms(self, q, dq):
            return np.eye(7), np.ones(7), 2*np.ones(7)

    zero, one = np.zeros(7), np.ones(7)
    only = JointController(Dynamics(), 'model_only')
    fuzzy = JointController(Dynamics(), 'joint_flc')
    np.testing.assert_array_equal(only.compute(zero, zero, one, one, one).correction, zero)
    assert np.all(fuzzy.compute(zero, zero, one, one, one).correction > 0)
    np.testing.assert_array_equal(only.compute(zero, zero, one, one, 100*one).command, LIMITS)
    np.testing.assert_array_equal(
        only.compute(zero, zero, one, zero, zero, terminal_hold=True).command,
        fuzzy.compute(zero, zero, one, zero, zero, terminal_hold=True).command,
    )


def test_sparse_aggregation_matches_dense_centroid():
    from panda_kinematics.fuzzy import AREA_WEIGHTS, GRID, OUTPUT_MEMBERSHIPS, RULE_MASKS

    rng = np.random.default_rng(911)
    fuzzy = JointFLC([1]*7, [1]*7, [1]*7)
    for _ in range(30):
        error, rate = rng.uniform(-1.5, 1.5, (2, 7))
        firing = np.minimum(memberships(error)[:, :, None], memberships(rate)[:, None, :])
        activation = np.stack([firing[:, mask].max(axis=1) for mask in RULE_MASKS], axis=1)
        aggregate = np.minimum(activation[:, :, None], OUTPUT_MEMBERSHIPS).max(axis=1)
        weights = aggregate * AREA_WEIGHTS
        moment = weights @ GRID - (GRID[1] - GRID[0])/6 * (aggregate[:, -1] - aggregate[:, 0])
        profile = {}
        np.testing.assert_allclose(fuzzy(error, rate, profile=profile),
                                   moment / weights.sum(axis=1), atol=1e-14, rtol=0)
        for phase in ('memberships', 'rules', 'aggregation', 'centroid'):
            assert profile[f'fuzzy_{phase}_ms'] >= 0
            assert profile[f'fuzzy_{phase}_cpu_ms'] >= 0
