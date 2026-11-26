"""Seven-channel Mamdani inference with a shared normalized rule base."""

import numpy as np

from .control_timing import record_timing, timing_start


LABELS = ('LN', 'MN', 'S', 'MP', 'LP')
CENTERS = np.linspace(-1, 1, 5)
# Rows are position error, columns are velocity error, both in LABELS order.
RULES = np.array([
    [0, 0, 0, 1, 2],
    [0, 0, 1, 2, 3],
    [0, 1, 2, 3, 4],
    [1, 2, 3, 4, 4],
    [2, 3, 4, 4, 4],
])


def memberships(values):
    """Overlapping triangles and outer shoulders on the normalized domain."""
    return np.maximum(0, 1 - 2*np.abs(np.clip(values, -1, 1)[..., None] - CENTERS))


GRID = np.linspace(-1, 1, 201)
OUTPUT_MEMBERSHIPS = memberships(GRID).T
AREA_WEIGHTS = np.ones(GRID.size)
AREA_WEIGHTS[[0, -1]] = 0.5
# Adjacent output sets are the only ones with nonzero support at any grid point.
OUTPUT_LABELS = np.argsort(OUTPUT_MEMBERSHIPS, axis=0)[-2:]
OUTPUT_HEIGHTS = np.take_along_axis(OUTPUT_MEMBERSHIPS, OUTPUT_LABELS, axis=0)
RULE_MASKS = tuple(RULES == i for i in range(5))


class JointFLC:
    """Scale position and velocity errors and return bounded torques."""

    channels = 7

    def __init__(self, position_scale, velocity_scale, torque_scale):
        self.position_scale, self.velocity_scale, self.torque_scale = (
            np.array(values, dtype=float, copy=True)
            for values in (position_scale, velocity_scale, torque_scale)
        )
        for scale in (self.position_scale, self.velocity_scale, self.torque_scale):
            if scale.shape != (self.channels,) or not np.all(np.isfinite(scale) & (scale > 0)):
                raise ValueError(f'Each scale must contain {self.channels} positive finite values')

    def __call__(self, error, velocity_error, *, profile=None):
        start = timing_start() if profile is not None else None
        error, velocity_error = np.asarray(error), np.asarray(velocity_error)
        for values in (error, velocity_error):
            if values.shape != (self.channels,) or not np.all(np.isfinite(values)):
                raise ValueError(f'Each error must contain {self.channels} finite values')
        position = memberships(error / self.position_scale)
        velocity = memberships(velocity_error / self.velocity_scale)
        if profile is not None:
            start = record_timing(profile, 'fuzzy_memberships', start)
        firing = np.minimum(position[:, :, None], velocity[:, None, :])
        activation = np.stack([firing[:, mask].max(axis=1) for mask in RULE_MASKS], axis=1)
        if profile is not None:
            start = record_timing(profile, 'fuzzy_rules', start)
        aggregate = np.minimum(activation[:, OUTPUT_LABELS], OUTPUT_HEIGHTS).max(axis=1)
        if profile is not None:
            start = record_timing(profile, 'fuzzy_aggregation', start)
        weights = aggregate * AREA_WEIGHTS
        # Integrate the first moment of the piecewise-linear sampled density.
        # The endpoint correction removes trapezoidal error for x * density.
        moment = np.einsum('ij,j->i', weights, GRID) - (GRID[1] - GRID[0])/6 * (aggregate[:, -1] - aggregate[:, 0])
        result = self.torque_scale * (moment / weights.sum(axis=1))
        if profile is not None:
            record_timing(profile, 'fuzzy_centroid', start)
        return result
