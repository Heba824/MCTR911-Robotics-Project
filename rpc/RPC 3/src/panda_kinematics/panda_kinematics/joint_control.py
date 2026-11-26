"""Stage 3 joint tracking law shared by ROS and numerical tests."""

from dataclasses import dataclass
from resource import RUSAGE_THREAD, getrusage
from threading import get_native_id

import numpy as np

from .control_timing import record_timing, timing_start
from .fuzzy import JointFLC


LIMITS = np.array([87, 87, 87, 87, 12, 12, 12])
POSITION_SCALE = [0.08] * 7
VELOCITY_SCALE = [0.35] * 7
TORQUE_SCALE = [10, 16, 10, 12, 3, 3, 2]


@dataclass(frozen=True)
class TorqueCommand:
    model: np.ndarray
    correction: np.ndarray
    requested: np.ndarray
    command: np.ndarray


def warm_controller(controller, trajectory):
    """Exercise rest, moving and terminal paths with no ROS interfaces."""
    times, q, dq, ddq = trajectory
    samples = []
    for index in (0, 1, len(times)//4, len(times)//2, len(times) - 2, len(times) - 1):
        for terminal in (False, True) if index == len(times) - 1 else (False,):
            profile = {'worker_native_id': get_native_id()}
            start = timing_start()
            controller.compute(q[index], dq[index], q[index], dq[index], ddq[index],
                               terminal_hold=terminal, profile=profile)
            record_timing(profile, 'total', start)
            samples.append(profile)
    return samples


class JointController:
    """Add optional fuzzy correction to the independent model torque."""

    def __init__(self, dynamics, mode, fuzzy=None):
        if mode not in ('model_only', 'joint_flc'):
            raise ValueError(f'Unknown Stage 3 control mode {mode!r}')
        self.dynamics = dynamics
        self.mode = mode
        self.fuzzy = (JointFLC(POSITION_SCALE, VELOCITY_SCALE, TORQUE_SCALE)
                      if fuzzy is None else fuzzy)

    def compute(self, q, dq, q_des, dq_des, ddq_des, *, terminal_hold=False, profile=None):
        start = timing_start() if profile is not None else None
        mass, coriolis, gravity = self.dynamics.terms(q, dq)
        if profile is not None:
            start = record_timing(profile, 'dynamics', start)
        q_des, dq_des, ddq_des = (np.asarray(values, dtype=float)
                                  for values in (q_des, dq_des, ddq_des))
        for desired in (q_des, dq_des, ddq_des):
            if desired.shape != (7,) or not np.all(np.isfinite(desired)):
                raise ValueError('Reference must contain seven finite values')
        model = mass @ ddq_des + coriolis + gravity
        if profile is not None:
            start = record_timing(profile, 'reference_and_model_numpy', start)
        if terminal_hold:
            correction = np.clip(12*(q_des - q) + 4*(dq_des - dq), -5, 5)
        elif self.mode == 'joint_flc':
            usage = getrusage(RUSAGE_THREAD) if profile is not None else None
            correction = self.fuzzy(q_des - q, dq_des - dq, profile=profile)
            if profile is not None:
                after = getrusage(RUSAGE_THREAD)
                profile.update(
                    correction_voluntary_switches=after.ru_nvcsw - usage.ru_nvcsw,
                    correction_involuntary_switches=after.ru_nivcsw - usage.ru_nivcsw,
                    correction_minor_faults=after.ru_minflt - usage.ru_minflt,
                    correction_major_faults=after.ru_majflt - usage.ru_majflt,
                )
        else:
            correction = np.zeros(7)
        if profile is not None:
            start = record_timing(profile, 'correction', start)
        requested = model + correction
        if not np.all(np.isfinite(requested)):
            raise ValueError('Nonfinite requested torque')
        command = np.clip(requested, -LIMITS, LIMITS)
        if profile is not None:
            record_timing(profile, 'torque_limits_numpy', start)
        return TorqueCommand(model, correction, requested, command)
