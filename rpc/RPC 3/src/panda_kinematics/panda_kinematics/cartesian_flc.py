"""Team 1 XYZ velocity FLC and independent inverse-dynamics torque control."""

import numpy as np

from .control_timing import record_timing, timing_start
from .fuzzy import JointFLC
from .joint_control import JointController, LIMITS, TorqueCommand
from .trajectory_node import Q_MAX, Q_MIN, compute_jacobian, forward_kinematics


TASK_SPEED = 0.18  # m/s, Euclidean bound
JOINT_SPEED = 0.65  # rad/s, per joint
JOINT_ACCEL = 4.0  # rad/s², per joint
NULL_SPEED = 0.20  # rad/s, Euclidean bound
LIMIT_MARGIN = 0.12  # rad
SINGULAR_THRESHOLD = 0.08  # m/rad
MAX_DAMPING = 0.10  # m/rad
CARTESIAN_FIELDS = (
    'cartesian_error_x', 'cartesian_error_y', 'cartesian_error_z',
    'cartesian_velocity_error_x', 'cartesian_velocity_error_y', 'cartesian_velocity_error_z',
    'cartesian_command_x', 'cartesian_command_y', 'cartesian_command_z',
    'jacobian_sigma_1', 'jacobian_sigma_2', 'jacobian_sigma_3', 'jacobian_damping',
    *(f'cartesian_joint_speed_{i}' for i in range(1, 8)),
    'cartesian_speed_limited', 'cartesian_joint_speed_limited',
    'cartesian_accel_limited', 'cartesian_torque_limited',
)


class CartesianVelocityFLC(JointFLC):
    """Three independent Mamdani channels whose output scales are in m/s."""

    channels = 3


def _bounded_norm(vector, limit):
    return vector * min(1.0, limit / max(limit, np.linalg.norm(vector)))


def damped_inverse(jacobian):
    """Return the adaptive DLS inverse, exact redundant-space projector and spectrum."""
    u, sigma, vt = np.linalg.svd(jacobian, full_matrices=False)
    proximity = np.clip(1.0 - sigma[-1]/SINGULAR_THRESHOLD, 0.0, 1.0)
    damping = MAX_DAMPING * proximity**2
    inverse = (vt.T * (sigma / (sigma**2 + damping**2))) @ u.T
    return inverse, np.eye(7) - vt.T @ vt, sigma, float(damping)


def joint_velocity(q, q_des, dq_des, jacobian, velocity):
    """Map task velocity with bounded posture bias and braking near joint limits."""
    inverse, null, sigma, damping = damped_inverse(jacobian)
    low = np.maximum(q - Q_MIN, 0.)
    high = np.maximum(Q_MAX - q, 0.)
    repulsion = np.clip(1 - low/LIMIT_MARGIN, 0, 1)**2
    repulsion -= np.clip(1 - high/LIMIT_MARGIN, 0, 1)**2
    posture = dq_des + 2*(q_des - q) + 0.6*repulsion
    requested = inverse @ velocity + _bounded_norm(null @ posture, NULL_SPEED)
    lower = np.minimum(JOINT_SPEED, np.sqrt(2*JOINT_ACCEL*np.maximum(low - 0.015, 0)))
    upper = np.minimum(JOINT_SPEED, np.sqrt(2*JOINT_ACCEL*np.maximum(high - 0.015, 0)))
    # Clip individual outward components so one joint at a limit cannot freeze the arm.
    bounded = np.clip(requested, -lower, upper)
    return bounded, sigma, damping, int(np.count_nonzero(bounded != requested))


class CartesianController:
    """Map XYZ fuzzy velocities to acceleration targets and bounded arm torques."""

    mode = 'cartesian_flc'

    def __init__(self, dynamics, fuzzy=None):
        self.dynamics = dynamics
        self.fuzzy = fuzzy if fuzzy is not None else CartesianVelocityFLC(
            [0.035]*3, [0.24]*3, [0.14]*3,
        )
        self.terminal = JointController(dynamics, 'model_only')

    def compute(self, q, dq, q_des, dq_des, ddq_des, *, terminal_hold=False, profile=None):
        if terminal_hold:
            return self.terminal.compute(q, dq, q_des, dq_des, ddq_des,
                                         terminal_hold=True, profile=profile)
        q, dq, q_des, dq_des, ddq_des = (np.asarray(v, dtype=float)
                                         for v in (q, dq, q_des, dq_des, ddq_des))
        if any(v.shape != (7,) or not np.all(np.isfinite(v))
               for v in (q, dq, q_des, dq_des, ddq_des)):
            raise ValueError('Cartesian control requires finite seven-joint states and references')
        start = timing_start() if profile is not None else None
        jacobian = compute_jacobian(q)
        desired_velocity = compute_jacobian(q_des) @ dq_des
        error = forward_kinematics(q_des)[:3, 3] - forward_kinematics(q)[:3, 3]
        rate = desired_velocity - jacobian @ dq
        if profile is not None:
            start = record_timing(profile, 'reference_and_model_numpy', start)
        requested_velocity = desired_velocity + self.fuzzy(error, rate, profile=profile)
        if profile is not None:
            start = record_timing(profile, 'correction', start)
        velocity = _bounded_norm(requested_velocity, TASK_SPEED)
        speeds, sigma, damping, speed_limited = joint_velocity(q, q_des, dq_des, jacobian, velocity)
        desired_accel = ddq_des + 18*(speeds - dq)
        accel = np.clip(desired_accel, -JOINT_ACCEL, JOINT_ACCEL)
        if profile is not None:
            start = record_timing(profile, 'cartesian_mapping', start)
        mass, coriolis, gravity = self.dynamics.terms(q, dq)
        if profile is not None:
            start = record_timing(profile, 'dynamics', start)
        model = mass @ ddq_des + coriolis + gravity
        requested = mass @ accel + coriolis + gravity
        if not np.all(np.isfinite(requested)):
            raise ValueError('Nonfinite Cartesian control torque')
        bounded = np.clip(requested, -LIMITS, LIMITS)
        if profile is not None:
            record_timing(profile, 'torque_limits_numpy', start)
            profile.update(zip(CARTESIAN_FIELDS, (
                *error, *rate, *velocity, *sigma, damping, *speeds,
                int(np.any(velocity != requested_velocity)), speed_limited,
                int(np.count_nonzero(accel != desired_accel)), int(np.count_nonzero(bounded != requested)),
            )))
        return TorqueCommand(model, requested - model, requested, bounded)
