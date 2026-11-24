"""Joint-space quintic trajectory generation for the MCTR911 UR5e project.

This module is independent of ROS 2 and MuJoCo. It generates a reference for
the six UR5e joints that can later be consumed by a simulator tracking node.

For normalized time tau=t/T:
    s(tau) = 10 tau^3 - 15 tau^4 + 6 tau^5
    q_d = q_0 + (q_f-q_0)s
    qdot_d = (q_f-q_0) ds/dt
    qddot_d = (q_f-q_0) d2s/dt2

After T, the reference holds q_f with zero velocity and acceleration.
Units: radians, radians/second, radians/second^2, and seconds.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

JOINT_COUNT = 6


def _six_vector(values: Iterable[float], name: str) -> np.ndarray:
    vector = np.asarray(values, dtype=float)
    if vector.shape != (JOINT_COUNT,):
        raise ValueError(f"{name} must contain exactly {JOINT_COUNT} values; got {vector.shape}")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain only finite values")
    return vector.copy()


def _positive_six_vector(values: float | Iterable[float], name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim == 0:
        array = np.full(JOINT_COUNT, float(array), dtype=float)
    if array.shape != (JOINT_COUNT,):
        raise ValueError(f"{name} must be a positive scalar or six values")
    if not np.all(np.isfinite(array)) or np.any(array <= 0.0):
        raise ValueError(f"{name} values must be positive and finite")
    return array


@dataclass(frozen=True)
class JointTrajectoryPoint:
    """One desired joint-space sample."""
    time_s: float
    progress: float
    position_rad: np.ndarray
    velocity_rad_s: np.ndarray
    acceleration_rad_s2: np.ndarray
    phase: str


@dataclass(frozen=True)
class QuinticJointTrajectory:
    """Synchronized six-joint quintic point-to-point trajectory.

    The class generates a reference only. It does not perform inverse
    kinematics, collision checking, ROS 2 publishing, or actuator control.
    """
    q_start_rad: np.ndarray
    q_goal_rad: np.ndarray
    duration_s: float

    def __post_init__(self) -> None:
        q_start = _six_vector(self.q_start_rad, "q_start_rad")
        q_goal = _six_vector(self.q_goal_rad, "q_goal_rad")
        duration = float(self.duration_s)
        if not np.isfinite(duration) or duration <= 0.0:
            raise ValueError("duration_s must be positive and finite")
        q_start.setflags(write=False)
        q_goal.setflags(write=False)
        object.__setattr__(self, "q_start_rad", q_start)
        object.__setattr__(self, "q_goal_rad", q_goal)
        object.__setattr__(self, "duration_s", duration)

    @property
    def delta_rad(self) -> np.ndarray:
        return self.q_goal_rad - self.q_start_rad

    @property
    def maximum_velocity_rad_s(self) -> np.ndarray:
        return 1.875 * np.abs(self.delta_rad) / self.duration_s

    @property
    def maximum_acceleration_rad_s2(self) -> np.ndarray:
        return (10.0 / np.sqrt(3.0)) * np.abs(self.delta_rad) / self.duration_s**2

    def evaluate(self, time_s: float) -> JointTrajectoryPoint:
        """Evaluate the reference at elapsed time, clamping before/after motion."""
        t = float(time_s)
        if not np.isfinite(t):
            raise ValueError("time_s must be finite")

        if t <= 0.0:
            progress = progress_dot = progress_ddot = 0.0
            phase = "before_start" if t < 0.0 else "start"
        elif t >= self.duration_s:
            progress = 1.0
            progress_dot = progress_ddot = 0.0
            phase = "hold"
        else:
            tau = t / self.duration_s
            tau2, tau3 = tau**2, tau**3
            tau4, tau5 = tau**4, tau**5
            progress = 10.0*tau3 - 15.0*tau4 + 6.0*tau5
            dprogress_dtau = 30.0*tau2 - 60.0*tau3 + 30.0*tau4
            d2progress_dtau2 = 60.0*tau - 180.0*tau2 + 120.0*tau3
            progress_dot = dprogress_dtau / self.duration_s
            progress_ddot = d2progress_dtau2 / self.duration_s**2
            if np.isclose(tau, 0.5, rtol=0.0, atol=1e-12):
                phase = "peak_velocity"
            else:
                phase = "accelerating" if tau < 0.5 else "decelerating"

        delta = self.delta_rad
        return JointTrajectoryPoint(
            time_s=t,
            progress=float(progress),
            position_rad=self.q_start_rad + delta*progress,
            velocity_rad_s=delta*progress_dot,
            acceleration_rad_s2=delta*progress_ddot,
            phase=phase,
        )

    def sample(
        self,
        sample_period_s: float,
        total_duration_s: float | None = None,
    ) -> list[JointTrajectoryPoint]:
        """Sample the movement, optionally followed by a stationary hold."""
        period = float(sample_period_s)
        if not np.isfinite(period) or period <= 0.0:
            raise ValueError("sample_period_s must be positive and finite")
        total = self.duration_s if total_duration_s is None else float(total_duration_s)
        if not np.isfinite(total) or total < self.duration_s:
            raise ValueError("total_duration_s must be finite and >= duration_s")

        count = int(np.floor(total / period))
        times = period * np.arange(count + 1, dtype=float)
        tolerance = 1e-12 * max(1.0, total)
        if total - times[-1] > tolerance:
            times = np.append(times, total)
        else:
            times[-1] = total
        return [self.evaluate(float(t)) for t in times]

    def validate_joint_limits(
        self,
        lower_limits_rad: Iterable[float],
        upper_limits_rad: Iterable[float],
    ) -> None:
        """Raise ValueError if either endpoint violates the six joint limits.

        Since the quintic scalar progress is monotonic, each joint position
        stays between its start and goal values throughout this trajectory.
        """
        lower = _six_vector(lower_limits_rad, "lower_limits_rad")
        upper = _six_vector(upper_limits_rad, "upper_limits_rad")
        if np.any(lower >= upper):
            raise ValueError("Every lower joint limit must be below its upper limit")
        for label, q in (("start", self.q_start_rad), ("goal", self.q_goal_rad)):
            invalid = np.flatnonzero((q < lower) | (q > upper))
            if invalid.size:
                details = ", ".join(
                    f"joint {i+1}: {q[i]:.6f} rad not in [{lower[i]:.6f}, {upper[i]:.6f}]"
                    for i in invalid
                )
                raise ValueError(f"{label} configuration violates joint limits: {details}")

    @staticmethod
    def minimum_duration_for_limits(
        q_start_rad: Iterable[float],
        q_goal_rad: Iterable[float],
        max_velocity_rad_s: float | Iterable[float],
        max_acceleration_rad_s2: float | Iterable[float],
    ) -> float:
        """Minimum T satisfying all per-joint velocity and acceleration limits.

        For this profile, max|ds/dt|=1.875/T and
        max|d2s/dt2|=(10/sqrt(3))/T^2.
        """
        q_start = _six_vector(q_start_rad, "q_start_rad")
        q_goal = _six_vector(q_goal_rad, "q_goal_rad")
        vmax = _positive_six_vector(max_velocity_rad_s, "max_velocity_rad_s")
        amax = _positive_six_vector(max_acceleration_rad_s2, "max_acceleration_rad_s2")
        delta = np.abs(q_goal - q_start)
        duration_v = np.max(1.875 * delta / vmax)
        duration_a = np.max(np.sqrt((10.0 / np.sqrt(3.0)) * delta / amax))
        return float(max(duration_v, duration_a))
