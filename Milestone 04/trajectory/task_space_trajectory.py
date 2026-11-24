"""World-frame Cartesian pose trajectory generation for the MCTR911 UR5e.

The position follows a straight line with quintic time scaling. Orientation
follows a shortest-path, fixed-world-axis rotation interpolation:

    R_d(t) = Exp([phi_world]x * s(t)) R_start
    phi_world = Log(R_goal R_start.T)

Because the rotation axis is fixed in the world frame:
    omega_d_world = phi_world * s_dot
    alpha_d_world = phi_world * s_ddot

The six-dimensional ordering matches Milestone 03:
    twist      = [vx, vy, vz, wx, wy, wz]
    acceleration = [ax, ay, az, alphax, alphay, alphaz]

This module generates references only. It does not solve inverse kinematics,
check collision constraints, publish ROS 2 commands, or control actuators.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np


def _finite_vector(values: Iterable[float], size: int, name: str) -> np.ndarray:
    vector = np.asarray(values, dtype=float)
    if vector.shape != (size,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain exactly {size} finite values")
    return vector.copy()


def _validate_pose(pose: np.ndarray, name: str) -> np.ndarray:
    transform = np.asarray(pose, dtype=float)
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        raise ValueError(f"{name} must be a finite 4x4 homogeneous transform")
    if not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=1e-9, rtol=0.0):
        raise ValueError(f"{name} has an invalid homogeneous bottom row")
    rotation = transform[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-7, rtol=0.0):
        raise ValueError(f"{name} rotation matrix is not orthonormal")
    determinant = float(np.linalg.det(rotation))
    if not np.isclose(determinant, 1.0, atol=1e-7, rtol=0.0):
        raise ValueError(f"{name} rotation matrix must have determinant +1")
    return transform.copy()


def _skew(vector: np.ndarray) -> np.ndarray:
    x, y, z = vector
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=float)


def so3_exp(rotation_vector: Iterable[float]) -> np.ndarray:
    """SO(3) exponential map: rotation vector [rad] -> 3x3 rotation matrix."""
    phi = _finite_vector(rotation_vector, 3, "rotation_vector")
    theta = float(np.linalg.norm(phi))
    phi_hat = _skew(phi)
    if theta < 1e-8:
        # Second-order series avoids cancellation for very small rotations.
        return np.eye(3) + phi_hat + 0.5 * (phi_hat @ phi_hat)
    a = np.sin(theta) / theta
    b = (1.0 - np.cos(theta)) / (theta * theta)
    return np.eye(3) + a * phi_hat + b * (phi_hat @ phi_hat)


def so3_log(rotation: np.ndarray) -> np.ndarray:
    """SO(3) logarithm map: rotation matrix -> shortest rotation vector [rad]."""
    matrix = np.asarray(rotation, dtype=float)
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        raise ValueError("rotation must be a finite 3x3 matrix")
    if not np.allclose(matrix.T @ matrix, np.eye(3), atol=1e-7, rtol=0.0):
        raise ValueError("rotation must be orthonormal")
    if not np.isclose(np.linalg.det(matrix), 1.0, atol=1e-7, rtol=0.0):
        raise ValueError("rotation must have determinant +1")

    cosine = float(np.clip((np.trace(matrix) - 1.0) / 2.0, -1.0, 1.0))
    theta = float(np.arccos(cosine))
    skew_vector = np.array([
        matrix[2, 1] - matrix[1, 2],
        matrix[0, 2] - matrix[2, 0],
        matrix[1, 0] - matrix[0, 1],
    ], dtype=float)

    if theta < 1e-8:
        return 0.5 * skew_vector

    if np.pi - theta < 1e-5:
        # At pi the skew part vanishes. The eigenvector of the symmetric
        # rotation matrix associated with eigenvalue +1 is the rotation axis.
        eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (matrix + matrix.T))
        axis = eigenvectors[:, int(np.argmax(eigenvalues))]
        axis /= np.linalg.norm(axis)
        if np.linalg.norm(skew_vector) > 1e-8:
            if float(np.dot(axis, skew_vector)) < 0.0:
                axis = -axis
        else:
            # Deterministic sign convention for an exact pi rotation.
            pivot = int(np.argmax(np.abs(axis)))
            if axis[pivot] < 0.0:
                axis = -axis
        return theta * axis

    axis = skew_vector / (2.0 * np.sin(theta))
    return theta * axis


def rotation_matrix_to_quaternion_xyzw(rotation: np.ndarray) -> np.ndarray:
    """Convert a proper rotation matrix to a normalized ROS-order [x,y,z,w]."""
    matrix = np.asarray(rotation, dtype=float)
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        raise ValueError("rotation must be a finite 3x3 matrix")
    if not np.allclose(matrix.T @ matrix, np.eye(3), atol=1e-7, rtol=0.0):
        raise ValueError("rotation must be orthonormal")
    if not np.isclose(np.linalg.det(matrix), 1.0, atol=1e-7, rtol=0.0):
        raise ValueError("rotation must have determinant +1")

    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = 2.0 * np.sqrt(trace + 1.0)
        w = 0.25 * scale
        x = (matrix[2, 1] - matrix[1, 2]) / scale
        y = (matrix[0, 2] - matrix[2, 0]) / scale
        z = (matrix[1, 0] - matrix[0, 1]) / scale
    else:
        diagonal_index = int(np.argmax(np.diag(matrix)))
        if diagonal_index == 0:
            scale = 2.0 * np.sqrt(max(0.0, 1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]))
            x = 0.25 * scale
            y = (matrix[0, 1] + matrix[1, 0]) / scale
            z = (matrix[0, 2] + matrix[2, 0]) / scale
            w = (matrix[2, 1] - matrix[1, 2]) / scale
        elif diagonal_index == 1:
            scale = 2.0 * np.sqrt(max(0.0, 1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]))
            x = (matrix[0, 1] + matrix[1, 0]) / scale
            y = 0.25 * scale
            z = (matrix[1, 2] + matrix[2, 1]) / scale
            w = (matrix[0, 2] - matrix[2, 0]) / scale
        else:
            scale = 2.0 * np.sqrt(max(0.0, 1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]))
            x = (matrix[0, 2] + matrix[2, 0]) / scale
            y = (matrix[1, 2] + matrix[2, 1]) / scale
            z = 0.25 * scale
            w = (matrix[1, 0] - matrix[0, 1]) / scale

    quaternion = np.array([x, y, z, w], dtype=float)
    norm = float(np.linalg.norm(quaternion))
    if norm < 1e-12:
        raise ValueError("Could not convert rotation matrix to quaternion")
    quaternion /= norm
    if quaternion[3] < 0.0:
        quaternion = -quaternion
    return quaternion


@dataclass(frozen=True)
class TaskSpaceTrajectoryPoint:
    """One desired world-frame end-effector pose/twist/acceleration sample."""
    time_s: float
    progress: float
    pose_world: np.ndarray
    position_world_m: np.ndarray
    quaternion_xyzw: np.ndarray
    linear_velocity_world_m_s: np.ndarray
    angular_velocity_world_rad_s: np.ndarray
    linear_acceleration_world_m_s2: np.ndarray
    angular_acceleration_world_rad_s2: np.ndarray
    phase: str

    @property
    def twist_world(self) -> np.ndarray:
        """Milestone 03 ordering [vx, vy, vz, wx, wy, wz]."""
        return np.concatenate((
            self.linear_velocity_world_m_s,
            self.angular_velocity_world_rad_s,
        ))

    @property
    def acceleration_world(self) -> np.ndarray:
        """Milestone 03 ordering [ax, ay, az, alphax, alphay, alphaz]."""
        return np.concatenate((
            self.linear_acceleration_world_m_s2,
            self.angular_acceleration_world_rad_s2,
        ))


@dataclass(frozen=True)
class QuinticTaskSpaceTrajectory:
    """Straight-line translation + shortest fixed-axis orientation trajectory."""
    start_pose_world: np.ndarray
    goal_pose_world: np.ndarray
    duration_s: float
    _rotation_vector_world: np.ndarray = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        start = _validate_pose(self.start_pose_world, "start_pose_world")
        goal = _validate_pose(self.goal_pose_world, "goal_pose_world")
        duration = float(self.duration_s)
        if not np.isfinite(duration) or duration <= 0.0:
            raise ValueError("duration_s must be positive and finite")

        rotation_delta_world = goal[:3, :3] @ start[:3, :3].T
        rotation_vector = so3_log(rotation_delta_world)
        start.setflags(write=False)
        goal.setflags(write=False)
        rotation_vector.setflags(write=False)
        object.__setattr__(self, "start_pose_world", start)
        object.__setattr__(self, "goal_pose_world", goal)
        object.__setattr__(self, "duration_s", duration)
        object.__setattr__(self, "_rotation_vector_world", rotation_vector)

    @property
    def translation_delta_world_m(self) -> np.ndarray:
        return self.goal_pose_world[:3, 3] - self.start_pose_world[:3, 3]

    @property
    def rotation_vector_world_rad(self) -> np.ndarray:
        return self._rotation_vector_world.copy()

    @property
    def rotation_angle_rad(self) -> float:
        return float(np.linalg.norm(self._rotation_vector_world))

    def evaluate(self, time_s: float) -> TaskSpaceTrajectoryPoint:
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
            tau4 = tau**4
            progress = 10.0*tau3 - 15.0*tau4 + 6.0*tau4*tau
            dprogress_dtau = 30.0*tau2 - 60.0*tau3 + 30.0*tau4
            d2progress_dtau2 = 60.0*tau - 180.0*tau2 + 120.0*tau3
            progress_dot = dprogress_dtau / self.duration_s
            progress_ddot = d2progress_dtau2 / self.duration_s**2
            phase = "peak_velocity" if np.isclose(tau, 0.5, atol=1e-12, rtol=0.0) else (
                "accelerating" if tau < 0.5 else "decelerating"
            )

        position_start = self.start_pose_world[:3, 3]
        translation_delta = self.translation_delta_world_m
        position = position_start + translation_delta * progress
        rotation = so3_exp(self._rotation_vector_world * progress) @ self.start_pose_world[:3, :3]

        pose = np.eye(4)
        pose[:3, :3] = rotation
        pose[:3, 3] = position

        linear_velocity = translation_delta * progress_dot
        angular_velocity = self._rotation_vector_world * progress_dot
        linear_acceleration = translation_delta * progress_ddot
        angular_acceleration = self._rotation_vector_world * progress_ddot

        return TaskSpaceTrajectoryPoint(
            time_s=t,
            progress=float(progress),
            pose_world=pose,
            position_world_m=position,
            quaternion_xyzw=rotation_matrix_to_quaternion_xyzw(rotation),
            linear_velocity_world_m_s=linear_velocity,
            angular_velocity_world_rad_s=angular_velocity,
            linear_acceleration_world_m_s2=linear_acceleration,
            angular_acceleration_world_rad_s2=angular_acceleration,
            phase=phase,
        )

    def sample(
        self,
        sample_period_s: float,
        total_duration_s: float | None = None,
    ) -> list[TaskSpaceTrajectoryPoint]:
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
