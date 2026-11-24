"""Validated CSV loading and C2 quintic-Hermite interpolation for M4 joint references.

The trajectory CSV stores q, qdot, and qddot at each sample. Between two
samples, this module reconstructs a quintic polynomial matching all six
boundary conditions (position, velocity, acceleration at both ends). This
preserves a continuous position, velocity, and acceleration reference when a
ROS 2 controller runs at a rate different from the CSV sampling rate.

This module is independent of ROS 2 and MuJoCo.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

JOINT_COUNT = 6


def _validate_joint_names(joint_names: Iterable[str]) -> tuple[str, ...]:
    names = tuple(str(name) for name in joint_names)
    if len(names) != JOINT_COUNT:
        raise ValueError(f"Exactly {JOINT_COUNT} joint names are required")
    if any(not name.strip() for name in names):
        raise ValueError("Joint names must be non-empty")
    if len(set(names)) != JOINT_COUNT:
        raise ValueError("Joint names must be unique")
    return names


@dataclass(frozen=True)
class JointReferencePoint:
    """Desired joint state evaluated at one reference time."""
    time_s: float
    phase: str
    position_rad: np.ndarray
    velocity_rad_s: np.ndarray
    acceleration_rad_s2: np.ndarray


class JointTrajectoryCSV:
    """Load and interpolate a six-joint reference CSV.

    Required columns follow the existing M4 generator naming convention:
      time_s
      q_desired_<joint_name>_rad
      qdot_desired_<joint_name>_rad_s
      qddot_desired_<joint_name>_rad_s2

    Each interval is reconstructed with a quintic polynomial in normalized
    interval time s in [0, 1]. The polynomial matches q, qdot, and qddot at
    both endpoints, so q, qdot, and qddot remain continuous at CSV knots.
    """

    def __init__(
        self,
        *,
        source_path: Path,
        joint_names: tuple[str, ...],
        times_s: np.ndarray,
        positions_rad: np.ndarray,
        velocities_rad_s: np.ndarray,
        accelerations_rad_s2: np.ndarray,
        phases: tuple[str, ...],
    ) -> None:
        self.source_path = Path(source_path).expanduser().resolve()
        self.joint_names = _validate_joint_names(joint_names)

        self.times_s = np.asarray(times_s, dtype=float).copy()
        self.positions_rad = np.asarray(positions_rad, dtype=float).copy()
        self.velocities_rad_s = np.asarray(velocities_rad_s, dtype=float).copy()
        self.accelerations_rad_s2 = np.asarray(accelerations_rad_s2, dtype=float).copy()
        self.phases = tuple(str(phase) for phase in phases)

        count = self.times_s.size
        expected_shape = (count, JOINT_COUNT)
        if count < 2:
            raise ValueError("The trajectory CSV must contain at least two samples")
        for name, array in (
            ("positions_rad", self.positions_rad),
            ("velocities_rad_s", self.velocities_rad_s),
            ("accelerations_rad_s2", self.accelerations_rad_s2),
        ):
            if array.shape != expected_shape:
                raise ValueError(f"{name} must have shape {expected_shape}; got {array.shape}")
            if not np.all(np.isfinite(array)):
                raise ValueError(f"{name} contains non-finite values")
        if self.times_s.shape != (count,) or not np.all(np.isfinite(self.times_s)):
            raise ValueError("time_s must contain one finite value per sample")
        if len(self.phases) != count:
            raise ValueError("phase must contain one value per sample")
        if np.any(np.diff(self.times_s) <= 0.0):
            raise ValueError("time_s values must be strictly increasing")
        if self.times_s[0] < 0.0 or not np.isclose(
            self.times_s[0], 0.0, rtol=0.0, atol=1e-9
        ):
            raise ValueError("The first trajectory sample must start at time_s = 0")

        self._interval_durations_s = np.diff(self.times_s)
        self._coefficients = self._build_quintic_coefficients()

        for array in (
            self.times_s,
            self.positions_rad,
            self.velocities_rad_s,
            self.accelerations_rad_s2,
            self._interval_durations_s,
            self._coefficients,
        ):
            array.setflags(write=False)

    @classmethod
    def load(
        cls,
        csv_path: str | Path,
        joint_names: Iterable[str],
    ) -> "JointTrajectoryCSV":
        """Load a CSV and validate its required columns and numeric content."""
        path = Path(csv_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Trajectory CSV does not exist: {path}")

        names = _validate_joint_names(joint_names)
        with path.open("r", newline="", encoding="utf-8-sig") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames is None:
                raise ValueError("Trajectory CSV is empty or has no header")

            required = ["time_s"]
            for joint_name in names:
                required.extend(
                    (
                        f"q_desired_{joint_name}_rad",
                        f"qdot_desired_{joint_name}_rad_s",
                        f"qddot_desired_{joint_name}_rad_s2",
                    )
                )
            missing = [column for column in required if column not in reader.fieldnames]
            if missing:
                raise ValueError(
                    "Trajectory CSV is missing required columns: " + ", ".join(missing)
                )

            times: list[float] = []
            positions: list[list[float]] = []
            velocities: list[list[float]] = []
            accelerations: list[list[float]] = []
            phases: list[str] = []

            for row_number, row in enumerate(reader, start=2):
                try:
                    times.append(float(row["time_s"]))
                    positions.append(
                        [float(row[f"q_desired_{name}_rad"]) for name in names]
                    )
                    velocities.append(
                        [float(row[f"qdot_desired_{name}_rad_s"]) for name in names]
                    )
                    accelerations.append(
                        [float(row[f"qddot_desired_{name}_rad_s2"]) for name in names]
                    )
                    phases.append((row.get("phase") or "unspecified").strip() or "unspecified")
                except (TypeError, ValueError, KeyError) as exc:
                    raise ValueError(
                        f"Invalid numeric or missing value in trajectory CSV row {row_number}"
                    ) from exc

        if not times:
            raise ValueError("Trajectory CSV contains no data rows")

        return cls(
            source_path=path,
            joint_names=names,
            times_s=np.asarray(times, dtype=float),
            positions_rad=np.asarray(positions, dtype=float),
            velocities_rad_s=np.asarray(velocities, dtype=float),
            accelerations_rad_s2=np.asarray(accelerations, dtype=float),
            phases=tuple(phases),
        )

    @property
    def start_position_rad(self) -> np.ndarray:
        return self.positions_rad[0].copy()

    @property
    def goal_position_rad(self) -> np.ndarray:
        return self.positions_rad[-1].copy()

    @property
    def start_time_s(self) -> float:
        return float(self.times_s[0])

    @property
    def end_time_s(self) -> float:
        return float(self.times_s[-1])

    @property
    def sample_count(self) -> int:
        return int(self.times_s.size)

    def _build_quintic_coefficients(self) -> np.ndarray:
        """Return coefficients [interval, joint, polynomial degree]."""
        h = self._interval_durations_s[:, None]
        q0 = self.positions_rad[:-1]
        q1 = self.positions_rad[1:]
        v0 = self.velocities_rad_s[:-1]
        v1 = self.velocities_rad_s[1:]
        a0 = self.accelerations_rad_s2[:-1]
        a1 = self.accelerations_rad_s2[1:]

        c0 = q0
        c1 = h * v0
        c2 = 0.5 * h**2 * a0

        delta_position = q1 - (c0 + c1 + c2)
        delta_velocity = h * v1 - (c1 + 2.0 * c2)
        delta_acceleration = h**2 * a1 - 2.0 * c2

        c3 = 10.0 * delta_position - 4.0 * delta_velocity + 0.5 * delta_acceleration
        c4 = -15.0 * delta_position + 7.0 * delta_velocity - delta_acceleration
        c5 = 6.0 * delta_position - 3.0 * delta_velocity + 0.5 * delta_acceleration

        return np.stack((c0, c1, c2, c3, c4, c5), axis=2)

    def evaluate(self, time_s: float) -> JointReferencePoint:
        """Evaluate the reference with C2 quintic-Hermite interpolation."""
        t = float(time_s)
        if not np.isfinite(t):
            raise ValueError("time_s must be finite")

        if t <= self.start_time_s:
            phase = "before_start" if t < self.start_time_s else self.phases[0]
            return JointReferencePoint(
                time_s=t,
                phase=phase,
                position_rad=self.positions_rad[0].copy(),
                velocity_rad_s=self.velocities_rad_s[0].copy(),
                acceleration_rad_s2=self.accelerations_rad_s2[0].copy(),
            )

        if t >= self.end_time_s:
            phase = "after_end" if t > self.end_time_s else self.phases[-1]
            return JointReferencePoint(
                time_s=t,
                phase=phase,
                position_rad=self.positions_rad[-1].copy(),
                velocity_rad_s=self.velocities_rad_s[-1].copy(),
                acceleration_rad_s2=self.accelerations_rad_s2[-1].copy(),
            )

        interval = int(np.searchsorted(self.times_s, t, side="right") - 1)
        h = float(self._interval_durations_s[interval])
        s = (t - float(self.times_s[interval])) / h
        coefficients = self._coefficients[interval]

        powers = np.array((1.0, s, s**2, s**3, s**4, s**5), dtype=float)
        first_derivative = np.array(
            (0.0, 1.0, 2.0*s, 3.0*s**2, 4.0*s**3, 5.0*s**4),
            dtype=float,
        )
        second_derivative = np.array(
            (0.0, 0.0, 2.0, 6.0*s, 12.0*s**2, 20.0*s**3),
            dtype=float,
        )

        position = coefficients @ powers
        velocity = (coefficients @ first_derivative) / h
        acceleration = (coefficients @ second_derivative) / (h**2)

        return JointReferencePoint(
            time_s=t,
            phase=self.phases[interval],
            position_rad=position,
            velocity_rad_s=velocity,
            acceleration_rad_s2=acceleration,
        )
