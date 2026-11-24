# Milestone 04 — Joint-space trajectory generation

This is the first implementation stage for Milestone 04. It provides a
reusable, synchronized six-joint quintic point-to-point reference for the
existing UR5e.

## Mathematical profile

For normalized time `tau = t / T`, where `0 <= tau <= 1`:

```text
s(tau) = 10 tau^3 - 15 tau^4 + 6 tau^5

s_dot(t)  = (30 tau^2 - 60 tau^3 + 30 tau^4) / T
s_ddot(t) = (60 tau - 180 tau^2 + 120 tau^3) / T^2
```

For six-joint start and goal vectors `q0` and `qf`:

```text
q_desired      = q0 + (qf - q0) s
qdot_desired   = (qf - q0) s_dot
qddot_desired  = (qf - q0) s_ddot
```

The profile starts and ends with zero velocity and acceleration. All joints
share the same scalar time law. After the movement duration, the reference
holds the goal position with zero derivatives.

## Relationship to Milestone 03

The quintic polynomial is reused from the M3 smooth-motion experiment. M3
files are not modified. This M4 implementation makes the reference reusable
and adds input validation, sampling, joint-limit endpoint checks, and
analytical peak velocity/acceleration calculations.

The default example reuses the M3 home pose and conservative joint
displacement so that its reference can be compared with the previous
experiment. This stage generates reference data only; it does not command the
simulator or claim closed-loop tracking.

## Generate a reference CSV

Run from the repository root:

```bash
python3 "Milestone 04/trajectory/generate_joint_space_trajectory.py"
```

Defaults: 6-second movement, 2-second final hold, 20 ms sample period, and
401 samples including both endpoints.

Output:
`Milestone 04/Results/Joint_Space/joint_space_reference.csv`

Custom start and goal positions can be supplied together using `--start` and
`--goal`, each followed by six angles in radians. Duration, hold duration, and
sample period are configurable.

## Plot the reference

```bash
python3 "Milestone 04/trajectory/plot_joint_space_trajectory.py"
```

The script saves position, velocity, and acceleration plots under:
`Milestone 04/Figures/Joint_Space/`

## Run mathematical tests

```bash
python3 -m unittest discover \
  -s "Milestone 04/validation" \
  -p "test_*.py" -v
```

Tests cover endpoint conditions, midpoint values, monotonic progress,
finite-difference agreement of analytical derivatives, final hold behavior,
sampling, analytical peak limits, joint-limit validation, and invalid input.

## Current scope

Implemented: joint-space trajectory reference generation and mathematical
validation.

Not yet implemented: task-space interpolation, inverse kinematics,
collision checking, ROS 2 publishing, motor-level control, feedback tracking,
or surgical instrument attachment. These remain Milestone 04 requirements
and will be implemented and validated in subsequent stages.
