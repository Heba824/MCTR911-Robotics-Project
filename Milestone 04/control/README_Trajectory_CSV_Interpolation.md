# MCTR911 Milestone 04 — Stage 4C.1: Runtime trajectory reference

## Purpose

The joint-space trajectory generator already creates a CSV sampled at 20 ms.
The ROS 2 controller will run at a separately configured rate (initially
100 Hz). The controller therefore needs a reliable way to evaluate the
reference at arbitrary elapsed times rather than assuming its timer callback
coincides with CSV sample times.

`trajectory_csv.py` loads the existing M4 CSV format and reconstructs each
interval with a quintic Hermite polynomial.

## Why quintic Hermite interpolation?

Each CSV interval contains the boundary values of:

- Joint position `q`
- Joint velocity `qdot`
- Joint acceleration `qddot`

A fifth-order polynomial has six coefficients per joint. These are determined
by matching position, velocity, and acceleration at both interval endpoints.
The resulting reference is C2-continuous across sample boundaries.

For normalized interval time `s = (t - t_i) / h`, the polynomial is:

`q(s) = c0 + c1*s + c2*s^2 + c3*s^3 + c4*s^4 + c5*s^5`

The implementation also evaluates its first and second time derivatives,
including the interval-duration scaling.

This preserves the generated reference's position, velocity, and acceleration
without requiring the ROS 2 timer to run at the CSV sampling rate.

## Files

- `Milestone 04/control/trajectory_csv.py`
- `Milestone 04/validation/test_trajectory_csv.py`

No existing milestone source, reference CSV, robot model, or ROS 2 bridge is
modified by this stage.

## Validation

Run the focused tests:

```bash
python -m unittest discover \
  -s "Milestone 04/validation" \
  -p "test_trajectory_csv.py" -v
```

The tests check the supplied reference CSV, all stored sample knots, agreement
with the original analytical quintic trajectory, endpoint clamping, malformed
columns, invalid joint names, duplicate timestamps, and non-finite values.

Run the complete M4 validation suite:

```bash
python -m unittest discover \
  -s "Milestone 04/validation" \
  -p "test_*.py" -v
```

## Scope boundary

This stage only loads and evaluates reference data. It does not publish ROS 2
commands, run MuJoCo, or claim closed-loop tracking. Those are implemented and
validated separately in the next stage.
