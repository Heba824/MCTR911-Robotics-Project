# MCTR911 Milestone 04 — Surgical handover Step 2D

This update addresses the first collision-preflight findings without changing the original UR5e model, the original surgical scene, the MuJoCo bridge, or existing experiment results.

## Changes

1. Adds an elevated `approach_clearance` waypoint at world position `(0.10, 0.20, 0.80)` before the existing tray approach. This is a candidate route only; it is not assumed collision-free until the MuJoCo preflight passes.
2. Reads the selected instrument body's initial pose from the loaded MuJoCo scene and computes the full `tool_from_instrument` transform.
3. Treats the handover target as the desired instrument-body pose and computes the compensated `attachment_site` target as `T_world_instrument_target * inverse(T_tool_instrument)`.
4. Extends collision preflight to validate the new waypoint sequence.
5. Explicitly permits the instrument's initial support contact with `tray_base` and its selected pad only during `pickup` and `lift`, until the instrument achieves the configured minimum clearance. Other tray geometry and all robot-to-surgeon checks remain enabled.
6. Records support-contact allowance and clearance timing in the preflight CSV/JSON.

## Replacement destinations

- `Milestone 04/application/surgical_handover_planner.py`
- `Milestone 04/application/plan_surgical_handover.py`
- `Milestone 04/application/surgical_handover_collision_preflight.py`
- `Milestone 04/validation/test_surgical_handover_step2d.py`

## Important

The elevated waypoint is only a candidate. Run IK planning and then the full collision preflight for each instrument. Do not execute motion unless the preflight result is `PASS`. This is sampled simulation geometry checking, not continuous collision detection or real-hardware safety certification.
