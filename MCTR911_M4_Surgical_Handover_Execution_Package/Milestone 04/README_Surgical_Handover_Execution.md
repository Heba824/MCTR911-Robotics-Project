# Milestone 04 — Surgical instrument handover execution

This addition provides a closed-loop ROS 2 execution node for the existing
nine-waypoint scalpel/forceps handover planner. It reuses the project's
`QuinticJointTrajectory` and `motor_level_controller` implementations and the
MuJoCo bridge attachment services.

## Safety behavior

- The program is motion-inert unless `--execute` is supplied.
- It requires a plan JSON and the matching collision-preflight JSON, and rejects a
  preflight artifact whose file timestamp predates the plan.
- The preflight must report MuJoCo native CCD enabled, zero collision events,
  zero joint-limit violations, no pickup-support clearance violation, and all
  reported clearances at or above the hard 20 mm minimum.
- A `WARNING_REVIEW_REQUIRED` report is not silently treated as a pass. The
  operator must inspect the printed close-pair list and explicitly supply
  `--approve-warnings` before execution. The flag cannot override any hard
  failure or clearance below 20 mm.
- Live `/joint_state` and localization feedback are checked before movement.
- Each segment uses the same sequential quintic point-to-point profile and
  duration as the collision preflight. The robot must settle at every waypoint.
- The instrument is attached only after live pickup-site alignment is checked.
  It is released only after both the tool-site pose and reconstructed
  instrument-body pose match the handover target.
- If execution aborts while carrying an instrument, the node holds the latest
  measured robot pose and deliberately does **not** release the instrument.
- A measured-state online collision guard uses the same MuJoCo native-CCD
  geometry selection as the offline preflight and aborts below the hard 20 mm
  clearance threshold or on penetration.
- Joint tracking error, stale feedback, actuator saturation, pickup alignment,
  handover pose, release stability, and final HOME pose are checked and logged.

This is a MuJoCo simulation workflow, not a safety-rated controller and not
authorization to operate a physical robot.

## Files added

- `Milestone 04/application/surgical_handover_execution_core.py`
- `Milestone 04/application/execute_surgical_handover.py`
- `Milestone 04/validation/test_surgical_handover_execution.py`
- `Milestone 04/README_Surgical_Handover_Execution.md`

## Offline tests

From the project root:

```bash
source ~/mctr911_kinematics_venv/bin/activate
source /opt/ros/jazzy/setup.bash

python "Milestone 04/validation/test_surgical_handover_execution.py"
```

These tests do not start ROS, publish commands, or move the simulation.

## Plan and collision preflight

Keep the MuJoCo handover scene running in one terminal. In another terminal:

```bash
cd ~/mctr911_project/MCTR911-Robotics-Project-Fresh
source ~/mctr911_kinematics_venv/bin/activate
source /opt/ros/jazzy/setup.bash
source ~/mctr911_ws/install/setup.bash

python "Milestone 04/application/plan_surgical_handover.py" --instrument scalpel
```

Use the exact plan JSON path printed by the planner:

```bash
python "Milestone 04/application/surgical_handover_collision_preflight.py" \
  --instrument scalpel \
  --plan-json "Milestone 04/Results/Surgical_Handover/Planning/PLAN_FILE.json"
```

Use the exact JSON path printed by the collision preflight in the next commands.

## Validate execution artifacts without moving the robot

```bash
python "Milestone 04/application/execute_surgical_handover.py" \
  --instrument scalpel \
  --plan-json "Milestone 04/Results/Surgical_Handover/Planning/PLAN_FILE.json" \
  --preflight-json "Milestone 04/Results/Surgical_Handover/Collision_Preflight/PREFLIGHT_FILE.json"
```

The default is dry-run. It validates the plan and preflight and publishes no ROS
joint commands.

## Execute the simulation

Only after the dry-run reports that the artifacts are consistent, and after
reviewing any close-pair warnings printed by the preflight/executor, run:

```bash
python "Milestone 04/application/execute_surgical_handover.py" \
  --instrument scalpel \
  --plan-json "Milestone 04/Results/Surgical_Handover/Planning/PLAN_FILE.json" \
  --preflight-json "Milestone 04/Results/Surgical_Handover/Collision_Preflight/PREFLIGHT_FILE.json" \
  --strategy inverse_dynamics \
  --execute
```

If and only if the matching preflight status is `WARNING_REVIEW_REQUIRED`,
add `--approve-warnings` after reviewing the reported pairs. This approval does
not override collisions, joint-limit violations, pickup-support violations, or
clearance below 20 mm. A `FAIL_*` report is always blocked.

The controller publishes to `/joint_commands` and uses `/joint_state`, the
`/mujoco/*_pickup_pose`, `/mujoco/handover_target_pose`, and
`/mujoco/attachment_site_pose` topics, plus `/mujoco/attach_scalpel` or
`/mujoco/attach_forceps` (`std_srvs/srv/SetBool`). It does not use the older
`/m4_joint_trajectory/start` service.

Execution CSV and JSON metrics are written under:

`Milestone 04/Results/Surgical_Handover/Execution/`
