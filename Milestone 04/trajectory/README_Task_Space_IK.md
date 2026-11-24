# Milestone 04 — Stage 3: full-pose IK for the task-space trajectory

This stage converts the Milestone 04 Cartesian end-effector reference into a
joint-space trajectory under the project's validated UR5e model.

## What this stage implements

For each desired task-space sample, the implementation:

1. Solves **full-pose inverse kinematics** for the desired homogeneous pose.
2. Uses the solved joint state to compute **inverse velocity kinematics** from
   the desired world-frame twist.
3. Uses the solved joint state and joint velocity to compute **inverse
   acceleration kinematics** from the desired world-frame twist derivative.
4. Records the recovered joint trajectory plus reconstruction diagnostics.

## Preserved project conventions

- End-effector: `attachment_site`
- Pose frame: world frame
- Jacobian ordering: `[vx, vy, vz, wx, wy, wz]`
- Orientation error: `Log(R_d R^T)` in the **world frame**
- Joint order: unchanged from Milestone 02 and 03
- Existing Milestone 01–03 files are **not** modified

## Main outputs

### Full-pose IK module

`Milestone 04/trajectory/full_pose_ik.py`

This provides a damped least-squares iterative solver with backtracking line
search and joint-limit clipping.

### Task-space to joint-space mapping

`Milestone 04/trajectory/generate_task_space_joint_trajectory.py`

By default this produces:

`Milestone 04/Results/Task_Space/task_space_joint_trajectory.csv`

The CSV contains, for each time sample:

- desired Cartesian pose
- recovered joint positions
- recovered joint velocities
- recovered joint accelerations
- FK position residual
- orientation residual
- Jacobian conditioning diagnostic
- continuity diagnostic between consecutive samples

### Plotting

`Milestone 04/trajectory/plot_task_space_joint_trajectory.py`

This generates:

- `task_space_joint_positions.png`
- `task_space_joint_velocities.png`
- `task_space_joint_accelerations.png`
- `task_space_ik_errors.png`
- `task_space_ik_diagnostics.png`

## Run the full current Milestone 04 test suite

```bash
python -m unittest discover \
  -s "Milestone 04/validation" \
  -p "test_*.py" -v
```

After installing this stage, the total should become:

- 24 previous tests
- 7 new full-pose IK tests
- **31 tests total**

## Generate the task-space joint trajectory

```bash
python "Milestone 04/trajectory/generate_task_space_joint_trajectory.py"
python "Milestone 04/trajectory/plot_task_space_joint_trajectory.py"
```

## Current limitations

This stage validates kinematic reachability under the implemented model, but it
still does **not**:

- run MuJoCo closed-loop tracking
- publish ROS 2 motion commands
- perform collision checking
- guarantee obstacle clearance in the surgical scene
- model grasping, handover, or force interaction

Those are later Milestone 04 stages.
