# Milestone 04 — Task-space trajectory generation

This stage implements the Project Description requirement to derive task-space
trajectories for the UR5e end effector using Python.

## Existing project conventions preserved

- End-effector frame: MuJoCo `attachment_site`.
- Pose convention: `T_world_attachment_site`, from the existing Milestone 02 FK.
- World-frame linear and angular quantities.
- Twist ordering: `[vx, vy, vz, wx, wy, wz]`.
- Acceleration ordering: `[ax, ay, az, alphax, alphay, alphaz]`.
- Existing six-joint order and DH constants are imported from Milestone 02.

No Milestone 01–03 source file is modified.

## Trajectory definition

Translation follows a straight line:
`p_d(t) = p_0 + (p_f - p_0) s(t)`.

Orientation follows a shortest fixed-axis world-frame interpolation:
`R_d(t) = Exp([phi_world]x s(t)) R_0`,
where `phi_world = Log(R_f R_0.T)`.

The same quintic scalar time law is used:
`s(tau) = 10 tau^3 - 15 tau^4 + 6 tau^5`, `tau=t/T`.

Its derivatives are:
`s_dot = (30 tau^2 - 60 tau^3 + 30 tau^4)/T`
and
`s_ddot = (60 tau - 180 tau^2 + 120 tau^3)/T^2`.

Because the orientation interpolation rotates around one fixed world-frame
axis, its world-frame angular velocity and acceleration are:
`omega_d = phi_world s_dot`
and
`alpha_d = phi_world s_ddot`.

The resulting task-space reference provides pose, quaternion, linear/angular
velocity, and linear/angular acceleration. Quaternion columns use ROS order
`[x, y, z, w]`.

## Default endpoint construction

The generator imports the existing Milestone 02 `forward_kinematics()` and
uses the same HOME and nearby goal joint configurations as the Milestone 04
joint-space reference. Therefore, both Cartesian endpoint poses are generated
from known joint configurations using the project's established DH model.

The path between those endpoint poses is independently interpolated in task
space. It is not the same curve as the joint-space path. Endpoint reachability
is known by construction, but intermediate inverse-kinematics feasibility and
collision clearance must be checked in the next stage before simulator motion.

## Generate a task-space reference

From the repository root, with the project Python environment activated:

```bash
python "Milestone 04/trajectory/generate_task_space_trajectory.py"
```

Default configuration: 6-second movement, 2-second final hold, 20 ms
sampling interval (401 samples). Output:

`Milestone 04/Results/Task_Space/task_space_reference.csv`

## Plot the reference

```bash
python "Milestone 04/trajectory/plot_task_space_trajectory.py"
```

Figures are written to:

`Milestone 04/Figures/Task_Space/`

## Run all current M4 trajectory tests

```bash
python -m unittest discover \
  -s "Milestone 04/validation" \
  -p "test_*.py" -v
```

## Current scope and limitations

Implemented: task-space position/orientation interpolation and analytical
world-frame twist and acceleration references, with unit tests.

Not yet implemented: full-pose inverse kinematics, intermediate-pose
reachability checks, collision/path constraints, ROS 2 command publishing,
motor-level control, closed-loop tracking, or tool handover. These are
subsequent Milestone 04 stages.
