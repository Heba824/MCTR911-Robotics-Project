# Milestone 04 — Stage 4D: Task-Space Closed-Loop Tracking

This stage adapts the existing full-pose IK trajectory to the already-tested
ROS 2 joint controller, then reconstructs the actual end-effector path from
measured joint feedback and evaluates Cartesian pose/twist tracking.

## Inputs preserved unchanged

- `Results/Task_Space/task_space_reference.csv`: desired Cartesian pose and
  world-frame twist reference.
- `Results/Task_Space/task_space_joint_trajectory.csv`: full-pose IK result,
  including six joint positions, velocities, accelerations and IK diagnostics.
- `control/trajectory_csv.py`: existing validated named-joint CSV loader and
  C2 quintic-Hermite interpolation.
- `control/ros2_joint_trajectory_controller.py`: existing IDLE-first ROS 2
  controller with baseline and inverse-dynamics strategies.
- Milestone 02 DH forward kinematics and Milestone 03 geometric Jacobian.

## 1. Convert the IK result to the controller reference schema

From the repository root, with the project Python environment active:

```bash
python "Milestone 04/trajectory/convert_task_space_ik_to_joint_reference.py"
```

The adapter writes:

`Milestone 04/Results/Task_Space/task_space_joint_reference.csv`

It validates row count, timestamps, phase labels, desired Cartesian pose
agreement, finite values, joint limits and maximum IK residuals. It maps
`q1_rad ... q6_rad`, `qdot1_rad_s ... qdot6_rad_s`, and
`qddot1_rad_s2 ... qddot6_rad_s2` into the existing named-joint schema.
The original task-space and IK CSV files are not modified.

## 2. Execute the task-space-derived joint reference in MuJoCo

Start the existing `mujoco_ros2` bridge in POSITION mode and confirm that the
robot is initialized from the named `home` keyframe. Then start the existing
controller with the task-space-derived reference and a separate output folder:

```bash
python "Milestone 04/control/ros2_joint_trajectory_controller.py" \
  --ros-args \
  -p trajectory_csv:="$PWD/Milestone 04/Results/Task_Space/task_space_joint_reference.csv" \
  -p output_directory:="$PWD/Milestone 04/Results/Tracking/Task_Space" \
  -p control_strategy:=inverse_dynamics \
  -p control_rate_hz:=100.0 \
  -p scene_xml:="$PWD/Milestone 01/Robot Model/scene_surgical_m3.xml"
```

The node must remain in `IDLE` at startup. Verify `/joint_state`,
`/joint_commands`, and the start/stop services. Only call the start service
after the dedicated collision preflight has been completed, visually confirming
the MuJoCo HOME pose, and confirming that no other node is publishing to
`/joint_commands`.

**Collision warning:** the current `scene_surgical_m3.xml` has 72 static
surgical-environment geoms explicitly configured with `contype="0"` and
`conaffinity="0"`. Normal MuJoCo contact detection therefore does not certify
robot clearance from those environment objects. This Stage 4D adapter/analyzer
does not enable collision checking. Do not treat a visually unobstructed path
or an empty contact list as collision-free; complete a separate geometry-aware
preflight before authorizing task-space motion.

```bash
ros2 service call /m4_joint_trajectory/start std_srvs/srv/Trigger "{}"
```

The controller's recorded joint-tracking CSV is the measured feedback source
for the Cartesian analyzer. The output directory is separated from the
joint-space experiment directory to preserve both datasets.

## 3. Reconstruct and analyze Cartesian tracking

Use the joint-tracking CSV produced by the ROS 2 controller:

```bash
python "Milestone 04/analysis/analyze_task_space_tracking.py" \
  --tracking-csv "Milestone 04/Results/Tracking/Task_Space/<tracking-file>.csv" \
  --task-space-reference "Milestone 04/Results/Task_Space/task_space_reference.csv"
```

The controller's raw joint-feedback files are written under
`Milestone 04/Results/Tracking/Task_Space/`. Its current legacy filename
prefix remains `joint_space_tracking_<strategy>_<run-id>`; the separate folder
identifies these as the task-space execution run. The analyzer writes:

- `task_space_tracking_<strategy>_<run-id>.csv`
- `task_space_tracking_<strategy>_<run-id>_metrics.json`

For every feedback sample, the analyzer:

1. Reconstructs the existing `QuinticTaskSpaceTrajectory` from the stored
   endpoint poses and movement duration, then verifies every stored reference
   sample (pose, twist and acceleration) against that model.
2. Evaluates the exact quintic Cartesian reference at each controller
   `trajectory_time_s`, avoiding piecewise-linear timing approximation.
3. Reconstructs the actual `attachment_site` pose with Milestone 02 standard-DH
   forward kinematics applied to measured joint positions.
4. Reconstructs actual world-frame end-effector twist with
   `J(q) @ qdot` using the Milestone 03 geometric Jacobian.
5. Computes Cartesian position error and the world-frame orientation error
   `Log(R_desired R_actual^T)`.
6. Reports overall, movement-only, final-hold and per-phase position,
   orientation, linear-velocity and angular-velocity metrics.

Position error is `desired - actual`. Orientation error uses the same
world-frame convention as the existing full-pose IK implementation.

## 4. Generate figures

```bash
python "Milestone 04/visualization/plot_task_space_tracking.py" \
  --csv "Milestone 04/Results/Tracking/Task_Space/<cartesian-tracking-file>.csv"
```

Figures are saved under `Milestone 04/Figures/Task_Space/Tracking/`:

- `task_space_position_tracking.png`
- `task_space_position_error.png`
- `task_space_cartesian_path_tracking.png`
- `task_space_orientation_velocity_errors.png`

## 5. Unit tests

```bash
PYTHONPATH="$PWD/Milestone 04/control:$PWD/Milestone 04/trajectory:$PWD/Milestone 04/analysis:$PYTHONPATH" \
python -m unittest discover \
  -s "Milestone 04/validation" \
  -p "test_task_space_tracking.py" -v
```

The tests verify adapter/schema compatibility, exact recovery of IK CSV knots,
source-file alignment checks, IK residual rejection, FK reconstruction of the
known IK path, quaternion sign invariance, non-finite feedback rejection and
reference-time bounds.

## Interpretation and limitations

The Cartesian analyzer evaluates the UR5e `attachment_site` using the same
validated DH/FK and world-frame Jacobian conventions used to generate the
reference. Its measured path is therefore a model-based reconstruction from
actual MuJoCo joint feedback, not a separate physical end-effector sensor.

This stage does not by itself certify collision-free motion, instrument grasping,
or a completed surgical-tool handover. The current task-space path is a
kinematic test motion between two known joint configurations. Collision and
application-specific handover validation remain separate tasks. In particular,
the 72 static environment geoms in the current surgical scene have contact
filtering disabled, so a geometry-aware collision audit is required before
running this trajectory in the scene.
