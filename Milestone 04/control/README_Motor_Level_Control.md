# Milestone 04 — Motor-level control law (Stage 4A)

## Requirement addressed

The Project Description's Milestone 04 requirements explicitly include deriving
the control algorithms for motor-level control. This stage derives the law for
the **existing UR5e MuJoCo actuators**. It does not add a new actuator model or
claim physical-hardware control.

## Existing actuator model

The supplied `Milestone 01/Robot Model/ur5e.xml` defines six affine MuJoCo
`general` joint actuators. The Milestone 03 report records the gains:

- `Kp = [2000, 2000, 2000, 500, 500, 500]`
- `Kd = [400, 400, 400, 100, 100, 100]`

The gains are read from the loaded MuJoCo model at runtime rather than
hard-coded by the controller.

For joint i, the actuator force is:

`tau_act_i = Kp_i (ctrl_i - q_i) - Kd_i qdot_i`

Here `ctrl_i` is the position command in radians, `q_i` is the actual joint
position, and `qdot_i` is the actual joint velocity.

## Inverse-dynamics feedforward

For a desired joint reference `(q_ref, qdot_ref, qddot_ref)`, MuJoCo provides
the mass matrix and generalized bias/passive forces at the reference state.
The required feedforward generalized force is:

`tau_ff = M(q_ref) qddot_ref + qfrc_bias(q_ref, qdot_ref) - qfrc_passive(q_ref, qdot_ref)`

The position command must compensate for the actuator's own velocity damping.
Solving the actuator equation gives:

`ctrl = q_ref + (tau_ff + Kd qdot_ref) / Kp`

Substituting this command back into the actuator model gives:

`tau_act = tau_ff + Kp (q_ref - q) + Kd (qdot_ref - qdot)`

This is the implemented control law: inverse-dynamics feedforward plus the
existing actuator's proportional/derivative tracking feedback.

## Baseline comparison

With feedforward disabled, the controller sends `ctrl = q_ref`, matching the
baseline position-reference strategy. With feedforward enabled, it applies the
computed torque as a position-command offset. Both strategies retain the
original actuator model.

The implementation supports actuator control-range clipping and returns a
per-joint clipping flag. Saturation must be reported in the experiment rather
than hidden.

## Reuse from Milestone 03

The formula is consistent with the already implemented and experimentally
validated `compute_position_control()` in:

`Milestone 03/visualization/ur5e_smooth_motion_demo.py`

The M3 smooth-motion report documents the same actuator gains and feedforward
method, including low joint-position and joint-velocity tracking errors in its
standalone MuJoCo experiment. M4 will reuse this validated approach when the
ROS 2 trajectory controller is implemented.

## Run the new pure-math tests

```bash
python -m unittest discover \
  -s "Milestone 04/validation" \
  -p "test_motor_level_controller.py" -v
```

Expected: 8 tests pass.

## ROS 2 bridge-rate diagnostic

The previous project terminal log showed `/joint_state` at approximately
1.2 Hz even though the bridge parameter was `simulation_frequency=1000`.
That rate is too low to claim high-rate ROS 2 trajectory feedback. Before
building the closed-loop ROS 2 runner, measure the rate again with the
visualizer slowed to 1 Hz.

Stop the current MuJoCo bridge, then in terminal 1 run:

```bash
source /opt/ros/jazzy/setup.bash
source ~/mctr911_ws/install/setup.bash

ros2 run mujoco_ros2 mujoco_node \
  "Milestone 01/Robot Model/scene_surgical_m3.xml" \
  --ros-args \
  -p control_mode:=POSITION \
  -p joint_state_topic_name:=joint_state \
  -p joint_command_topic_name:=joint_commands \
  -p simulation_frequency:=1000 \
  -p visualisation_frequency:=1
```

In terminal 2, from the project root with the project environment active:

```bash
source /opt/ros/jazzy/setup.bash
source ~/mctr911_ws/install/setup.bash

python "Milestone 04/validation/ros2_bridge_rate_diagnostic.py" \
  --topic /joint_state --duration 10
```

This diagnostic is read-only. It does not publish commands.

If the rate increases substantially, the renderer/single-threaded executor is
likely limiting the state stream at the default 20 Hz visualization setting.
If it remains low, we will inspect and modify the bridge timing architecture
before making any closed-loop tracking claims.

## Important integration constraints discovered

1. The existing ROS 2 bridge is launched in `POSITION` mode and exposes
   `/joint_commands` (`Float64MultiArray`) and `/joint_state` (`JointState`).
2. The bridge's `mj_makeData()` initialization starts from the model's default
   zero joint state; the existing `home` keyframe is not automatically selected.
   The M4 trajectory references currently start at the M3 HOME vector, so a
   safe pre-positioning or explicit keyframe initialization is required before
   trajectory playback.
3. The existing surgical scene's scalpel/forceps are visual-only, and the M3
   GUI explicitly states that instrument grasping is not implemented. A later
   application stage must implement and clearly label a virtual attachment /
   release mechanism or another demonstrable tool-transfer mechanism.
4. The current 6-second M4 trajectories are kinematic test paths inherited
   from the M3 smooth-motion start/goal. They are not yet the complete
   scalpel-pickup-to-handover application path.

These are intentionally recorded as integration tasks; they are not silently
treated as completed functionality.
