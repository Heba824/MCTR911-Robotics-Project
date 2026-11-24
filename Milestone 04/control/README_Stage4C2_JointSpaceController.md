# MCTR911 Milestone 04 — Stage 4C.2: ROS 2 joint-space trajectory controller

## Scope

This stage adds a ROS 2 Python node that executes the already generated
`joint_space_reference.csv` against the existing MuJoCo bridge.

It does **not** change the C++ bridge, UR5e model, reference CSV, or existing
motor-level control equations.

The node supports two strategies:

- `baseline`: publishes the desired joint positions directly to the existing
  POSITION-mode MuJoCo bridge.
- `inverse_dynamics`: computes the existing M4 inverse-dynamics feedforward
  and maps it to the existing position actuators using the derived motor-level
  servo equation.

The default is `baseline`. Validate baseline tracking first. Then repeat from a
fresh HOME initialization with `inverse_dynamics` to compare performance.

## Files and destinations

Copy these files into the existing project:

- `Milestone 04/control/joint_trajectory_controller_core.py`
- `Milestone 04/control/ros2_joint_trajectory_controller.py`
- `Milestone 04/validation/test_joint_trajectory_controller_core.py`

## Safety and execution behavior

- The node starts in `IDLE` and publishes no trajectory commands automatically.
- `/m4_joint_trajectory/start` is a `std_srvs/srv/Trigger` service.
- Starting requires fresh `/joint_state` feedback.
- Starting also requires every joint to be within `start_tolerance_rad` of the
  first reference sample (default 0.05 rad).
- Starting requires the arm to be nearly stationary: every joint velocity
  must be within `start_velocity_tolerance_rad_s` (default 0.05 rad/s).
- JointState values are mapped by joint name, not assumed array order.
- A stale feedback watchdog aborts the run (default timeout 0.10 s).
- Excessive tracking error aborts after a configurable number of consecutive
  samples (default 0.25 rad for 10 samples).
- Actuator command saturation aborts inverse-dynamics execution by default.
- `/m4_joint_trajectory/stop` requests a hold at the latest measured joint pose.
- Completion holds the final reference joint configuration.
- CSV tracking samples and a JSON metrics summary are written under
  `Milestone 04/Results/Tracking/Joint_Space/`.

This is a simulation experiment, not a safety-rated robot controller. Keep the
MuJoCo simulation visible and do not connect this node to physical hardware.

## 1. Install

From WSL:

```bash
cd ~/mctr911_project/MCTR911-Robotics-Project-Fresh

unzip -o /mnt/c/Users/bedo/Downloads/MCTR911_M4_Stage4C2_JointSpaceController.zip \
  -d .
```

Activate the project Python environment and source ROS 2:

```bash
source ~/mctr911_kinematics_venv/bin/activate
source /opt/ros/jazzy/setup.bash
source ~/mctr911_ws/install/setup.bash
```

Run the pure helper tests:

```bash
python -m unittest discover \
  -s "Milestone 04/validation" \
  -p "test_joint_trajectory_controller_core.py" -v
```

Run the complete validation suite:

```bash
python -m unittest discover \
  -s "Milestone 04/validation" \
  -p "test_*.py" -v
```

## 2. Start the MuJoCo bridge

Use terminal 1. The bridge must be in POSITION mode and must print the HOME
keyframe initialization message.

```bash
source /opt/ros/jazzy/setup.bash
source ~/mctr911_ws/install/setup.bash

ros2 run mujoco_ros2 mujoco_node \
  "/home/bedo/mctr911_project/MCTR911-Robotics-Project-Fresh/Milestone 01/Robot Model/scene_surgical_m3.xml" \
  --ros-args \
  -p control_mode:=POSITION \
  -p joint_state_topic_name:=joint_state \
  -p joint_command_topic_name:=joint_commands \
  -p simulation_frequency:=1000 \
  -p visualisation_frequency:=20
```

Confirm the message:

```text
Initialized MuJoCo from the 'home' keyframe.
```

In terminal 2, check that six joint states are arriving:

```bash
source /opt/ros/jazzy/setup.bash
source ~/mctr911_ws/install/setup.bash
ros2 topic echo /joint_state --once
```

Do not continue if the six joints do not appear in the expected HOME vicinity.

## 3. Start the controller in baseline mode

Use terminal 3:

```bash
cd ~/mctr911_project/MCTR911-Robotics-Project-Fresh

source ~/mctr911_kinematics_venv/bin/activate
source /opt/ros/jazzy/setup.bash
source ~/mctr911_ws/install/setup.bash

python "Milestone 04/control/ros2_joint_trajectory_controller.py" \
  --ros-args \
  -p control_strategy:=baseline \
  -p control_rate_hz:=100.0
```

The node must report that it is ready in `IDLE`. It must not move the robot
until the start service is called.

Check the command topic connections in another terminal:

```bash
ros2 topic info /joint_commands -v
```

There should be one controller publisher and one MuJoCo bridge subscriber.
Stop any other publisher before proceeding.

## 4. Start the trajectory explicitly

Only after visually confirming the robot is at HOME and the correct bridge and
controller are running:

```bash
ros2 service call /m4_joint_trajectory/start std_srvs/srv/Trigger "{}"
```

The controller rejects the request if feedback is stale or the initial pose
is outside the configured tolerance.

The reference is 6 seconds of movement plus 2 seconds of final hold. At the
end, the controller holds the final reference configuration and saves:

- `joint_space_tracking_baseline_<run-id>.csv`
- `joint_space_tracking_baseline_<run-id>_metrics.json`

The exact paths are printed in the controller terminal.

## 5. Stop manually if needed

```bash
ros2 service call /m4_joint_trajectory/stop std_srvs/srv/Trigger "{}"
```

The stop service publishes a position hold at the latest measured joint state
and saves the partial run.

## 6. Inverse-dynamics comparison (run only after baseline is reviewed)

Stop the baseline controller and stop/restart the MuJoCo bridge so the robot
is reinitialized at HOME. Start the same node with:

```bash
python "Milestone 04/control/ros2_joint_trajectory_controller.py" \
  --ros-args \
  -p control_strategy:=inverse_dynamics \
  -p control_rate_hz:=100.0
```

This strategy requires the Python `mujoco` package to be importable in the
active environment. It loads the same `scene_surgical_m3.xml`, reads actuator
gains and control ranges from the actual model, and uses the existing
`motor_level_controller.py` functions. It aborts if the computed command
saturates an actuator, unless explicitly configured otherwise.

Do not run this comparison until the baseline result has been inspected.

## 7. What to send back for review

For the first baseline run, send:

1. Controller startup output.
2. `ros2 topic info /joint_commands -v`.
3. Start-service response.
4. Controller completion output.
5. The generated tracking CSV and metrics JSON.

Do not claim successful tracking from a successful service call alone. Review
the actual-vs-reference plots and numerical metrics after the run.
