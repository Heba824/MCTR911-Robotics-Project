# Team 1 Cartesian-space FLC and graphical bridge

`cartesian_flc` is a separate mode. The required seven-joint `joint_flc`
and the `model_only` comparison remain selectable. All modes use the
same four-second, limit-checked joint quintic. The Cartesian reference
is its hand XYZ path in the base frame. There is no orientation command.

The controller computes position error `p(q_des) - p(q)` and velocity
error `J(q_des) @ dq_des - J(q) @ dq`. Three Mamdani channels use the
same five membership labels, rule table, min/max inference, and centroid
integration as the joint FLC. Their gains are in metres, metres/second,
and metres/second of correction. The desired XYZ velocity plus fuzzy
correction is limited to 0.18 m/s.

SVD implements the damped inverse. For smallest singular value (s),
the damping is `0.10 * clip(1 - s/0.08, 0, 1)**2`. The four-dimensional
redundant space biases the solution toward the joint reference and away
from joint limits. The posture bias is capped at 0.20 rad/s in norm.
Joint-speed commands are limited to 0.65 rad/s per joint with braking
near limits, and acceleration commands to 4 rad/s² per joint. The
independent model maps acceleration to torque. Torque is limited to
87 N·m on joints 1–4 and 12 N·m on joints 5–7. Saturated or damped
commands can leave a task-space residual. Those limits do not guarantee
actual joint speed or acceleration in the presence of disturbance.

The singularity experiment searches seeded legal configurations and
rejects poses with MuJoCo contacts. It evaluates the weakest task
direction at the selected pose with an undamped pseudoinverse and
the adaptive damped inverse. The undamped velocity is an offline
number and is never issued to the simulator.

The old trace records about 295 ms between DDS receive and subscriber
callback for one queued torque command. DDS source-to-receive time was
about 0.15 ms, and the command had no diagnostic sequence tag, so the
trace cannot conclusively identify its publisher. The original graphical
path ran rendering and VSync alongside physics in the same executor.
The TORQUE graphical bridge now gives the ROS executor its own thread.
GLFW stays on the main thread with a private MuJoCo snapshot copied
under a mutex. The copy lock is released before rendering, swapping
buffers, or polling events. TORQUE rendering defaults to 15 Hz with VSync
disabled. Legacy graphical modes retain their former execution path.
DDS-stamped torque packets older than 50 ms at the bridge cannot activate
or refresh the watchdog. The bridge's 100 ms watchdog remains latched
until relaunch. The original assignment template is unchanged.

## Build and verification on WSL

```bash
cd ~/ros2_ws
source /opt/ros/jazzy/setup.bash
source ~/mctr911_kinematics_venv/bin/activate
export RPC3_SCENE="$HOME/mujoco_menagerie/franka_emika_panda/scene.xml"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export ROS_DOMAIN_ID=87 ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
export LD_LIBRARY_PATH="/opt/mujoco/mujoco-3.2.0/lib:$LD_LIBRARY_PATH"

PYTHONPATH="/usr/lib/python3/dist-packages:$PYTHONPATH" python3 /usr/bin/colcon \
  --log-base /tmp/rpc4-log build --build-base /tmp/rpc4-build \
  --install-base /tmp/rpc4-install --packages-select panda_kinematics mujoco_ros2 \
  --cmake-args -DPython3_EXECUTABLE=/usr/bin/python3 -DBUILD_TESTING=ON
source /tmp/rpc4-install/local_setup.bash
export PYTHONPATH="$PWD/src/panda_kinematics:/usr/lib/python3/dist-packages:$PYTHONPATH"

python3 -B -m panda_kinematics.validate_dynamics --scene "$RPC3_SCENE" --seed 911 --samples 200
python3 -B -m pytest -q -p no:cacheprovider src/panda_kinematics/test/test_rpc{3,4}_*.py

out="$PWD/docs/rpc3/stage4_wsl_$(date +%Y%m%d_%H%M%S)"
python3 -B docs/rpc3/experiments/run_stage3_headless.py \
  --bridge /tmp/rpc4-build/mujoco_ros2/rpc3_bridge_callback_test \
  --scene "$RPC3_SCENE" --pace --modes model_only joint_flc cartesian_flc --output "$out/headless"
python3 -B docs/rpc3/experiments/evaluate_stage4.py \
  --input "$out/headless" --scene "$RPC3_SCENE" --output "$out/headless_figures"
```

Stop manual bridges and controllers before running automated trials.
Check any publishers shown by `ros2 topic info /rpc3/joint_commands -v`
and `ros2 topic info /rpc3/joint_state -v` before starting a fresh bridge.

```bash
python3 -B docs/rpc3/experiments/run_stage3_ros.py --scene "$RPC3_SCENE" \
  --repeats 3 --seconds 10 --modes model_only joint_flc cartesian_flc --output "$out/ros"
python3 -B docs/rpc3/experiments/evaluate_stage4.py \
  --input "$out/ros" --scene "$RPC3_SCENE" --output "$out/ros_figures"

python3 -B docs/rpc3/experiments/run_stage3_ros.py --scene "$RPC3_SCENE" \
  --gui --repeats 1 --seconds 10 --modes joint_flc cartesian_flc --output "$out/gui"
```

For a manual graphical demo, repeat the environment setup in two
terminals. Start one bridge in terminal 1:

```bash
ros2 launch mujoco_ros2 rpc3_torque.launch.py scene:="$RPC3_SCENE" \
  headless:=false namespace:=rpc3 visualisation_frequency:=15 \
  torque_trace_csv:="$PWD/docs/rpc3/gui_bridge_$(date +%Y%m%d_%H%M%S).csv"
```

Choose a mode in terminal 2 and run the controller. Stop it first to
save telemetry, then stop the bridge to flush its trace. Relaunch the
bridge before running the other mode.

```bash
mode=joint_flc              # Standard seven-joint FLC
# mode=cartesian_flc        # Team 1 XYZ FLC
ros2 run panda_kinematics controller_node --ros-args -r __ns:=/rpc3 \
  -p scene:="$RPC3_SCENE" -p control_mode:="$mode" \
  -p results_csv:="$PWD/docs/rpc3/gui_${mode}_$(date +%Y%m%d_%H%M%S).csv"
```

## WSL Cartesian feedback-stall retest

In the prior live experiment, Cartesian control stopped on a normal
10 ms tick after bridge state and status became 53.70 and 52.97 ms old.
The bridge's monotonic publication gap was 340.03 ms. The affected state
callback lasted 330.45 ms, of which 0.27 ms was measured in MuJoCo physics.
The previous trace did not timestamp the individual DDS send calls, so
it cannot separate a blocked send from scheduling delay within the callback.

The TORQUE launch now uses asynchronous Fast DDS publication for the
bridge process. This removes synchronous network sending from the bridge
callback path while preserving the command topic, QoS depth and all
freshness checks. The trace records preparation time, status and state
publication wall time, and callback thread CPU time. The launch argument
`dds_publication_mode` and runner argument `--dds-publication-mode` let
you compare the old synchronous setting on the same host.

After rebuilding and sourcing `/tmp/rpc4-install` as above, run:

```bash
out="$PWD/docs/rpc3/stage4_repair_wsl_$(date +%Y%m%d_%H%M%S)"
python3 -B docs/rpc3/experiments/run_stage3_ros.py --scene "$RPC3_SCENE" \
  --repeats 3 --seconds 10 --modes cartesian_flc --output "$out/cartesian"
python3 -B docs/rpc3/experiments/run_stage3_ros.py --scene "$RPC3_SCENE" \
  --repeats 1 --seconds 10 --modes model_only joint_flc --output "$out/regression"
python3 -B docs/rpc3/experiments/evaluate_stage4.py \
  --input "$out" --scene "$RPC3_SCENE" --output "$out/figures"
```
