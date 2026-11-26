# Stage 3 ROS controller

The runnable controller is installed as `panda_kinematics/controller_node`.
Both modes use the independent Panda dynamics and the verified four-second
quintic. The moving-phase torque is `M @ ddq_des + C @ dq + g`.
`joint_flc` adds the bounded seven-channel Mamdani correction.
`model_only` uses the same trajectory and torque limits with zero fuzzy
correction. Both retain their selected law for a one-second final observation
window, then use the same bounded endpoint hold.

## Approved exceptions in the packaged copy

| Exception | Reason |
|---|---|
| Import the verified RPC 2 start and end vectors above the marker in place of the two zero placeholders | The original zero vector violates Panda joint 4 limits. The verified start matches the simulator home keyframe. |
| Delegate named, timestamped feedback validation and startup gating to `TrackingRun` | Prevents commands from zero-initialized state, incomplete or stale feedback, an already-active bridge, and a robot away from home. The controller also consumes the bridge's existing `torque_status` diagnostic topic. |
| Keep references to the startup and control timers and cancel them explicitly | The first timer stops only after successful gating. A fault or shutdown stops command publication so the bridge watchdog can pause the simulation. Buffering keeps CSV file I/O out of the timed loop. |
| Delegate trajectory indexing to monotonic elapsed time and submit the final reference explicitly | A delayed callback advances to the current reference. The final sample is sent before the observation window and terminal hold. |
| Use the common bounded terminal hold | A gravity-only command cannot correct terminal pose drift. Both experiment modes receive the same bounded correction after the measurement window. |

The original template in this directory is the supplied reference. The
packaged node retains `joint_state` and `joint_commands` and publishes the
seven arm torques, gripper field, and reserved zero field at a nominal
10 ms period. Its command and feedback queues keep the latest sample.

## Build on Ubuntu WSL

Use a Python environment that can import MuJoCo and Jazzy's `rclpy`.
The commands below use the existing MCTR911 virtual environment. Its Python
runs colcon so the installed controller entry point uses the same interpreter.

```bash
cd ~/ros2_ws
source /opt/ros/jazzy/setup.bash
source ~/mctr911_kinematics_venv/bin/activate
python3 -c 'import mujoco, numpy, rclpy; print(mujoco.__version__)'
PYTHONPATH="/usr/lib/python3/dist-packages:${PYTHONPATH}" python3 /usr/bin/colcon \
  --log-base /tmp/mctr911-rpc3-log build \
  --build-base /tmp/mctr911-rpc3-build --install-base /tmp/mctr911-rpc3-install \
  --packages-select mujoco_ros2 panda_kinematics \
  --cmake-args -DPython3_EXECUTABLE=/usr/bin/python3 -DBUILD_TESTING=ON
```

Run this setup in both terminals.

```bash
cd ~/ros2_ws
source /opt/ros/jazzy/setup.bash
source ~/mctr911_kinematics_venv/bin/activate
source /tmp/mctr911-rpc3-install/local_setup.bash
export ROS_DOMAIN_ID=87 ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
export LD_LIBRARY_PATH="/opt/mujoco/mujoco-3.2.0/lib:${LD_LIBRARY_PATH}"
export RPC3_SCENE="$HOME/mujoco_menagerie/franka_emika_panda/scene.xml"
```

In terminal 1, start the bridge. It publishes feedback while paused at home.

```bash
ros2 launch mujoco_ros2 rpc3_torque.launch.py scene:="$RPC3_SCENE" headless:=true namespace:=rpc3
```

In terminal 2, run the model-only controller.

```bash
ros2 run panda_kinematics controller_node --ros-args -r __ns:=/rpc3 \
  -p scene:="$RPC3_SCENE" -p control_mode:=model_only \
  -p results_csv:="$PWD/docs/rpc3/stage3_results/ros_model_only.csv"
```

Wait at least six seconds after the startup message, then press Ctrl+C in
terminal 2 to save the CSV and its companion JSON. Stop and restart the
bridge in terminal 1 to reset the home pose and watchdog. Then run the FLC
controller in terminal 2 and collect its files in the same way.

```bash
ros2 run panda_kinematics controller_node --ros-args -r __ns:=/rpc3 \
  -p scene:="$RPC3_SCENE" -p control_mode:=joint_flc \
  -p results_csv:="$PWD/docs/rpc3/stage3_results/ros_joint_flc.csv"

python3 -m json.tool docs/rpc3/stage3_results/ros_model_only.json
python3 -m json.tool docs/rpc3/stage3_results/ros_joint_flc.json
```

The JSON records callback-to-publication and torque-computation latency,
counts above 10 ms, skipped elapsed-time reference slots, publication cadence,
joint tracking errors, saturation and observed watchdog events. The CSV includes
the torque reported by the feedback sample available *before* each new command.
Use its timestamp when associating effort with an earlier command. The bridge
may report a watchdog timeout after controller shutdown because commands have
ceased. Check terminal 1 for that event. Each run requires a fresh bridge.

The read-only startup parameters `position_scale`, `velocity_scale`, and
`torque_scale` accept seven positive double values for tuning. Keep them
fixed for a paired comparison.

## Diagnosing feedback freshness

The bridge stamps each joint-state sample with ROS time immediately before
publishing. The controller compares that stamp with its ROS clock at
receipt. Thereafter it adds time elapsed since receipt from the local
monotonic clock. A sample must be no more than **50 ms** old when selected
and when its torque is published. The separate bridge status must also
have been received within **50 ms**. Run both nodes with their default
`use_sim_time:=false` setting so their ROS clocks share the host time.
Samples with a source stamp more than 10 ms in the future are rejected.

The packaged node uses a single-threaded executor for ROS feedback and
startup, and a dedicated monotonic-clock thread for torque computation and
publication. The worker skips missed ticks without sending a burst. This
removes live control-tick dispatch from the ROS executor while preserving
concurrent feedback updates. When the sample used in computation
has expired but newer feedback and bridge status are fresh, the controller
discards the computed torque and retries with current feedback on the next
timer tick. Genuine loss of fresh feedback stops commands. A bridge timeout
is latched, and an observed timeout or command gap of 100 ms prevents a
retry. Restart both processes with the bridge back at home.

The CSV contains the source stamp, local receipt and callback timestamps,
source age at receipt, age of the sample used before and after computation,
latest feedback age after computation, computations that overlap receive
callbacks, and skipped reference slots. Warnings report discarded cycles.
A fault logs the latest timing and joint-state receipt details. Its first
fault snapshot and the discarded-command count are also saved in the JSON.

Interpret the paired clocks together. A growing source age at receipt can
indicate queued feedback or delayed callback service. A gap between source
stamps can indicate missed samples or a gap in bridge publication. A long
compute or callback interval with newer fresh samples indicates controller
or host scheduling delay. JointState has no sample sequence number, so
source-stamp gaps alone cannot distinguish publisher delay from DDS loss.

## Diagnosing computation and scheduling stalls

Before creating any command publisher or timer, the controller evaluates
seven warm-up cases covering zero velocity, moving states, and terminal
hold. The startup log reports the first and maximum durations. The run JSON
contains the component timings for all seven warm-up cases.

Use one BLAS/OpenMP thread for WSL comparisons. Set the environment before
starting Python so NumPy sees it at import time.

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
```

The tracking CSV and fault snapshot include wall and current-thread CPU
times for dynamics, correction (Mamdani inference in joint FLC), NumPy
reference/model and limit operations, reference selection, lock waits and
work, and ROS publication. A large wall/CPU gap locates time the controller
thread spent off CPU or waiting, but alone cannot distinguish scheduler
preemption, GIL waits, or a native-library wait. An expensive component in
both clocks indicates CPU work in that phase. Dynamics uses the same
Lagrangian terms with batched body transforms and Jacobians. The Mamdani
rule table and centroid are unchanged.

For paced headless profiling after the build, run the following in the
configured WSL shell. This uses the bridge callbacks and the real watchdog
with a test pipe. Its cycle and command-gap figures are independent of DDS
and do not establish live ROS deadlines.

```bash
PYTHONPATH="$PWD/src/panda_kinematics:$PYTHONPATH" python3 -B \
  docs/rpc3/experiments/run_stage3_headless.py \
  --bridge /tmp/mctr911-rpc3-build/mujoco_ros2/rpc3_bridge_callback_test \
  --scene "$RPC3_SCENE" --pace --repeats 3 --output /tmp/mctr911-rpc3-paced
```

## First-control-callback stalls

The trajectory clock starts after the first successful command publication.
If a first attempt ages out while the last bridge state is `waiting`, the
controller discards the candidate and allows up to two seconds for a new
attempt. An attempt still requires fresh joint state and bridge status at
home before calculation and a fresh sample at publication. This startup
window applies only before the first command. After publication, the
normal stale-feedback and watchdog rules apply immediately.

The JSON keeps the first eight attempts, including discarded candidates
and their worker IDs. Compare those with the warm-up worker and subsequent
CSV samples. Fuzzy profiles split membership calculation, rule evaluation,
aggregation, and centroid. The correction profile also records Linux
per-thread voluntary and involuntary context switches and minor/major page
faults. These counters help distinguish preemption and allocation evidence
from waits, but a voluntary switch alone does not identify a GIL wait.

Stop any other bridge/controller pair on the selected ROS domain. After
building and sourcing as above, compare both installed modes through
actual ROS topics with fresh bridge processes for every trial.

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python3 -B docs/rpc3/experiments/run_stage3_ros.py \
  --scene "$RPC3_SCENE" --repeats 3 --seconds 10 \
  --output "docs/rpc3/stage3_results/ros_startup_$(date +%Y%m%d_%H%M%S)"
```

Each trial saves bridge and controller logs, the tracking CSV, and a JSON
summary. The runner requires a published endpoint, a terminal hold, and
no observed watchdog or controller fault. A watchdog after controller
shutdown is expected and appears in the bridge log. For comparison with
a real CPU-contention workload, repeat on Linux with the runner and two
external CPU burners pinned to one CPU allowed by `taskset -pc $$`.
This intentionally stresses both bridge and controller.

```bash
cpu=0  # replace with an allowed CPU from taskset -pc $$
(
  taskset -c "$cpu" yes >/dev/null &
  load_a=$!
  taskset -c "$cpu" yes >/dev/null &
  load_b=$!
  trap 'kill "$load_a" "$load_b" 2>/dev/null || true' EXIT
  taskset -c "$cpu" python3 -B docs/rpc3/experiments/run_stage3_ros.py \
    --scene "$RPC3_SCENE" --repeats 1 --seconds 10 \
    --output "docs/rpc3/stage3_results/ros_contention_$(date +%Y%m%d_%H%M%S)"
)
```

The controller executable caps CPython's thread switch interval at 1 ms
while it runs. This gives the feedback executor and control thread shorter
GIL handoff intervals under Python-thread contention. It does not set Linux thread
priority or bound OS scheduling latency. The previous interpreter setting
is restored at shutdown. To reproduce the isolated GIL contention probe:

```bash
PYTHONPATH="$PWD/src/panda_kinematics:$PYTHONPATH" python3 -B \
  docs/rpc3/experiments/profile_stage3_gil.py /tmp/rpc3-gil.json
```


## Live repair diagnostics and graphical demonstration

The packaged copy replaces the live ROS timer below the assignment marker
with a 10 ms monotonic worker. The startup timer still gates the worker at
fresh, paused home. Shutdown signals and joins the worker before writing
telemetry. The command remains a nine-field array. Its optional layout
label carries sequence and ROS publication timestamp for tracing only.
The supplied template file is preserved.

The live runner automatically enables a bounded bridge trace. It records
valid command callback entry, available DDS timestamps, physics and state
callback duration, state publication, and watchdog latches. Joins use the
command sequence and the shared host ROS clock. If DDS receive timestamps
are empty, publication-to-bridge-callback latency includes both transport
and callback service delays. Timing intervals within each process use a
monotonic clock. Trace rows are flushed at bridge shutdown. The watchdog
is considered unexpected only if it latches before controller shutdown.

For a scheduling A/B comparison, first run the normal paired trial. Then
repeat with disjoint CPU lists using `--bridge-cpus` and
`--controller-cpus`. Select lists from `taskset -pc $$`. Affinity can
reduce competition between the two processes, and it cannot guarantee
scheduling against Windows or WSL host load.

For a graphical demonstration, use the same shell setup and launch:

```bash
ros2 launch mujoco_ros2 rpc3_torque.launch.py   scene:="$RPC3_SCENE" headless:=false namespace:=rpc3   torque_trace_csv:="$PWD/docs/rpc3/stage3_results/gui_bridge_timing.csv"
```

Launch the joint FLC controller from the second terminal as documented
above. Use a fresh bridge for a model-only demonstration.
