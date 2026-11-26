# RPC 3 demonstration guide

## Build once on WSL

```bash
cd /home/bedo/ros2_ws
source /opt/ros/jazzy/setup.bash
source /home/bedo/mctr911_kinematics_venv/bin/activate
PYTHONPATH="/usr/lib/python3/dist-packages:$PYTHONPATH" python3 /usr/bin/colcon \
  --log-base /tmp/rpc4-log build --build-base /tmp/rpc4-build \
  --install-base /tmp/rpc4-install --packages-select panda_kinematics mujoco_ros2 \
  --cmake-args -DPython3_EXECUTABLE=/usr/bin/python3 -DBUILD_TESTING=ON
```

## Set up each terminal

```bash
cd /home/bedo/ros2_ws
source /opt/ros/jazzy/setup.bash
source /home/bedo/mctr911_kinematics_venv/bin/activate
source /tmp/rpc4-install/local_setup.bash
export ROS_DOMAIN_ID=87 ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export LD_LIBRARY_PATH="/opt/mujoco/mujoco-3.2.0/lib:$LD_LIBRARY_PATH"
export RPC3_SCENE="/home/bedo/mujoco_menagerie/franka_emika_panda/scene.xml"
export PYTHONPATH="$PWD/src/panda_kinematics:/usr/lib/python3/dist-packages:$PYTHONPATH"
mkdir -p docs/rpc3/demo_runs
```

## Graphical run

Terminal 1 starts the bridge in TORQUE mode at home. Stop any previous controllers or bridges on domain 87 before starting.

```bash
ros2 launch mujoco_ros2 rpc3_torque.launch.py scene:="$RPC3_SCENE" \
  headless:=false namespace:=rpc3 visualisation_frequency:=15 \
  torque_trace_csv:="$PWD/docs/rpc3/demo_runs/bridge_$(date +%Y%m%d_%H%M%S).csv"
```

In terminal 2, define this helper and run **one mode per fresh bridge**:

```bash
run_rpc3_mode() {
  local mode="$1"
  ros2 run panda_kinematics controller_node --ros-args -r __ns:=/rpc3 \
    -p scene:="$RPC3_SCENE" -p control_mode:="$mode" \
    -p results_csv:="$PWD/docs/rpc3/demo_runs/${mode}_$(date +%Y%m%d_%H%M%S).csv"
}

run_rpc3_mode model_only
```

Stop the controller and restart the bridge before the standard FLC:

```bash
run_rpc3_mode joint_flc
```

Stop the controller and restart the bridge again before the Team 1 variant:

```bash
run_rpc3_mode cartesian_flc
```

The window opens at home. The controller starts after receiving fresh home-state feedback. The four-second reference is followed by a one-second observation window and terminal hold. The baseline can lag during tracking. Expect a tighter joint path with joint FLC and close XYZ tracking with Cartesian FLC. Wait at least six seconds after the startup message before pressing Ctrl+C in terminal 2. The CSV and JSON are saved on exit. Then stop terminal 1 to flush the bridge trace. A watchdog after commands cease is expected. A watchdog during motion invalidates the run. Relaunch the bridge to clear a latch and reset to home. Successful controller telemetry shows an endpoint, terminal-hold commands, active bridge status, and no in-run fault or observed watchdog.

## Numerical and offline checks

```bash
python3 -B -m panda_kinematics.validate_dynamics \
  --scene "$RPC3_SCENE" --seed 911 --samples 200
python3 -B -m pytest -q -p no:cacheprovider \
  src/panda_kinematics/test/test_rpc{3,4}_*.py
```

This reproduces the collision-free pose search and numerical speed sweep without issuing a torque command. Supply a directory with a complete tracking CSV because the evaluator also writes tracking figures.

```bash
python3 -B docs/rpc3/experiments/evaluate_stage4.py \
  --input docs/rpc3/final_evaluation/evidence/pipe --scene "$RPC3_SCENE" \
  --output "docs/rpc3/demo_runs/offline_$(date +%Y%m%d_%H%M%S)"
```

## Automated live comparison and final figures

Stop manual runs before starting the automated runner. It launches fresh headless bridges and checks endpoint receipt, terminal hold, torque bounds, and the watchdog.

```bash
out="$PWD/docs/rpc3/demo_runs/live_$(date +%Y%m%d_%H%M%S)"
python3 -B docs/rpc3/experiments/run_stage3_ros.py --scene "$RPC3_SCENE" \
  --repeats 3 --seconds 10 --modes model_only joint_flc cartesian_flc --output "$out"
python3 -B docs/rpc3/experiments/evaluate_stage4.py \
  --input "$out" --scene "$RPC3_SCENE" --output "$out/figures"
```

Regenerate the **presentation package from the saved evidence** without launching ROS nodes or MuJoCo:

```bash
MPLCONFIGDIR=/tmp/rpc3-final-mpl python3 -B docs/rpc3/final_evaluation/scripts/generate.py
```

If the window is unavailable, use the verified WSL headless runner and the saved graphical evidence during the presentation. [Final evaluation](FINAL_EVALUATION.md)
