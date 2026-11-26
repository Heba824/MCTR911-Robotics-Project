RPC2 TEAM 1 — SIDE-BY-SIDE POSITION / VELOCITY DEMO
===================================================

FILES TO PLACE IN /home/bedo/ros2_ws/
  1) run_rpc2_side_by_side.sh      (new launcher)
  2) rpc2_dual_mode.launch.py      (new ROS2 launch wrapper)
  3) rpc2_team1_demo.py           (your previously working script; NOT in this ZIP)

This does NOT modify or replace:
  - src/panda_kinematics/panda_kinematics/trajectory_node.py
  - src/mujoco_ros2/src/mujoco_ros.cpp
  - src/mujoco_ros2/launch/example.py

WHAT IT DOES
  - Starts the same MuJoCo Panda scene from your existing example.py twice.
  - Forces POSITION mode in one process and VELOCITY in the other.
  - Uses ROS_DOMAIN_ID 61 and 62 to prevent commands from crossing between robots.
  - Waits for both simulators to become ready, then starts both trajectories.
  - Leaves both windows open after the motion, and writes separate logs.

ONE COMMAND
  cd ~/ros2_ws
  bash run_rpc2_side_by_side.sh

OPTIONAL: RETURN TO HOME
  bash run_rpc2_side_by_side.sh --return-home

NOTE
  - Shut down any existing MuJoCo/demonstration processes first.
  - The two MuJoCo GLFW windows have the same title. POSITION opens first,
    VELOCITY second; resize and drag them side-by-side manually.
  - Running two 3D simulations may slow down WSL/your GPU. This is a visual
    comparison, not a rigorous timing/performance benchmark.
  - For quantitative metrics, record separate single-mode runs as well.
  - You do not need to edit example.py or rebuild either ROS package.
  - Terminate with Ctrl+C in the runner terminal.
  - If a window fails to start, check rpc2_side_by_side_logs/*_bridge.log.
  - If the demo ends unexpectedly, check *_demo.log.
