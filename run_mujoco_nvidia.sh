#!/bin/bash
set -e

source /opt/ros/jazzy/setup.bash
source "$HOME/mctr911_ws/install/setup.bash"

cd "$HOME/mctr911_project/MCTR911-Robotics-Project-git"

export GALLIUM_DRIVER=d3d12
export MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA
unset LIBGL_ALWAYS_SOFTWARE
unset LIBGL_ALWAYS_INDIRECT

exec ros2 run mujoco_ros2 mujoco_node \
  "$PWD/Milestone 01/Robot Model/scene.xml" \
  --ros-args \
  -p control_mode:=POSITION \
  -p joint_state_topic_name:=joint_state \
  -p joint_command_topic_name:=joint_commands \
  -p simulation_frequency:=1000 \
  -p visualisation_frequency:=20
