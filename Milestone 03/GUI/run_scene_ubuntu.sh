#!/usr/bin/env bash
# Launch Milestone 03 surgical workstation using the same ROS 2 / MuJoCo bridge as Milestone 02.
# ROS 2 setup.bash expects some unset environment variables; avoid `set -u` here.
set -Eeo pipefail

GUI_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$GUI_DIR/../.." && pwd)"
SCENE="$REPO_ROOT/Milestone 01/Robot Model/scene_surgical_m3.xml"

if [[ ! -f "$SCENE" ]]; then
  echo "Scene not found: $SCENE" >&2
  exit 1
fi

if [[ -z "${ROS_DISTRO:-}" ]]; then
  for distro in jazzy humble; do
    if [[ -f "/opt/ros/$distro/setup.bash" ]]; then
      # shellcheck disable=SC1090
      source "/opt/ros/$distro/setup.bash"
      break
    fi
  done
fi
if [[ -n "${ROS_DISTRO:-}" && -f "/opt/ros/$ROS_DISTRO/setup.bash" ]]; then
  # shellcheck disable=SC1090
  source "/opt/ros/$ROS_DISTRO/setup.bash"
fi

if [[ -f "$HOME/mctr911_ws/install/setup.bash" ]]; then
  # shellcheck disable=SC1090
  source "$HOME/mctr911_ws/install/setup.bash"
fi

if ! command -v ros2 >/dev/null 2>&1; then
  echo "ROS 2 was not found. Source your ROS installation, then try again." >&2
  exit 1
fi
if ! ros2 pkg prefix mujoco_ros2 >/dev/null 2>&1; then
  echo "Package 'mujoco_ros2' was not found in your sourced workspace." >&2
  echo "Build/source your existing Milestone 02 ROS workspace first." >&2
  exit 1
fi

# Only enable the team's existing D3D12 settings on compatible WSL installations.
# Do not force these variables on native Ubuntu, where OpenGL is often already configured.
if grep -qi microsoft /proc/version 2>/dev/null && [[ -e /dev/dxg ]]; then
  export GALLIUM_DRIVER="${GALLIUM_DRIVER:-d3d12}"
  export MESA_D3D12_DEFAULT_ADAPTER_NAME="${MESA_D3D12_DEFAULT_ADAPTER_NAME:-NVIDIA}"
fi

echo "Launching MCTR911 surgical scene: $SCENE"
echo "ROS distro: ${ROS_DISTRO:-unknown} | package: mujoco_ros2"
exec ros2 run mujoco_ros2 mujoco_node "$SCENE" \
  --ros-args \
  -p control_mode:=POSITION \
  -p joint_state_topic_name:=joint_state \
  -p joint_command_topic_name:=joint_commands \
  -p simulation_frequency:=1000 \
  -p visualisation_frequency:=20
