#!/usr/bin/env bash
# Run Milestone 03 joint control GUI in a second Ubuntu terminal.
# ROS 2 setup.bash expects some unset environment variables; avoid `set -u` here.
set -Eeo pipefail
GUI_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

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

exec python3 "$GUI_DIR/surgical_dashboard.py" "$@"
