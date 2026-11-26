#!/usr/bin/env bash
# Run ONE graphical MuJoCo instance and a repeated Team 1 tracking demo.
# Uses the original installed example.py scene and the verified trajectory planner.
set -Eeuo pipefail

usage() {
  echo 'Usage: bash run_rpc2_comparison.sh POSITION|VELOCITY [--cycles 4] [--leg-duration 2.0] [--hold 0.4]'
}
if (( $# < 1 )); then usage >&2; exit 2; fi
MODE="${1^^}"
shift
if [[ "$MODE" != POSITION && "$MODE" != VELOCITY ]]; then usage >&2; exit 2; fi
WORKSPACE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
for f in "$WORKSPACE/rpc2_mode_once.launch.py" "$WORKSPACE/rpc2_repeated_comparison.py" "$WORKSPACE/rpc2_plot_comparison.py"; do
  if [[ ! -f "$f" ]]; then echo "Missing file: $f" >&2; exit 1; fi
done
# ROS setup references unset variables; disable nounset temporarily.
set +u
source /opt/ros/jazzy/setup.bash
source "$WORKSPACE/install/setup.bash"
set -u
export LD_LIBRARY_PATH="/opt/mujoco/mujoco-3.2.0/lib:${LD_LIBRARY_PATH:-}"
export RPC2_COMPARISON_MODE="$MODE"
LOGS="$WORKSPACE/rpc2_comparison_results"
mkdir -p "$LOGS"
SIM_PID=""
cleanup() {
  trap - EXIT INT TERM
  if [[ -n "$SIM_PID" ]]; then
    kill -INT "$SIM_PID" 2>/dev/null || true
    sleep 1
    if kill -0 "$SIM_PID" 2>/dev/null; then kill -TERM "$SIM_PID" 2>/dev/null || true; fi
    wait "$SIM_PID" 2>/dev/null || true
  fi
  echo '[RPC2] Simulator closed; tracking data remains saved.'
}
trap cleanup EXIT INT TERM

echo "[RPC2] Launching one MuJoCo window in $MODE mode..."
ros2 launch "$WORKSPACE/rpc2_mode_once.launch.py" >"$LOGS/${MODE,,}_bridge.log" 2>&1 &
SIM_PID=$!

echo '[RPC2] Running repeated reach/return; waits for HOME feedback...'
STATUS=0
python3 "$WORKSPACE/rpc2_repeated_comparison.py" --mode "$MODE" --output "$LOGS" "$@" || STATUS=$?
if (( STATUS != 0 )); then
  echo "[RPC2] Demo failed. Bridge log: $LOGS/${MODE,,}_bridge.log" >&2
  tail -n 25 "$LOGS/${MODE,,}_bridge.log" || true
  exit "$STATUS"
fi

if [[ -f "$LOGS/position_summary.json" && -f "$LOGS/velocity_summary.json" ]]; then
  python3 "$WORKSPACE/rpc2_plot_comparison.py" --dir "$LOGS" || true
else
  echo "[RPC2] Recorded $MODE. Run the other mode next to generate comparison graph."
fi
if [[ -t 0 ]]; then
  echo "[RPC2] $MODE run finished. Press Enter to close this MuJoCo window."
  read -r _ || true
fi
