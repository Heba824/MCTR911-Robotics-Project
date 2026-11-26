#!/usr/bin/env bash
# Optional visual demo: run POSITION and VELOCITY in two separate MuJoCo windows.
# V3: no ROS CLI type inference; software OpenGL on the second instance by default.
# No changes to the tested bridge, instructor trajectory, or original example.py.
set -Eeuo pipefail

WORKSPACE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$WORKSPACE/rpc2_side_by_side_logs"
SHARED_DOMAIN="${RPC2_SHARED_DOMAIN:-${ROS_DOMAIN_ID:-0}}"
READY_TIMEOUT="${RPC2_READY_TIMEOUT:-40}"
RETURN_ARGS=()
SOFTWARE_GL=1

usage() {
    echo "Usage: bash run_rpc2_side_by_side.sh [--return-home] [--no-software-gl]"
}
for arg in "$@"; do
    case "$arg" in
        --return-home) RETURN_ARGS+=(--return-home) ;;
        --no-software-gl) SOFTWARE_GL=0 ;;
        -h|--help) usage; exit 0 ;;
        *) usage >&2; exit 2 ;;
    esac
done

for file in /opt/ros/jazzy/setup.bash "$WORKSPACE/install/setup.bash" \
            "$WORKSPACE/rpc2_dual_mode.launch.py" "$WORKSPACE/rpc2_team1_demo.py" \
            "$WORKSPACE/rpc2_wait_for_joint_state.py"; do
    if [[ ! -f "$file" ]]; then
        echo "[RPC2] Missing required file: $file" >&2
        exit 1
    fi
done

# ROS 2 setup may reference previously unset variables, so disable nounset just here.
set +u
source /opt/ros/jazzy/setup.bash
source "$WORKSPACE/install/setup.bash"
set -u
export LD_LIBRARY_PATH="/opt/mujoco/mujoco-3.2.0/lib:${LD_LIBRARY_PATH:-}"
export ROS_DOMAIN_ID="$SHARED_DOMAIN"

PIDS=()
cleanup() {
    trap - EXIT INT TERM
    echo
    echo '[RPC2] Stopping only processes started by this launcher...'
    local pid
    for pid in "${PIDS[@]}"; do
        kill -INT "$pid" 2>/dev/null || true
    done
    sleep 2
    for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill -TERM "$pid" 2>/dev/null || true
        fi
    done
    for pid in "${PIDS[@]}"; do
        wait "$pid" 2>/dev/null || true
    done
    echo "[RPC2] Logs: $LOG_DIR"
}
trap cleanup EXIT INT TERM

mkdir -p "$LOG_DIR"
printf '[RPC2] Shared ROS_DOMAIN_ID=%s; separate /rpc2_position and /rpc2_velocity topics\n' "$ROS_DOMAIN_ID"

wait_ready() {
    local name="$1" namespace="$2" launcher_pid="$3" logfile="$4"
    local readylog="$LOG_DIR/${name}_ready.log"
    local errlog="$LOG_DIR/${name}_ready.err"
    echo "[RPC2] Waiting up to ${READY_TIMEOUT}s for /${namespace}/joint_state ..."
    if python3 "$WORKSPACE/rpc2_wait_for_joint_state.py" \
            "/${namespace}/joint_state" --timeout "$READY_TIMEOUT" \
            > "$readylog" 2> "$errlog"; then
        echo "[RPC2] ${name^^} ready."
        return 0
    fi
    echo "[RPC2] ERROR: ${name^^} did not publish joint states." >&2
    if ! kill -0 "$launcher_pid" 2>/dev/null; then
        echo "[RPC2] ROS launch process exited." >&2
    fi
    if ! grep -q 'MuJoCo simulation initiated' "$logfile"; then
        echo '[RPC2] Bridge did not reach the post-OpenGL initialization message.' >&2
        echo '[RPC2] Possible WSL/GLFW/OpenGL window initialization stall.' >&2
    fi
    echo '[RPC2] Last bridge messages:' >&2
    tail -n 20 "$logfile" >&2 || true
    cat "$errlog" >&2 || true
    return 1
}

launch_bridge() {
    local mode="$1" namespace="$2" logfile="$3" use_sw="$4"
    (
        export RPC2_CONTROL_MODE="$mode" RPC2_BRIDGE_NAMESPACE="$namespace"
        if (( use_sw )); then
            export LIBGL_ALWAYS_SOFTWARE=1
        fi
        exec ros2 launch "$WORKSPACE/rpc2_dual_mode.launch.py"
    ) > "$logfile" 2>&1 &
    LAUNCHED_PID=$!
    PIDS+=("$LAUNCHED_PID")
}

echo '[RPC2] Opening POSITION window (normal OpenGL)...'
launch_bridge POSITION rpc2_position "$LOG_DIR/position_bridge.log" 0
POS_BRIDGE="$LAUNCHED_PID"
wait_ready position rpc2_position "$POS_BRIDGE" "$LOG_DIR/position_bridge.log"

if (( SOFTWARE_GL )); then
    echo '[RPC2] Opening VELOCITY window (software OpenGL fallback for WSL)...'
else
    echo '[RPC2] Opening VELOCITY window (normal OpenGL)...'
fi
launch_bridge VELOCITY rpc2_velocity "$LOG_DIR/velocity_bridge.log" "$SOFTWARE_GL"
VEL_BRIDGE="$LAUNCHED_PID"
wait_ready velocity rpc2_velocity "$VEL_BRIDGE" "$LOG_DIR/velocity_bridge.log"

sleep 2
echo '[RPC2] Starting Team 1 trajectories in both simulations...'
(
    export RPC2_TOPIC_NS=rpc2_position
    exec python3 "$WORKSPACE/rpc2_team1_demo.py" --mode POSITION "${RETURN_ARGS[@]}"
) > "$LOG_DIR/position_demo.log" 2>&1 &
POS_DEMO=$!
PIDS+=("$POS_DEMO")
(
    export RPC2_TOPIC_NS=rpc2_velocity
    exec python3 "$WORKSPACE/rpc2_team1_demo.py" --mode VELOCITY "${RETURN_ARGS[@]}"
) > "$LOG_DIR/velocity_demo.log" 2>&1 &
VEL_DEMO=$!
PIDS+=("$VEL_DEMO")

pos_status=0
vel_status=0
wait "$POS_DEMO" || pos_status=$?
wait "$VEL_DEMO" || vel_status=$?

echo '================ POSITION ================'
tail -n 12 "$LOG_DIR/position_demo.log"
echo '================ VELOCITY ================'
tail -n 12 "$LOG_DIR/velocity_demo.log"
if (( pos_status != 0 || vel_status != 0 )) || \
    ! grep -q 'Team 1 demonstration finished' "$LOG_DIR/position_demo.log" || \
    ! grep -q 'Team 1 demonstration finished' "$LOG_DIR/velocity_demo.log"; then
    echo '[RPC2] One or both demos did not complete; see logs.' >&2
    exit 1
fi

echo '[RPC2] Both completed. Arrange windows side by side; Ctrl+C closes them.'
while kill -0 "$POS_BRIDGE" 2>/dev/null && kill -0 "$VEL_BRIDGE" 2>/dev/null; do
    sleep 1
done
