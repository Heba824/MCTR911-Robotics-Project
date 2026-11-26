#!/usr/bin/env python3
"""RPC 2 Franka Panda: linear Jacobian, Jdot and quintic trajectory.

Self-contained: reuses the tested mathematical implementation from RPC 1.
The provided ROS 2 infrastructure is preserved below its section marker.
Configured start/end poses are for Team 1: Standard Pick-and-Place,
a controlled reach-and-lift from the neutral home configuration.
"""
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
import numpy as np

# ==========================================================
#  EDIT ONLY THIS SECTION
# ==========================================================

# Rows: [alpha_i, a_i, d_i]; q_i is supplied at runtime.
# Values equivalent to the link0 -> link7 transforms in the pinned
# MuJoCo Menagerie franka_emika_panda/panda.xml.
DH_PARAMS = [
    [-np.pi / 2,  0.0,     0.333],   # joint 1
    [ np.pi / 2,  0.0,     0.0  ],   # joint 2
    [ np.pi / 2,  0.0825,  0.316],   # joint 3
    [-np.pi / 2, -0.0825,  0.0  ],   # joint 4
    [ np.pi / 2,  0.0,     0.384],   # joint 5
    [ np.pi / 2,  0.088,   0.0  ],   # joint 6
    [ 0.0,        0.0,     0.0  ],   # joint 7
]

# Fixed pose of the hand child of link7 in panda.xml:
#     pos="0 0 0.107" quat="0.9238795 0 0 -0.3826834"
# This is translation 0.107 m along link7 z, and Rz(-pi/4).
_hand_angle = -np.pi / 4
T_L7_HAND = np.array([
    [np.cos(_hand_angle), -np.sin(_hand_angle), 0.0, 0.0],
    [np.sin(_hand_angle),  np.cos(_hand_angle), 0.0, 0.0],
    [0.0,      0.0,     1.0, 0.107],
    [0.0,      0.0,     0.0, 1.0],
], dtype=float)

# Arm joint limits copied from panda.xml (default joint range and overrides).
Q_MIN = np.array([-2.8973, -1.7628, -2.8973, -3.0718,
                  -2.8973, -0.0175, -2.8973], dtype=float)
Q_MAX = np.array([ 2.8973,  1.7628,  2.8973, -0.0698,
                   2.8973,  3.7525,  2.8973], dtype=float)



def dh_transform(alpha, a, d, theta):
    """Standard-DH homogeneous transformation for one revolute joint."""
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array([
        [ct, -st * ca,  st * sa, a * ct],
        [st,  ct * ca, -ct * sa, a * st],
        [0.0, sa,       ca,      d],
        [0.0, 0.0,      0.0,     1.0],
    ], dtype=float)


def _as_q(q, name):
    q = np.asarray(q, dtype=float)
    if q.shape != (7,) or not np.all(np.isfinite(q)):
        raise ValueError(f"{name} must be 7 finite angles (radians)")
    return q


def forward_kinematics(q):
    """Return the 4x4 world(link0)-to-hand pose for seven arm angles."""
    q = np.asarray(q, dtype=float)
    if q.shape != (7,) or not np.all(np.isfinite(q)):
        raise ValueError('q must contain exactly seven finite joint angles')
    T = np.eye(4)
    for theta, (alpha, a, d) in zip(q, DH_PARAMS):
        T = T @ dh_transform(alpha, a, d, theta)
    return T @ T_L7_HAND



def _fk_and_jacobian(q):
    """Hand pose and 6x7 spatial/geometric Jacobian in link0 coordinates.

    A revolute joint's world z-axis is the joint rotation axis. Its velocity
    effect at the hand is [ z_i x (p_hand - p_i) ; z_i ].
    """
    T = np.eye(4)
    origins = np.empty((7, 3))
    axes = np.empty((7, 3))
    for i, (alpha, a, d) in enumerate(DH_PARAMS):
        origins[i] = T[:3, 3]
        axes[i] = T[:3, 2]
        T = T @ dh_transform(alpha, a, d, q[i])
    hand_pose = T @ T_L7_HAND
    hand_pos = hand_pose[:3, 3]
    J = np.empty((6, 7))
    J[:3, :] = np.cross(axes, hand_pos - origins).T
    J[3:, :] = axes.T
    return hand_pose, J


def _rotation_error(target_R, actual_R):
    """SO(3) logarithm as a world-frame rotation vector, robust near 180 deg.

    Convert R_target R_actual.T to its shortest-arc unit quaternion and then
    to axis*angle. Unlike asin/skew-only formulas, this also works near pi.
    """
    R = target_R @ actual_R.T
    tr = np.trace(R)
    if tr > 0:
        s = np.sqrt(max(0.0, 1.0 + tr)) * 2.0
        quat = np.array([(R[2, 1] - R[1, 2]) / s,
                         (R[0, 2] - R[2, 0]) / s,
                         (R[1, 0] - R[0, 1]) / s,
                         0.25 * s])
    else:
        i = int(np.argmax(np.diag(R)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = np.sqrt(max(0.0, 1.0 + R[i, i] - R[j, j] - R[k, k])) * 2.0
        quat = np.zeros(4)
        quat[i] = 0.25 * s
        quat[j] = (R[i, j] + R[j, i]) / s
        quat[k] = (R[i, k] + R[k, i]) / s
        quat[3] = (R[k, j] - R[j, k]) / s
    quat /= np.linalg.norm(quat)
    if quat[3] < 0.0:
        quat = -quat
    norm_v = np.linalg.norm(quat[:3])
    if norm_v < 1e-12:
        return 2.0 * quat[:3]
    angle = 2.0 * np.arctan2(norm_v, quat[3])
    return quat[:3] * (angle / norm_v)


def _pose_error(target, actual):
    pos = target[:3, 3] - actual[:3, 3]
    rot = _rotation_error(target[:3, :3], actual[:3, :3])
    return pos, rot


def _solve_from_seed(target, seed, *, max_iterations=180):
    """One bounded, joint-limit-aware damped least-squares solve attempt."""
    q = np.clip(seed.copy(), Q_MIN, Q_MAX)
    best_q = q.copy()
    best_cost = float('inf')
    I6 = np.eye(6)
    position_tol = 0.002           # 2 mm
    orientation_tol = 0.01         # about 0.57 degrees
    orientation_weight = 0.40      # metres / radian for least squares
    damping = 0.055               # fixed positive damping at singularities

    for _ in range(max_iterations):
        T, J = _fk_and_jacobian(q)
        dp, dr = _pose_error(target, T)
        p_err, r_err = np.linalg.norm(dp), np.linalg.norm(dr)
        cost = p_err**2 + (orientation_weight * r_err)**2
        if cost < best_cost:
            best_q, best_cost = q.copy(), cost
        if p_err < position_tol and r_err < orientation_tol:
            return q, True, cost

        ew = np.r_[dp, orientation_weight * dr]
        Jw = J.copy()
        Jw[3:, :] *= orientation_weight
        # J^T (J J^T + lambda^2 I)^-1 e, no undamped inverse.
        dq = Jw.T @ np.linalg.solve(Jw @ Jw.T + damping*damping*I6, ew)
        # Limit any one iteration's joint motion to avoid unstable jumps.
        max_abs = float(np.max(np.abs(dq)))
        if max_abs > 0.25:
            dq *= 0.25 / max_abs

        improved = False
        for fraction in (1.0, 0.5, 0.25, 0.125, 0.0625):
            q_next = np.clip(q + fraction*dq, Q_MIN, Q_MAX)
            if np.max(np.abs(q_next - q)) < 1e-11:
                continue
            next_pose = forward_kinematics(q_next)
            next_dp, next_dr = _pose_error(target, next_pose)
            next_cost = np.dot(next_dp, next_dp) + (orientation_weight**2)*np.dot(next_dr, next_dr)
            if next_cost < cost - 1e-14:
                q = q_next
                improved = True
                break
        if not improved:
            break

    T_best = forward_kinematics(best_q)
    dp, dr = _pose_error(target, T_best)
    ok = np.linalg.norm(dp) < position_tol and np.linalg.norm(dr) < orientation_tol
    return best_q, bool(ok), best_cost


def inverse_kinematics(target_pose, q0):
    """Return (7-joint solution, success flag) for the desired hand pose.

    Uses the initial seed first, then a small number of deterministic backup
    seeds if it gets stuck in a local minimum. Joint 3 is NOT fixed. Solutions
    are checked against position and orientation tolerances and joint limits.
    Failure always returns a *valid* joint vector and False: do not command it.

    Numerical IK is local and not collision-aware. False may mean that this
    solver failed to find a pose, not a mathematical proof of unreachability.
    """
    q0 = _as_q(q0, 'q0')
    target = np.asarray(target_pose, dtype=float)
    if target.shape != (4, 4) or not np.all(np.isfinite(target)):
        return np.clip(q0.copy(), Q_MIN, Q_MAX), False
    R = target[:3, :3]
    if (not np.allclose(target[3], [0, 0, 0, 1], atol=1e-5)
        or not np.allclose(R.T @ R, np.eye(3), atol=1e-3)
        or np.linalg.det(R) < 0.999):
        return np.clip(q0.copy(), Q_MIN, Q_MAX), False

    seed = np.clip(q0, Q_MIN, Q_MAX)
    # Conservative bound: the Panda hand cannot be this far from link0.
    # Only a fast rejection, never used as proof for less obvious positions.
    if np.linalg.norm(target[:3, 3]) > 1.5:
        return seed.copy(), False

    # Near seed solution preferred. Fallbacks help local minimum and singularities.
    home = np.array([0., 0., 0., -1.57079, 0., 1.57079, -0.7853])
    starts = [seed, home]
    # Try another wrist/elbow branch if the current seed gets stuck. A Panda
    # has one redundant joint: different joint vectors can reach the same pose.
    starts += [
        np.clip(np.array([seed[0], -seed[1], -seed[2], -1.6,
                          -seed[4], 1.8, seed[6]]), Q_MIN, Q_MAX),
        np.clip(np.array([seed[0], 0.5, 0.8, -2.0, 0.5, 2.4, 0.0]), Q_MIN, Q_MAX),
    ]
    # Additional deterministic restarts around the target direction. These
    # don't guarantee a solution but help escape some local minima without
    # requiring SciPy or random/non-reproducible initializations.
    azimuth = float(np.arctan2(target[1, 3], target[0, 3]))
    for shoulder_offset in (0.0, -0.7, 0.7):
        for elbow_branch in (-1.0, 1.0):
            starts.append(np.clip(np.array([
                azimuth + shoulder_offset,
                0.5 * elbow_branch,
                elbow_branch,
                -1.6,
                -0.8 * elbow_branch,
                1.6,
                0.5 * elbow_branch,
            ]), Q_MIN, Q_MAX))
    best_q, best_cost = seed.copy(), float('inf')
    for start in starts:
        q, ok, cost = _solve_from_seed(target, start)
        if cost < best_cost:
            best_q, best_cost = q, cost
        if ok:
            return q, True
    return best_q, False


# ==========================================================
# RPC 2: Velocity kinematics, acceleration kinematics, planning
# ==========================================================


def compute_jacobian(q):
    """3x7 LINEAR geometric Jacobian of the hand, in link0 coordinates.

    For revolute joint i: Jv[:,i] = z_i cross (p_hand - p_i).
    The axes/origins are evaluated BEFORE applying joint i's DH transform.
    Multiplying Jv @ dq yields hand linear velocity in metres/second.
    """
    q = _as_q(q, 'q')
    T = np.eye(4)
    origins = np.empty((7, 3), dtype=float)
    z_axes = np.empty((7, 3), dtype=float)

    for i, (alpha, a, d) in enumerate(DH_PARAMS):
        origins[i] = T[:3, 3]
        z_axes[i] = T[:3, 2]
        T = T @ dh_transform(alpha, a, d, q[i])

    p_hand = (T @ T_L7_HAND)[:3, 3]
    return np.cross(z_axes, p_hand - origins).T


def compute_jacobian_dot(q, dq):
    """3x7 time derivative of Jv (1/second), using central differences.

    q_plus/minus = q +/- h*dq, so (J(q+h*dq)-J(q-h*dq))/(2*h)
    approximates dJ/dt along the CURRENT joint velocity direction.
    Thus hand linear acceleration = Jv(q) @ ddq + Jv_dot(q,dq) @ dq.
    """
    q = _as_q(q, 'q')
    dq = _as_q(dq, 'dq')
    if not np.any(dq):
        return np.zeros((3, 7), dtype=float)

    # Choose a small perturbation in joint-angle space even at high speeds.
    h = 1e-4 / max(1.0, float(np.max(np.abs(dq))))  # seconds
    return (compute_jacobian(q + h*dq) -
            compute_jacobian(q - h*dq)) / (2.0*h)


def plan_trajectory(q_start, q_end, T, dt):
    """Joint-space quintic with zero velocity AND acceleration at both ends.

    u=t/T, s=10u^3-15u^4+6u^5, q=q_start+(q_end-q_start)*s.
    Also return analytic dq/dt and ddq/dt^2, not noisy differences.
    Uses a final sample at exactly T, even when T/dt is not an integer.
    Note: the provided ROS timer runs at dt, so choose T divisible by dt
    for time-exact live playback (e.g. T=4, dt=0.05).
    """
    q_start = _as_q(q_start, 'q_start')
    q_end = _as_q(q_end, 'q_end')
    T, dt = float(T), float(dt)
    if not np.isfinite(T) or T <= 0 or not np.isfinite(dt) or dt <= 0:
        raise ValueError('T and dt must be positive, finite seconds')
    if np.any(q_start < Q_MIN) or np.any(q_start > Q_MAX):
        raise ValueError('q_start is outside the Panda joint limits')
    if np.any(q_end < Q_MIN) or np.any(q_end > Q_MAX):
        raise ValueError('q_end is outside the Panda joint limits')

    n = int(np.floor(T / dt))
    times = np.arange(n + 1, dtype=float) * dt
    # Floating-point guard: avoid including a sample slightly beyond T.
    times = times[times < T - 1e-10]
    times = np.concatenate([times, [T]])

    u = times / T
    s = 10*u**3 - 15*u**4 + 6*u**5
    ds = (30*u**2 - 60*u**3 + 30*u**4) / T
    dds = (60*u - 180*u**2 + 120*u**3) / (T*T)
    delta = q_end - q_start

    q_array = q_start[None, :] + s[:, None] * delta[None, :]
    dq_array = ds[:, None] * delta[None, :]
    ddq_array = dds[:, None] * delta[None, :]

    # Remove round-off at boundary points to show exact boundary conditions.
    q_array[0], q_array[-1] = q_start, q_end
    dq_array[[0, -1], :] = 0.0
    ddq_array[[0, -1], :] = 0.0
    return times, q_array, dq_array, ddq_array


# ==========================================================
# INFRASTRUCTURE — DO NOT MODIFY BELOW
# ==========================================================

CMD_SIZE = 9
GRIPPER_CTRL = 255.0

# UPDATE THESE BASED ON YOUR ASSIGNED VARIANT
# You must choose reachable configurations within joint limits.
# TEAM 1: neutral home pose -> moderate forward reach and lift, near centerline.
# Endpoints are chosen within joint limits; quintic interpolation cannot overshoot.
Q_START = np.array([0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853])
Q_END   = np.array([0.14, 0.16, -0.18, -1.31, 0.21, 1.83, -0.59])

TRAJ_DURATION = 4.0
TRAJ_DT       = 0.05


class TrajectoryNode(Node):
    def __init__(self):
        super().__init__('trajectory_node')
        self.q_current = np.zeros(7)
        self.dq_current = np.zeros(7)

        self.create_subscription(
            JointState, 'joint_state', self.on_joint_state, 10)
        self.cmd_pub = self.create_publisher(
            Float64MultiArray, 'joint_commands', 10)

        self.t_array = None
        self.q_traj = None
        self.dq_traj = None
        self.ddq_traj = None
        self.step_idx = 0

        # Wait 0.5s before starting to ensure the sim is up
        self.create_timer(0.5, self.start_trajectory_once)

    def on_joint_state(self, msg):
        self.q_current = np.array(msg.position[:7])
        if hasattr(msg, 'velocity') and len(msg.velocity) >= 7:
            self.dq_current = np.array(msg.velocity[:7])

    def start_trajectory_once(self):
        if self.t_array is not None:
            return

        result = plan_trajectory(Q_START, Q_END, TRAJ_DURATION, TRAJ_DT)
        if result is None:
            return

        t_arr, q_arr, dq_arr, ddq_arr = result

        if q_arr.shape[1] != 7 or dq_arr.shape[1] != 7 or ddq_arr.shape[1] != 7:
            self.get_logger().error(
                "plan_trajectory returned wrong shape. Expected (N, 7).")
            return

        self.t_array = t_arr
        self.q_traj = q_arr
        self.dq_traj = dq_arr
        self.ddq_traj = ddq_arr

        self.get_logger().info(
            f"Trajectory generated: {len(t_arr)} samples over "
            f"{TRAJ_DURATION:.1f} s at {1.0/TRAJ_DT:.0f} Hz.")

        self.create_timer(TRAJ_DT, self.tick)

    def tick(self):
        if self.step_idx >= len(self.t_array):
            # Hold final target
            cmd = Float64MultiArray()
            
            # ==================================================
            # DEPENDING ON YOUR BRIDGE MODE, UNCOMMENT ONE BELOW:
            # ==================================================
            
            # -- POSITION MODE (Default) --
            cmd.data = Q_END.tolist() + [GRIPPER_CTRL, 0.0]
            
            # -- VELOCITY MODE --
            # cmd.data = [0.0] * 7 + [GRIPPER_CTRL, 0.0]
            
            self.cmd_pub.publish(cmd)
            return

        cmd = Float64MultiArray()
        
        # ==================================================
        # DEPENDING ON YOUR BRIDGE MODE, UNCOMMENT ONE BELOW:
        # ==================================================
        
        # -- POSITION MODE (Default) --
        q_target = self.q_traj[self.step_idx]
        
        # -- VELOCITY MODE --
        # q_target = self.dq_traj[self.step_idx]
        
        cmd.data = q_target.tolist() + [GRIPPER_CTRL, 0.0]
        self.cmd_pub.publish(cmd)

        self.step_idx += 1


def main(args=None):
    rclpy.init(args=args)
    node = TrajectoryNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
