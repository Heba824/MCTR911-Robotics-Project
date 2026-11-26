#!/usr/bin/env python3
"""RPC 1: Franka Panda hand forward and inverse position kinematics.

Works with ROS 2 Jazzy and the assignment's MuJoCo ROS 2 bridge.
The arm has seven revolute joints; the two remaining controls are for the gripper.
All lengths are metres, all joint angles are radians, and poses are relative to
MuJoCo's `link0` body frame.

The DH parameters below use *standard*, not modified, DH:
    A_i(q_i) = Rz(q_i) Tz(d_i) Tx(a_i) Rx(alpha_i)
"""

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray

# ==========================================================
# KINEMATICS IMPLEMENTATION
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

# A demonstrably reachable target is generated from a joint configuration
# strictly inside the limits. For another target change Q_TARGET, not the FK.


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


# This is a proper hand target, not a link7 target.
TARGET_POSE = forward_kinematics(
    np.array([0.8, -0.3, 0.0, -1.8, 0.0, 1.9, 0.785], dtype=float)
)


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
# ROS 2 Jazzy node / bridge interface
# ==========================================================
CMD_SIZE = 9
GRIPPER_CTRL = 255.0


class PandaKinematicsNode(Node):
    def __init__(self):
        super().__init__('panda_kinematics')
        self.q_current = None
        self.q_command = None
        self.last_solve_time_ns = None

        self.create_subscription(JointState, 'joint_state', self.on_joint_state, 10)
        self.cmd_pub = self.create_publisher(Float64MultiArray, 'joint_commands', 10)
        self.create_timer(0.05, self.tick)
        self.get_logger().info(f"Target hand pose relative to link0:\n{np.round(TARGET_POSE, 4)}")

    def on_joint_state(self, msg):
        if len(msg.position) < 7:
            return
        q = np.asarray(msg.position[:7], dtype=float)
        if np.all(np.isfinite(q)):
            self.q_current = q

    def tick(self):
        # Don't solve until the simulator sends a real joint configuration.
        if self.q_current is None:
            return

        if self.q_command is None:
            # Don't burn all CPU trying unreachable targets at 20 Hz.
            now_ns = self.get_clock().now().nanoseconds
            if (self.last_solve_time_ns is not None and
                now_ns - self.last_solve_time_ns < 2_000_000_000):
                return
            self.last_solve_time_ns = now_ns
            q_solution, ok = inverse_kinematics(TARGET_POSE, self.q_current)
            if not ok:
                self.get_logger().warn('IK failed: no joint command published.', throttle_duration_sec=2.0)
                return
            self.q_command = q_solution
            self.get_logger().info(f"IK solution (rad): {np.round(q_solution, 4)}")

        # The bridge requires exactly 9 floats: 7 arm joints + 2 gripper slots.
        command = Float64MultiArray()
        command.data = self.q_command.tolist() + [GRIPPER_CTRL, 0.0]
        assert len(command.data) == CMD_SIZE
        self.cmd_pub.publish(command)


def main(args=None):
    rclpy.init(args=args)
    node = PandaKinematicsNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
