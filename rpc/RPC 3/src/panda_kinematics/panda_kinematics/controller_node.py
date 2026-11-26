#!/usr/bin/env python3
"""Runnable RPC 3 controller. The supplied template is retained under docs/rpc3."""

from contextlib import contextmanager
from functools import lru_cache
import math
import sys
import time
from threading import Event, Thread

import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import ExternalShutdownException, SingleThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray, MultiArrayDimension, String

from .dynamics import PandaDynamics
from .joint_control import JointController, warm_controller
from .panda_model import PandaModel
from .rpc3_runtime import TrackingRun, configured_controller, save_tracking
from .trajectory_node import DH_PARAMS as DH_PARAMS
from .trajectory_node import Q_END, Q_START
from .trajectory_node import T_L7_HAND as T_L7_HAND
from .trajectory_node import plan_trajectory as quintic_trajectory


def plan_trajectory(q_start, q_end, T, dt):
    """Use the verified, limit-checked quintic planner."""
    return quintic_trajectory(q_start, q_end, T, dt)


@lru_cache(maxsize=1)
def _default_controller():
    return JointController(PandaDynamics(PandaModel.from_scene()), 'joint_flc')


def compute_dynamics(q, dq, ddq):
    """Return mass and bias using the template's three-argument contract."""
    mass, coriolis, gravity = _default_controller().dynamics.terms(q, dq)
    return mass, coriolis + gravity


def fuzzy_controller(error, delta_error, dt):
    """Return the torque correction from simultaneous position and velocity errors."""
    if not math.isfinite(dt) or dt <= 0:
        raise ValueError('dt must be positive and finite')
    return _default_controller().fuzzy(error, delta_error)


def compute_torques(q, dq, q_des, dq_des, ddq_des):
    """Return bounded model-plus-FLC torque with the template's defaults."""
    return _default_controller().compute(q, dq, q_des, dq_des, ddq_des).command


@contextmanager
def control_thread_switch_interval():
    """Give the ROS feedback and control threads shorter GIL handoff intervals."""
    previous = sys.getswitchinterval()
    sys.setswitchinterval(min(previous, 0.001))
    try:
        yield
    finally:
        sys.setswitchinterval(previous)


def steady_control_loop(stop, callback, period):
    """Run one control tick per deadline, dropping missed ticks without bursts."""
    deadline = time.monotonic() + period
    while not stop.wait(max(0.0, deadline - time.monotonic())):
        callback(deadline)
        now = time.monotonic()
        deadline += max(1, math.floor((now - deadline)/period) + 1)*period


# ==========================================================
# DO NOT MODIFY BELOW
# ==========================================================

CMD_SIZE = 9
GRIPPER_CTRL = 255.0
TRAJ_DURATION = 4.0
TRAJ_DT = 0.01


class ControllerNode(Node):
    def __init__(self):
        super().__init__('controller_node')
        controller, output = configured_controller(self)
        trajectory = plan_trajectory(Q_START, Q_END, TRAJ_DURATION, TRAJ_DT)
        self.run = TrackingRun(controller, trajectory, TRAJ_DT, output)
        # No timer, subscriber or command publisher exists during warm-up.
        self.run.warmup = warm_controller(controller, trajectory)
        warm_ms = [sample['total_ms'] for sample in self.run.warmup]
        self.get_logger().info(f'Warm-up complete before ROS command setup: '
                               f'first={warm_ms[0]:.3f} ms, max={max(warm_ms):.3f} ms')

        self.feedback_group = MutuallyExclusiveCallbackGroup()
        self.control_group = MutuallyExclusiveCallbackGroup()
        self.create_subscription(JointState, 'joint_state', self.on_joint_state, 1,
                                 callback_group=self.feedback_group)
        self.cmd_pub = self.create_publisher(Float64MultiArray, 'joint_commands', 1)
        self.create_subscription(String, 'torque_status', self.run.on_status, 1,
                                 callback_group=self.feedback_group)

        self.start_timer = self.create_timer(0.5, self.start_trajectory_once,
                                             callback_group=self.control_group)
        self.control_stop = Event()
        self.control_thread = None

    def on_joint_state(self, msg):
        self.run.on_feedback(msg, time.monotonic(), self.get_clock().now().nanoseconds)

    def start_trajectory_once(self):
        if self.run.ready(time.monotonic()):
            self.start_timer.cancel()
            self.control_thread = Thread(target=steady_control_loop,
                                         args=(self.control_stop, self.control_loop, TRAJ_DT),
                                         name='rpc3-control')
            self.get_logger().info(f'Starting {self.run.controller.mode} from verified home')
            self.control_thread.start()
        elif self.run.fault:
            self.start_timer.cancel()
            self.get_logger().error(self.run.fault)

    def control_loop(self, deadline):
        try:
            outcome = self.run.step(time.monotonic(), self.publish_torque, deadline=deadline)
            if outcome in ('discarded', 'startup_deferred'):
                self.get_logger().warning(f'{outcome}: {self.run.diagnostic_message()}',
                                          throttle_duration_sec=1.0)
        except (ValueError, RuntimeError, FloatingPointError) as error:
            if self.control_stop.is_set():
                return
            self.run.latch_fault(str(error))
            self.control_stop.set()
            self.get_logger().error(
                f'{error}. Commands stopped for the bridge watchdog. '
                f'{self.run.diagnostic_message()}'
            )

    def publish_torque(self, torques):
        if self.control_stop.is_set():
            raise RuntimeError('Command publication stopped')
        cmd = Float64MultiArray()
        cmd.data = torques.tolist() + [GRIPPER_CTRL, 0.0]
        if len(cmd.data) != CMD_SIZE:
            raise ValueError('Invalid torque packet length')
        stamp = self.get_clock().now().nanoseconds
        cmd.layout.dim = [MultiArrayDimension(
            label=f'rpc3:{self.run.command_count + 1}:{stamp}', size=CMD_SIZE, stride=CMD_SIZE,
        )]
        self.cmd_pub.publish(cmd)
        return stamp


def main(args=None):
    with control_thread_switch_interval():
        spin_controller(args)


def spin_controller(args):
    rclpy.init(args=args)
    node = None
    executor = None
    try:
        node = ControllerNode()
        executor = SingleThreadedExecutor()
        executor.add_node(node)
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        try:
            if node is not None:
                node.start_timer.cancel()
                node.control_stop.set()
                node.run.shutdown_ros_ns = node.get_clock().now().nanoseconds
            if executor is not None:
                executor.shutdown()
            if node is not None:
                if node.control_thread is not None:
                    node.control_thread.join()
                save_tracking(node.run)
        finally:
            if node is not None:
                node.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()


if __name__ == '__main__':
    main()
