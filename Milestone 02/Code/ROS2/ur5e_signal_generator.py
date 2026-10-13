#!/usr/bin/env python3
"""Milestone 02: UR5e ROS 2 position-command signal generator.

Run while the MuJoCo bridge is active in POSITION mode. One command publisher
at a time. Records requested and observed joint angles (all six) to CSV.
"""

import argparse
import csv
import math
import time
from datetime import datetime
from pathlib import Path

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray

JOINT_COUNT = 6


def signal_offset(mode, t, amplitude, frequency, ramp_time):
    """Desired displacement from the initial joint position, in radians."""
    if mode == "step":
        return amplitude if t >= 1.0 else 0.0
    if mode == "ramp":
        return amplitude * min(t / ramp_time, 1.0)
    if mode == "sine":
        return amplitude * math.sin(2.0 * math.pi * frequency * t)
    raise ValueError(f"Unsupported mode: {mode}")


class UR5eSignalGenerator(Node):
    def __init__(self, args):
        super().__init__("ur5e_signal_generator")
        self.args = args
        self.joint_index = args.joint - 1
        self.initial = None
        self.actual = None
        self.start = None
        self.last_log = 0.0
        self.finished = False
        self.records = 0

        self.publisher = self.create_publisher(
            Float64MultiArray, "/joint_commands", 10
        )
        self.subscription = self.create_subscription(
            JointState, "/joint_state", self.on_joint_state, 10
        )
        # /Milestone 02/Code/ROS2/file.py -> /Milestone 02/Figures
        figures_dir = Path(__file__).resolve().parents[2] / "Figures"
        figures_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_path = (
            figures_dir / f"signal_{args.mode}_j{args.joint}_{timestamp}.csv"
        )
        self.file = self.csv_path.open("w", newline="", encoding="utf-8")
        self.writer = csv.writer(self.file)
        self.writer.writerow(
            ["time_s", "mode"]
            + [f"desired_j{i}_rad" for i in range(1, 7)]
            + [f"actual_j{i}_rad" for i in range(1, 7)]
            + [f"error_j{i}_rad" for i in range(1, 7)]
        )
        self.timer = self.create_timer(1.0 / args.rate, self.on_timer)
        self.get_logger().info(
            f"Waiting for six-joint feedback; mode={args.mode}, "
            f"joint=J{args.joint}, amplitude={args.amplitude:.3f} rad"
        )

    def on_joint_state(self, msg):
        # This bridge reports UR5e positions in model/actuator order.
        if len(msg.position) != JOINT_COUNT:
            return
        values = [float(q) for q in msg.position]
        if not all(math.isfinite(q) for q in values):
            return
        self.actual = values
        if self.initial is None:
            self.initial = values.copy()
            self.start = time.monotonic()
            self.get_logger().info(f"Initial joint angles [rad]: {self.initial}")

    def on_timer(self):
        if self.initial is None or self.actual is None or self.finished:
            return
        elapsed = time.monotonic() - self.start
        if elapsed > self.args.duration:
            self.finished = True
            self.get_logger().info(
                f"Experiment finished: {self.records} samples saved to {self.csv_path}"
            )
            return

        offset = signal_offset(
            self.args.mode, elapsed, self.args.amplitude,
            self.args.frequency, self.args.ramp_time
        )
        desired = self.initial.copy()
        desired[self.joint_index] += offset

        command = Float64MultiArray()
        command.data = desired
        self.publisher.publish(command)

        errors = [d - a for d, a in zip(desired, self.actual)]
        self.writer.writerow(
            [f"{elapsed:.5f}", self.args.mode]
            + [f"{v:.8f}" for v in desired]
            + [f"{v:.8f}" for v in self.actual]
            + [f"{v:.8f}" for v in errors]
        )
        self.records += 1
        if elapsed - self.last_log >= 1.0:
            self.last_log = elapsed
            j = self.joint_index
            self.get_logger().info(
                f"t={elapsed:5.1f}s | desired J{j+1}={desired[j]:+.4f} "
                f"| actual={self.actual[j]:+.4f} | error={errors[j]:+.4f} rad"
            )
            self.file.flush()

    def close(self):
        if not self.file.closed:
            self.file.flush()
            self.file.close()
            self.get_logger().info(f"CSV saved: {self.csv_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Publish UR5e step, ramp or sine joint-position references."
    )
    parser.add_argument("--mode", choices=["step", "ramp", "sine"], default="sine")
    parser.add_argument("--joint", type=int, choices=range(1, 7), metavar="1..6", default=1)
    parser.add_argument("--amplitude", type=float, default=0.04, help="Max angle offset (radians).")
    parser.add_argument("--frequency", type=float, default=0.025, help="Sine frequency in Hz.")
    parser.add_argument("--ramp-time", type=float, default=12.0, help="Ramp duration in seconds.")
    parser.add_argument("--duration", type=float, default=40.0, help="Total run time in seconds.")
    parser.add_argument("--rate", type=float, default=20.0, help="Command publishing rate (Hz).")
    args = parser.parse_args()
    if not math.isfinite(args.amplitude) or not 0 < args.amplitude <= 0.3:
        parser.error("--amplitude must be between 0 and 0.3 radians")
    for name in ("frequency", "ramp_time", "duration", "rate"):
        v = getattr(args, name)
        if not math.isfinite(v) or v <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    return args


def main():
    args = parse_args()
    rclpy.init()
    node = UR5eSignalGenerator(args)
    try:
        while rclpy.ok() and not node.finished:
            rclpy.spin_once(node, timeout_sec=0.2)
    except KeyboardInterrupt:
        node.get_logger().info("Stopped by user")
    finally:
        node.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
