"""Measure the live ROS 2 /joint_state publication rate.

Run this from the MCTR911 repository root after sourcing ROS 2 Jazzy and the
mujoco_ros2 workspace. Start the MuJoCo bridge in another terminal first.
This is a read-only diagnostic; it publishes no joint commands.
"""
from __future__ import annotations

import argparse
import math
import time

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState


class JointStateRateMonitor(Node):
    def __init__(self, topic: str):
        super().__init__("m4_joint_state_rate_monitor")
        self.topic = topic
        self.count = 0
        self.first_time: float | None = None
        self.last_time: float | None = None
        self.intervals: list[float] = []
        self.create_subscription(JointState, topic, self._callback, 50)

    def _callback(self, message: JointState) -> None:
        now = time.monotonic()
        if self.first_time is None:
            self.first_time = now
        if self.last_time is not None:
            interval = now - self.last_time
            if math.isfinite(interval) and interval > 0.0:
                self.intervals.append(interval)
        self.last_time = now
        self.count += 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topic", default="/joint_state")
    parser.add_argument("--duration", type=float, default=10.0)
    args = parser.parse_args()
    if not math.isfinite(args.duration) or args.duration <= 0.0:
        parser.error("--duration must be positive and finite")

    rclpy.init()
    node = JointStateRateMonitor(args.topic)
    started = time.monotonic()
    print(f"Monitoring {args.topic} for {args.duration:.1f} seconds; read-only.")
    try:
        while rclpy.ok() and time.monotonic() - started < args.duration:
            rclpy.spin_once(node, timeout_sec=0.05)
    except KeyboardInterrupt:
        pass
    finally:
        elapsed = time.monotonic() - started
        node.destroy_node()
        rclpy.shutdown()

    print("\n========== ROS 2 JOINT-STATE RATE ==========")
    print("Topic:", args.topic)
    print(f"Measurement duration: {elapsed:.3f} s")
    print("Received messages:", node.count)
    if node.count >= 2 and node.intervals:
        intervals = sorted(node.intervals)
        mean_interval = sum(intervals) / len(intervals)
        median_interval = intervals[len(intervals) // 2]
        p95_interval = intervals[min(len(intervals) - 1, int(0.95 * len(intervals)))]
        print(f"Average rate: {1.0 / mean_interval:.3f} Hz")
        print(f"Median rate: {1.0 / median_interval:.3f} Hz")
        print(f"Minimum interval: {min(intervals):.6f} s")
        print(f"Median interval: {median_interval:.6f} s")
        print(f"95th-percentile interval: {p95_interval:.6f} s")
        print(f"Maximum interval: {max(intervals):.6f} s")
    else:
        print("Insufficient messages to estimate a rate.")
    print("============================================")


if __name__ == "__main__":
    main()
