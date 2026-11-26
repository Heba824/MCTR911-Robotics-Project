#!/usr/bin/env python3
"""Wait for one ROS 2 JointState by explicit type (no CLI type discovery)."""
import argparse
import sys
import time

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('topic')
    parser.add_argument('--timeout', type=float, default=40.0)
    args = parser.parse_args()

    rclpy.init()
    node = Node('rpc2_joint_state_ready')
    received = False

    def callback(msg):
        nonlocal received
        if len(msg.position) >= 7:
            received = True
            print(f'READY {args.topic}: received {len(msg.position)} joint positions', flush=True)

    subscriber = node.create_subscription(JointState, args.topic, callback, 10)
    try:
        deadline = time.monotonic() + args.timeout
        while rclpy.ok() and time.monotonic() < deadline and not received:
            rclpy.spin_once(node, timeout_sec=0.2)
        if not received:
            print(f'TIMEOUT: no JointState received on {args.topic} after {args.timeout:g} s', file=sys.stderr)
            return 1
        return 0
    finally:
        node.destroy_subscription(subscriber)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    sys.exit(main())
