
import csv
import time
from pathlib import Path

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState


class ConstantController(Node):
    def __init__(self):
        super().__init__("ur5e_constant_controller")

        self.publisher = self.create_publisher(
            Float64MultiArray, "/joint_commands", 10
        )

        self.subscription = self.create_subscription(
            JointState,
            "/joint_state",
            self.feedback_callback,
            10
        )

        self.actual = None
        self.target = None
        self.start_time = None
        self.counter = 0

        output = Path("Milestone 02/Figures")
        output.mkdir(parents=True, exist_ok=True)

        self.csv_file = open(
            output / "constant_j1_response.csv",
            "w",
            newline=""
        )
        self.writer = csv.writer(self.csv_file)
        self.writer.writerow([
            "time_s", "desired_j1_rad",
            "actual_j1_rad", "error_rad"
        ])

        self.timer = self.create_timer(
            0.1, self.publish_command
        )

        self.get_logger().info(
            "Waiting for joint-state feedback..."
        )

    def feedback_callback(self, msg):
        if len(msg.position) != 6:
            return

        self.actual = list(msg.position)

        if self.target is None:
            self.target = self.actual.copy()
            self.target[0] += 0.2

            self.start_time = time.monotonic()

            self.get_logger().info(
                f"Fixed target: {self.target}"
            )

    def publish_command(self):
        if self.target is None or self.actual is None:
            return

        command = Float64MultiArray()
        command.data = self.target
        self.publisher.publish(command)

        elapsed = time.monotonic() - self.start_time
        error = self.target[0] - self.actual[0]

        self.writer.writerow([
            elapsed,
            self.target[0],
            self.actual[0],
            error
        ])

        self.counter += 1

        if self.counter % 10 == 0:
            self.get_logger().info(
                f"t={elapsed:.1f}s | "
                f"Desired J1={self.target[0]:.4f} | "
                f"Actual J1={self.actual[0]:.4f} | "
                f"Error={error:.4f}"
            )

    def destroy_node(self):
        self.csv_file.close()
        return super().destroy_node()


def main():
    rclpy.init()
    node = ConstantController()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()

