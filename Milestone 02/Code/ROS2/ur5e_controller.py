
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState


class UR5eController(Node):
    def __init__(self):
        super().__init__("ur5e_python_controller")

        self.publisher = self.create_publisher(
            Float64MultiArray, "/joint_commands", 10
        )

        self.subscriber = self.create_subscription(
            JointState,
            "/joint_state",
            self.joint_state_callback,
            10
        )

        self.current_position = None
        self.target_position = None
        self.counter = 0

        # Publish joint targets at 10 Hz.
        self.timer = self.create_timer(0.1, self.send_command)

        self.get_logger().info(
            "Waiting for UR5e joint feedback..."
        )

    def joint_state_callback(self, msg):
        if len(msg.position) != 6:
            return

        self.current_position = list(msg.position)

        # Set the target once, using the initial robot pose.
        if self.target_position is None:
            self.target_position = self.current_position.copy()

            # Rotate the first joint by +0.3 rad.
            self.target_position[0] += 0.3

            self.get_logger().info(
                f"Target joint angles: {self.target_position}"
            )

    def send_command(self):
        if self.target_position is None:
            return

        command = Float64MultiArray()
        command.data = self.target_position
        self.publisher.publish(command)

        self.counter += 1

        if self.counter % 20 == 0:
            self.get_logger().info(
                f"Actual J1: {self.current_position[0]:.4f} rad | "
                f"Target J1: {self.target_position[0]:.4f} rad"
            )


def main(args=None):
    rclpy.init(args=args)
    node = UR5eController()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

