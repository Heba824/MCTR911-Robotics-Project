#!/usr/bin/env python3
"""Interactive six-joint UR5e position controller for mujoco_ros2.

Run from the repository root after sourcing ROS 2 Jazzy.
ROS topics:
  /joint_state    sensor_msgs/msg/JointState
  /joint_commands std_msgs/msg/Float64MultiArray

Each slider sets an offset relative to the joint angle observed when the GUI
starts (or when 'Hold current pose' is pressed). The default range is +/-0.30 rad.
"""

import csv
import math
import time
from datetime import datetime
from pathlib import Path

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
import tkinter as tk
from tkinter import ttk


JOINT_NAMES = (
    "Shoulder pan (J1)",
    "Shoulder lift (J2)",
    "Elbow (J3)",
    "Wrist 1 (J4)",
    "Wrist 2 (J5)",
    "Wrist 3 (J6)",
)
SLIDER_RANGE_RAD = 0.30  # offsets, not absolute joint limits
PUBLISH_INTERVAL_MS = 100  # 10 Hz
ROS_POLL_INTERVAL_MS = 25


class UR5eSliderGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("UR5e Joint Sliders — ROS 2 / MuJoCo")
        self.root.geometry("800x650")

        rclpy.init()
        self.node = Node("ur5e_slider_controller")
        self.publisher = self.node.create_publisher(
            Float64MultiArray, "/joint_commands", 10
        )
        self.subscriber = self.node.create_subscription(
            JointState, "/joint_state", self._joint_state_callback, 10
        )

        self.actual = None
        self.reference = None
        self.target = None
        self.last_feedback_at = None
        self.started_at = None
        self.rows_written = 0
        self._resetting_sliders = False
        self._closed = False

        output_dir = Path("Milestone 02/Figures")
        output_dir.mkdir(parents=True, exist_ok=True)
        filename = "slider_motion_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".csv"
        self.output_path = output_dir / filename
        self.csv_file = self.output_path.open("w", newline="")
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(
            ["time_s"]
            + [f"desired_j{i}_rad" for i in range(1, 7)]
            + [f"actual_j{i}_rad" for i in range(1, 7)]
        )

        content = ttk.Frame(self.root, padding=14)
        content.pack(fill="both", expand=True)
        ttk.Label(
            content,
            text="UR5e — Live Joint Position Control",
            font=("TkDefaultFont", 14, "bold"),
        ).grid(row=0, column=0, columnspan=4, sticky="w", pady=(0, 10))
        ttk.Label(content, text="Joint").grid(row=1, column=0, sticky="w")
        ttk.Label(content, text="Offset from hold pose (rad)").grid(row=1, column=1)
        ttk.Label(content, text="Target (rad)").grid(row=1, column=2)
        ttk.Label(content, text="Actual (rad)").grid(row=1, column=3)

        self.sliders = []
        self.target_labels = []
        self.actual_labels = []
        for i, name in enumerate(JOINT_NAMES):
            row = i + 2
            ttk.Label(content, text=name).grid(
                row=row, column=0, sticky="w", padx=(0, 12), pady=6
            )
            slider = tk.Scale(
                content,
                from_=-SLIDER_RANGE_RAD,
                to=SLIDER_RANGE_RAD,
                resolution=0.01,
                orient="horizontal",
                length=330,
                showvalue=True,
                command=lambda _value, j=i: self._slider_changed(j),
            )
            slider.set(0.0)
            slider.configure(state="disabled")
            slider.grid(row=row, column=1, sticky="ew")
            target_label = ttk.Label(content, text="--")
            target_label.grid(row=row, column=2, padx=16)
            actual_label = ttk.Label(content, text="--")
            actual_label.grid(row=row, column=3, padx=16)
            self.sliders.append(slider)
            self.target_labels.append(target_label)
            self.actual_labels.append(actual_label)

        self.hold_button = ttk.Button(
            content,
            text="Hold current pose / Recenter sliders",
            command=self._hold_current_pose,
            state="disabled",
        )
        self.hold_button.grid(row=8, column=0, columnspan=2, sticky="w", pady=(15, 8))

        self.status_var = tk.StringVar(value="Waiting for /joint_state feedback...")
        ttk.Label(
            content, textvariable=self.status_var, wraplength=720
        ).grid(row=9, column=0, columnspan=4, sticky="w", pady=8)
        ttk.Label(
            content,
            text="Move one slider at a time. The slider range is ±0.30 rad from the hold pose. "
                 "Close this window to stop publishing.",
            wraplength=720,
        ).grid(row=10, column=0, columnspan=4, sticky="w")

        content.columnconfigure(1, weight=1)
        self.root.protocol("WM_DELETE_WINDOW", self._close)
        self.root.after(ROS_POLL_INTERVAL_MS, self._spin_ros)
        self.root.after(PUBLISH_INTERVAL_MS, self._publish_and_log)

        self.node.get_logger().info(
            f"Waiting for feedback; logging to {self.output_path.resolve()}"
        )

    def _joint_state_callback(self, msg):
        if len(msg.position) != 6 or not all(math.isfinite(v) for v in msg.position):
            return
        self.actual = list(msg.position)
        self.last_feedback_at = time.monotonic()

        if self.reference is None:
            self.reference = self.actual.copy()
            self.target = self.actual.copy()
            self.started_at = time.monotonic()
            for slider in self.sliders:
                slider.configure(state="normal")
            for i, angle in enumerate(self.target):
                self.target_labels[i].configure(text=f"{angle:+.3f}")
            self.hold_button.configure(state="normal")
            self.status_var.set("Feedback received. Move a slider to command that joint.")
            self.node.get_logger().info(f"Initial joint angles: {self.reference}")

        for i, angle in enumerate(self.actual):
            self.actual_labels[i].configure(text=f"{angle:+.3f}")

    def _slider_changed(self, _joint_index):
        if self.reference is None or self._resetting_sliders:
            return
        self.target = [
            self.reference[i] + self.sliders[i].get() for i in range(6)
        ]
        for i, angle in enumerate(self.target):
            self.target_labels[i].configure(text=f"{angle:+.3f}")

    def _hold_current_pose(self):
        if self.actual is None:
            return
        self.reference = self.actual.copy()
        self.target = self.actual.copy()
        self._resetting_sliders = True
        try:
            for slider in self.sliders:
                slider.set(0.0)
        finally:
            self._resetting_sliders = False
        for i, angle in enumerate(self.target):
            self.target_labels[i].configure(text=f"{angle:+.3f}")
        self.status_var.set("Current measured pose is now the reference; slider offsets reset.")

    def _spin_ros(self):
        if self._closed:
            return
        try:
            rclpy.spin_once(self.node, timeout_sec=0.0)
        except Exception as exc:
            self.status_var.set(f"ROS 2 error: {exc}")
        self.root.after(ROS_POLL_INTERVAL_MS, self._spin_ros)

    def _publish_and_log(self):
        if self._closed:
            return
        if self.target is not None and self.actual is not None:
            feedback_age = time.monotonic() - self.last_feedback_at
            if feedback_age < 2.0:
                command = Float64MultiArray()
                command.data = list(self.target)
                self.publisher.publish(command)
                elapsed = time.monotonic() - self.started_at
                self.csv_writer.writerow([elapsed] + self.target + self.actual)
                self.rows_written += 1
                if self.rows_written % 10 == 0:
                    self.csv_file.flush()
            else:
                self.status_var.set(
                    "Joint feedback is more than 2 seconds old; publishing paused."
                )
        self.root.after(PUBLISH_INTERVAL_MS, self._publish_and_log)

    def _close(self):
        if self._closed:
            return
        self._closed = True
        self.csv_file.close()
        self.node.get_logger().info(
            f"Saved {self.rows_written} samples to {self.output_path.resolve()}"
        )
        self.node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        self.root.destroy()

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    UR5eSliderGUI().run()
