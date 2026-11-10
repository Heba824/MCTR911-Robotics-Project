#!/usr/bin/env python3
"""MCTR911 Milestone 03: UR5e Surgical Handover control console.

GUI-only extension of the team's Milestone 02 six-joint ROS controller.
No robot model, forward/inverse kinematics, or handover automation is altered.

Normal mode (Ubuntu ROS 2):
    python3 "Milestone 03/GUI/surgical_dashboard.py"

UI preview mode (does NOT connect to ROS/MuJoCo or command hardware):
    python3 "Milestone 03/GUI/surgical_dashboard.py" --demo

ROS topics (unchanged):
    /joint_state     sensor_msgs/JointState
    /joint_commands  std_msgs/Float64MultiArray
"""

from __future__ import annotations

import argparse
import csv
import math
import time
from datetime import datetime
from pathlib import Path
import tkinter as tk


JOINT_LABELS = (
    "Shoulder pan",
    "Shoulder lift",
    "Elbow",
    "Wrist 1",
    "Wrist 2",
    "Wrist 3",
)
JOINT_ROS_NAMES = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
)
# MuJoCo actuator control ranges in ur5e.xml: elbow limited to +/- pi.
ACTUATOR_LIMITS = (6.2831, 6.2831, 3.1415, 6.2831, 6.2831, 6.2831)
SLIDER_OFFSET_RANGE = 0.30
SEND_EVERY_MS = 100
ROS_POLL_EVERY_MS = 25
STALE_FEEDBACK_SECONDS = 2.0

COLORS = {
    "bg": "#0A1323",
    "panel": "#132137",
    "panel_soft": "#1A2A42",
    "border": "#293B50",
    "white": "#ECF5FB",
    "muted": "#92A7BA",
    "teal": "#46D2C2",
    "teal_dark": "#133E43",
    "amber": "#FFD18A",
    "red": "#FF8E95",
    "blue": "#85AFFF",
}
FONT = "DejaVu Sans"


def cap(value: float, minimum: float, maximum: float) -> float:
    return min(max(value, minimum), maximum)


class SurgicalDashboard:
    def __init__(self, demo: bool = False):
        self.demo = demo
        self.root = tk.Tk()
        self.root.title("MCTR911  |  Surgical Handover Console")
        self.root.configure(bg=COLORS["bg"])
        self.root.minsize(880, 600)
        width = min(1120, max(900, self.root.winfo_screenwidth() - 80))
        height = min(760, max(650, self.root.winfo_screenheight() - 95))
        self.root.geometry(f"{width}x{height}")

        self.closed = False
        self.enabled = False
        self.reference: list[float] | None = None
        self.target: list[float] | None = None
        self.actual: list[float] | None = None
        self.feedback_time: float | None = None
        self.started_at: float | None = None
        self.rows_written = 0
        self.resetting = False
        self.csv_file = None
        self.csv_writer = None
        self.publisher = None
        self.node = None
        self.ros_module = None
        self.ros_message_class = None

        self.status_text = tk.StringVar(value="DEMO PREVIEW" if demo else "WAITING FOR ROS")
        self.status_description = tk.StringVar(
            value="Preview only. No ROS commands are sent." if demo
            else "Start MuJoCo and wait for fresh /joint_state messages."
        )
        self.current_mode = tk.StringVar(value="SAFE / OUTPUT OFF")
        self.actual_values = [tk.StringVar(value="--") for _ in range(6)]
        self.target_values = [tk.StringVar(value="--") for _ in range(6)]
        self.error_values = [tk.StringVar(value="--") for _ in range(6)]
        self.delta_values = [tk.StringVar(value="+0.00") for _ in range(6)]
        self.log_value = tk.StringVar(value="Recording begins with live feedback")

        if not demo:
            self._setup_ros()
        self._setup_ui()
        self.root.protocol("WM_DELETE_WINDOW", self._close)

        if demo:
            self._on_feedback([0.0, -0.8, 1.2, -1.25, -1.2, 0.0])
            self.status_text.set("DEMO PREVIEW")
            self.root.after(SEND_EVERY_MS, self._demo_tick)
        else:
            self.root.after(ROS_POLL_EVERY_MS, self._poll_ros)
            self.root.after(SEND_EVERY_MS, self._publish)
        self.root.after(300, self._refresh_status)

    def _setup_ros(self):
        try:
            import rclpy
            from rclpy.node import Node
            from sensor_msgs.msg import JointState
            from std_msgs.msg import Float64MultiArray
        except ImportError as exc:
            self.root.destroy()
            raise SystemExit(
                "ROS 2 Python packages are missing. First source /opt/ros/<distro>/setup.bash "
                "and your workspace's install/setup.bash, or use --demo. "
                f"Details: {exc}"
            ) from exc

        rclpy.init()
        self.ros_module = rclpy
        self.ros_message_class = Float64MultiArray
        self.node = Node("ur5e_surgical_dashboard")
        self.publisher = self.node.create_publisher(Float64MultiArray, "/joint_commands", 10)
        self.node.create_subscription(JointState, "/joint_state", self._receive_ros_state, 10)

        logs = Path(__file__).resolve().parent / "logs"
        logs.mkdir(parents=True, exist_ok=True)
        filename = "gui_joint_tracking_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".csv"
        path = logs / filename
        self.csv_file = path.open("w", newline="", encoding="utf-8")
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(
            ["elapsed_seconds"]
            + [f"desired_j{i}_rad" for i in range(1, 7)]
            + [f"actual_j{i}_rad" for i in range(1, 7)]
        )
        self.log_value.set(str(path))
        self.node.get_logger().info("GUI connected; waiting for /joint_state feedback")

    @staticmethod
    def _label(parent, text, size=10, color="white", weight="normal", **kwargs):
        return tk.Label(
            parent, text=text, bg=parent.cget("bg"), fg=COLORS[color],
            font=(FONT, size, weight), anchor="w", **kwargs
        )

    @staticmethod
    def _panel(parent, **grid_options):
        panel = tk.Frame(
            parent, bg=COLORS["panel"], highlightbackground=COLORS["border"],
            highlightthickness=1, bd=0
        )
        panel.grid(**grid_options)
        return panel

    def _setup_ui(self):
        root = self.root
        header = tk.Frame(root, bg=COLORS["bg"], padx=28, pady=14)
        header.pack(fill="x")
        heading = tk.Frame(header, bg=COLORS["bg"])
        heading.pack(side="left", fill="x", expand=True)
        self._label(heading, "MCTR911  /  ROBOTICS PROGRAMMING", 9, "teal", "bold").pack(anchor="w")
        self._label(heading, "Surgical handover console", 23, "white", "bold").pack(anchor="w", pady=(6, 3))
        self._label(heading, "UR5e   •   6-DOF robot   •   ROS 2 + MuJoCo   •   Milestone 03", 10, "muted").pack(anchor="w")

        status_box = tk.Frame(header, bg=COLORS["panel_soft"], padx=16, pady=12)
        status_box.pack(side="right", padx=(14, 0), anchor="n")
        self.status_chip = tk.Label(
            status_box, textvariable=self.status_text, bg=COLORS["panel_soft"],
            fg=COLORS["amber"], font=(FONT, 10, "bold"), width=19, anchor="center"
        )
        self.status_chip.pack()

        content = tk.Frame(root, bg=COLORS["bg"], padx=28)
        content.pack(fill="both", expand=True)
        content.grid_rowconfigure(0, weight=1)
        content.grid_columnconfigure(0, weight=3, minsize=520)
        content.grid_columnconfigure(1, weight=2, minsize=300)

        joints_panel = self._panel(content, row=0, column=0, sticky="nsew", padx=(0, 14))
        joints_panel.grid_columnconfigure(0, weight=1)
        top = tk.Frame(joints_panel, bg=COLORS["panel"], padx=20, pady=18)
        top.grid(row=0, column=0, sticky="ew")
        self._label(top, "JOINT POSITION CONTROL", 12, "white", "bold").pack(anchor="w")
        self._label(top, "Offsets are relative to the current hold pose (radians).", 9, "muted").pack(anchor="w", pady=(5, 0))

        self.sliders = []
        for i, name in enumerate(JOINT_LABELS):
            section = tk.Frame(joints_panel, bg=COLORS["panel"], padx=20)
            section.grid(row=i + 1, column=0, sticky="ew", pady=(2, 5))
            section.grid_columnconfigure(0, weight=1)

            titleline = tk.Frame(section, bg=COLORS["panel"])
            titleline.grid(row=0, column=0, sticky="ew")
            self._label(titleline, f"J{i + 1:02d}  {name}", 10, "white", "bold").pack(side="left")
            tk.Label(
                titleline, textvariable=self.delta_values[i], bg=COLORS["panel"],
                fg=COLORS["teal"], font=(FONT, 10, "bold")
            ).pack(side="right")

            slider = tk.Scale(
                section, from_=-SLIDER_OFFSET_RANGE, to=SLIDER_OFFSET_RANGE,
                resolution=0.01, length=300, orient="horizontal", showvalue=False,
                bg=COLORS["panel"], fg=COLORS["white"],
                troughcolor=COLORS["panel_soft"],
                activebackground=COLORS["teal"],
                highlightthickness=0, borderwidth=0,
                sliderlength=17, width=12, relief="flat",
                command=lambda _val, idx=i: self._slider_changed(idx),
            )
            slider.set(0)
            slider.configure(state="disabled")
            slider.grid(row=1, column=0, sticky="ew")
            self.sliders.append(slider)

            readouts = tk.Frame(section, bg=COLORS["panel"])
            readouts.grid(row=2, column=0, sticky="ew", pady=(0, 2))
            for text, value, color in (
                ("TARGET", self.target_values[i], "blue"),
                ("ACTUAL", self.actual_values[i], "white"),
                ("ERROR", self.error_values[i], "muted"),
            ):
                item = tk.Frame(readouts, bg=COLORS["panel"])
                item.pack(side="left", fill="x", expand=True)
                self._label(item, text, 8, "muted", "bold").pack(side="left", padx=(0, 6))
                tk.Label(item, textvariable=value, bg=COLORS["panel"],
                         fg=COLORS[color], font=(FONT, 9, "bold")).pack(side="left")

        right = tk.Frame(content, bg=COLORS["bg"])
        right.grid(row=0, column=1, sticky="nsew")
        right.grid_columnconfigure(0, weight=1)

        action_panel = self._panel(right, row=0, column=0, sticky="ew")
        action_inner = tk.Frame(action_panel, bg=COLORS["panel"], padx=18, pady=14)
        action_inner.pack(fill="both", expand=True)
        self._label(action_inner, "ROBOT CONTROLS", 12, "white", "bold").pack(anchor="w", pady=(0, 6))
        self._label(action_inner, "Publish commands only when you explicitly enable them.",
                    9, "muted", wraplength=270, justify="left").pack(anchor="w", pady=(0, 8))

        self.enable_button = tk.Button(
            action_inner, text="ENABLE JOINT COMMANDS", command=self._toggle_enabled,
            font=(FONT, 10, "bold"), bg=COLORS["teal"], fg=COLORS["bg"],
            disabledforeground=COLORS["muted"], activebackground="#7DE5D9",
            relief="flat", borderwidth=0, cursor="hand2", pady=6, state="disabled"
        )
        self.enable_button.pack(fill="x", pady=(0, 6))
        self.hold_button = tk.Button(
            action_inner, text="HOLD MEASURED POSE / RECENTER", command=self._hold_pose,
            font=(FONT, 9, "bold"), bg=COLORS["panel_soft"], fg=COLORS["white"],
            activebackground=COLORS["border"], activeforeground=COLORS["white"],
            relief="flat", borderwidth=0, pady=6, state="disabled"
        )
        self.hold_button.pack(fill="x", pady=(0, 6))
        self.stop_button = tk.Button(
            action_inner, text="STOP COMMANDING / HOLD", command=self._stop_commanding,
            font=(FONT, 9, "bold"), bg=COLORS["panel_soft"], fg=COLORS["red"],
            activebackground=COLORS["border"], activeforeground=COLORS["red"],
            relief="flat", borderwidth=0, pady=6, state="disabled"
        )
        self.stop_button.pack(fill="x")
        self._label(action_inner, "Mode:", 9, "muted").pack(anchor="w", pady=(10, 1))
        tk.Label(action_inner, textvariable=self.current_mode, bg=COLORS["panel"],
                 fg=COLORS["teal"], font=(FONT, 10, "bold"), anchor="w").pack(fill="x")

        scene_panel = self._panel(right, row=1, column=0, sticky="ew", pady=(14, 0))
        scene_inner = tk.Frame(scene_panel, bg=COLORS["panel"], padx=18, pady=10)
        scene_inner.pack(fill="both", expand=True)
        self._label(scene_inner, "SURGICAL WORKSTATION", 12, "white", "bold").pack(anchor="w", pady=(0, 6))
        for number, line in (
            ("01", "UR5e mounted on support pedestal"),
            ("02", "Sterile tray with scalpel and forceps"),
            ("03", "Predefined surgeon handover location"),
            ("04", "Operating table, surgeon and monitor"),
            ("05", "Three selectable MuJoCo camera views"),
        ):
            row = tk.Frame(scene_inner, bg=COLORS["panel"], pady=1)
            row.pack(fill="x")
            self._label(row, number, 9, "teal", "bold").pack(side="left", padx=(0, 12))
            self._label(row, line, 9, "white", wraplength=260,
                        justify="left").pack(side="left", fill="x", expand=True)
        self._label(scene_inner, "Scene props are visual-only; instrument grasping is not implemented.",
                    9, "amber", wraplength=265, justify="left").pack(anchor="w", pady=(6, 0))

        footer = tk.Frame(root, bg=COLORS["bg"], padx=30, pady=10)
        footer.pack(fill="x")
        status = tk.Label(footer, textvariable=self.status_description,
                          font=(FONT, 9), bg=COLORS["bg"], fg=COLORS["muted"],
                          anchor="w", justify="left")
        status.pack(side="left", fill="x", expand=True)
        self._label(footer, "MCTR911 · MILESTONE 03", 9, "teal", "bold").pack(side="right")

    def _receive_ros_state(self, message):
        # ROS JointState.name may not be in joint 1..6 order.
        if message.name and all(name in message.name for name in JOINT_ROS_NAMES):
            indexes = [message.name.index(name) for name in JOINT_ROS_NAMES]
            if any(index >= len(message.position) for index in indexes):
                return
            angles = [float(message.position[index]) for index in indexes]
        elif len(message.position) == 6 and not message.name:
            angles = list(map(float, message.position))
        elif message.name and all(name in message.name for name in (
            "shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3"
        )):
            actuator_order = ("shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3")
            indexes = [message.name.index(name) for name in actuator_order]
            if any(index >= len(message.position) for index in indexes):
                return
            angles = [float(message.position[index]) for index in indexes]
        elif len(message.position) == 6:
            # Backward compatibility with the Milestone 02 controller, which
            # uses the fixed six-element order supplied by its MuJoCo bridge.
            angles = list(map(float, message.position))
        else:
            return
        if all(math.isfinite(q) for q in angles):
            self._on_feedback(angles)

    def _on_feedback(self, angles: list[float]):
        self.actual = angles
        self.feedback_time = time.monotonic()
        if self.reference is None:
            self.reference = angles.copy()
            self.target = angles.copy()
            self.started_at = self.feedback_time
            for slider in self.sliders:
                slider.configure(state="normal")
            self.enable_button.configure(state="normal")
            self.hold_button.configure(state="normal")
            self.stop_button.configure(state="normal")
            self.status_description.set("Feedback received. Inspect the robot before enabling commands.")
        self._draw_joint_values()

    def _draw_joint_values(self):
        if self.actual is None or self.target is None:
            return
        for i in range(6):
            self.actual_values[i].set(f"{self.actual[i]:+.3f}")
            self.target_values[i].set(f"{self.target[i]:+.3f}")
            self.error_values[i].set(f"{(self.target[i] - self.actual[i]):+.3f}")

    def _slider_changed(self, idx: int):
        if self.reference is None or self.resetting:
            return
        self.delta_values[idx].set(f"{self.sliders[idx].get():+.2f}")
        self.target = [
            cap(self.reference[j] + self.sliders[j].get(),
                -ACTUATOR_LIMITS[j], ACTUATOR_LIMITS[j])
            for j in range(6)
        ]
        self._draw_joint_values()

    def _toggle_enabled(self):
        if self.reference is None or not self._fresh():
            return
        self.enabled = not self.enabled
        self._refresh_buttons()
        self.status_description.set(
            "Joint position commands enabled. Move sliders in small steps."
            if self.enabled else "Commands paused. Simulator keeps its last position target."
        )

    def _refresh_buttons(self):
        self.enable_button.configure(
            text="PAUSE JOINT COMMANDS" if self.enabled else "ENABLE JOINT COMMANDS",
            bg=COLORS["amber"] if self.enabled else COLORS["teal"]
        )
        self.current_mode.set("POSITION CONTROL / ACTIVE" if self.enabled else "SAFE / OUTPUT OFF")

    def _hold_pose(self):
        if self.actual is None:
            return
        self.reference = self.actual.copy()
        self.target = self.actual.copy()
        self.resetting = True
        try:
            for i, slider in enumerate(self.sliders):
                slider.set(0)
                self.delta_values[i].set("+0.00")
        finally:
            self.resetting = False
        self._draw_joint_values()
        if self.enabled:
            self._send_target()
        self.status_description.set("Reference recentered at measured joint angles.")

    def _stop_commanding(self):
        if self.actual is None:
            return
        self.target = self.actual.copy()
        # Publish one measured-pose command before stopping; this is not an E-stop.
        if not self.demo and self._fresh():
            self._send_target()
        self.enabled = False
        self._hold_pose()
        self._refresh_buttons()
        self.status_description.set("Command stream paused; simulator retains last hold target. Not an emergency stop.")

    def _fresh(self):
        return (self.feedback_time is not None
                and time.monotonic() - self.feedback_time < STALE_FEEDBACK_SECONDS)

    def _poll_ros(self):
        if self.closed:
            return
        try:
            self.ros_module.spin_once(self.node, timeout_sec=0.0)
        except Exception as exc:
            self.enabled = False
            self._refresh_buttons()
            self.status_description.set(f"ROS 2 error: {exc}")
        self.root.after(ROS_POLL_EVERY_MS, self._poll_ros)

    def _send_target(self):
        if self.publisher is None or self.target is None:
            return
        message = self.ros_message_class()
        message.data = [float(q) for q in self.target]
        self.publisher.publish(message)

    def _publish(self):
        if self.closed:
            return
        if self.enabled and self._fresh() and self.target is not None and self.actual is not None:
            self._send_target()
            if self.csv_writer is not None and self.started_at is not None:
                self.csv_writer.writerow(
                    [time.monotonic() - self.started_at] + self.target + self.actual
                )
                self.rows_written += 1
                if self.rows_written % 10 == 0:
                    self.csv_file.flush()
        elif self.enabled and not self._fresh():
            self.enabled = False
            self._refresh_buttons()
            self.status_description.set("Feedback is stale: commands automatically paused.")
        self.root.after(SEND_EVERY_MS, self._publish)

    def _demo_tick(self):
        if self.closed:
            return
        if self.enabled and self.actual is not None and self.target is not None:
            self._on_feedback([
                self.actual[i] + 0.18 * (self.target[i] - self.actual[i])
                for i in range(6)
            ])
        self.root.after(SEND_EVERY_MS, self._demo_tick)

    def _refresh_status(self):
        if self.closed:
            return
        if self.demo:
            self.status_text.set("DEMO PREVIEW")
            self.status_chip.configure(fg=COLORS["blue"])
        elif self._fresh():
            self.status_text.set("ROS CONNECTED")
            self.status_chip.configure(fg=COLORS["teal"])
        elif self.feedback_time is not None:
            self.status_text.set("FEEDBACK LOST")
            self.status_chip.configure(fg=COLORS["red"])
        else:
            self.status_text.set("WAITING FOR ROS")
            self.status_chip.configure(fg=COLORS["amber"])
        self.root.after(300, self._refresh_status)

    def _close(self):
        if self.closed:
            return
        self.closed = True
        # Disconnect GUI only; robot simulator remains independently running.
        if self.csv_file is not None:
            self.csv_file.flush()
            self.csv_file.close()
        if self.node is not None:
            self.node.destroy_node()
        if self.ros_module is not None and self.ros_module.ok():
            self.ros_module.shutdown()
        self.root.destroy()

    def run(self):
        self.root.mainloop()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo", action="store_true", help="GUI-only preview without ROS")
    args = parser.parse_args()
    SurgicalDashboard(demo=args.demo).run()


if __name__ == "__main__":
    main()
