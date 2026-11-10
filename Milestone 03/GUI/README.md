# MCTR911 Milestone 03 — Surgical Handover GUI

This is an **additional GUI/environment deliverable** for the original UR5e
robotic scrub nurse project. It does not overwrite Milestone 01 or 02 files.

## Files and exact locations

- `Milestone 01/Robot Model/scene_surgical_m3.xml` — polished MuJoCo environment, beside `ur5e.xml` and its `assets/` folder.
- `Milestone 03/GUI/surgical_dashboard.py` — modern Tkinter dashboard, compatible with the previous project's `/joint_state` and `/joint_commands` ROS topics.
- `Milestone 03/GUI/run_scene_ubuntu.sh` — start the scene using the existing `mujoco_ros2` ROS 2 package.
- `Milestone 03/GUI/run_dashboard_ubuntu.sh` — start the dashboard in a separate terminal.
- `Milestone 03/GUI/preview_scene_no_ros.py` — optional MuJoCo-only preview (requires Python `mujoco`).

## How to open on Ubuntu

1. Open a terminal in the **repository root** (the directory containing `Milestone 01` and `Milestone 02`).
2. Confirm that your ROS 2 workspace from Milestone 02 is already built and contains `mujoco_ros2`.
3. Launch the simulator:

   ```bash
   bash 'Milestone 03/GUI/run_scene_ubuntu.sh'
   ```

4. Open a **second terminal** at the repository root and launch the controls:

   ```bash
   bash 'Milestone 03/GUI/run_dashboard_ubuntu.sh'
   ```

5. Wait for **ROS CONNECTED**. Move a slider to set a small target offset; click **ENABLE JOINT COMMANDS** to start publishing. Use **HOLD MEASURED POSE / RECENTER** between pose changes, and **STOP COMMANDING / HOLD** to pause commands.

   **Note:** STOP COMMANDING is **not a hardware emergency stop**; it commands a hold at the measured pose and pauses publishing. Use only in simulation.

### If your existing `mujoco_ros2` workspace is somewhere else

The launch scripts try to source `/opt/ros/jazzy` (or `/opt/ros/humble`) and `$HOME/mctr911_ws/install/setup.bash`. If your bridge is in another workspace, source that workspace manually before running the script:

```bash
source /opt/ros/jazzy/setup.bash  # change to humble if applicable
source ~/YOUR_WORKSPACE/install/setup.bash
bash 'Milestone 03/GUI/run_scene_ubuntu.sh'
```

Do not run the old `run_mujoco_nvidia.sh` to demonstrate Milestone 03: it launches the original plain `scene.xml`.

## Preview the dashboard without ROS

Run this even on a computer that does not have ROS 2 installed:

```bash
python3 'Milestone 03/GUI/surgical_dashboard.py' --demo
```

The **DEMO PREVIEW** indicator means the GUI is showing example joint values only; it never publishes commands. On Ubuntu, install the standard `python3-tk` package if Tkinter is missing.

## Environment contents

- UR5e manipulator and original robot frame: **unchanged**.
- Surgical workstation with a robot pedestal, instrument trolley and sterile tray.
- Two simplified fixed instruments (scalpel and forceps), each with its original pickup target.
- Static scrub surgeon with a gloved hand; fixed `handover_target_site` at `(0.30, 0.43, 0.38)` m.
- Operating table with sterile drape, surgical lamp, vital-signs monitor, storage and medical waste bin.
- Three MuJoCo camera views: `overview`, `instrument_tray`, `handover_closeup`.
- Visual-only clinical props with collisions disabled to avoid disturbing the existing motion model.

The scene retains the robot base at `(0, 0, 0)` and keeps `ur5e.xml` intact. The floor is at `z=-0.65` and the support tabletop at `z=0`.

## ROS 2 interface

The existing bridge is used without modification:

| ROS topic | Type | Purpose |
| --- | --- | --- |
| `/joint_state` | `sensor_msgs/msg/JointState` | Measured joint angles |
| `/joint_commands` | `std_msgs/msg/Float64MultiArray` | Six position targets (radians) |

The GUI displays target, actual and error for six joints, only publishes after explicit enabling, automatically pauses on stale feedback, and records feedback when publishing in `Milestone 03/GUI/logs/`.

Only run **one** joint-command publisher at a time. Close the older Milestone 02 slider/constant/signal controllers while using this dashboard.

## What this delivers and does not deliver

This work addresses the **Milestone 03 GUI/environment** portion of the project description: visualize the robot and build a scene that suits the surgical tool handover application, with props, a surgeon, equipment, and a control interface. You can record a walkthrough of the scene and demonstrate moving joints with the slider dashboard.

This package does **not** implement forward/inverse velocity or acceleration kinematics, validation, autonomous pick-and-place, tool grasping, hand tracking, or collision-safe planning. Those are separate requirements or future milestones and must not be claimed as completed by the GUI alone.

## Suggested scene video

1. Show the `overview` camera and identify each element of the surgical workstation.
2. Switch to `instrument_tray` and point out scalpel, forceps and pickup sites.
3. Switch to `handover_closeup` and identify the fixed handover site.
4. Show the joint-control dashboard receiving feedback. Enable commands and move one slider slowly to demonstrate that the UR5e moves while the workstation remains in place.
5. Pause commands, close the dashboard, and explain that grasping and autonomous transfer are not part of the GUI implementation.
