#!/usr/bin/env python3
"""Optional MuJoCo-only scene preview. No ROS, control, or tool attachment."""
from pathlib import Path
import sys

SCENE = Path(__file__).resolve().parents[2] / "Milestone 01" / "Robot Model" / "scene_surgical_m3.xml"
try:
    import mujoco
    import mujoco.viewer
except ImportError:
    sys.exit("Missing Python MuJoCo package. Use the ROS 2 launcher or install mujoco in a Python environment.")

print(f"Opening {SCENE}")
print("Static scene preview only. ROS position-command controls are not connected.")
mujoco.viewer.launch_from_path(str(SCENE))
