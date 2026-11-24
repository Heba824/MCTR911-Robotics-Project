# MCTR911 Milestone 04 — Stage 4D Collision Preflight

## Purpose

`check_task_space_collision.py` performs a sampled, kinematic collision preflight of the controller-ready task-space joint reference against the static geometries in `scene_surgical_m3.xml`.

The scene contains 72 environment geoms, all configured with `contype="0"` and `conaffinity="0"`. These settings exclude them from the normal MuJoCo contact pipeline. The preflight does **not** rely on `data.contact`; it queries signed pairwise geometry distance with `mujoco.mj_geomDistance`, which computes a distance between two geoms independently of the standard collision filtering pipeline.

The UR5e is represented by its simplified collision proxies in geom group 3. Its visual meshes in group 2 are not used for this check. The `robot_station` body is excluded because it is the intended mounting support for the robot base. All other scene geoms are included as environment obstacles, regardless of their contact masks. This first pass checks robot-to-environment pairs only; it does not evaluate robot self-collision pairs.

## Files

- `check_task_space_collision.py` — collision-preflight executable.
- `test_task_space_collision.py` — unit tests for geometry selection, sampling, home-pose guard, joint limits, clearance thresholds and report output.

## Run the unit tests

From the repository root:

```bash
source ~/mctr911_kinematics_venv/bin/activate
source /opt/ros/jazzy/setup.bash

PYTHONPATH="$PWD/Milestone 04/control:$PWD/Milestone 02/kinematics:$PYTHONPATH" \
python -m unittest discover \
  -s "Milestone 04/validation" \
  -p "test_task_space_collision.py" -v
```

## Run the collision preflight

The adapter must already have generated:

`Milestone 04/Results/Task_Space/task_space_joint_reference.csv`

Run:

```bash
source ~/mctr911_kinematics_venv/bin/activate
source /opt/ros/jazzy/setup.bash

PYTHONPATH="$PWD/Milestone 04/control:$PWD/Milestone 02/kinematics:$PYTHONPATH" \
python "Milestone 04/validation/check_task_space_collision.py"
```

Defaults:

- Scene: `Milestone 01/Robot Model/scene_surgical_m3.xml`
- Reference: `Milestone 04/Results/Task_Space/task_space_joint_reference.csv`
- Sampling period: 0.005 s (200 Hz), including the exact final time
- Geometry-distance cap: 0.05 m
- Minimum required clearance: 0.02 m
- Review-warning clearance: 0.05 m
- UR5e collision-proxy group: 3
- Excluded mounting body: `robot_station`

The script reconstructs the controller's C2 quintic-Hermite reference between CSV knots using the existing `JointTrajectoryCSV` class. It checks each sample against every selected robot/environment geom pair using MuJoCo signed geometry distance. It also checks that the first reference pose matches the scene's `home` keyframe and that the trajectory respects the MuJoCo joint limits.

## Reports

Generated under `Milestone 04/Results/Collision_Preflight/`:

- `task_space_collision_preflight.csv` — one row per sampled time.
- `task_space_collision_preflight.json` — summary, closest pairs, collision events, joint-limit violations and limitations.

Status interpretation:

- `PASS`: no collision or joint-limit violation, and all queried clearances are at least 0.05 m. When no pair comes within the 0.05 m query cap, the result is reported as a lower bound (`>= 0.05 m`), not an exact global minimum.
- `WARNING_REVIEW_REQUIRED`: no penetration, but at least one pair is closer than 0.05 m while remaining at or above 0.02 m. Review before execution.
- `FAIL_CLEARANCE`: at least one pair is closer than 0.02 m.
- `FAIL_COLLISION`: a signed distance is zero or negative.
- `FAIL_JOINT_LIMIT`: at least one reference sample violates a MuJoCo joint limit.

The process exits with code 0 only for `PASS`; 2 for a warning requiring review; and 3 for a failed preflight.

## Important limitations

This is a **sampled kinematic geometry sweep**, not a dynamic simulation of controller tracking error, actuator lag, contact response or transient motion between samples. The UR5e collision proxies are simplified approximations, not the rendered visual meshes. The intended mounting body is excluded. The surgical instrument is not attached to the end effector in the current scene, so held-tool clearance is not checked. This analysis is not a safety-rated monitor and does not authorize real hardware operation.

A passing result is a necessary simulation check before the controlled MuJoCo run, not a guarantee that the robot cannot collide under dynamic tracking error.
