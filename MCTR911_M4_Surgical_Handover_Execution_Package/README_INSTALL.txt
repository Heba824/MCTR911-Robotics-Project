MCTR911 M4 Surgical Handover Execution Package

Extract this ZIP directly into:
  ~/mctr911_project/MCTR911-Robotics-Project-Fresh

Preserve the folder paths and allow the existing collision-preflight test file
to be replaced. This package adds the closed-loop execution node/core, its unit
tests, and documentation. It also updates the older Step 2C collision-preflight
test fixture from the superseded 8-waypoint sequence to the current 9-waypoint
sequence. It does not replace the scene, ROS bridge, planner, or existing results.

Run tests from the project root:
  source ~/mctr911_kinematics_venv/bin/activate
  source /opt/ros/jazzy/setup.bash
  python "Milestone 04/validation/test_surgical_handover_step2d.py"
  python "Milestone 04/validation/test_surgical_handover_collision_preflight.py"
  python "Milestone 04/validation/test_surgical_handover_execution.py"

Read Milestone 04/README_Surgical_Handover_Execution.md before executing.
The executor is dry-run by default. Actual simulation motion requires --execute.
A WARNING_REVIEW_REQUIRED preflight additionally requires --approve-warnings;
that flag cannot override collisions, joint-limit violations, pickup-support
violations, or clearances below the hard 20 mm minimum.
