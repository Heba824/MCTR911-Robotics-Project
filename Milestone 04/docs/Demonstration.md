# Surgical handover demonstration

Use two Ubuntu WSL terminals for the MuJoCo bridge and the operator. Keep both on ROS domain 71. Each demonstration starts with a newly launched copy of the corrected handover scene, both instruments on their pads, and the UR5e near HOME. The saved plans below match that scene.

## Terminal 1 · start the fresh scene

~~~bash
cd ~/mctr911_project/MCTR911-Robotics-Project-Fresh
source /opt/ros/jazzy/setup.bash
source ~/mctr911_ws/install/setup.bash
export ROS_DOMAIN_ID=71
ros2 run mujoco_ros2 mujoco_node \
  "$PWD/Milestone 01/Robot Model/scene_surgical_handover.xml" \
  --ros-args -p control_mode:=POSITION \
  -p joint_state_topic_name:=joint_state -p joint_command_topic_name:=joint_commands \
  -p simulation_frequency:=1000 -p visualisation_frequency:=20 -p pose_publish_frequency:=20
~~~

Between instruments, wait for the executor to exit and save its results. Stop the bridge with Ctrl-C in terminal 1. Relaunch with the same command and verify HOME and the instruments’ initial placement. The HOME return at the end of a handover leaves the released instrument at the receiver. A fresh process restores both instrument mocap poses and resets the attachment state.

## Terminal 2 · verify feedback and prepare either instrument

~~~bash
cd ~/mctr911_project/MCTR911-Robotics-Project-Fresh
source /opt/ros/jazzy/setup.bash
source ~/mctr911_ws/install/setup.bash
source ~/mctr911_kinematics_venv/bin/activate
export ROS_DOMAIN_ID=71
export PYTHONDONTWRITEBYTECODE=1

ros2 node list
ros2 topic info /joint_state --verbose
ros2 topic info /joint_commands --verbose
ros2 topic echo /joint_state --once
~~~

Expect one bridge in this domain, a live joint-state publisher, the bridge command subscriber, and no joint-command publisher before starting the executor. Check positions by joint name in shoulder-pan, shoulder-lift, elbow, wrist-1, wrist-2, wrist-3 order. Nominal scene HOME in radians is **[−1.5708, −1.5708, 1.5708, −1.5708, −1.5708, 0]**. Small settling offsets can occur. For the archived plans the executor checks each joint against its plan start within 0.05 rad and each speed within 0.05 rad/s. Inspect the tray and confirm that both instruments are still in their pickup positions. [Scene keyframe](<../../Milestone 01/Robot Model/ur5e.xml>) · [Executor](../application/execute_surgical_handover.py)

Select **scalpel** in the following block. For the separate forceps demonstration, start a fresh scene and use **instrument=forceps**.

~~~bash
instrument=scalpel
results='Milestone 04/Results/Surgical_Handover'
case "$instrument" in
  scalpel) plan_id=20261124_134621_068881 ;;
  forceps) plan_id=20261124_134622_317512 ;;
esac
plan="$results/Planning/surgical_handover_plan_${instrument}_$plan_id.json"
preflight="$results/Collision_Preflight/surgical_handover_collision_${instrument}_20261124_114651.json"
executor='Milestone 04/application/execute_surgical_handover.py'
args=(--instrument "$instrument" --plan-json "$plan" --preflight-json "$preflight")

ros2 topic echo "/mujoco/${instrument}_pickup_pose" --qos-reliability best_effort --once
ros2 topic echo /mujoco/attachment_site_pose --qos-reliability best_effort --once
ros2 topic echo /mujoco/handover_target_pose --qos-reliability best_effort --once
python "$executor" "${args[@]}"
~~~

The dry gate prints plan and preflight compatibility, all nine segments, and the clearance review pairs. It does not inspect live joint feedback. Check the live outputs above separately. Review **WARNING_REVIEW_REQUIRED**, native CCD, the 20 mm hard minimum and 50 mm warning threshold, and the zero collision and joint-limit event counts in the selected JSON. The archived reports show carried minima of 20.477 mm for scalpel and 20.504 mm for forceps. A failed or mismatched preflight blocks motion. [Scalpel report](../Results/Surgical_Handover/Collision_Preflight/surgical_handover_collision_scalpel_20261124_114651.json) · [Forceps report](../Results/Surgical_Handover/Collision_Preflight/surgical_handover_collision_forceps_20261124_114651.json)

### Explicit motion · operator authorization required

Run this only after a fresh readiness and clearance review for the selected instrument. This command publishes motion and lets the executor call the simulation attachment and release services.

~~~bash
python "$executor" "${args[@]}" --approve-warnings --execute
~~~

Expected milestones in order:

1. Live start checks passed.
2. The selected instrument reports **attachment confirmed at pickup** after the pickup settles.
3. **Online pickup-support envelope cleared** during lift.
4. The executor verifies **release at handover target**.
5. **Released instrument remained stationary during robot retreat.**
6. **All nine waypoints completed; instrument released and held at handover; robot returned HOME.**

Confirm that the metrics report **COMPLETED**, the release flag is true, and the attachment-at-end flag is false. Execution writes a timestamped **.csv** and **_metrics.json** under [Execution](../Results/Surgical_Handover/Execution/). Preserve both output files and record the terminal output separately when numerical service-time pose checks or the support-transition reading are needed.

## Failure and reset procedure

Allow the executor to report its abort and write results. It attempts a hold using feedback. Stop the bridge with Ctrl-C and preserve the logs and artifacts. If the abort reports an attached instrument, leave it attached for inspection. Avoid manual service calls, repeated starts or loading a different scene into the running bridge. Diagnose the failure, then start a fresh scene for a separately reviewed retry. A hold is an application behavior and does not establish readiness to continue.

## Troubleshooting

| Symptom | Checks |
|---|---|
| Python import fails | Use the project venv with Python 3.12. Source Jazzy and the ROS overlay before running the executor. Confirm imports with **python -c 'import rclpy, mujoco; print(mujoco.__version__)'**. Offline preflight needs a Python MuJoCo with native CCD available. Run the application script from the repository root. |
| Pose topics visible but echo or executor receives no samples | Use **best_effort** for the bridge’s pose topics. Inspect offered QoS with **ros2 topic info … --verbose**. Joint state and command endpoints use reliable QoS. |
| Plan or preflight is stale or mismatched | Check that instrument, plan, preflight and exact scene XML match. Replan and regenerate preflight when the scene changes. Re-run the gate with the new explicit filenames. |
| Pickup-frame validation fails | Check the selected instrument’s world-frame pickup pose and the attachment-site pose. Compare the live pose with the correct planned site. A nearly opposed tool orientation is expected for this configuration and is distinct from a moved or rotated pickup site. |
| Grasp-transform mismatch or missing synchronized pose pair | Inspect both stamped pose topics, freshness, scene version and pickup alignment. Keep the aborted instance stopped if attached. Preserve the terminal error and metrics rather than retrying services. |
| Pickup-support penetration | Save the named pair, signed distance and metrics. Check the corrected handover XML and initial pad geometry. Use the matching regenerated offline preflight. Keep the hard minimum and penetration guard in place. |
| Second run fails at pickup or starts away from HOME | Stop the old bridge after the executor ends, then launch a fresh bridge. Verify HOME, stationary feedback, and both instruments on the tray. |
| Bridge or topics missing | Check both terminals source Jazzy and the same overlay and export domain 71. Check for competing bridges or publishers and a simulator that exited during launch. |

### Generate new artifacts if needed

The archived artifacts are sufficient for the unchanged scene. For a new plan, work from the repository root and use the same instrument setting. The planner command is offline. Use the printed exact plan path in the next command.

~~~bash
python 'Milestone 04/application/plan_surgical_handover.py' --offline --instrument "$instrument"
~~~

~~~bash
plan='Milestone 04/Results/Surgical_Handover/Planning/<printed-plan-filename>.json'
python 'Milestone 04/application/surgical_handover_collision_preflight.py' \
  --instrument "$instrument" --plan-json "$plan"
~~~

The warning status exits with code 2. Review the report and use its printed JSON path with the new plan in the dry gate. Collision, insufficient clearance, support-clearance or joint-limit failures require resolution before any motion.

## Presentation sequence

Show the scene and HOME check, then the warning pairs and planned time. During execution, narrate pickup alignment, attachment, support clearance, handover, release and HOME return. After completion, show the measured duration and online minima in the metrics alongside the planned values. Reset the scene for the other instrument. Close with the controller and clearance figures from the [evaluation report](Evaluation.md).
