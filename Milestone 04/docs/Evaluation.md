# Milestone 04 evaluation

The UR5e completed both archived surgical handovers with inverse dynamics plus actuator PD feedback. Both executions met the hard geometry gate and required approval of sub-50 mm clearances. In earlier matched-reference experiments, inverse dynamics reduced joint-position RMSE by about 90%. The task-space experiment also exposed a larger angular-velocity startup transient.

## Project scope and evidence

The Milestone 01 literature review proposes the simulated scrub-nurse workflow with simulator localization, typed instrument selection, a simplified grasp and a fixed receiver. The application implements those elements through CLI selection and a sequential ROS 2 executor. Milestone 02 derived and validated standard-DH forward kinematics and position-only Jacobian-transpose IK. Milestone 03 added geometric Jacobian, velocity and acceleration validation, and a standalone smooth-motion actuator experiment. Milestone 04 adds full-pose IK, quintic references, motor-level control and closed-loop handover. Stage 4A documents the motor-control derivation requirement. The local literature review and milestone procedures supply the project scope. [Milestone 01 literature review](<../../Milestone 01/Literature Review/MCTR911_Literature_Review_Revised_Section_7.pdf>) · [Milestone 02 report](<../../Milestone 02/MS2Report.pdf>) · [Milestone 03 Jacobian](<../../Milestone 03/kinematics/ur5e_jacobian.py>) · [Milestone 03 smooth-motion experiment](<../../Milestone 03/visualization/ur5e_smooth_motion_demo.py>) · [Stage 4A control derivation](../control/README_Motor_Level_Control.md)

Existing [reference figures](../Figures/) show planned kinematics and IK residuals. The saved [controller and handover feedback](../Results/) supplies the measured joint tracking in this report. The Milestone 03 standalone run and Stage 4D collision checks use separate procedures. The controller comparisons below use the two archived Milestone 04 matched pairs.

## System, planning, and control

The operator selects scalpel or forceps. The planner calibrates the validated DH model to the MuJoCo world frame and uses full-pose IK, seeding each waypoint from its predecessor. The route is **approach clearance → pregrasp → pickup → lift → transfer clearance → handover approach → handover → retreat → return HOME**. Quintic point-to-point references have zero endpoint velocity and acceleration. Segment durations account for displacement and the planner defaults of 0.50 rad/s maximum joint speed and 0.80 rad/s² maximum joint acceleration. Earlier sample-based tracking references use quintic-Hermite interpolation. [Planner](../application/surgical_handover_planner.py) · [Quintic trajectory](../trajectory/joint_space_trajectory.py) · [Reference interpolation](../control/trajectory_csv.py)

The planner aligns the attachment site’s position with the pickup site while preserving the intended tool orientation. The two frames can be nearly opposed, as in the historical 178.93° case. At pickup the bridge captures the rigid transform from attachment site to instrument body and carries the mocap body with the current world-from-site transform. The planner applies the inverse grasp transform to the desired instrument body pose to obtain the handover tool target. On release, the bridge leaves the mocap body at its last pose. The executor checks instrument stability after the robot retreats. [Planner](../application/surgical_handover_planner.py) · [Executor](../application/execute_surgical_handover.py) · [Scene](<../../Milestone 01/Robot Model/scene_surgical_handover.xml>)

The bridge publishes **/joint_state** and world-frame pose topics. The controller sends six position commands in radians on **/joint_commands** in POSITION mode. Baseline control sends the desired joint positions directly. The actuator applies **Kp(ctrl − q) − Kd q̇**. The inverse-dynamics strategy computes **τff = M(qref) q̈ref + bias − passive** and commands **ctrl = qref + (τff + Kd q̇ref)/Kp**. This gives **τff + Kp(qref − q) + Kd(q̇ref − q̇)** at the actuator. The saved torque is computed feedforward. Total motor torque, actuator power, and energy are unavailable in these run files. The executor gates motion on fresh feedback, compatible artifacts and sampled clearance. It checks sustained tracking error, settling and command range. On abort it attempts a position hold and leaves an attached instrument in place for inspection. [Control derivation](../control/README_Motor_Level_Control.md) · [Executor](../application/execute_surgical_handover.py)

The setup uses Ubuntu WSL, ROS 2 Jazzy, Python 3.12 and the project venv. The archived handover preflights record MuJoCo Python 3.2.7. The C++ bridge is built separately in the ROS workspace. The six completed runs report a configured 100 Hz control rate. The demonstration uses a 1 kHz simulator setting and 20 Hz pose and visualization settings. The saved metrics retain control rates and feedback ages. They do not preserve an independent bridge build or physics-setting snapshot for every run. [Demo environment](Demonstration.md) · [Run summary](../Results/Final_Evaluation/summary.json)

## Controller results

Each pair used its own common 8 s reference. The pipeline checked logged desired joint positions and velocities against the same interpolator evaluated at each run’s timestamps. Paired runs match in initial measured joint feedback, configured control rate and feedback timeout. Observed control rates ranged from 98.63 to 100.11 Hz. These are descriptive comparisons of one run per controller per reference. Between-run variance and unrecorded configuration differences cannot be quantified. The task-space experiment executes a Cartesian-derived joint reference using joint feedback. Cartesian pose and twist are reconstructed from FK(q) and J(q) q̇. [Controller runs](../Results/Final_Evaluation/controller_runs.csv) · [Cartesian method](../analysis/README_Stage4D_TaskSpaceTracking.md)

| Reference | Controller | Position RMSE (mrad) | Peak position error (mrad) | Velocity RMSE (mrad/s) | Peak velocity error (mrad/s) | Peak command (rad) | Peak feedforward (N·m) |
|---|---|---:|---:|---:|---:|---:|---:|
| Joint-space | Baseline PD | 8.674 | 20.984 | 4.700 | 15.888 | 1.871 | 0 |
| Joint-space | Inverse dynamics + PD | 0.895 | 8.079 | 3.536 | 36.438 | 1.871 | 21.896 |
| Task-space-derived | Baseline PD | 8.986 | 27.766 | 6.087 | 32.366 | 1.871 | 0 |
| Task-space-derived | Inverse dynamics + PD | 0.941 | 8.079 | 3.690 | 36.426 | 1.871 | 22.021 |

No logged command clipped. Final-hold joint-position RMSE was about 5.685 mrad under PD and under 0.008 mrad with feedforward. The terminal joint-position error was about 11.145 mrad under PD and below 0.001 mrad with feedforward. The historical controller logged through reference completion and did not record a tolerance-based settling time. Peak joint-velocity error increased with inverse dynamics despite lower overall velocity RMSE. [Per-joint metrics](../Results/Final_Evaluation/joint_metrics.csv) · [Per-joint figure](../Results/Final_Evaluation/figures/per_joint_rmse.png)

| Cartesian error norm, whole run | Baseline PD RMSE | Inverse dynamics + PD RMSE |
|---|---:|---:|
| Position (mm) | 11.573 | 1.272 |
| Orientation (mrad) | 18.367 | 2.438 |
| Linear velocity (mm/s) | 6.160 | 5.491 |
| Angular velocity (mrad/s) | 5.344 | 11.404 |

The inverse-dynamics angular error peaked at 84.869 mrad/s at 0.039 s. The measured world-X angular velocity reconstructed from joint feedback was +84.844 mrad/s, while the desired X component was −0.026 mrad/s. The first 0.25 s contributed 85.3% of the squared angular-velocity error for that run. For the remainder of the 6 s motion, inverse dynamics gave 5.160 mrad/s RMSE versus 6.297 mrad/s for baseline. Final-hold RMSE was 0.033 versus 0.430 mrad/s. Both runs started at nearly stationary feedback with an approximately 8.08 mrad joint offset. The first inverse-dynamics command included a gravity/feedforward position offset. A startup correction is consistent with these observations, but the two runs do not isolate its cause. The windowed comparison explains the transient while retaining the worse overall angular-velocity result as the primary comparison. [Cartesian window metrics](../Results/Final_Evaluation/cartesian_metrics.csv) · [Cartesian positions](../Results/Final_Evaluation/figures/cartesian_position_tracking.png) · [Pose and twist errors](../Results/Final_Evaluation/figures/cartesian_tracking_errors.png)

## Handover results

| Metric | Scalpel | Forceps |
|---|---:|---:|
| Completed waypoints | 9 | 9 |
| Nominal trajectory duration (s) | 22.763 | 23.207 |
| Recorded wall duration (s) | 25.113 | 25.575 |
| Last control sample elapsed time (s) | 23.849 | 24.285 |
| Joint-position RMSE / peak (mrad) | 9.410 / 39.202 | 9.560 / 40.851 |
| Joint-velocity RMSE / peak (mrad/s) | 21.571 / 85.627 | 20.862 / 86.443 |
| Grasp translation error (mm) | 0.475 | 0.444 |
| Grasp orientation error (°) | 0.0040 | 0.0044 |
| FK pickup tool to planned site at last pickup settle row (mm) | 0.640 | 0.507 |
| FK handover body position error at last handover settle row (mm) | 0.059 | 0.110 |
| FK handover body orientation error (°) | 0.0035 | 0.0042 |
| HOME maximum joint error at last logged settle row (mrad) | 0.785 | 0.758 |
| Logged clipped command samples | 0 | 0 |
| Attachment, release, and release-stability checks | Completed | Completed |

Wall duration comes from recorded start and finish timestamps. It includes settling, service and pose-check waits, and end-of-run work before the metrics timestamp. The last CSV sample provides a separate control-loop bound. Grasp errors compare the bridge transform stored in the metrics to the planned grasp. Pickup and handover estimates use the plan calibration, the last measured settle joint state, and, for handover, the saved grasp transform. They are model-derived estimates against planned targets. Direct service-time pose errors, numeric post-release drift, and exact online support-transition samples are unavailable in the saved artifacts. The operator-reported transition readings of 22.90 mm and 21.04 mm agree to rounding with the saved carried minima. Those minima alone do not identify the transition samples. [Run table](../Results/Final_Evaluation/handover_runs.csv) · [Raw executions](../Results/Surgical_Handover/Execution/)

Scalpel had slightly lower joint-position RMSE and forceps had slightly lower velocity RMSE. Forceps took 0.463 s longer in wall time, with a 0.445 s longer plan. Approach clearance had the largest motion-only position RMSE for both runs (11.862 and 12.437 mrad). Each waypoint has eight logged settle rows. The largest final settle error occurred on retreat (1.702 and 1.594 mrad). The forceps peak crossed the default 40 mrad tracking threshold. The default abort requires ten consecutive over-threshold samples. These comparisons describe two distinct plans and runs. [Per-waypoint measurements](../Results/Final_Evaluation/handover_stages.csv) · [Stage durations](../Results/Final_Evaluation/figures/handover_stage_durations.png) · [Scalpel traces](../Results/Final_Evaluation/figures/scalpel_joint_tracking.png) · [Forceps traces](../Results/Final_Evaluation/figures/forceps_joint_tracking.png)

## Collision evidence and limitations

| Minimum clearance (mm) | Scalpel offline | Scalpel online | Forceps offline | Forceps online |
|---|---:|---:|---:|---:|
| Robot–environment | 22.274 | 22.548 | 22.274 | 22.545 |
| Robot self | 45.026 | 45.026 | 45.026 | 45.026 |
| Carried instrument–environment | 20.477 | 22.901 | 20.504 | 21.038 |

The preflight used MuJoCo native CCD and queried signed distances with **mj_geomDistance** at 5 ms trajectory samples, independently of normal contact masks. Negative separation denotes penetration. Positive separation below 20 mm fails under the general policy. Separation from 20 mm up to 50 mm requires review. Robot–environment checks use collision proxies against the scene. Self checks omit same-link and direct parent–child overlaps. Carried checks evaluate instrument geometry against scene objects, with phase-specific intentional-contact handling for the grasp and receiver. The robot–receiver checks remain active. [Collision implementation](../application/surgical_handover_collision_preflight.py) · [Clearance table](../Results/Final_Evaluation/collision_clearances.csv) · [Clearance figure](../Results/Final_Evaluation/figures/collision_clearances.png)

The pickup allowance applies to the tray base, selected instrument pad, and trolley top during pickup and lift. These pairs may approach or touch while the allowance is open. Any negative distance fails. Once every support pair reaches 20 mm, ordinary clearance applies. The planned supports cleared at 10.871 s for scalpel and 10.528 s for forceps. Both preflights had zero collision events and zero joint-limit violations and retained **WARNING_REVIEW_REQUIRED**. The executor checked measured joint states and the recovered grasp transform before sending sampled control commands. Both runs explicitly approved the warnings and completed without an online abort. Offline and online minima differ because the sample states and timing differ. [Scalpel preflight](../Results/Surgical_Handover/Collision_Preflight/surgical_handover_collision_scalpel_20261124_114651.json) · [Forceps preflight](../Results/Surgical_Handover/Collision_Preflight/surgical_handover_collision_forceps_20261124_114651.json)

Two earlier scalpel attempts remain in the archive. One stopped before logging joint samples after reporting a 178.93° pickup-frame mismatch. The other ended attached after reporting support penetration. Its old preflight CSV shows −3.000 mm at attachment with zero reported collisions. The corrected policy counts negative support distances, and the current scene positions both instruments clear of their pads. The historical aborted carried-distance accumulator at its initial cap is excluded from exact measured minima. These archived examples do not define a trial protocol or success rate. [Alignment abort](../Results/Surgical_Handover/Execution/surgical_handover_execution_scalpel_20261124_111756_178765_metrics.json) · [Support abort](../Results/Surgical_Handover/Execution/surgical_handover_execution_scalpel_20261124_113401_483815_metrics.json) · [Historical preflight CSV](../Results/Surgical_Handover/Collision_Preflight/surgical_handover_collision_scalpel_20261124_034050.csv)

The fixed scene, simplified proxies, rigid mocap grasp, static receiver and calibration bound the interpretation. The system has no physical gripper, force or slip validation, independent pose sensing, physical robot validation, or clinical safety certification. Sampled clearances establish separation at the evaluated states. The distance backend does not provide a continuous collision guarantee for these sampled runs.

## Reproduction

Run the single command in the [Milestone 04 README](../README.md). The pipeline validates schema, units, timestamps, and cross-file consistency, then reuses the reference interpolator and FK. Each CSV row carries source paths and the JSON includes SHA-256 hashes. Joint RMSE pools samples and joints with equal weight. Cartesian RMSE uses the Euclidean error norm over samples. Logged phase span is last minus first timestamp in each phase and excludes boundary gaps. Blank fields are unavailable or inapplicable. [Analysis code](../analysis/evaluate_milestone04.py) · [Source manifest and summary](../Results/Final_Evaluation/summary.json)
