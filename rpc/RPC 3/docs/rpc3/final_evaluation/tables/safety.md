## Safety

| condition | mode | trial | total_commands | missed_reference_slots | skipped_scheduler_ticks | publish_deadline_misses | callback_overruns | compute_overruns | torque_saturated_joint_samples | task_speed_limited_ticks | joint_speed_limited_components | accel_limited_components | controller_watchdogs | bridge_watchdogs_before_shutdown | bridge_watchdogs_total | bridge_stale_torques | rejected_feedback_samples | discarded_commands | fault | source_csv |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| test pipe | model_only | 1 | 551 | — | — | 0 | — | 0 | 0 | — | — | — | 0 | — | — | — | — | — | — | stage4_repair_20261124_211356/headless_isolated/model_only.csv |
| test pipe | joint_flc | 1 | 551 | — | — | 0 | — | 0 | 0 | — | — | — | 0 | — | — | — | — | — | — | stage4_repair_20261124_211356/headless_isolated/joint_flc.csv |
| test pipe | cartesian_flc | 1 | 551 | — | — | 0 | — | 0 | 0 | 0 | 0 | 0 | 0 | — | — | — | — | — | — | stage4_repair_20261124_211356/headless_isolated/cartesian_flc.csv |
| live ROS stage3 | model_only | 1 | 842 | 0 | 0 | 0 | 0 | 0 | 0 | — | — | — | 0 | 0 | 1 | — | 0 | 0 | — | stage3_results/wsl_live_repair_20261124_204350/01_model_only/tracking.csv |
| live ROS stage3 | joint_flc | 1 | 838 | 0 | 0 | 0 | 0 | 0 | 0 | — | — | — | 0 | 0 | 1 | — | 0 | 0 | — | stage3_results/wsl_live_repair_20261124_204350/01_joint_flc/tracking.csv |
| live ROS stage3 | model_only | 2 | 835 | 0 | 0 | 0 | 0 | 0 | 0 | — | — | — | 0 | 0 | 1 | — | 0 | 0 | — | stage3_results/wsl_live_repair_20261124_204350/02_model_only/tracking.csv |
| live ROS stage3 | joint_flc | 2 | 825 | 0 | 0 | 0 | 0 | 0 | 0 | — | — | — | 0 | 0 | 1 | — | 0 | 0 | — | stage3_results/wsl_live_repair_20261124_204350/02_joint_flc/tracking.csv |
| live ROS stage3 | model_only | 3 | 840 | 1 | 0 | 0 | 0 | 0 | 0 | — | — | — | 0 | 0 | 1 | — | 0 | 0 | — | stage3_results/wsl_live_repair_20261124_204350/03_model_only/tracking.csv |
| live ROS stage3 | joint_flc | 3 | 836 | 0 | 0 | 0 | 0 | 0 | 0 | — | — | — | 0 | 0 | 1 | — | 0 | 0 | — | stage3_results/wsl_live_repair_20261124_204350/03_joint_flc/tracking.csv |
| live ROS stage4_cartesian | cartesian_flc | 1 | 834 | 2 | 1 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | — | stage4_repair_wsl_20261124_211821/cartesian/01_cartesian_flc/tracking.csv |
| live ROS stage4_cartesian | cartesian_flc | 2 | 828 | 5 | 5 | 4 | 4 | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | — | stage4_repair_wsl_20261124_211821/cartesian/02_cartesian_flc/tracking.csv |
| live ROS stage4_cartesian | cartesian_flc | 3 | 835 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | — | stage4_repair_wsl_20261124_211821/cartesian/03_cartesian_flc/tracking.csv |
| live ROS stage4_regression | model_only | 1 | 838 | 10 | 1 | 1 | 0 | 0 | 0 | — | — | — | 0 | 0 | 1 | 0 | 0 | 0 | — | stage4_repair_wsl_20261124_211821/regression/01_model_only/tracking.csv |
| live ROS stage4_regression | joint_flc | 1 | 841 | 0 | 0 | 0 | 0 | 0 | 0 | — | — | — | 0 | 0 | 1 | 0 | 0 | 0 | — | stage4_repair_wsl_20261124_211821/regression/01_joint_flc/tracking.csv |
| graphical ROS | cartesian_flc | 1 | 2174 | 224 | 11 | 9 | 3 | 2 | 0 | 0 | 0 | 0 | 0 | — | — | — | 0 | 0 | — | gui_cartesian_flc.csv |
