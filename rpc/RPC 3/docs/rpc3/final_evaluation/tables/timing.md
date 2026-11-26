## Timing

| condition | mode | trial | timed_rows | compute_mean_ms | compute_p95_ms | compute_p99_ms | compute_max_ms | command_gap_mean_ms | command_gap_max_ms | bridge_receipt_gap_max_ms | feedback_receive_hz | source_csv |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| test pipe | model_only | 1 | 551 | 0.874389 | 2.04236 | 2.95896 | 3.80826 | 10.0003 | 12.6098 | — | — | stage4_repair_20261124_211356/headless_isolated/model_only.csv |
| test pipe | joint_flc | 1 | 551 | 0.808218 | 1.13057 | 1.50477 | 2.27367 | 9.99967 | 15.3691 | — | — | stage4_repair_20261124_211356/headless_isolated/joint_flc.csv |
| test pipe | cartesian_flc | 1 | 551 | 1.38767 | 1.75114 | 2.43988 | 3.46406 | 9.99812 | 11.5439 | — | — | stage4_repair_20261124_211356/headless_isolated/cartesian_flc.csv |
| live ROS stage3 | model_only | 1 | 551 | 0.529718 | 0.75939 | 0.981466 | 1.129 | 9.99912 | 11.4503 | 11.4887 | 100.014 | stage3_results/wsl_live_repair_20261124_204350/01_model_only/tracking.csv |
| live ROS stage3 | joint_flc | 1 | 551 | 0.673664 | 0.901327 | 1.12158 | 1.25531 | 9.9991 | 10.8038 | 11.7494 | 100.003 | stage3_results/wsl_live_repair_20261124_204350/01_joint_flc/tracking.csv |
| live ROS stage3 | model_only | 2 | 551 | 0.495898 | 0.702127 | 1.02993 | 1.18716 | 9.99982 | 10.7128 | 10.8251 | 99.9973 | stage3_results/wsl_live_repair_20261124_204350/02_model_only/tracking.csv |
| live ROS stage3 | joint_flc | 2 | 551 | 0.661265 | 0.824153 | 1.15888 | 1.83043 | 10.0007 | 11.3722 | 11.3522 | 99.9986 | stage3_results/wsl_live_repair_20261124_204350/02_joint_flc/tracking.csv |
| live ROS stage3 | model_only | 3 | 551 | 0.530865 | 0.734924 | 1.03557 | 1.33815 | 10.0004 | 11.3221 | 11.4744 | 99.9999 | stage3_results/wsl_live_repair_20261124_204350/03_model_only/tracking.csv |
| live ROS stage3 | joint_flc | 3 | 551 | 0.639048 | 0.807218 | 0.947322 | 1.22979 | 9.99991 | 11.0616 | 11.0636 | 100.001 | stage3_results/wsl_live_repair_20261124_204350/03_joint_flc/tracking.csv |
| live ROS stage4_cartesian | cartesian_flc | 1 | 550 | 1.2987 | 2.06533 | 5.70202 | 9.63998 | 10.0172 | 15.4272 | 15.485 | 99.9999 | stage4_repair_wsl_20261124_211821/cartesian/01_cartesian_flc/tracking.csv |
| live ROS stage4_cartesian | cartesian_flc | 2 | 546 | 2.4099 | 5.18698 | 8.64583 | 12.2421 | 10.0889 | 22.8384 | 23.0162 | 100 | stage4_repair_wsl_20261124_211821/cartesian/02_cartesian_flc/tracking.csv |
| live ROS stage4_cartesian | cartesian_flc | 3 | 551 | 1.72681 | 2.16971 | 2.2925 | 2.46802 | 9.99741 | 11.1322 | 11.2608 | 100.003 | stage4_repair_wsl_20261124_211821/cartesian/03_cartesian_flc/tracking.csv |
| live ROS stage4_regression | model_only | 1 | 550 | 0.635234 | 0.891883 | 2.92071 | 8.15251 | 10.0184 | 13.8086 | 13.7773 | 100.02 | stage4_repair_wsl_20261124_211821/regression/01_model_only/tracking.csv |
| live ROS stage4_regression | joint_flc | 1 | 551 | 0.664336 | 0.929274 | 1.17841 | 1.41993 | 9.99969 | 10.778 | 11.5239 | 100.001 | stage4_repair_wsl_20261124_211821/regression/01_joint_flc/tracking.csv |
| graphical ROS | cartesian_flc | 1 | 549 | 1.78872 | 3.65625 | 5.79506 | 10.6512 | 10.0366 | 18.866 | — | 98.4673 | gui_cartesian_flc.csv |
