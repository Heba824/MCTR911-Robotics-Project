# Milestone 04 · UR5e surgical instrument handover

The archived scalpel and forceps runs completed all nine waypoints. Both runs passed the 20 mm hard clearance gate and required review of clearances below 50 mm.

- [Evaluation report](docs/Evaluation.md) covers the implementation, controller experiments, handovers, failures, and limitations.
- [Demonstration guide](docs/Demonstration.md) covers scene reset, readiness checks, and operator-authorized motion.
- [Evaluation outputs](Results/Final_Evaluation/) contain six CSV tables, a JSON source manifest and summary, and seven figures.

From the repository root, regenerate the evaluation with the project venv. This command needs neither a ROS node nor a simulator.

~~~bash
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/mctr911-mpl \
  ~/mctr911_kinematics_venv/bin/python 'Milestone 04/analysis/evaluate_milestone04.py'
~~~

Add **--output-dir /tmp/mctr911-evaluation-check** to generate a separate copy. The earlier [reference trajectories](trajectory/README.md), [full-pose IK](trajectory/README_Task_Space_IK.md), [control derivation](control/README_Motor_Level_Control.md), and [Cartesian reconstruction procedure](analysis/README_Stage4D_TaskSpaceTracking.md) remain available.
