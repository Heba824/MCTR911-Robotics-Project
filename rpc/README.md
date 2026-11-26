# Robotics Programming Challenges

- [RPC 1](RPC%201/) implements Franka Panda forward and inverse kinematics with numerical checks and a MuJoCo POSITION bridge.
- [RPC 2](RPC%202/) implements the linear Jacobian, its time derivative, and a quintic trajectory for Team 1's pick-and-place motion. The [side-by-side instructions](RPC%202/RPC2_SIDE_BY_SIDE_README.txt) describe both command modes. The [comparison plot](RPC%202/rpc2_comparison_results/position_vs_velocity.svg) shows recorded tracking.
- [RPC 3](RPC%203/) implements Panda dynamics, torque control, and joint and Cartesian fuzzy controllers. The [evaluation package](RPC%203/docs/rpc3/final_evaluation/README.md) contains the report, demo guide, figures, and data.

Each challenge is a separate ROS 2 Jazzy workspace. Build from inside the challenge directory so that Colcon discovers only that challenge's packages. The launch files expect a Franka Panda scene at `~/mujoco_menagerie/franka_emika_panda/scene.xml`. The bridge CMake file uses MuJoCo 3.2.0 under `/opt/mujoco`.

```bash
source /opt/ros/jazzy/setup.bash
colcon build --packages-select panda_kinematics mujoco_ros2
source install/setup.bash
```

For RPC 1, launch the bridge with `ros2 launch mujoco_ros2 example.py` and run `ros2 run panda_kinematics kinematics` in a second sourced terminal. RPC 2 defaults to POSITION in both the launch file and the trajectory node. Its side-by-side script selects the control mode and matching commands for each simulator.
