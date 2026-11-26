# RPC 3 final evaluation · Team 1
MCTR911 Robotics Programming · Franka Panda simulation · 25 November 2026

## 1. Assignment objectives and Team 1 requirements

RPC 3 models the Panda arm’s Lagrangian dynamics, applies physical joint torque in MuJoCo, and compares a model-only controller with a seven-channel Mamdani controller on the same reference. Team 1 adds a separate XYZ Cartesian FLC with Jacobian-based velocity mapping and singularity handling. The three runnable modes are `model_only`, `joint_flc`, and `cartesian_flc`.

## 2. System architecture and ROS topics

ROS 2 Jazzy runs on Ubuntu 24.04 under WSL. The simulated Panda bridge publishes named position, velocity, and effort on `/rpc3/joint_state`, plus watchdog state on `/rpc3/torque_status`. The controller publishes seven bounded arm torques, a separate gripper command, and a reserved zero in the nine-field `/rpc3/joint_commands` packet. The bridge converts only the named arm actuators to unit-gain, zero-bias torque actuation. The controller sends torque from fresh state on a nominal 10 ms schedule. Graphical rendering uses a private state snapshot and runs separately from TORQUE physics and ROS callbacks.

## 3. Franka Panda Lagrangian dynamics

For body i, compute the world COM position and the linear and angular COM Jacobians from the MJCF body and joint geometry. Resolve body masses, COM offsets and inertia tensors from the compiled Panda model, then rotate inertia into world axes. With gravity acceleration vector **g₀**, potential energy is −Σ mᵢ **g₀**ᵀ **pᵢ**. Its gradient gives the gravity torque. The Hessian of kinetic energy with respect to joint velocity gives

```text
M(q) = Σ [mᵢ Jvᵢᵀ Jvᵢ + Jωᵢᵀ Iᵢ(world) Jωᵢ] + diag(armature)
τ = M(q) q̈ + C(q,q̇)q̇ + g(q)
```

The implementation evaluates COM and angular accelerations at zero joint acceleration through centered directional Jacobian differences. Projecting inertial forces and moments, including the angular gyroscopic term, gives the Coriolis and centrifugal torque. Fingers are included as bodies at the fixed home opening in this seven-coordinate model.

## 4. Standard seven-joint Mamdani FLC

The feedforward command is `M(q) @ ddq_des + C(q,dq) @ dq + g(q)`. Joint FLC adds seven independent corrective torques. Each channel uses the desired-minus-actual position and velocity errors, scales them into a normalized domain, and fuzzifies them with LN, MN, S, MP and LP. The two outer sets are shoulders and the interior sets are triangles. The explicit 5×5 rule table combines position and velocity labels. Min evaluates conjunction and clips each consequent. Max aggregates the consequents. Centroid integration converts the aggregate to a bounded scalar correction. The fuzzy channels are decoupled, while the Lagrangian model retains joint coupling.

## 5. Team 1 Cartesian FLC

The four-second joint quintic defines the desired hand position and velocity in the robot base frame. The three fuzzy channels use `p(q_des) - p(q)` and `J(q_des) @ dq_des - J(q) @ dq`. Their outputs are XYZ velocity corrections in m/s. The controller adds feedforward XYZ velocity, limits the task request, maps it to joint velocity, and uses joint velocity feedback and reference acceleration to obtain a bounded acceleration command. The independent model converts that acceleration to torque. The mode controls translation only and biases redundant motion toward the joint reference.

## 6. Damped least-squares Jacobian inverse

For the linear 3×7 Jacobian `J = U Σ Vᵀ`, each singular-direction gain of the inverse is `σ / (σ² + λ²)`. This approaches the ordinary pseudoinverse away from singularities and attenuates the weakest directions as damping increases. The null-space bias uses `I − VᵣVᵣᵀ`, where the columns of `Vᵣ` are the three selected right-singular vectors.

## 7. Singularity protection and trade-offs

Damping follows `λ = 0.10 × clip(1 − σmin/0.08, 0, 1)²` in m/rad. The controller bounds task speed at 0.18 m/s, posture bias at 0.20 rad/s in norm, joint speed at 0.65 rad/s per joint and commanded acceleration at 4 rad/s² per joint. Outward speed is reduced near each joint limit. Arm torque commands are bounded to ±87 N·m on joints 1–4 and ±12 N·m on joints 5–7. These are command limits. The achieved motion depends on the simulator and applied torque.

## 8. Experimental setup

The nominal reference is the same legal four-second quintic, starting from the verified Panda home pose. The matched three-mode comparison in Figures 01–17 uses the **headless MuJoCo test pipe**. Separate tables identify **live ROS headless** and **graphical ROS** runs. The singularity analysis is **offline numerical**. The evidence manifest maps bundled files to the original workspace paths. Metrics use the rows tagged `tracking` through the four-second reference endpoint. Timing uses all saved rows. Joint RMSE pools scalar errors over joints and samples. XYZ RMSE uses the Euclidean position-error norm, with XYZ recomputed using the implementation’s FK. Publication gaps use monotonic times or recorded monotonic send offsets. Missing values appear blank in CSV and as em dashes in Markdown. [Evidence manifest](evidence/sources.json) · [Tracking metrics](tables/tracking.md) · [Timing](tables/timing.md) · [Safety](tables/safety.md)

## 9. Dynamics validation results

With seed 911, 200 legal states were sampled with joint velocities and accelerations. Contacts and constraints were disabled for this comparison, and passive effects were reconciled. The analytical matrix was compared with `mj_fullM`, the bias with `qfrc_bias`, and total torque with `mj_inverse`. The saved check’s maximum errors were **5.77 × 10⁻¹⁵ kg·m²** for mass and **5.87 × 10⁻¹⁰ N·m** for both bias and total torque. The independent rerun also passed all thresholds. [Validation table](tables/dynamics.md) · [Recheck, including per-joint errors](evidence/dynamics_recheck.json)

## 10. Model-only versus joint-FLC results

| Matched test-pipe metric | Model only | Joint FLC | Cartesian FLC |
|---|---:|---:|---:|
| Joint RMSE (rad) | 0.11687 | 0.001275 | 0.01211 |
| XYZ RMSE (mm) | 51.904 | 0.358 | 0.929 |
| XYZ peak (mm) | 92.884 | 0.531 | 1.410 |
| XYZ final (mm) | 92.884 | 0.080 | 0.220 |

- [01 · Joint tracking](figures/01_joint_positions.png) shows the seven reference and actual paths. Feedback reduces the lag visible with model feedforward alone.
- [02 · Joint errors](figures/02_joint_errors.png) shows the signed errors, including residual errors at the endpoint.
- [03 · Joint RMS](figures/03_joint_rms.png) and [04 · Joint peaks](figures/04_joint_maximum.png) show lower RMS and peak error on each joint with joint FLC.
- [05 · Joint torque](figures/05_joint_torques.png) shows the different arm torque bounds and the commanded effort below them.

[Per-joint RMSE](tables/per_joint_rmse.md)

## 11. Cartesian FLC results

- [06 · XYZ tracking](figures/06_cartesian_position.png) shows the Cartesian controller following the hand path derived from the same joint reference.
- [07 · Signed XYZ error](figures/07_cartesian_error.png) identifies each axis’s contribution. [08 · XYZ error norm](figures/08_xyz_error_norm.png) gives a direction-independent measure and peaks at 1.410 mm in the test pipe.
- [09 · Cartesian-mode joint motion](figures/09_cartesian_joint_motion.png) shows that similar XYZ motion can result from different joint trajectories.
- [10 · Fuzzy velocity](figures/10_fuzzy_velocity.png) contrasts the fuzzy correction recomputed from logged XYZ errors with the recorded total velocity command, which includes feedforward.
- [11 · Minimum singular value](figures/11_sigma_min.png) remains above the activation threshold along this nominal path. [12 · Damping](figures/12_damping.png) accordingly stays at zero.
- [13 · Command limits](figures/13_limit_behavior.png) shows task velocity, mapped joint speed, and normalized torque below their bounds. The trace recorded zero task-speed, joint-speed, acceleration, and torque limit activations.
- [14 · XYZ RMSE](figures/14_xyz_rmse.png) and [15 · Joint RMSE](figures/15_joint_rmse.png) show that both FLC modes improve on feedforward. Joint FLC gives the smallest errors on this reference, which directly specifies all seven joint positions.

## 12. Numerical singularity experiment

The seeded search selected a legal contact-free pose with `σmin = 0.00230 m/rad`. With a 0.08 m/s request along its weakest direction, the undamped inverse gives a joint-speed norm of **34.76 rad/s**. The damped inverse gives **0.0207 rad/s** and a Cartesian residual of **0.07995 m/s**. Neither output was commanded to a robot. [Numerical values](tables/singularity.md)

- [18 · Joint-speed norm](figures/18_singular_speed.png) shows amplification for eight requests at that single near-singular pose.
- [19 · Task residual](figures/19_singular_residual.png) shows the accuracy sacrificed to suppress weak-direction motion.
- [20 · Damping law](figures/20_damping_law.png) evaluates the implemented equation on a new offline parameter grid. The marker is the actual test pose.

## 13. Live ROS verification

Stage 3 completed 3/3 model-only and 3/3 joint-FLC runs. The latest Stage 4 run completed **3/3 Cartesian** trials with 834, 828 and 835 commands, plus **1/1 model-only** and **1/1 joint-FLC** regressions with 838 and 841 commands. All eleven saved passes show the bridge accepted the trajectory endpoint and no watchdog occurred before controller shutdown. The latest live Cartesian XYZ RMSE ranged from **0.903 to 0.927 mm** and joint RMSE from **0.01210 to 0.01242 rad**. [Per-trial acceptance and provenance](tables/live_trials.md) · [Pass counts](tables/trial_counts.md)

## 14. Graphical MuJoCo verification

The Cartesian GUI run recorded 2,174 total commands and 549 saved rows. Its controller JSON records the endpoint, terminal-hold commands, no fault, and zero controller-observed watchdogs. Over the tracking interval, XYZ RMSE was **0.946 mm** and the worst joint endpoint error was **0.009917 rad**. The last recorded terminal-hold sample, at **5.498 s** with reference **4.0 s**, had maximum and RMS joint errors of **0.000705 rad** and **0.000463 rad**. Those are terminal-hold statistics. [Terminal sample](tables/gui_terminal_sample.md)

The GUI operator reported a stale torque rejection and a bridge watchdog after the recorded tracking interval. No matching bridge trace was available for this successful GUI file, so the packet count and watchdog timing relative to controller shutdown cannot be independently verified. There is no complete graphical data set for standard joint FLC in this package.

## 15. Timing and safety observations

- [16 · Compute latency](figures/16_latency.png) compares wall-time statistics across the test-pipe runs. Their maximum torque computations were below 10 ms.
- [17 · Command timing](figures/17_command_timing.png) shows actual send gaps and zero missed send deadlines in those matched runs. A gap above 10 ms can occur even when the next send meets its own deadline.
- [21 · Graphical timing and hold](figures/21_graphical_timing_hold.png) shows recorded command gaps, computation and missed-reference counts alongside the terminal-hold error. The full GUI JSON reports **224 missed reference slots**, **11 skipped scheduler ticks**, **9 publication deadline misses**, **3 callback overruns**, and **2 computation overruns**. These counters have different definitions and the full-run totals extend beyond the saved plot window.

The controller rejects expired feedback at 50 ms. The bridge rejects aged DDS-stamped torque packets and latches at a 100 ms command timeout. The scheduled controller and rendering snapshot allow feedback and physics to proceed separately. Passing runs still show scheduling variation and do not guarantee a hard 100 Hz deadline. [Complete counters](tables/safety.md)

## 16. Limitations and conclusions

The evidence establishes repeatable live ROS headless execution for all required modes and a completed recorded Cartesian graphical trajectory. Joint FLC performed best on this joint-defined reference. Cartesian FLC nevertheless tracked XYZ closely and the numerical experiment demonstrates how damping trades task accuracy for bounded motion near a singularity. The nominal Cartesian run did not engage damping, and the separate near-singular experiment did not actuate the simulator. The model assumes a fixed gripper opening for arm dynamics and does not model arbitrary contact or payload. All reported tracking was in simulation. No physical Franka hardware was used.
