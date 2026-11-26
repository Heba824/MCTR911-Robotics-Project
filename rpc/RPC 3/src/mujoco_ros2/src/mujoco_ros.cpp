/**
 * @file   mujoco_ros.cpp
 * @author Jon Woolfrey
 * @email  jonathan.woolfrey@gmail.com
 * @date   April 2025
 * @version 1.1
 * @brief  A class for connecting a MuJoCo simulation with ROS2 communication.
 * 
 * @details This class launches a MuJoCo simulation and provides communication channels in ROS2 for controlling it.
 * 
 * @copyright Copyright (c) 2025 Jon Woolfrey
 * 
 * @license GNU General Public License V3
 * 
 * @see https://mujoco.org/ for more information about MuJoCo
 * @see https://docs.ros.org/en/humble/index.html for ROS 2 documentation
 */
 
#include <mujoco_ros2/mujoco_ros.hpp>
#include <algorithm>
#include <cmath>
#include <charconv>
#include <string_view>
#include <fstream>
#include <thread>
#include <ctime>
namespace
{
int64_t monotonic_ns()
{
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

int64_t thread_cpu_ns()
{
    timespec stamp{};
    clock_gettime(CLOCK_THREAD_CPUTIME_ID, &stamp);
    return static_cast<int64_t>(stamp.tv_sec) * 1'000'000'000 + stamp.tv_nsec;
}

}

  ////////////////////////////////////////////////////////////////////////////////////////////////////
 //                                         Constructor                                            //
////////////////////////////////////////////////////////////////////////////////////////////////////
MuJoCoROS::MuJoCoROS(const std::string &xmlLocation) : Node("mujoco_node")
{
    // Declare & get parameters for this node
    std::string jointStateTopicName   = this->declare_parameter<std::string>("joint_state_topic_name", "joint_state");
    std::string jointCommandTopicName = this->declare_parameter<std::string>("joint_command_topic_name", "joint_commands");
    std::string controlMode           = this->declare_parameter<std::string>("control_mode", "POSITION");
    _simFrequency                     = this->declare_parameter<int>("simulation_frequency", 1000);
    int visualisationFrequency        = this->declare_parameter<int>("visualisation_frequency", 20);

    // Set the control mode
    if (controlMode == "POSITION") _controlMode = POSITION;
    else if (controlMode == "VELOCITY") _controlMode = VELOCITY;
    else if (controlMode == "TORQUE"  ) _controlMode = TORQUE;
    else
    {
        throw std::invalid_argument("[ERROR] [MuJoCo NODE] Unknown control mode. "
                                    "Argument was '" + controlMode + "', but expected 'POSITION', 'VELOCITY', or 'TORQUE'.");
    }

    const bool headless = this->declare_parameter<bool>("headless", false);
    if (_simFrequency <= 0 || (!headless && visualisationFrequency <= 0))
        throw std::invalid_argument("Simulation and enabled visualization rates must be positive.");

    // Load the robot model     
    char errorMessage[1000] = "Could not load model.";                                              // We need this as an input argument.   
    _model = mj_loadXML(xmlLocation.c_str(), nullptr, errorMessage, 1000);                          // Try to load the model  
    if (not _model) throw std::runtime_error("[ERROR] [MuJoCo NODE] Problem loading model: " + std::string(errorMessage)); 
    _model->opt.timestep = 1/((double)_simFrequency);                                               // Match MuJoCo to node frequency
    
    if (_controlMode == TORQUE)
    {
        configure_torque_model();
        _torqueTracePath = declare_parameter<std::string>("torque_trace_csv", "");
        if (!_torqueTracePath.empty()) _torqueTrace.reserve(12000);
    }

    _jointState = mj_makeData(_model);
    const int home_id = mj_name2id(_model, mjOBJ_KEY, "home");
    if (_controlMode == TORQUE) mj_setConst(_model, _jointState);
    if (home_id >= 0) mj_resetDataKeyframe(_model, _jointState, home_id);

    // Legacy modes use position servos, so start their targets at home.
    for (int i = 0; _controlMode != TORQUE && i < 7 && i < _model->nu; ++i)
        _jointState->ctrl[i] = _jointState->qpos[i];
    _jointStateMessage.name.resize(_model->nq);
    _jointStateMessage.position.resize(_model->nq);
    _jointStateMessage.velocity.resize(_model->nq);
    _jointStateMessage.effort.resize(_model->nq);    
    if (_controlMode == TORQUE)
    {
        for (int i = 0; i < 7; ++i)
        {
            _jointState->ctrl[_armActuators[i]] = 0;
            _jointStateMessage.name[i] = "joint" + std::to_string(i + 1);
        }
        _jointStateMessage.name[7] = "finger_joint1";
        _jointStateMessage.name[8] = "finger_joint2";
        _torqueStatus = create_publisher<std_msgs::msg::String>("torque_status", 1);
        mj_forward(_model, _jointState);
    }

    // Record joint names
    for (int i = 0; _controlMode != TORQUE && i < _model->nq; ++i) _jointStateMessage.name[i] = mj_id2name(_model, mjOBJ_JOINT, i);
       
    // Batch physics steps for nominal 100 Hz torque feedback or 50 Hz legacy feedback.
    _physicsStepsPerUpdate = std::max(1, _simFrequency / (_controlMode == TORQUE ? 100 : 50));
    const int updatePeriodMs = std::max(1, static_cast<int>(
        1000.0 * _physicsStepsPerUpdate / _simFrequency));
    _simTimer = this->create_wall_timer(std::chrono::milliseconds(updatePeriodMs),
                                        std::bind(&MuJoCoROS::update_simulation, this));
    RCLCPP_INFO(this->get_logger(),
        "Bridge: %d physics steps per ROS callback, nominal update %d ms",
        _physicsStepsPerUpdate, updatePeriodMs);
                                        
    _visualisationFrequency = visualisationFrequency;
    if (!headless && _controlMode != TORQUE) _visTimer = this->create_wall_timer(std::chrono::milliseconds(static_cast<int>(1000/visualisationFrequency)),
                                        std::bind(&MuJoCoROS::update_visualization, this));

   // Create joint state publisher and joint command subscriber
    // Keep the RPC 2 trajectory buffer. Torque commands use only the most recent packet.
    _jointCommandSubscriber = create_subscription<std_msgs::msg::Float64MultiArray>(
        jointCommandTopicName, _controlMode == TORQUE ? 1 : 100,
        [this](std_msgs::msg::Float64MultiArray::SharedPtr msg, const rclcpp::MessageInfo &info) {
            joint_command_callback(msg, info);
        });

    _jointStatePublisher = this->create_publisher<sensor_msgs::msg::JointState>(jointStateTopicName, 1);
                      
             
    if (headless) return;

    // Initialize Graphics Library FrameWork (GLFW)
    if (not glfwInit()) throw std::runtime_error("Failed to initialise Graphics Library Framework (GLFW).");

    _window = glfwCreateWindow(1200, 900, "MuJoCo Visualization", nullptr, nullptr);
    
    if (not _window) throw std::runtime_error("Failed to create Graphics Library Framework (GLFW) window.");
    
    // Make the OpenGL context current
    glfwMakeContextCurrent(_window);
    // Render at the configured rate, without coupling TORQUE frames to VSync.
    glfwSwapInterval(_controlMode == TORQUE ? 0 : 1);
    if (_controlMode == TORQUE)
    {
        _renderData = mj_makeData(_model);
        if (!_renderData) throw std::runtime_error("Could not allocate MuJoCo render snapshot.");
        if (!_torqueTracePath.empty()) _renderTrace.reserve(2000);
    }

    // Initialize MuJoCo rendering context
    mjv_defaultCamera(&_camera);
    mjv_defaultOption(&_renderingOptions);
    mjv_defaultPerturb(&_perturbation);
    mjr_defaultContext(&_context);
    mjv_makeScene(_model, &_scene, 1000);
    
    // Declare & get parameters for camera, visualisation
    _camera.azimuth      = this->declare_parameter<double>("camera_azimuth", 135);    
    _camera.distance     = this->declare_parameter<double>("camera_distance", 2.5);
    _camera.elevation    = this->declare_parameter<double>("camera_elevation", -35);
    _camera.orthographic = this->declare_parameter<bool>("camera_orthographic", true);
    
    auto focalPoint = this->declare_parameter<std::vector<double>>("camera_focal_point", {0.0, 0.0, 0.5});
    
    for(int i = 0; i < 3; ++i) _camera.lookat[i] = focalPoint[i]; 

    // Create MuJoCo rendering context
    glfwMakeContextCurrent(_window);
    mjr_makeContext(_model, &_context, mjFONTSCALE_100);
    
    RCLCPP_INFO(this->get_logger(), "MuJoCo simulation initiated. "
                                    "Publishing the joint state to '%s' topic. "
                                    "Subscribing to joint %s commands via '%s' topic.",
                                    jointStateTopicName.c_str(), controlMode.c_str(), jointCommandTopicName.c_str());

}

void MuJoCoROS::spin_bridge()
{
    if (!_renderData)
    {
        rclcpp::spin(shared_from_this());
        return;
    }

    // GLFW event processing and the GL context stay on the main thread.
    // The executor keeps physics, commands, and watchdog serial on its own thread.
    rclcpp::executors::SingleThreadedExecutor executor;
    executor.add_node(shared_from_this());
    std::exception_ptr spin_error;
    std::thread simulation([&]() {
        try { executor.spin(); }
        catch (...) { spin_error = std::current_exception(); rclcpp::shutdown(); }
    });
    try
    {
        const auto frame_period = std::chrono::duration<double>(1.0 / _visualisationFrequency);
        while (rclcpp::ok() && !glfwWindowShouldClose(_window))
        {
            const auto frame_start = std::chrono::steady_clock::now();
            update_visualization();
            std::this_thread::sleep_until(frame_start + frame_period);
        }
        rclcpp::shutdown();
        simulation.join();
    }
    catch (...)
    {
        rclcpp::shutdown();
        if (simulation.joinable()) simulation.join();
        throw;
    }
    if (spin_error) std::rethrow_exception(spin_error);
}

void MuJoCoROS::configure_torque_model()
{
    if (_model->nq != 9 || _model->nv != 9 || _model->nu != 8 ||
        mj_name2id(_model, mjOBJ_KEY, "home") < 0)
        throw std::invalid_argument("TORQUE requires the Panda arm, gripper and home keyframe.");

    for (int i = 0; i < 7; ++i)
    {
        const auto suffix = std::to_string(i + 1);
        const int joint = mj_name2id(_model, mjOBJ_JOINT, ("joint" + suffix).c_str());
        const int actuator = mj_name2id(_model, mjOBJ_ACTUATOR, ("actuator" + suffix).c_str());
        const double limit = i < 4 ? 87.0 : 12.0;
        if (joint < 0 || actuator < 0 || _model->jnt_type[joint] != mjJNT_HINGE ||
            _model->actuator_trntype[actuator] != mjTRN_JOINT ||
            _model->actuator_trnid[2 * actuator] != joint ||
            _model->actuator_gear[6 * actuator] != 1.0 ||
            _model->actuator_dyntype[actuator] != mjDYN_NONE ||
            !_model->actuator_forcelimited[actuator] ||
            _model->actuator_forcerange[2 * actuator] != -limit ||
            _model->actuator_forcerange[2 * actuator + 1] != limit)
            throw std::invalid_argument("Unexpected Panda transmission or force limit for joint " + suffix);

        _armActuators[i] = actuator;
        _armPositions[i] = _model->jnt_qposadr[joint];
        _armDofs[i] = _model->jnt_dofadr[joint];
        _model->actuator_gaintype[actuator] = mjGAIN_FIXED;
        _model->actuator_gainprm[mjNGAIN * actuator] = 1.0;
        _model->actuator_biastype[actuator] = mjBIAS_NONE;
        _model->actuator_ctrllimited[actuator] = 1;
        _model->actuator_ctrlrange[2 * actuator] = -limit;
        _model->actuator_ctrlrange[2 * actuator + 1] = limit;
    }

    _gripperActuator = mj_name2id(_model, mjOBJ_ACTUATOR, "actuator8");
    _gripperJoint = mj_name2id(_model, mjOBJ_JOINT, "finger_joint1");
    _secondFingerJoint = mj_name2id(_model, mjOBJ_JOINT, "finger_joint2");
    const int tendon = mj_name2id(_model, mjOBJ_TENDON, "split");
    if (_gripperActuator < 0 || _gripperJoint < 0 || _secondFingerJoint < 0 ||
        tendon < 0 || _model->actuator_trntype[_gripperActuator] != mjTRN_TENDON ||
        _model->actuator_trnid[2 * _gripperActuator] != tendon ||
        _model->jnt_type[_gripperJoint] != mjJNT_SLIDE ||
        _model->jnt_type[_secondFingerJoint] != mjJNT_SLIDE ||
        !_model->actuator_ctrllimited[_gripperActuator] ||
        _model->actuator_ctrlrange[2 * _gripperActuator] != 0 ||
        _model->actuator_ctrlrange[2 * _gripperActuator + 1] != 255)
        throw std::invalid_argument("Unexpected Panda gripper configuration.");

    RCLCPP_INFO(get_logger(), "TORQUE ready at home, waiting for a valid command. Watchdog: 100 ms.");
}

  ////////////////////////////////////////////////////////////////////////////////////////////////////
 //                                    Update the simulation                                       //
////////////////////////////////////////////////////////////////////////////////////////////////////
void MuJoCoROS::record_torque_trace(const TorqueTrace &event)
{
    if (_torqueTrace.size() < _torqueTrace.capacity()) _torqueTrace.push_back(event);
}

void MuJoCoROS::write_torque_trace() const
{
    if (_torqueTracePath.empty()) return;
    std::ofstream output(_torqueTracePath);
    if (!output)
    {
        RCLCPP_ERROR(get_logger(), "Could not write torque trace: %s", _torqueTracePath.c_str());
        return;
    }
    output << "event,monotonic_ns,ros_ns,sequence,publish_ros_ns,dds_source_ns,dds_receive_ns,"
              "physics_ns,callback_ns,pre_status_ns,status_publish_ns,state_publish_ns,callback_cpu_ns\n";
    for (const auto *events : {&_torqueTrace, &_renderTrace})
        for (const auto &e : *events)
            output << e.event << ',' << e.monotonic_ns << ',' << e.ros_ns << ',' << e.sequence << ','
               << e.publish_ros_ns << ',' << e.dds_source_ns << ',' << e.dds_receive_ns << ','
               << e.physics_ns << ',' << e.callback_ns << ',' << e.pre_status_ns << ','
               << e.status_publish_ns << ',' << e.state_publish_ns << ',' << e.callback_cpu_ns
               << '\n';
    if (_torqueTrace.size() == _torqueTrace.capacity())
        RCLCPP_WARN(get_logger(), "Torque trace buffer filled; later events were not recorded.");
}

void MuJoCoROS::check_torque_timeout()
{
    if (_torqueState == TorqueState::Active &&
        std::chrono::steady_clock::now() - _lastTorqueCommand >= std::chrono::milliseconds(100))
    {
        for (int actuator : _armActuators) _jointState->ctrl[actuator] = 0;
        _torqueState = TorqueState::TimedOut;
        if (!_torqueTracePath.empty())
            record_torque_trace({"watchdog", monotonic_ns(), get_clock()->now().nanoseconds()});
        RCLCPP_ERROR(get_logger(), "TORQUE watchdog expired. Simulation paused until relaunch.");
    }
}

void MuJoCoROS::update_simulation()
{
    if (not _model or not _jointState)
    {
        RCLCPP_ERROR(this->get_logger(), "MuJoCo model or data is not initialized.");
        return;
    }

    const bool trace = !_torqueTracePath.empty();
    const auto tick_start = trace ? monotonic_ns() : 0;
    const auto cpu_start = trace ? thread_cpu_ns() : 0;
    int64_t physics_ns = 0, pre_status_ns = 0, status_publish_ns = 0;
    if (_controlMode == TORQUE)
    {
        std::unique_lock<std::mutex> lock(_simulationMutex, std::defer_lock);
        if (_renderData) lock.lock();
        check_torque_timeout();
        const auto physics_start = trace ? monotonic_ns() : 0;
        if (_torqueState == TorqueState::Active)
        {
            for (int step = 0; step < _physicsStepsPerUpdate; ++step)
                mj_step(_model, _jointState);
        }
        mj_forward(_model, _jointState);
        if (trace) physics_ns = monotonic_ns() - physics_start;
        for (int i = 0; i < 7; ++i)
        {
            _jointStateMessage.position[i] = _jointState->qpos[_armPositions[i]];
            _jointStateMessage.velocity[i] = _jointState->qvel[_armDofs[i]];
            _jointStateMessage.effort[i] = _jointState->qfrc_actuator[_armDofs[i]];
        }
        const std::array<int, 2> fingers{_gripperJoint, _secondFingerJoint};
        for (int i = 0; i < 2; ++i)
        {
            const int dof = _model->jnt_dofadr[fingers[i]];
            _jointStateMessage.position[7 + i] = _jointState->qpos[_model->jnt_qposadr[fingers[i]]];
            _jointStateMessage.velocity[7 + i] = _jointState->qvel[dof];
            _jointStateMessage.effort[7 + i] = _jointState->qfrc_actuator[dof];
        }
        if (lock.owns_lock()) lock.unlock();
        std_msgs::msg::String status;
        status.data = _torqueState == TorqueState::Waiting ? "waiting" :
                      _torqueState == TorqueState::Active ? "active" : "timed_out";
        const auto status_start = trace ? monotonic_ns() : 0;
        if (trace) pre_status_ns = status_start - tick_start;
        _torqueStatus->publish(status);
        if (trace) status_publish_ns = monotonic_ns() - status_start;
    }
    else
    {
        for (int step = 0; step < _physicsStepsPerUpdate; ++step)
            mj_step(_model, _jointState);
        for (int i = 0; i < _model->nq; ++i)
        {
            _jointStateMessage.position[i] = _jointState->qpos[i];
            _jointStateMessage.velocity[i] = _jointState->qvel[i];
            _jointStateMessage.effort[i] =
                (i < _model->nu) ? _jointState->actuator_force[i] : 0.0;
        }
    }

    _jointStateMessage.header.stamp = this->get_clock()->now();
    const auto published_at = trace ? monotonic_ns() : 0;
    _jointStatePublisher->publish(_jointStateMessage);
    if (trace)
    {
        const auto done = monotonic_ns();
        record_torque_trace({"state", published_at,
            rclcpp::Time(_jointStateMessage.header.stamp).nanoseconds(), 0, 0, 0, 0,
            physics_ns, done - tick_start, pre_status_ns, status_publish_ns,
            done - published_at, thread_cpu_ns() - cpu_start});
    }
}

  ////////////////////////////////////////////////////////////////////////////////////////////////////
 //                                    Handle joint commands                                       //
////////////////////////////////////////////////////////////////////////////////////////////////////
void
MuJoCoROS::joint_command_callback(const std_msgs::msg::Float64MultiArray::SharedPtr msg,
                                  const rclcpp::MessageInfo &info)
{
    if (_controlMode == TORQUE)
    {
        const auto received_mono = _torqueTracePath.empty() ? 0 : monotonic_ns();
        const auto received_ros = received_mono ? get_clock()->now().nanoseconds() : 0;
        auto trace_command = [&](const char *event) {
            if (!received_mono) return;
            int64_t sequence = 0, published = 0;
            if (!msg->layout.dim.empty())
            {
                const std::string_view label = msg->layout.dim.front().label;
                if (label.substr(0, 5) == "rpc3:")
                {
                    const auto end = label.data() + label.size();
                    const auto first = std::from_chars(label.data() + 5, end, sequence);
                    if (first.ec == std::errc{} && first.ptr != end && *first.ptr == ':')
                    {
                        const auto second = std::from_chars(first.ptr + 1, end, published);
                        if (second.ec != std::errc{} || second.ptr != end || published <= 0 ||
                            sequence <= 0)
                            sequence = published = 0;
                    }
                    else sequence = 0;
                }
            }
            const auto &middleware = info.get_rmw_message_info();
            record_torque_trace({event, received_mono, received_ros, sequence, published,
                                 middleware.source_timestamp, middleware.received_timestamp,
                                 0, monotonic_ns() - received_mono});
        };
        std::unique_lock<std::mutex> lock(_simulationMutex, std::defer_lock);
        if (_renderData) lock.lock();
        check_torque_timeout();
        if (_torqueState == TorqueState::TimedOut)
        {
            trace_command("rejected_latched");
            return;
        }
        // Depth-one QoS still permits a single queued sample to arrive late.
        // Fast DDS source time is on the same system clock as this process.
        const auto stamp = info.get_rmw_message_info().source_timestamp;
        const auto system_now = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        if (stamp > 0 && system_now - stamp > 50'000'000)
        {
            trace_command("rejected_stale");
            RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000,
                                 "Rejected TORQUE packet older than 50 ms at callback entry.");
            return;
        }
        const auto &values = msg->data;
        bool valid = values.size() == 9;
        if (valid)
        {
            valid = values[8] == 0 && std::isfinite(values[7]) &&
                    values[7] >= 0 && values[7] <= 255;
            for (int i = 0; i < 7; ++i)
            {
                const double limit = _model->actuator_ctrlrange[2 * _armActuators[i] + 1];
                valid = valid && std::isfinite(values[i]) && std::abs(values[i]) <= limit;
            }
        }
        if (!valid)
        {
            trace_command("rejected_invalid");
            RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000, "Rejected invalid TORQUE packet.");
            return;
        }
        for (int i = 0; i < 7; ++i) _jointState->ctrl[_armActuators[i]] = values[i];
        _jointState->ctrl[_gripperActuator] = values[7];
        _lastTorqueCommand = std::chrono::steady_clock::now();
        if (_torqueState == TorqueState::Waiting)
            RCLCPP_INFO(get_logger(), "First valid TORQUE packet received. Simulation active.");
        _torqueState = TorqueState::Active;
        trace_command("valid_command");
        return;
    }

    if (msg->data.size() != _model->nq)
    {
        RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
                             "Received joint command with incorrect size.");
        return;
    }
    else
    {
        switch(_controlMode)
        {
            case POSITION:
            {
                for(int i = 0; i < _model->nu; ++i)
                {
                    _jointState->ctrl[i] = msg->data[i];                                            // Assign control input directly
                }
                break;
            }
            case VELOCITY:
            {
                if (msg->data.size() < 9 || _model->nu < 8)
                {
                    RCLCPP_WARN(this->get_logger(), "Invalid velocity command.");
                    break;
                }

                // Each published velocity is one 0.05 s sample of the
                // instructor's planned trajectory (TRAJ_DT = 0.05).
                // Integrating here makes the *commanded* displacement
                // independent of MuJoCo's actual wall-clock step rate.
                constexpr double commandDt = 0.05;
                bool nonzeroArmVelocity = false;
                for (int i = 0; i < 7; ++i)
                {
                    if (!std::isfinite(msg->data[i]))
                    {
                        RCLCPP_WARN(this->get_logger(), "Nonfinite velocity command rejected.");
                        return;
                    }
                }
                for (int i = 0; i < 7; ++i)
                {
                    const double dq = msg->data[i];
                    nonzeroArmVelocity |= std::abs(dq) > 1e-8;
                    _jointState->ctrl[i] += dq * commandDt;

                    // Position-type arm actuators still need legal targets.
                    if (_model->jnt_limited[i])
                    {
                        _jointState->ctrl[i] = std::clamp(
                            _jointState->ctrl[i],
                            _model->jnt_range[2 * i],
                            _model->jnt_range[2 * i + 1]);
                    }
                }

                // Gripper actuator is commanded directly, never integrated.
                _jointState->ctrl[7] = msg->data[7];

                // One short built-in diagnostic at the end of the trajectory.
                // Repeated zero commands are ignored by this diagnostic.
                static int receivedSamples = 0;
                static bool sawMovement = false;
                static bool loggedFinish = false;
                ++receivedSamples;
                if (nonzeroArmVelocity)
                    sawMovement = true;
                else if (sawMovement && !loggedFinish)
                {
                    loggedFinish = true;
                    RCLCPP_INFO(this->get_logger(),
                        "VELOCITY complete: %d commands; final actuator targets "
                        "[%.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f]",
                        receivedSamples,
                        _jointState->ctrl[0], _jointState->ctrl[1],
                        _jointState->ctrl[2], _jointState->ctrl[3],
                        _jointState->ctrl[4], _jointState->ctrl[5],
                        _jointState->ctrl[6]);
                }
                break;
            }
            default:
            {
                RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
                                     "Unknown control mode. Unable to set joint commands.");
                break;
            }
        }
    }
}

  ////////////////////////////////////////////////////////////////////////////////////////////////////
 //                                    Update the 3D simulation                                    //
////////////////////////////////////////////////////////////////////////////////////////////////////
void MuJoCoROS::update_visualization()
{
    glfwMakeContextCurrent(_window);                                                                // Ensure OpenGL context is current
    
    const auto start = _renderTrace.capacity() ? monotonic_ns() : 0;
    mjData *view = _jointState;
    if (_renderData)
    {
        // No GLFW call runs under this lock, and MuJoCo physics never touches _renderData.
        {
            std::lock_guard<std::mutex> lock(_simulationMutex);
            mj_copyData(_renderData, _model, _jointState);
        }
        view = _renderData;
    }

    mjv_updateScene(_model, view, &_renderingOptions, NULL, &_camera,
                    mjCAT_ALL, &_scene);   // Update 3D rendering

    // Get framebuffer size
    int width, height;
    glfwGetFramebufferSize(_window, &width, &height);
    mjrRect viewport = {0, 0, width, height};

    mjr_render(viewport, &_scene, &_context); // Render scene
    
    // Swap buffers and process events
    glfwSwapBuffers(_window);
    glfwPollEvents();
    if (start && _renderTrace.size() < _renderTrace.capacity())
        _renderTrace.push_back({"render", start, get_clock()->now().nanoseconds(),
                                0, 0, 0, 0, 0, monotonic_ns() - start});
}

  ////////////////////////////////////////////////////////////////////////////////////////////////////
 //                                           Destructor                                           //
////////////////////////////////////////////////////////////////////////////////////////////////////
MuJoCoROS::~MuJoCoROS()
{
    write_torque_trace();
    if (_renderData) mj_deleteData(_renderData);
    mj_deleteData(_jointState);
    mj_deleteModel(_model);
    if (_window)
    {
        mjv_freeScene(&_scene);
        mjr_freeContext(&_context);
        glfwDestroyWindow(_window);
        glfwTerminate();
    }
}     
