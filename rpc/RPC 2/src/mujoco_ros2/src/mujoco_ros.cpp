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
  ////////////////////////////////////////////////////////////////////////////////////////////////////
 //                                         Constructor                                            //
////////////////////////////////////////////////////////////////////////////////////////////////////
MuJoCoROS::MuJoCoROS(const std::string &xmlLocation) : Node("mujoco_node")
{
    // Declare & get parameters for this node
    std::string jointStateTopicName   = this->declare_parameter<std::string>("joint_state_topic_name", "joint_state");
    std::string jointCommandTopicName = this->declare_parameter<std::string>("joint_command_topic_name", "joint_commands");
    std::string controlMode           = this->declare_parameter<std::string>("control_mode", "TORQUE");
    _simFrequency                     = this->declare_parameter<int>("simulation_frequency", 1000);
    int visualisationFrequency        = this->declare_parameter<int>("visualisation_frequency", 20);

    // Load the robot model     
    char errorMessage[1000] = "Could not load model.";                                              // We need this as an input argument.   
    _model = mj_loadXML(xmlLocation.c_str(), nullptr, errorMessage, 1000);                          // Try to load the model  
    if (not _model) throw std::runtime_error("[ERROR] [MuJoCo NODE] Problem loading model: " + std::string(errorMessage)); 
    _model->opt.timestep = 1/((double)_simFrequency);                                               // Match MuJoCo to node frequency
    
     // Resize arrays based on the number of joints in the model
    _jointState = mj_makeData(_model);
	int home_id = mj_name2id(_model, mjOBJ_KEY, "home");

	if (home_id >= 0)
	{
   	 mj_resetDataKeyframe(_model, _jointState, home_id);
	}

	// Initialize the seven arm position-actuator targets
	// to match the robot's actual starting joint positions.
	for (int i = 0; i < 7 && i < _model->nu; ++i)
	{
   	 _jointState->ctrl[i] = _jointState->qpos[i];
	}
    _jointStateMessage.name.resize(_model->nq);
    _jointStateMessage.position.resize(_model->nq);
    _jointStateMessage.velocity.resize(_model->nq);
    _jointStateMessage.effort.resize(_model->nq);    
    _torqueInput.resize(_model->nq, 0.0);

    // Record joint names
    for (int i = 0; i < _model->nq; ++i) _jointStateMessage.name[i] = mj_id2name(_model, mjOBJ_JOINT, i);
       
    // Set the control mode  
         if (controlMode == "POSITION") _controlMode = POSITION;
    else if (controlMode == "VELOCITY") _controlMode = VELOCITY;
    else if (controlMode == "TORQUE"  ) _controlMode = POSITION;
    else
    {   
        throw std::invalid_argument("[ERROR] [MuJoCo NODE] Unknown control mode. "
                                    "Argument was '" + controlMode + "', but expected 'POSITION', 'VELOCITY', or 'TORQUE'.");
    }

    // Keep the physics timestep at 1 / _simFrequency (normally 0.001 s),
    // but batch physics steps so ROS only schedules/publishes at about 50 Hz.
    // This avoids flooding the executor and Python node with 1000 JointState/s.
    const int physicsStepsPerUpdate = std::max(1, _simFrequency / 50);
    const int updatePeriodMs = std::max(1, static_cast<int>(
        1000.0 * physicsStepsPerUpdate / _simFrequency));
    _simTimer = this->create_wall_timer(std::chrono::milliseconds(updatePeriodMs),
                                        std::bind(&MuJoCoROS::update_simulation, this));
    RCLCPP_INFO(this->get_logger(),
        "RPC2 bridge: %d physics steps per ROS callback, nominal update %d ms",
        physicsStepsPerUpdate, updatePeriodMs);
                                        
    _visTimer = this->create_wall_timer(std::chrono::milliseconds(static_cast<int>(1000/visualisationFrequency)),
                                        std::bind(&MuJoCoROS::update_visualization, this));

   // Create joint state publisher and joint command subscriber
    // Buffer a complete 81-sample trajectory if rendering briefly delays the callback.
    _jointCommandSubscriber = this->create_subscription<std_msgs::msg::Float64MultiArray>(jointCommandTopicName, 100, std::bind(&MuJoCoROS::joint_command_callback, this, std::placeholders::_1));
    
    _jointStatePublisher = this->create_publisher<sensor_msgs::msg::JointState>(jointStateTopicName, 1);
                      
             
    // Initialize Graphics Library FrameWork (GLFW)
    if (not glfwInit()) throw std::runtime_error("Failed to initialise Graphics Library Framework (GLFW).");

    _window = glfwCreateWindow(1200, 900, "MuJoCo Visualization", nullptr, nullptr);
    
    if (not _window) throw std::runtime_error("Failed to create Graphics Library Framework (GLFW) window.");
    
    // Make the OpenGL context current
    glfwMakeContextCurrent(_window);
    glfwSwapInterval(1);                    // NOTE TO SELF: CHECK THIS ARGUMENT

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

  ////////////////////////////////////////////////////////////////////////////////////////////////////
 //                                    Update the simulation                                       //
////////////////////////////////////////////////////////////////////////////////////////////////////
void MuJoCoROS::update_simulation()
{
    if (not _model or not _jointState)
    {
        RCLCPP_ERROR(this->get_logger(), "MuJoCo model or data is not initialized.");
        return;
    }

    // Execute multiple 1 ms MuJoCo steps in each 20 ms ROS callback.
    // The velocity command is already integrated into position actuator
    // targets by joint_command_callback() exactly once per received sample.
    const int physicsStepsPerUpdate = std::max(1, _simFrequency / 50);
    for (int step = 0; step < physicsStepsPerUpdate; ++step)
    {
        if (_controlMode == TORQUE)
        {
            for (int i = 0; i < _model->nu; ++i)
            {
                _jointState->ctrl[i] = _torqueInput[i]
                    + _jointState->qfrc_bias[i] - 0.01 * _jointState->qvel[i];
                _torqueInput[i] = 0.0;
            }
        }
        mj_step(_model, _jointState);
    }

    // Publish at ~50 Hz, not once per 1 ms physics step.
    // The reduced JointState stream leaves time for trajectory callbacks.
    for (int i = 0; i < _model->nq; ++i)
    {
        _jointStateMessage.position[i] = _jointState->qpos[i];
        _jointStateMessage.velocity[i] = _jointState->qvel[i];
        _jointStateMessage.effort[i] =
            (i < _model->nu) ? _jointState->actuator_force[i] : 0.0;
    }

    _jointStateMessage.header.stamp = this->get_clock()->now();
    _jointStatePublisher->publish(_jointStateMessage);
}

  ////////////////////////////////////////////////////////////////////////////////////////////////////
 //                                    Handle joint commands                                       //
////////////////////////////////////////////////////////////////////////////////////////////////////
void
MuJoCoROS::joint_command_callback(const std_msgs::msg::Float64MultiArray::SharedPtr msg)
{
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
            case TORQUE:
            {
                for(int i = 0; i < _model->nq; ++i)
                {
                    _torqueInput[i] = msg->data[i];
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
    
    mjv_updateScene(_model, _jointState, &_renderingOptions, NULL, &_camera, mjCAT_ALL, &_scene);   // Update 3D rendering

    // Get framebuffer size
    int width, height;
    glfwGetFramebufferSize(_window, &width, &height);
    mjrRect viewport = {0, 0, width, height};

    mjr_render(viewport, &_scene, &_context); // Render scene
    
    // Swap buffers and process events
    glfwSwapBuffers(_window);
    glfwPollEvents();
}

  ////////////////////////////////////////////////////////////////////////////////////////////////////
 //                                           Destructor                                           //
////////////////////////////////////////////////////////////////////////////////////////////////////
MuJoCoROS::~MuJoCoROS()
{
    mj_deleteData(_jointState);
    mj_deleteModel(_model);
    mjv_freeScene(&_scene);
    mjr_freeContext(&_context);
    glfwDestroyWindow(_window);
    glfwTerminate();
}     
