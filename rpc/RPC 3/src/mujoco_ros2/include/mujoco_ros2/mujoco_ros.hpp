/**
 * @file   mujoco_ros.hpp
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

#ifndef MUJOCO_NODE_H
#define MUJOCO_NODE_H

#include <GLFW/glfw3.h>                                                                             // Graphics Library Framework; for visualisation
#include <iostream>                                                                                 // std::cerr, std::cout
#include <mujoco/mujoco.h>                                                                          // Dynamic simulation library
#include <rclcpp/rclcpp.hpp>                                                                        // ROS2 C++ libraries.
#include <sensor_msgs/msg/joint_state.hpp>                                                          // For publishing / subscribing to joint states.
#include <array>
#include <chrono>
#include <mutex>
#include <std_msgs/msg/float64_multi_array.hpp>
#include <std_msgs/msg/string.hpp>

enum ControlMode {POSITION, VELOCITY, TORQUE, UNKNOWN};                                             // This needs a global scope
        
/**
 * @brief This class launches both a MuJoCo simulation, and ROS2 node for communication.
 */
class MuJoCoROS: public rclcpp::Node
{
    public:
            
        /**
         * @brief Contructor.
         * @param xmlLocation Where to find the XML file that defines the MuJoCo model.
         * @param jointStateTopicName The name of the topic to which joint state data is published.
         * @param jointControlTopicName The name of the topic to subscribe to for control inputs.
         * @param controlMode POSITION, VELOCITY, or TORQUE
         * @param simulationFrequency The rate in Hz to update the simulation
         * @param visualisationFrequency The rate with which to refresh the 3D rendering of the robot, environment
         */
        MuJoCoROS(const std::string &xmlLocation);
        
       /**
        * @brief Deconstructor.
        */
        ~MuJoCoROS();

        void spin_bridge();
        
    private:

        enum class TorqueState { Waiting, Active, TimedOut };
        TorqueState _torqueState = TorqueState::Waiting;
        std::array<int, 7> _armActuators{}, _armPositions{}, _armDofs{};
        int _gripperActuator = -1;
        int _gripperJoint = -1;
        int _secondFingerJoint = -1;
        int _physicsStepsPerUpdate = 1;
        std::chrono::steady_clock::time_point _lastTorqueCommand{};
        struct TorqueTrace
        {
            const char *event;
            int64_t monotonic_ns;
            int64_t ros_ns;
            int64_t sequence = 0;
            int64_t publish_ros_ns = 0;
            int64_t dds_source_ns = 0;
            int64_t dds_receive_ns = 0;
            int64_t physics_ns = 0;
            int64_t callback_ns = 0;
            int64_t pre_status_ns = 0;
            int64_t status_publish_ns = 0;
            int64_t state_publish_ns = 0;
            int64_t callback_cpu_ns = 0;
        };
        std::string _torqueTracePath;
        std::vector<TorqueTrace> _torqueTrace, _renderTrace;
        void record_torque_trace(const TorqueTrace &event);
        void write_torque_trace() const;
        rclcpp::Publisher<std_msgs::msg::String>::SharedPtr _torqueStatus;
        void configure_torque_model();
        void check_torque_timeout();
        friend struct TorqueBridgeTest;
        std::array<double, 7> _armVelocityCommand{};
	std::chrono::steady_clock::time_point _lastVelocityCommand{};
        ControlMode _controlMode;                                                                   ///< POSITION, VELOCITY, or TORQUE
        
        mjModel *_model;                                                                            ///< Underlying model of the robot.
        mjData  *_jointState;                                                                       ///< Joint state data (position, velocity, acceleration)

        mjvCamera  _camera;                                                                         ///< Camera for viewing
        mjvOption  _renderingOptions;                                                               ///< As it says
        mjvPerturb _perturbation;                                                                   ///< Allows manual interaction
        mjvScene   _scene;                                                                          ///< The environment that the robot is rendered in

        mjrContext _context;                                                                        ///< No idea what this does.

        // Only the live-state copy is locked. GLFW and rendering use a private snapshot.
        mjData *_renderData = nullptr;
        std::mutex _simulationMutex;
        int _visualisationFrequency = 20;
        GLFWwindow *_window = nullptr;                                                                        ///< This displays the robot and environment.

        rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr _jointStatePublisher;            ///< As it says on the label

        rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr _jointCommandSubscriber;  ///< Subscriber for joint commands
        
        rclcpp::TimerBase::SharedPtr _simTimer, _visTimer;                                          ///< Regulates the ROS2 node

        sensor_msgs::msg::JointState _jointStateMessage;                                            ///< For publishing joint state data over ROS2
        
        int _simFrequency = 1000;                                                                   ///< Speed at which the frequency runs
    
        /**
         * @brief Updates the robot state, publishes joint state information.
         */
        void
        update_simulation();
        
        /**
         * @brief Updates the visualisation.
         */
        void
        update_visualization();
        
        /**
         * @brief Callback function to handle incoming joint commands.
         * @param msg The message containing joint commands.
         */
        void
        joint_command_callback(const std_msgs::msg::Float64MultiArray::SharedPtr msg,
                               const rclcpp::MessageInfo &info = rclcpp::MessageInfo(rmw_message_info_t{}));
};

#endif
