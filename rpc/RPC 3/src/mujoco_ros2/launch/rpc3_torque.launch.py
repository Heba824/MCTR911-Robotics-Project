"""Launch an isolated Panda bridge with physical arm torque control."""

from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    scene = Path.home() / 'mujoco_menagerie/franka_emika_panda/scene.xml'
    return LaunchDescription([
        DeclareLaunchArgument('scene', default_value=str(scene)),
        DeclareLaunchArgument('visualisation_frequency', default_value='15'),
        DeclareLaunchArgument('dds_publication_mode', default_value='ASYNCHRONOUS',
                              choices=['ASYNCHRONOUS', 'SYNCHRONOUS']),
        DeclareLaunchArgument('headless', default_value='true'),
        DeclareLaunchArgument('torque_trace_csv', default_value=''),
        DeclareLaunchArgument('namespace', default_value='rpc3'),
        Node(
            package='mujoco_ros2',
            executable='mujoco_node',
            namespace=LaunchConfiguration('namespace'),
            arguments=[LaunchConfiguration('scene')],
            parameters=[{
                'control_mode': 'TORQUE',
                'headless': ParameterValue(LaunchConfiguration('headless'), value_type=bool),
                'simulation_frequency': 1000,
                'visualisation_frequency': ParameterValue(
                    LaunchConfiguration('visualisation_frequency'), value_type=int),

                'torque_trace_csv': ParameterValue(
                    LaunchConfiguration('torque_trace_csv'), value_type=str),

            }],
            additional_env={
                'RMW_FASTRTPS_PUBLICATION_MODE': LaunchConfiguration('dds_publication_mode'),
            },
            output='screen',
        ),
    ])
