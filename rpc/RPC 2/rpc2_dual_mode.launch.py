#!/usr/bin/env python3
"""Run the existing MuJoCo example scene with isolated namespaced ROS topics.

The wrapper does NOT modify the original mujoco_ros2/launch/example.py.
For each instance set RPC2_CONTROL_MODE to POSITION or VELOCITY and
RPC2_BRIDGE_NAMESPACE to rpc2_position or rpc2_velocity.
"""

import os
import runpy
from pathlib import Path
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    mode = os.environ.get('RPC2_CONTROL_MODE', '').strip().upper()
    namespace = os.environ.get('RPC2_BRIDGE_NAMESPACE', '').strip('/')
    if mode not in ('POSITION', 'VELOCITY'):
        raise ValueError('RPC2_CONTROL_MODE must be POSITION or VELOCITY')
    if namespace not in ('rpc2_position', 'rpc2_velocity'):
        raise ValueError('RPC2_BRIDGE_NAMESPACE must be rpc2_position or rpc2_velocity')

    original = (Path(get_package_share_directory('mujoco_ros2'))
                / 'launch' / 'example.py')
    if not original.is_file():
        raise FileNotFoundError(f'Missing original MuJoCo launch file: {original}')
    scope = runpy.run_path(str(original))
    factory = scope.get('generate_launch_description')
    node_factory = scope.get('Node')
    if not callable(factory) or not callable(node_factory):
        raise RuntimeError('Original example.py launch structure is not supported')

    patched = []

    def isolated_node(*args, **kwargs):
        if (kwargs.get('package') == 'mujoco_ros2' and
                kwargs.get('executable') == 'mujoco_node'):
            params = []
            for entry in kwargs.get('parameters', []):
                if isinstance(entry, dict):
                    settings = dict(entry)
                    settings.pop('control_mode', None)
                    settings.pop('joint_state_topic_name', None)
                    settings.pop('joint_command_topic_name', None)
                    if settings:
                        params.append(settings)
                else:
                    params.append(entry)
            # Names must be relative so the namespace isolates both simulations.
            params.append({
                'control_mode': mode,
                'joint_state_topic_name': 'joint_state',
                'joint_command_topic_name': 'joint_commands',
            })
            kwargs['parameters'] = params
            kwargs['namespace'] = '/' + namespace
            patched.append(True)
        return node_factory(*args, **kwargs)

    factory.__globals__['Node'] = isolated_node
    description = factory()
    if not patched:
        raise RuntimeError('MuJoCo Node action not found in original example.py')
    print(f'[RPC2] MuJoCo {mode} in /{namespace} '
          f'(topics /{namespace}/joint_state and /{namespace}/joint_commands)',
          flush=True)
    return description
