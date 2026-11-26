#!/usr/bin/env python3
"""Launch the installed MuJoCo example scene, overriding only its control mode.

Matches the working V3 side-by-side wrapper's runpy strategy, without namespaces:
only ONE simulator is launched at a time on /joint_state and /joint_commands.
"""
import os
import runpy
from pathlib import Path
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    mode = os.environ.get('RPC2_COMPARISON_MODE', '').strip().upper()
    if mode not in ('POSITION', 'VELOCITY'):
        raise RuntimeError('RPC2_COMPARISON_MODE must be POSITION or VELOCITY')

    original = Path(get_package_share_directory('mujoco_ros2')) / 'launch' / 'example.py'
    if not original.is_file():
        raise FileNotFoundError(f'Missing installed MuJoCo example.py: {original}')

    namespace = runpy.run_path(str(original))
    factory = namespace.get('generate_launch_description')
    original_node = namespace.get('Node')
    if not callable(factory) or not callable(original_node):
        raise RuntimeError('Unexpected structure in installed MuJoCo example.py')
    found = []

    def override_node(*args, **kwargs):
        if kwargs.get('package') == 'mujoco_ros2' and kwargs.get('executable') == 'mujoco_node':
            parameters = []
            for item in kwargs.get('parameters', []):
                if isinstance(item, dict):
                    params = dict(item)
                    params.pop('control_mode', None)
                    if params:
                        parameters.append(params)
                else:
                    parameters.append(item)
            parameters.append({'control_mode': mode,
                               'joint_state_topic_name': 'joint_state',
                               'joint_command_topic_name': 'joint_commands'})
            kwargs['parameters'] = parameters
            # Preserve the original scene and rendering properties.
            found.append(True)
        return original_node(*args, **kwargs)

    factory.__globals__['Node'] = override_node
    description = factory()
    if not found:
        raise RuntimeError('mujoco_ros2/mujoco_node not found in original example.py')
    print(f'[RPC2 comparison] Starting original MuJoCo scene in {mode} mode', flush=True)
    return description
