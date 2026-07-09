import os
import yaml
import pprint
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import SetEnvironmentVariable
# from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
# from launch.launch_description_sources import PythonLaunchDescriptionSource

def generate_launch_description():

    synapse_config_path = os.path.join(
        get_package_share_directory('synapse'),
        'config',
        'synapse_config.yaml'
    )

    with open(synapse_config_path, 'r') as f:
        config = yaml.safe_load(f)
    try:
        muscle_option = config['synapse_bt_node']['ros__parameters']['muscle_option']

    except KeyError:
        muscle_option = "DUMMY"  

    ld = LaunchDescription([
        SetEnvironmentVariable('RCUTILS_CONSOLE_OUTPUT_FORMAT', '[{severity}]: {message}'),
        Node(
            package='synapse',
            executable='synapse_bt_node.py',
            # name='synapse_bt_node',
            output='screen',
            # emulate_tty=True,
            prefix='gnome-terminal --wait --',
            parameters=[ synapse_config_path ],
        )
    ])

    match muscle_option:
        case "DUMMY":
            ld.add_action(
                Node(
                    package='synapse',
                    executable='dummy_muscle_node.py',
                    output='screen',
                    emulate_tty=True,
                    # prefix=["gnome-terminal -- bash -c"],
                    parameters=[ synapse_config_path ],
                )
            )
        case "ISAAC":
            ld.add_action(
                Node(
                    package='synapse',
                    executable='isaac_node.py',
                    output='screen',
                    emulate_tty=True,
                    # prefix=["gnome-terminal -- bash -c"],
                    parameters=[ synapse_config_path ],
                )
            )
        case "REAL_ROBOT":
            ld.add_action(
                Node(
                    package='synapse',
                    executable='real_robot_node.py',
                    output='screen',
                    emulate_tty=True,
                    # prefix=["gnome-terminal -- bash -c"],
                    parameters=[ synapse_config_path ],
                )
            )
        case "RVIZ":
            pass

    return ld

