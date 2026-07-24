import os
import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription, LaunchContext
from launch_ros.actions import Node
from launch.actions import SetEnvironmentVariable, DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration

def setup_launch(context: LaunchContext, *args, **kwargs):
    config_nickname = LaunchConfiguration('config').perform(context)
    debug_arg = LaunchConfiguration('debug')

    synapse_config_path = os.path.join(
        get_package_share_directory('synapse'), 
        'configs', 
        'config_collections.yaml'
    )

    # 2. Read the YAML to determine the muscle option
    with open(synapse_config_path, 'r') as f:
        config = yaml.safe_load(f)

    if config_nickname not in config:
        raise ValueError(f"Nickname '{config_nickname}' not found in config_collections.yaml")

    active_config = config[config_nickname]
    muscle_option = active_config.get('muscle_option', 'DUMMY')

    nodes = [
        SetEnvironmentVariable('RCUTILS_CONSOLE_OUTPUT_FORMAT', '[{severity}]: {message}'),
        Node(
            package='synapse',
            executable='synapse_main_node.py',
            output='screen',
            prefix='gnome-terminal --wait --',
            # prefix="gnome-terminal --wait -- bash -c '\"$@\"; echo \"\"; echo \"Node exited or crashed. Press Enter to close...\"; read' bash ",
            parameters=[active_config, {'debug_mode': debug_arg}],
        )
    ]

    # 4. Conditionally append the Muscle Node
    match muscle_option:
        case "DUMMY":
            nodes.append(
                Node(
                    package='synapse',
                    executable='dummy_muscle_node.py',
                    output='screen',
                    emulate_tty=True,
                    # parameters=[synapse_config_path],
                    parameters=[active_config],
                )
            )
        case "ISAAC":
            nodes.append(
                Node(
                    package='synapse',
                    executable='isaac_node.py',
                    output='screen',
                    emulate_tty=True,
                    parameters=[active_config],
                )
            )
        case "REAL_ROBOT":
            nodes.append(
                Node(
                    package='synapse',
                    executable='real_robot_node.py',
                    output='screen',
                    emulate_tty=True,
                    parameters=[active_config],
                )
            )
            
    return nodes

def generate_launch_description():
    return LaunchDescription([
        # Declare the command line argument
        DeclareLaunchArgument(
            'config',
            default_value='default',
            description='Nickname of the configuration to load'
        ),
        DeclareLaunchArgument(
            'debug',
            default_value='false',
            description='Debug mode'
        ),
        # Use OpaqueFunction to evaluate the argument before building the graph
        OpaqueFunction(function=setup_launch)
    ])
