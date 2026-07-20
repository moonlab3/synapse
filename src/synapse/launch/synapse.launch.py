import os
import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription, LaunchContext
from launch_ros.actions import Node
from launch.actions import SetEnvironmentVariable, DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration

def setup_launch(context: LaunchContext, *args, **kwargs):
    # 1. Resolve the config file name from the command line argument
    config_filename = LaunchConfiguration('config_file').perform(context)

    synapse_config_path = os.path.join(
        get_package_share_directory('synapse'),
        'config',
        config_filename
    )

    # 2. Read the YAML to determine the muscle option
    with open(synapse_config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    try:
        muscle_option = config['synapse_bt_node']['ros__parameters']['muscle_option']
    except KeyError:
        muscle_option = "DUMMY"  

    # 3. Base Nodes list
    nodes = [
        SetEnvironmentVariable('RCUTILS_CONSOLE_OUTPUT_FORMAT', '[{severity}]: {message}'),
        Node(
            package='synapse',
            executable='synapse_bt_node.py',
            output='screen',
            prefix='gnome-terminal --wait --',
            # prefix="gnome-terminal --wait -- bash -c '\"$@\"; echo \"\"; echo \"Node exited or crashed. Press Enter to close...\"; read' bash ",
            parameters=[synapse_config_path],
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
                    parameters=[synapse_config_path],
                )
            )
        case "ISAAC":
            nodes.append(
                Node(
                    package='synapse',
                    executable='isaac_node.py',
                    output='screen',
                    emulate_tty=True,
                    parameters=[synapse_config_path],
                )
            )
        case "REAL_ROBOT":
            nodes.append(
                Node(
                    package='synapse',
                    executable='real_robot_node.py',
                    output='screen',
                    emulate_tty=True,
                    parameters=[synapse_config_path],
                )
            )
            
    return nodes

def generate_launch_description():
    return LaunchDescription([
        # Declare the command line argument
        DeclareLaunchArgument(
            'config_file',
            default_value='default.yaml',
            description='Name of the YAML config file to load from the config directory'
        ),
        # Use OpaqueFunction to evaluate the argument before building the graph
        OpaqueFunction(function=setup_launch)
    ])
