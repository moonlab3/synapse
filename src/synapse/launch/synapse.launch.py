import os
import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription, LaunchContext
from launch_ros.actions import Node
from launch.actions import SetEnvironmentVariable, DeclareLaunchArgument, OpaqueFunction, LogInfo
from launch.substitutions import LaunchConfiguration

def setup_launch(context: LaunchContext, *args, **kwargs):
    config_nickname = LaunchConfiguration('config').perform(context)
    debug_mode = LaunchConfiguration('debug').perform(context).lower()
    ui_mode = LaunchConfiguration('ui').perform(context).lower()

    synapse_config_path = os.path.join(
        get_package_share_directory('synapse'), 
        'configs', 
        'config_main.yaml'
    )

    # 2. Read the YAML to determine the muscle option
    with open(synapse_config_path, 'r') as f:
        config = yaml.safe_load(f)

    if config_nickname not in config:
        raise ValueError(f"Nickname '{config_nickname}' not found in config_main.yaml")

    active_config = config[config_nickname]
    muscle_option = active_config.get('muscle_option', 'DUMMY')

    # The TUI reads keys from a terminal of its own. The GUI reads them from
    # its window, so a spawned terminal would only sit there empty.
    main_node_kwargs = {'prefix': 'gnome-terminal --wait --'} if ui_mode == 'tui' else {}

    nodes = [
        SetEnvironmentVariable('RCUTILS_CONSOLE_OUTPUT_FORMAT', '[{severity}]: {message}'),
        Node(
            package='synapse',
            executable='synapse_main_node.py',
            output='screen',
            # ui/debug are process options (argv), not node parameters: the UI
            # is built before the node exists.
            arguments=['--ui', ui_mode, '--debug', debug_mode],
            parameters=[active_config],
            **main_node_kwargs,
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
        case "RVIZ":
            # TODO: DO WHATEVER I NEED TO DO FOR RVIZ LAUNCHING
            nodes.append(
                Node(
                    package='synapse',
                    executable='rviz_node.py',
                    output='screen',
                    emulate_tty=True,
                    parameters=[active_config],
                )
            )
        case "REAL_ROBOT":
            nodes.append(LogInfo(msg="====================================="))
            nodes.append(LogInfo(msg="🚨 PLEASE PREPARE YOUR REAL ROBOT! 🚨"))
            nodes.append(LogInfo(msg="====================================="))
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
            description='false | true (DEBUG logs + debugpy on 127.0.0.1:5678) | wait (also block until attached)'
        ),
        DeclareLaunchArgument(
            'ui',
            default_value='tui',
            description='Front-end: tui (curses, own terminal) | gui (Qt window)'
        ),
        # Use OpaqueFunction to evaluate the argument before building the graph
        OpaqueFunction(function=setup_launch)
    ])
