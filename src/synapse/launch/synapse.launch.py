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
        # print("🚀🚀 Synapse Config 🚀🚀")
        config = yaml.safe_load(f)
        # pprint.pprint(config)
    try:
        muscle_option = config['synapse_bt_node']['ros__parameters']['muscle_option']

    except KeyError:
        # print("❌❌ Error: 'muscle_option' not found in config")
        muscle_option = "DUMMY"  

    ld = LaunchDescription([
        SetEnvironmentVariable('RCUTILS_CONSOLE_OUTPUT_FORMAT', '[{severity}]: {message}'),
        Node(
            package='synapse',
            executable='synapse_bt_node.py',
            # name='synapse_bt_node',
            output='screen',
            emulate_tty=True,
            parameters=[ synapse_config_path ],
            # prefix=["gnome-terminal -- bash -c"],
        )
    ])

    match muscle_option:
        case "DUMMY":
            # print("🚀🚀 Launching with Dummy muscle node")
            ld.add_action(
                Node(
                    package='synapse',
                    executable='dummy_muscle_node.py',
                    # name='dummy_muscle_node',
                    output='screen',
                    emulate_tty=True,
                    prefix=["gnome-terminal -- bash -c"],
                    parameters=[ synapse_config_path ],
                )
            )
        case "ISAAC":
            # print("🚀🚀 Launching with Isaac Sim node")
            ld.add_action(
                Node(
                    package='synapse',
                    executable='isaac_node.py',
                    # name='isaac_node',
                    output='screen',
                    emulate_tty=True,
                    prefix=["gnome-terminal -- bash -c"],
                    # prefix=['gnome-terminal', '--', 'bash', '-c', '"$@"; echo "Process exited with code $?"; read -p "Press Enter to close..."', 'bash'],
                    parameters=[ synapse_config_path ],
                )
            )
        case "REAL_ROBOT":
            # print("🚀🚀 Launching with Real Robot node")
            ld.add_action(
                Node(
                    package='synapse',
                    executable='real_robot_node.py',
                    # name='real_robot_node',
                    output='screen',
                    emulate_tty=True,
                    prefix=["gnome-terminal -- bash -c"],
                    parameters=[ synapse_config_path ],
                )
            )

    return ld






"""


import os
import yaml
import pprint
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import SetEnvironmentVariable, ExecuteProcess

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

    # Extract the embodiment/policy option for the bash script (Defaulting to OXE)
    try:
        policy_option = config['synapse_bt_node']['ros__parameters']['policy_option']
    except KeyError:
        policy_option = "OXE"

    ld = LaunchDescription([
        SetEnvironmentVariable('RCUTILS_CONSOLE_OUTPUT_FORMAT', '[{severity}]: {message}'),
        
        # 1. Behavior Tree Node
        Node(
            package='synapse',
            executable='synapse_bt_node.py',
            name='synapse_bt_node',
            output='screen',
            emulate_tty=True,
            parameters=[ synapse_config_path ],
        ),
        
        # 2. GR00T Policy Server Shell Script
        ExecuteProcess(
            cmd=['./start_policyserver.sh', policy_option],
            cwd=os.path.expanduser('~/vla/Isaac-GR00T/'),
            output='screen',
            emulate_tty=True,
            # Uncomment the line below if you want the server to open in its own separate terminal
            # prefix=["gnome-terminal -- bash -c"], 
        )
    ])

    match muscle_option:
        case "DUMMY":
            ld.add_action(
                Node(
                    package='synapse',
                    executable='dummy_muscle_node.py',
                    name='dummy_muscle_node',
                    output='screen',
                    emulate_tty=True,
                    prefix=["gnome-terminal -- bash -c"],
                    parameters=[ synapse_config_path ],
                )
            )
        case "ISAAC":
            ld.add_action(
                Node(
                    package='synapse',
                    executable='isaac_node.py',
                    name='isaac_node',
                    output='screen',
                    emulate_tty=True,
                    prefix=["gnome-terminal -- bash -c"],
                    parameters=[ synapse_config_path ],
                )
            )
        case "REAL_ROBOT":
            ld.add_action(
                Node(
                    package='synapse',
                    executable='real_robot_node.py',
                    name='real_robot_node',
                    output='screen',
                    emulate_tty=True,
                    prefix=["gnome-terminal -- bash -c"],
                    parameters=[ synapse_config_path ],
                )
            )

    return ld
    """