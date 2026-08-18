#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Header, String
import numpy as np
import copy
from synapse.utils.embodiment_parser import EmbodimentParser
import functools
import subprocess
import os

class RvizNode(Node):
    def __init__(self, node_name="rviz_node", parameter_overrides=None):
        super().__init__(
            node_name, 
            parameter_overrides=parameter_overrides,
            allow_undeclared_parameters=True,
            automatically_declare_parameters_from_overrides=True
            )
        embodiment_name = self.get_parameter('embodiment_name').value
        freq = self.get_parameter('publish_rate_hz').value

        self.parser = EmbodimentParser(embodiment_name)
        self.robots_cfg = self.parser.get_robots()

        self.subprocesses = []

        self.current_joints = {}
        self.target_joints = {}
        self.joint_names = {}

        self.action_subs = {}
        self.state_pubs = {}

        self._launch_rviz_environment()
        self._setup_ros_interfaces()
        self.sub_synapse_command = self.create_subscription(String, '/synapse/command', self._synapse_command_callback, 10)

        self.timer = self.create_timer(1.0/freq, self.spin_and_step)

        self.interpolation_alpha = 0.5
        self.get_logger().info(f"RViz Muscle Node initialized for embodiment: {embodiment_name}")

    def _setup_ros_interfaces(self):

        for name, cfg in self.robots_cfg.items():
            self.joint_names[name] = cfg.get('joint_names', [])
            dof = len(self.joint_names[name])

            self.current_joints[name] = np.zeros(dof, dtype=np.float32)
            self.target_joints[name] = np.zeros(dof, dtype=np.float32)

            target = cfg.get('target', f'/synapse/target/{name}')
            joint_states = cfg.get('states', f'/synapse/joint_states/{name}')
            self.action_subs[name] = self.create_subscription(
                JointState, target,
                functools.partial(self._action_callback, robot_name=name), 10
            )
            self.state_pubs[name] = self.create_publisher(
                JointState, joint_states, 10
            )

    def _launch_rviz_environment(self):
        self.get_logger().info("Spawning RViz and Robot State Publishers...")
        self.subprocesses.append(subprocess.Popen(['rviz2']))

        for name, cfg in self.robots_cfg.items():
            urdf_path = cfg.get('urdf_path')
            states_topic = cfg.get('states', f'/synapse/joint_states/{name}')

            if urdf_path and os.path.exists(urdf_path):
                cmd = [
                    'ros2', 'run', 'robot_state_publisher', 'robot_state_publisher', 
                    urdf_path, '--ros-args',
                    '-r', f'__node:={name}_state_publisher',
                    '-r', f'__ns:=/{name}',
                    '-r', f'joint_states:={states_topic}'
                ]
                self.subprocesses.append(subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
            else:
                self.get_logger().warning(f"URDF not found {name}")

    def _synapse_command_callback(self, msg):
        command = msg.data
        match command:
            case "START":
                self.get_logger().info("Received START command. Resuming dummy simulation.")
                self.is_playing = True
            case "PAUSE":
                self.get_logger().info("Received PAUSE command. Pausing dummy simulation.")
                self.is_playing = False
            case "QUIT":
                self.get_logger().info("Received QUIT command. Shutting down Dummy Node.")
                raise KeyboardInterrupt
            case "RESET":
                self.get_logger().info("Received RESET command. Zeroing all dummy joints.")
                # Hard reset all simulated robots to zero
                for name in self.robots_cfg.keys():
                    if len(self.current_joints[name]) > 0:
                        self.current_joints[name] = np.zeros_like(self.current_joints[name])
                        self.target_joints[name] = np.zeros_like(self.target_joints[name])
            case _:
                pass

    def _action_callback(self, msg: JointState, robot_name: str):
        if not msg.position:
            return
        expected_dofs = len(self.target_joints[robot_name])
        raw_q = np.array(msg.position, dtype=np.float32)
        if raw_q.shape[0] > expected_dofs:
            q_target = raw_q[:expected_dofs]
            self.get_logger().info(f"Different DOF {expected_dofs} vs {raw_q.shape[0]}")
        elif raw_q.shape[0] < expected_dofs:
            q_target = np.pad(raw_q, (0, expected_dofs - raw_q.shape[0]))
            self.get_logger().info(f"Different DOF {expected_dofs} vs {raw_q.shape[0]}")
        else:
            q_target = raw_q
        self.target_joints[robot_name] = q_target

        if msg.name and len(msg.name) == expected_dofs:
            self.joint_names[robot_name] = msg.name

        if len(self.current_joints[robot_name]) != len(q_target):
            self.current_joints[robot_name] = q_target.copy()

    def spin_and_step(self):

        for name in self.robots_cfg.keys():
            self.current_joints[name] += (self.target_joints[name] - self.current_joints[name]) * self.interpolation_alpha

            msg = JointState()
            msg.header = Header()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.name = self.joint_names[name]
            msg.position = self.current_joints[name].tolist()

            self.state_pubs[name].publish(msg)

    def destroy_node(self):
        self.get_logger().info("Shutting down RViz and TF publishers...")
        for p in self.subprocesses:
            p.terminate()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)

    node = RvizNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()



