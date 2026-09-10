#!/usr/bin/env python3
import rclpy
import numpy as np
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import JointState
from synapse.utils.embodiment_parser import EmbodimentParser
import functools

class DummyMuscleNode(Node):
    def __init__(self, node_name="dummy_muscle_node", parameter_overrides=None):
        super().__init__(
            node_name,
            parameter_overrides=parameter_overrides,
            allow_undeclared_parameters=True,
            automatically_declare_parameters_from_overrides=True
        )

        freq = self.get_parameter('publish_rate_hz').value
        embodiment_name = self.get_parameter('embodiment_name').value
        parser = EmbodimentParser(embodiment_name)
        self.robots_cfg = parser.get_robots()

        articulation_groups = parser.get_articulations()
        self.groups = {}
        grouped_component_names = set()

        for group_name, group_cfg in articulation_groups.items():
            self.groups[group_name] = {
                'states': parser.resolve_states(group_cfg, group_name),
                'components': group_cfg.get('components', {}),
            }
            grouped_component_names.update(self.groups[group_name]['components'].keys())

        for name, cfg in self.robots_cfg.items():
            if name in grouped_component_names:
                continue
            self.groups[name] = {
                'states': parser.resolve_states(cfg, name),
                'components': {name: cfg},
            }

        self.current_joints = {}
        self.target_joints = {}
        self.joint_names = {}
        for name, cfg in self.robots_cfg.items():
            self.joint_names[name] = cfg.get('joint_names', [])
            dof = len(self.joint_names[name])
            self.current_joints[name] = np.zeros(dof, dtype=np.float32)
            self.target_joints[name] = np.zeros(dof, dtype=np.float32)

        # ---- One subscriber per COMPONENT (was per group). No more
        # joint-name matching needed in target_callback -- the topic itself
        # tells us which component this is. ----
        self.component_targets = parser.get_component_targets('DUMMY')
        self.target_subscribers = {}
        for component_name, resolved in self.component_targets.items():
            if resolved is None:
                continue
            self.target_subscribers[component_name] = self.create_subscription(
                JointState, resolved.topic,
                functools.partial(self.target_callback, component_name=component_name), 10)

        # ---- One publisher per GROUP (states stay combined, unchanged) ----
        self.state_publishers = {}
        for group_name, group in self.groups.items():
            if group['states'] is None:
                continue
            self.state_publishers[group_name] = self.create_publisher(JointState, group['states'], 10)

        self.sub_synapse_command = self.create_subscription(String, '/synapse/command', self.synapse_command_callback, 10)
        self.timer = self.create_timer(1.0 / freq, self.spin_and_step)

        self.get_logger().info(f"💪 Dummy Muscle Node Ready. Publishing at {freq}Hz.")
        self.is_playing = False

    def synapse_command_callback(self, msg):
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
                for name in self.robots_cfg.keys():
                    if len(self.current_joints[name]) > 0:
                        self.current_joints[name] = np.zeros_like(self.current_joints[name])
                        self.target_joints[name] = np.zeros_like(self.target_joints[name])
            case _:
                pass

    def spin_and_step(self):
        for group_name, group in self.groups.items():
            if group_name not in self.state_publishers:
                continue

            combined = JointState()
            combined.header.stamp = self.get_clock().now().to_msg()
            debug_parts = []

            for component_name in group['components']:
                if len(self.current_joints[component_name]) == 0:
                    continue
                self.current_joints[component_name] += 0.15 * (
                    self.target_joints[component_name] - self.current_joints[component_name]
                )
                combined.name.extend(self.joint_names.get(component_name, []))
                combined.position.extend(self.current_joints[component_name].tolist())
                output = ",".join(f"{x:.2f}" for x in self.current_joints[component_name])
                debug_parts.append(f"{component_name}[{output}]")

            if not combined.name:
                continue

            if debug_parts:
                self.get_logger().info("ROBOT " + " | ".join(debug_parts))
            self.state_publishers[group_name].publish(combined)

    def target_callback(self, msg: JointState, component_name: str):
        """One subscription = one component. No demux needed anymore."""
        if not msg.position:
            return
        incoming_target = np.array(msg.position, dtype=np.float32)
        self.target_joints[component_name] = incoming_target
        self.joint_names[component_name] = msg.name
        if len(self.current_joints[component_name]) != len(incoming_target):
            self.current_joints[component_name] = incoming_target.copy()

def main(args=None):
    rclpy.init(args=args)
    node = DummyMuscleNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()