#!/usr/bin/env python3
import rclpy
import numpy as np
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import JointState
from rclpy.duration import Duration
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
        
        # Increased default rate to 100Hz to match the BT Node tick frequency
        freq = self.get_parameter('publish_rate_hz').value
        embodiment_name = self.get_parameter('embodiment_name').value
        parser = EmbodimentParser(embodiment_name)
        self.robots_cfg = parser.get_robots()          # flattened, per-component

        articulation_groups = parser.get_articulations()
        self.groups = {}
        grouped_component_names = set()

        for group_name, group_cfg in articulation_groups.items():
            self.groups[group_name] = {
                'states': group_cfg.get('states', f'/synapse/joint_states/{group_name}'),
                'target': group_cfg.get('target', f'/synapse/target/{group_name}'),
                'components': group_cfg.get('components', {}),
            }
            grouped_component_names.update(self.groups[group_name]['components'].keys())

        for name, cfg in self.robots_cfg.items():
            if name in grouped_component_names:
                continue
            self.groups[name] = {
                'states': cfg.get('states', f'/synapse/joint_states/{name}'),
                'target': cfg.get('target', f'/synapse/target/{name}'),
                'components': {name: cfg},
            }

        self.target_subscribers = {}
        self.state_publishers = {}

        self.current_joints = {}
        self.target_joints = {}
        self.joint_names = {}

        for name, cfg in self.robots_cfg.items():
            self.joint_names[name] = cfg.get('joint_names', [])
            dof = len(self.joint_names[name])
            self.current_joints[name] = np.zeros(dof, dtype=np.float32)
            self.target_joints[name] = np.zeros(dof, dtype=np.float32)

        # ---- One subscriber and one publisher per GROUP, not per component.
        for group_name, group in self.groups.items():
            self.target_subscribers[group_name] = self.create_subscription(
                JointState, group['target'],
                functools.partial(self.target_callback, group_name=group_name), 10)
            self.state_publishers[group_name] = self.create_publisher(
                JointState, group['states'], 10)

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
                # Hard reset all simulated robots to zero
                for name in self.robots_cfg.keys():
                    if len(self.current_joints[name]) > 0:
                        self.current_joints[name] = np.zeros_like(self.current_joints[name])
                        self.target_joints[name] = np.zeros_like(self.target_joints[name])
            case _:
                pass

    def spin_and_step(self):
        """100Hz loop to simulate physics and publish states, one combined
        message per physical articulation group (mirrors isaac_node.py)."""
        for group_name, group in self.groups.items():
            combined = JointState()
            combined.header.stamp = self.get_clock().now().to_msg()
            debug_parts = []

            for component_name in group['components']:
                if len(self.current_joints[component_name]) == 0:
                    continue
                # ⚡ Dummy "Physics": Simple P-Controller interpolation
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
                self.get_logger().info(f"ROBOT " + " | ".join(debug_parts))
            self.state_publishers[group_name].publish(combined)

    def target_callback(self, msg: JointState, group_name: str):
        if not msg.position:
            return

        components = self.groups[group_name]['components']

        if len(components) == 1:
            component_name = next(iter(components))
            incoming_target = np.array(msg.position, dtype=np.float32)
            self.target_joints[component_name] = incoming_target
            self.joint_names[component_name] = msg.name
            if len(self.current_joints[component_name]) != len(incoming_target):
                self.current_joints[component_name] = incoming_target.copy()
            return

        per_component_names = {c: [] for c in components}
        per_component_positions = {c: [] for c in components}

        for joint_name, pos in zip(msg.name, msg.position):
            component_name = next(
                (c for c, cfg in components.items()
                 if joint_name in cfg.get('joint_names', [])),
                None
            )
            if component_name is None:
                continue
            per_component_names[component_name].append(joint_name)
            per_component_positions[component_name].append(pos)

        for component_name in components:
            if not per_component_names[component_name]:
                continue  # this message carried no data for this component
            incoming_target = np.array(per_component_positions[component_name], dtype=np.float32)
            self.target_joints[component_name] = incoming_target
            self.joint_names[component_name] = per_component_names[component_name]
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