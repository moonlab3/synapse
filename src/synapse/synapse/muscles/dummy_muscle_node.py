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
        self.robots_cfg = parser.get_robots()

        self.target_subscribers = {}
        self.robot_publishers = {}

        self.current_joints = {}
        self.target_joints = {}
        self.joint_names = {}

        for name, cfg in self.robots_cfg.items():
            target = cfg.get('target', f'/synapse/target/{name}')
            joint_states = cfg.get('states', f'/synapse/joint_states/{name}')

            self.target_subscribers[name] = self.create_subscription(
                JointState, target,
                functools.partial(self.target_callback, robot_name=name), 10)

            self.robot_publishers[name] = self.create_publisher(JointState, joint_states, 10)
            self.joint_names[name] = cfg.get('joint_names', [])
            dof = len(self.joint_names[name])

            self.current_joints[name] = np.zeros(dof, dtype=np.float32)
            self.target_joints[name] = np.zeros(dof, dtype=np.float32)

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
        """100Hz loop to simulate physics and publish states."""
        for name in self.robots_cfg.keys():
            if len(self.current_joints[name]) == 0:
                continue
            # ⚡ Dummy "Physics": Simple P-Controller interpolation
            # Moves the current state 15% closer to the target every frame (visual smoothing)
            self.current_joints[name] += 0.15 * (self.target_joints[name] - self.current_joints[name])
            
            msg = JointState()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.name = self.joint_names.get(name, [])
            msg.position = self.current_joints[name].tolist()
            
            output = ",".join(f"{x:.2f}" for x in msg.position)
            self.get_logger().info(f"ROBOT[{name}] joint_states[{output}]")
            self.robot_publishers[name].publish(msg)

    def target_callback(self, msg: JointState, robot_name: str):
        """Receives target joints from synapse_main_node and updates specific robot."""
        if msg.position:
            incoming_target = np.array(msg.position, dtype=np.float32)
            self.target_joints[robot_name] = incoming_target
            self.joint_names[robot_name] = msg.name
            
            # If this is the very first message, snap the current joints to the target 
            # to initialize the array shape and prevent a wild jump from zero.
            if len(self.current_joints[robot_name]) != len(incoming_target):
                self.current_joints[robot_name] = incoming_target.copy()

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