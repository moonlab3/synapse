#!/usr/bin/env python3
import rclpy
import numpy as np
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import JointState

class DummyMuscleNode(Node):
    def __init__(self):
        super().__init__('dummy_muscle_node')
        
        # Increased default rate to 100Hz to match the BT Node tick frequency
        self.declare_parameter('publish_rate_hz', 100)
        freq = self.get_parameter('publish_rate_hz').value

        self.pub_joint_states = self.create_publisher(JointState, '/synapse/joint_states', 10)
        
        # Subscribers: Listening for actions and commands from Synapse
        self.sub_brain_output = self.create_subscription(JointState, '/synapse/brain_output', self.brain_output_callback, 10)
        self.sub_synapse_command = self.create_subscription(String, '/synapse/command', self.synapse_command_callback, 10)

        # Timer loop for physical simulation/publishing
        self.timer = self.create_timer(1.0 / freq, self.publish_state)
        
        # Internal State: 7-DOF arm 
        self.joint_names = ['joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6', 'joint_7', 'joint_8']  # Assuming 9 joints for the dummy arm
        self.current_joints = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.target = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        self.get_logger().info(f"💪 Dummy Muscle Node Ready. Publishing at {freq}Hz.")

    def synapse_command_callback(self, msg):
        if msg.data == "QUIT":
            self.get_logger().info("💪 Received QUIT command. Shutting down Dummy Muscle Node.")
            raise KeyboardInterrupt

    def brain_output_callback(self, msg: JointState):
        # Architecture 3.3: Translate actions into robot-specific commands
        positions = ", ".join(f"{p:.3f}" for p in msg.position)
        self.get_logger().info(f"💪 Received {positions}")
        if msg.position:
            # Handle potential length mismatches cleanly during development
            length = min(len(msg.position), len(self.target))
            self.target[:length] = np.array(msg.position)[:length]

    def publish_state(self):
        # Simulate physical movement (P-controller towards target)
        error = self.target - self.current_joints
        
        # Proportional step simulating motor movement over the dt window
        # Kp = 0.1 for smooth dummy interpolation
        step = 0.1 * error 
        self.current_joints += step

        # Construct and publish observation back to BT Node
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = self.joint_names
        msg.position = self.current_joints.tolist()
        
        self.pub_joint_states.publish(msg)

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