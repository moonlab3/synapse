from rclpy.node import Node
from std_msgs.msg import String
from abc import ABC, abstractmethod

class BaseMuscle(ABC, Node):
    def __init__(self, node_name=None, parameter_overrides=None):
        super().__init__(
            node_name,
            parameter_overrides=parameter_overrides,
            allow_undeclared_parameters=True,
            automatically_declare_parameters_from_overrides=True
        )
        # Every muscle must have a system command publisher (e-stop, quit)
        self.synapse_command_sub = self.create_subscription(String, '/synapse/command', self.synapse_command_callback, 10)

        self.get_logger().info(f"{node_name} initialized. Listening for commands on /synapse/command.")

    def synapse_command_callback(self, msg: String):
        command = msg.data
        match command:
            case "QUIT":
                self.get_logger().info("Received QUIT command. Shutting down muscle node.")
                self.hardware_shutdown()
                raise KeyboardInterrupt
            case "ESTOP":
                self.get_logger().error("🚨 EMERGENCY STOP TRIGGERED 🚨")
                self.hardware_estop()
            case _:
                self.get_logger().warning(f"Unknown command received: {command}")
    
    @abstractmethod
    def publish_trajectory(self, action_chunk):
        """Translates the AI's numpy array into the specific ROS message for this hardware."""
        pass

    @abstractmethod
    def hardware_estop(self):
        """Standardized E-Stop sequence."""
        pass

    @abstractmethod
    def hardware_shutdown(self):
        """Standardized shutdown sequence."""
        pass