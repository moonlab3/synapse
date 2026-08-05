from abc import ABC, abstractmethod
from rclpy.node import Node

class BaseBrainAdapter(ABC, Node):
    def __init__(self, terminal, node_name=None, parameter_overrides=None):
        super().__init__(
            node_name,
            parameter_overrides=parameter_overrides,
            allow_undeclared_parameters=True,
            automatically_declare_parameters_from_overrides=True
        )
        self.embodiment_name = self.get_parameter('embodiment_name').value

        self.robot_description = self.get_parameter('robot_description').value
        self.terminal = terminal
        terminal.wait_debug("BaseBrain Initiation Done")

    def infer(self, obs_history: list, get_default: False) -> list:
        if not obs_history:
            return []

        # 1. Convert observations into model-specific formats (and apply FK if needed)
        formatted_obs = self._format_for_policy(obs_history, get_default)
        # self.terminal.wait_debug("after format policy")
        
        # 2. Communicate with AI Policy (ZMQ for RL/VLA, bypass for Manual)
        raw_action = self._communicate_with_policy(formatted_obs)
        
        # 3. Format action for muscle wrapper (and apply IK if needed)
        action_chunk = self._format_for_muscle(raw_action)

        if action_chunk and isinstance(action_chunk[0], dict):
            debug_strings = []
            # Iterate through all robots in the timestep dictionary
            for robot_name, msg in action_chunk[0].items():
                if hasattr(msg, 'position') and msg.position:
                    pos_str = ", ".join(f"{v:.3f}" for v in msg.position)
                    debug_strings.append(f"{robot_name}: [{pos_str}]")
            _chunk = " | ".join(debug_strings) 
        
        return action_chunk

    @abstractmethod
    def _format_for_policy(self, obs_history: list):
        """Extracts history, applies FK if needed, formats for policy/ZMQ."""
        pass

    @abstractmethod
    def _communicate_with_policy(self, formatted_obs):
        """Sends data over ZMQ (or handles locally) and returns model output."""
        pass

    @abstractmethod
    def _format_for_muscle(self, raw_action) -> list:
        """Applies IK if needed, converts to ROS2 msgs, returns as an action chunk list."""
        pass