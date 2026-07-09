from abc import ABC, abstractmethod
from rclpy.node import Node

class BaseBrainAdapter(ABC, Node):
    def __init__(self, node_name=None):
        super().__init__(node_name)
        self.declare_parameter('muscle_embodiment', "LIBERO_PANDA")  # Placeholder for future muscle embodiment options
        self.muscle_embodiment = self.get_parameter('muscle_embodiment').value

    def infer(self, obs_history: list) -> list:
        if not obs_history:
            return []

        # 1. Convert observations into model-specific formats (and apply FK if needed)
        formatted_obs = self._format_for_policy(obs_history)
        
        # 2. Communicate with AI Policy (ZMQ for RL/VLA, bypass for Manual)
        raw_action = self._communicate_with_policy(formatted_obs)
        
        # 3. Format action for muscle wrapper (and apply IK if needed)
        action_chunk = self._format_for_muscle(raw_action)
        if action_chunk and action_chunk[0].position:
            _chunk = ", ".join(f"{v:.3f}" for v in action_chunk[0].position)
        
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