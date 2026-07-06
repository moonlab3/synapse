from ..base_brain_adapter import BaseBrainAdapter
import numpy as np
import os
import sys
gr00t_path = os.path.abspath("/home/rog-sf/vla/Isaac-GR00T")
if gr00t_path not in sys.path:
    sys.path.append(gr00t_path)
from gr00t.policy.server_client import PolicyClient

class VLAAdapter(BaseBrainAdapter):
    def __init__(self, config, checkpoint_path):
        super().__init__(config, 'VLA_Adapter')
        self.task_instruction = "pick up the red cube"
        # self.model = load_model(checkpoint_path)
        self.policy_client = PolicyClient("0.0.0.0", 8088)
        self.connected = self.policy_client.ping()
        if not self.connected:
            self.get_logger().info("🧠 Failed to connect to GR00T policy server. Please ensure it's running and accessible.")
        else:
            self.get_logger().info("🧠 Successfully connected to GR00T policy server.")

    def infer(self, obs_dict) -> list:
        if not self.connected:
            self.get_logger().info("🧠 Not connected to GR00T server. Cannot perform inference.")
            return None

        if not obs_dict:
            return None

        # Extract sequences
        images = [frame["image"] for frame in obs_dict]
        states = [frame["state"] for frame in obs_dict]

        # Collate: Stack (T) and add Batch dimension (B=1)
        video_tensor = np.expand_dims(np.stack(images, axis=0), axis=0) # (1, T, H, W, 3)
        state_tensor = np.expand_dims(np.stack(states, axis=0), axis=0) # (1, T, D)
        
        gr00t_input = {
            "video": {"camera_name": video_tensor},
            "state": {"state_name": state_tensor},
            "language": {"task": [[self.task_instruction]]}
        }
        raw_action = self.policy_client.get_action(gr00t_input)
        return raw_action
