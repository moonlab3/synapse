from ..base_brain_adapter import BaseBrainAdapter
from ..base_brain_adapter import InferenceOption

class DummyBrainAdapter(BaseBrainAdapter):
    def __init__(self, terminal, name="dummy_brain_adapter", params=None, ros_node=None):
        super().__init__(terminal, name, params, ros_node)

        self.terminal.log(
            f"🧠 [{self.name}] initialized. Yielding target control to external ROS 2 applications."
        )

    def _format_for_policy(self, obs_history: list, inference_option: InferenceOption) -> dict:
        # No formatting needed, we aren't querying an AI policy
        return {}

    def _communicate_with_policy(self, formatted_obs: dict) -> dict:
        # No AI policy to communicate with
        return {}

    def _format_for_muscle(self, raw_action: dict) -> list:
        # Returning an empty list starves the BT Node's action buffer.
        # This prevents the BT Node from publishing, keeping the target topics clear.
        return []