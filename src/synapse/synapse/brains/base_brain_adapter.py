from abc import ABC, abstractmethod
from dataclasses import dataclass

from synapse.utils.paths import resolve_path

_REQUIRED = object()


@dataclass
class InferenceOption:
    default_command: bool = True
    restart: bool = False
    params: dict = None      # per-Action knobs from the scenario YAML; None outside the BT


class BaseBrainAdapter(ABC):
    """A plain object, not a ROS node. Parameters arrive as a flat dict of the main
    node's parameters ('<name>.<key>', 'paths.<key>', ...); anything that needs ROS
    (clock, subscriptions) goes through `self.node`, the main node, so its callbacks
    run on the executor that is actually spinning."""

    def __init__(self, terminal, name, params: dict, ros_node):
        self.terminal = terminal
        self.name = name
        self.params = params
        self.node = ros_node

        self.embodiment_name = params['embodiment_name']
        self.chunk_dt = 1.0 / params.get('bt_tick_frequency_hz', 50)

    def param(self, key, default=_REQUIRED):
        """This adapter's own setting, i.e. '<name>.<key>' in the profile."""
        full = f"{self.name}.{key}"
        if full in self.params:
            return self.params[full]
        if default is _REQUIRED:
            raise KeyError(f"[{self.name}] missing required parameter '{full}'")
        return default

    def path(self, key) -> str:
        """A shared path from the `paths:` block in config_main.yaml, resolved."""
        full = f"paths.{key}"
        if full not in self.params:
            raise KeyError(f"[{self.name}] missing '{key}' in the paths: block of config_main.yaml")
        return resolve_path(self.params[full])

    def infer(self, obs_history: list, inference_option: InferenceOption = None) -> list:
        if not obs_history:
            return []

        if inference_option is None:
            inference_option = InferenceOption()

        formatted_obs = self._format_for_policy(obs_history, inference_option)
        raw_action = self._communicate_with_policy(formatted_obs)
        action_chunk = self._format_for_muscle(raw_action)

        # if action_chunk and isinstance(action_chunk[0], dict):
        #     debug_strings = []
        #     # Iterate through all robots in the timestep dictionary
        #     for robot_name, msg in action_chunk[0].items():
        #         if hasattr(msg, 'position') and msg.position:
        #             pos_str = ", ".join(f"{v:.3f}" for v in msg.position)
        #             debug_strings.append(f"{robot_name}: [{pos_str}]")
        #     _chunk = " | ".join(debug_strings)

        # self.terminal.log(f"🧠🧠 return {_chunk}")

        return action_chunk

    def reset(self):
        pass
    @abstractmethod
    def _format_for_policy(self, obs_history: list, inference_option: InferenceOption):
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
