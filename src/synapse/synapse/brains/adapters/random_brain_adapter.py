import numpy as np
from sensor_msgs.msg import JointState
from ..base_brain_adapter import BaseBrainAdapter
from ..base_brain_adapter import InferenceOption
from synapse.utils.embodiment_parser import EmbodimentParser


class RandomBrainAdapter(BaseBrainAdapter):
    def __init__(self, terminal, name="dummy_brain_adapter", params=None, ros_node=None):
        super().__init__(terminal, name, params, ros_node)

        parser = EmbodimentParser(self.embodiment_name)
        self.robots_cfg = parser.get_robots()  # flattened, per component

        self.step_size = self.param('step_size', 0.005)       # radians per micro-step
        self.flip_prob = self.param('flip_prob', 0.05)         # chance to reverse direction each step
        self.chunk_length = self.param('chunk_length', 1)     # micro-steps produced per inference call
        self.bound_radius = self.param('bound_radius', 0.3)    # soft +/- range around start pose (radians)

        self.joint_names = {}
        self.direction = {}          # component -> np.array of +1/-1 per joint
        self.current_position = {}   # component -> np.array, running target
        self.origin_position = {}    # component -> np.array, soft-bound center
        self._seeded = set()

        self.terminal.log(
            f"🎲 [{self.name}] initialized. Random-walk joint driver "
            f"(step={self.step_size}, flip_p={self.flip_prob}, "
            f"chunk={self.chunk_length}, bound=±{self.bound_radius})."
        )

    def _seed_component(self, name, cfg, joint_state):
        n = len(cfg.get('joint_names', []))
        if joint_state is not None and joint_state.position and len(joint_state.position) == n:
            start = np.array(joint_state.position, dtype=np.float32)
        else:
            start = np.zeros(n, dtype=np.float32)

        self.joint_names[name] = cfg.get('joint_names', [])
        self.current_position[name] = start.copy()
        self.origin_position[name] = start.copy()
        self.direction[name] = np.random.choice([-1.0, 1.0], size=n).astype(np.float32)
        self._seeded.add(name)

    def _format_for_policy(self, obs_history: list, inference_option: InferenceOption) -> dict:
        latest_obs = obs_history[-1]
        joints_dict = latest_obs.get("joints", {})

        # Seed each component's walk from its first real observed pose,
        # so the soft bound is centered on where the robot actually is.
        for name, cfg in self.robots_cfg.items():
            if name not in self._seeded:
                self._seed_component(name, cfg, joints_dict.get(name))

        return {}  # no external policy input needed

    def _communicate_with_policy(self, formatted_obs: dict) -> dict:
        # No AI policy -- generate the random-walk chunk locally.
        chunk = []

        for _ in range(self.chunk_length):
            step = {}
            for name in self.robots_cfg:
                n = len(self.joint_names.get(name, []))
                if n == 0:
                    continue

                # 10% chance per joint to flip direction this micro-step.
                flips = np.random.random(n) < self.flip_prob
                self.direction[name][flips] *= -1.0

                candidate = self.current_position[name] + self.direction[name] * self.step_size

                # Soft-bound reflection: joints heading out of their band
                # flip direction instead of walking past it.
                lower = self.origin_position[name] - self.bound_radius
                upper = self.origin_position[name] + self.bound_radius
                out_of_bounds = (candidate < lower) | (candidate > upper)
                self.direction[name][out_of_bounds] *= -1.0

                candidate = np.clip(
                    self.current_position[name] + self.direction[name] * self.step_size,
                    lower, upper
                )

                self.current_position[name] = candidate
                step[name] = candidate.copy()

            chunk.append(step)

        return {"chunk": chunk}

    def _format_for_muscle(self, raw_action: dict) -> list:
        chunk = raw_action.get("chunk", [])
        action_chunk = []

        for step in chunk:
            step_dict = {}
            for name, positions in step.items():
                msg = JointState()
                msg.header.stamp = self.node.get_clock().now().to_msg()
                msg.name = self.joint_names[name]
                msg.position = positions.tolist()
                step_dict[name] = msg
            action_chunk.append(step_dict)

        return action_chunk