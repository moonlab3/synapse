#!/usr/bin/env python3
import json
import os
import time
import numpy as np
from sensor_msgs.msg import JointState

from synapse.brains.base_brain_adapter import BaseBrainAdapter
from synapse.utils.embodiment_parser import EmbodimentParser


class RosbagAdapter(BaseBrainAdapter):

    STATE_SEEKING = "SEEKING"
    STATE_PLAYING = "PLAYING"
    STATE_DONE = "DONE"

    def __init__(self, terminal, node_name="rosbag_adapter", parameter_overrides=None):
        super().__init__(terminal, node_name, parameter_overrides)

        def _p(name, default):
            full = f"{self.get_name()}.{name}"
            return self.get_parameter(full).value if self.has_parameter(full) else default

        self.bag_path = _p('bag_path', None)
        raw_map = _p('topic_component_map', [])       # ["topic:component", ...]
        self.seek_velocity = _p('seek_velocity', 0.2)         # rad/s
        self.seek_min_duration = _p('seek_min_duration', 0.5)  # sec
        self.denormal_epsilon = _p('denormal_epsilon', 1e-6)

        self.topic_component_map = {}
        for entry in raw_map:
            topic, component = entry.split(':', 1)
            self.topic_component_map[topic] = component

        parser = EmbodimentParser(self.embodiment_name)
        self.robots_cfg = parser.get_robots()

        self.state = None                # set once seeded from first observation
        self.bag_data = {}               # component -> sorted [{t, data}, ...]
        self.cursor = {}                 # component -> next unread index
        self.first_position = {}         # component -> bag's t≈0 pose
        self.current_values = {}         # component -> latest emitted pose
        self.seek_start_pose = {}        # component -> pose we started seeking from
        self.seek_duration = 0.0
        self.seek_start_time = None
        self.playback_start_time = None
        self.last_done = False

        self._load_bag()
        self.terminal.log(
            f"🎞️ [{node_name}] loaded bag '{self.bag_path}' "
            f"({len(self.bag_data)} mapped components), seek_velocity={self.seek_velocity} rad/s"
        )

    def _load_bag(self):
        if not self.bag_path or not os.path.exists(self.bag_path):
            self.terminal.log(f"❌ Rosbag JSON not found at: '{self.bag_path}'")
            return

        with open(self.bag_path, 'r') as f:
            raw = json.load(f)

        messages = raw.get('messages', {})

        for topic, component in self.topic_component_map.items():
            if topic not in messages:
                self.terminal.log(f"⚠️ Topic '{topic}' not in bag; skipping component '{component}'")
                continue
            if component not in self.robots_cfg:
                self.terminal.log(f"⚠️ Component '{component}' not in embodiment '{self.embodiment_name}'; skipping")
                continue

            msgs = sorted(messages[topic], key=lambda m: m['t'])

            # Sanitize denormal near-zero artifacts seen in recorded data.
            for m in msgs:
                data = np.array(m['data'], dtype=np.float64)
                data[np.abs(data) < self.denormal_epsilon] = 0.0
                m['data'] = data.tolist()

            self.bag_data[component] = msgs
            self.cursor[component] = 0
            self.first_position[component] = msgs[0]['data'] if msgs else None

    # ------------------------------------------------------------------

    def _seed_if_needed(self, joints_dict):
        if self.state is not None:
            return

        for component in self.bag_data:
            joint_state = joints_dict.get(component)
            n = len(self.first_position[component])
            if joint_state is not None and joint_state.position and len(joint_state.position) == n:
                self.seek_start_pose[component] = list(joint_state.position)
            else:
                # No live observation yet -- start exactly at the bag's
                # first pose so this component skips seeking.
                self.seek_start_pose[component] = list(self.first_position[component])

            self.current_values[component] = list(self.seek_start_pose[component])

        max_delta = 0.0
        for component, start_pose in self.seek_start_pose.items():
            target_pose = self.first_position[component]
            for a, b in zip(start_pose, target_pose):
                max_delta = max(max_delta, abs(b - a))

        self.seek_duration = max(self.seek_min_duration, max_delta / max(self.seek_velocity, 1e-6))
        self.seek_start_time = time.monotonic()
        self.state = self.STATE_SEEKING
        self.terminal.log(f"🐢 [rosbag] seeking to bag start over {self.seek_duration:.2f}s")

    @staticmethod
    def _smoothstep(t):
        t = max(0.0, min(1.0, t))
        return t * t * (3.0 - 2.0 * t)  # zero velocity at both ends

    def _step_seek(self):
        elapsed = time.monotonic() - self.seek_start_time
        alpha = self._smoothstep(elapsed / self.seek_duration)

        for component, start_pose in self.seek_start_pose.items():
            target_pose = self.first_position[component]
            self.current_values[component] = [
                s + (t - s) * alpha for s, t in zip(start_pose, target_pose)
            ]

        if alpha >= 1.0:
            self.state = self.STATE_PLAYING
            self.playback_start_time = time.monotonic()
            self.terminal.log("▶️ [rosbag] seek complete, starting playback")

    def _step_playback(self):
        elapsed = time.monotonic() - self.playback_start_time
        all_done = True

        for component, msgs in self.bag_data.items():
            cursor = self.cursor[component]
            n = len(msgs)
            while cursor < n and msgs[cursor]['t'] <= elapsed:
                self.current_values[component] = msgs[cursor]['data']
                cursor += 1
            self.cursor[component] = cursor
            if cursor < n:
                all_done = False

        if all_done:
            self.state = self.STATE_DONE
            self.last_done = True
            self.terminal.log("✅ [rosbag] playback finished, holding last pose")

    # ------------------------------------------------------------------

    def _format_for_policy(self, obs_history: list, get_default: bool) -> dict:
        latest_obs = obs_history[-1]
        joints_dict = latest_obs.get("joints", {})
        self._seed_if_needed(joints_dict)
        return {}

    def _communicate_with_policy(self, formatted_obs: dict) -> dict:
        if self.state == self.STATE_SEEKING:
            self._step_seek()
        elif self.state == self.STATE_PLAYING:
            self._step_playback()
        elif self.state == self.STATE_DONE:
            return {}  # nothing further -- buffer holds last pose

        return {"values": {k: list(v) for k, v in self.current_values.items()}}

    def _format_for_muscle(self, raw_action: dict) -> list:
        values = raw_action.get("values")
        if not values:
            return []

        step = {}
        for component, positions in values.items():
            msg = JointState()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.name = self.robots_cfg.get(component, {}).get('joint_names', [])
            msg.position = positions
            step[component] = msg

        return [step]