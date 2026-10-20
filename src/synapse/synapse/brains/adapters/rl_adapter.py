import os
import re
import sys
import time

import numpy as np
from geometry_msgs.msg import PointStamped
from sensor_msgs.msg import JointState

from ..base_brain_adapter import BaseBrainAdapter
from ..base_brain_adapter import InferenceOption
from synapse.utils.embodiment_parser import EmbodimentParser
from synapse.utils.paths import resolve_path


def _canonical(joint_name: str) -> str:
    """'right_joint1' / 'joint1' -> 'joint1'; 'joint00' -> 'joint00' (policy joint names)."""
    m = re.search(r'(joint\d+)$', joint_name)
    return m.group(1) if m else joint_name


class RLAdapter(BaseBrainAdapter):
    """Runs an rl-platform student policy (numpy, from an export dir with spec.json + policy.npz).

    The policy expects joint states of both arm+hand sides and the ball position in the table
    frame (the frame of dual_xarm7_v6f.usd: table centre at x=y=0, z up from the floor).
    It outputs absolute joint targets at spec['control_hz'] (30 Hz); between policy steps the
    last targets are held.
    """

    def __init__(self, terminal, name="rl_adapter", params=None, ros_node=None):
        super().__init__(terminal, name, params, ros_node)

        rl_path = self.path('rl_platform')
        if rl_path not in sys.path:
            sys.path.append(rl_path)
        from rl_platform.deploy.student_runtime import StudentRuntime  # numpy only

        export_dir = resolve_path(self.param('export_dir'))
        self.runtime = StudentRuntime(os.path.join(export_dir, 'spec.json'), os.path.join(export_dir, 'policy.npz'))
        self.period = 1.0 / float(self.runtime.spec['control_hz'])

        # 'right:right_arm,right_hand' -> policy side -> synapse components (arm first, then hand)
        self.side_map = {}
        for entry in self.param('side_map', ['right:right_arm,right_hand', 'left:left_arm,left_hand']):
            side, comps = entry.split(':', 1)
            self.side_map[side] = comps.split(',')
        self.action_gain = float(self.param('action_gain', 1.0))         # <1 slows every joint (first real runs)
        self.max_start_error = float(self.param('max_start_error', 0.15))  # rad from the training start pose
        self.ball_timeout = float(self.param('ball_timeout', 0.2))       # s before the ball counts as not seen
        self.default_command = self.param('default_command', 'handover')

        self.robots_cfg = EmbodimentParser(self.embodiment_name).get_robots()
        for side, comps in self.side_map.items():
            missing = [c for c in comps if c not in self.robots_cfg]
            if missing:
                raise ValueError(f"rl_adapter: components {missing} for side '{side}' not in {self.embodiment_name}")

        self._ball = None
        self._ball_stamp = 0.0
        self.node.create_subscription(PointStamped, self.param('ball_topic', '/rl/ball_position'), self._ball_cb, 10)

        self._started = False
        self._last_step = 0.0
        self._prev_pos = None
        self._targets = None
        self.terminal.log(
            f"🤖 [{self.name}] policy {export_dir} ({self.runtime.spec['obs_dim']} obs -> "
            f"{self.runtime.spec['action_dim']} act, {self.runtime.spec['control_hz']} Hz), gain {self.action_gain}"
        )

    def reset(self):
        self._started = False
        self._prev_pos = None
        self._targets = None

    def _ball_cb(self, msg: PointStamped):
        self._ball = np.array([msg.point.x, msg.point.y, msg.point.z])
        self._ball_stamp = time.monotonic()

    # ------------------------------------------------------------------ policy-side joint vectors
    def _side_positions(self, joints_dict: dict) -> dict | None:
        """Per side: joint positions in the policy's order, or None if any joint is missing."""
        out = {}
        for side, comps in self.side_map.items():
            by_name = {}
            for comp in comps:
                msg = joints_dict.get(comp)
                if msg is None or not msg.position:
                    return None
                names = list(msg.name) or self.robots_cfg[comp].get('joint_names', [])
                by_name.update({_canonical(n): p for n, p in zip(names, msg.position)})
            order = self.runtime.side[side]['order']
            try:
                out[side] = np.array([by_name[j] for j in order])
            except KeyError as e:
                self.terminal.log(f"❌ rl_adapter: joint {e} missing for side {side}")
                return None
        return out

    # ------------------------------------------------------------------ BaseBrainAdapter
    def _format_for_policy(self, obs_history: list, inference_option: InferenceOption):
        joints = self._side_positions(obs_history[-1].get('joints', {}))
        return {'pos': joints, 'now': time.monotonic()}

    def _communicate_with_policy(self, formatted_obs):
        pos, now = formatted_obs['pos'], formatted_obs['now']
        if pos is None:
            return None
        if not self._started:
            err = self.runtime.start_pose_error(pos)
            if max(err.values()) > self.max_start_error:
                self.terminal.log(
                    f"⚠️ rl_adapter: robot is {err} rad from the training start pose; move it there first"
                )
                return None
            ball = self._ball if self._fresh_ball(now) else None
            self.runtime.reset(pos, ball)
            self._started, self._prev_pos, self._last_step = True, pos, now
            self._targets = {s: q.copy() for s, q in pos.items()}
            return self._targets
        if now - self._last_step < self.period:
            return self._targets  # hold between 30 Hz policy steps
        dt = now - self._last_step
        vel = {s: (pos[s] - self._prev_pos[s]) / dt for s in pos}
        ball = self._ball if self._fresh_ball(now) else None
        self._targets = self.runtime.step(pos, vel, ball, action_gain=self.action_gain)
        self._prev_pos, self._last_step = pos, now
        return self._targets

    def _fresh_ball(self, now: float) -> bool:
        return self._ball is not None and now - self._ball_stamp < self.ball_timeout

    def _format_for_muscle(self, raw_action) -> list:
        if raw_action is None:
            return []
        step = {}
        stamp = self.node.get_clock().now().to_msg()
        for side, comps in self.side_map.items():
            order = self.runtime.side[side]['order']
            target = dict(zip(order, raw_action[side]))
            for comp in comps:
                names = self.robots_cfg[comp].get('joint_names', [])
                msg = JointState()
                msg.header.stamp = stamp
                msg.name = list(names)
                msg.position = [float(target[_canonical(n)]) for n in names]
                step[comp] = msg
        return [step]
