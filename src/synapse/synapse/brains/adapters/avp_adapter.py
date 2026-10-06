import re
import threading
import time

import numpy as np
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import JointState

from ..base_brain_adapter import BaseBrainAdapter
from ..base_brain_adapter import InferenceOption
from synapse.utils.embodiment_parser import EmbodimentParser
from synapse.utils.kinematics import ArmKinematics, interpolate_pose, pose_distance

# avp_stream's wrist-local hand skeleton: the thumb is knuckle..tip, the four
# fingers metacarpal..tip. Listed in robot finger order, X of joint_XY.
THUMB = (1, 2, 3, 4)
FINGERS = ((5, 6, 7, 8, 9), (10, 11, 12, 13, 14), (15, 16, 17, 18, 19), (20, 21, 22, 23, 24))


class AvpTeleopAdapter(BaseBrainAdapter):
    """Apple Vision Pro teleoperation: wrists drive the arms, fingers the hands.

    Arms follow the wrist's offset from where it was at engage, never its
    absolute pose, so engaging cannot jump the arm and the operator can stand
    anywhere. References are taken on the first tracked frame after a restart
    and again whenever a hand comes back from a tracking loss.

    Hands are joint-space: bend angles measured on the operator's fingers go
    straight to the robot's. Only hands with joint_XY names (X finger, thumb
    first; Y joint, base first) are supported, with joint_limits in the
    embodiment config giving each joint's range and bend direction.

    A side that is not tracked gets no target at all and the muscle holds its
    last one. One waypoint leaves per infer call, as in MoveAdapter.
    """

    def __init__(self, terminal, node_name="avp_adapter", parameter_overrides=None):
        super().__init__(terminal, node_name, parameter_overrides)

        def _p(name, default):
            full = f"{self.get_name()}.{name}"
            return self.get_parameter(full).value if self.has_parameter(full) else default

        self.avp_ip = _p('avp_ip', None)
        self.arm_map = dict(e.split(':', 1) for e in _p('arm_map', []))    # 'left'|'right' -> component
        self.hand_map = dict(e.split(':', 1) for e in _p('hand_map', []))
        # AVP frame is x right, y forward, z up; -90 deg suits a base whose x points forward.
        self.r_map = Rotation.from_euler('z', _p('avp_yaw_deg', -90.0), degrees=True).as_matrix()
        self.scale = _p('scale', 1.0)
        self.max_linear_step = _p('max_linear_step', 0.01)    # m per waypoint
        self.max_angular_step = _p('max_angular_step', 0.1)   # rad per waypoint
        self.stale_timeout = _p('stale_timeout', 0.2)         # s without a new frame
        self.ik_tolerance = _p('ik_tolerance', 0.005)         # m, residual that holds the arm
        self.ik_iterations = _p('ik_iterations', 4)           # re-seeded solves per waypoint
        # Per joint Y, applied to the operator's angle: robot = gain * angle + offset.
        self.finger_gain = _p('finger_gain', [1.0] * 4)       # spread, then three bends
        self.finger_offset = _p('finger_offset', [0.0] * 4)
        self.thumb_gain = _p('thumb_gain', [1.0] * 4)         # bend whole thumb, rotate, two bends
        self.thumb_offset = _p('thumb_offset', [0.0] * 4)

        self.robots_cfg = EmbodimentParser(self.embodiment_name).get_robots()
        for hand in self.hand_map.values():
            names = self.robots_cfg.get(hand, {}).get('joint_names', [])
            if not all(re.fullmatch(r'joint_[0-4][0-3]', n) for n in names):
                raise ValueError(f"'{hand}' joints are not named joint_XY, cannot map fingers onto it")

        self.kin = ArmKinematics(self.robots_cfg, terminal)
        self.current_eef_poses = self.kin.current_eef_poses  # the BT pose checks read this

        self.streamer = None
        threading.Thread(target=self._connect, daemon=True).start()

        self.reset()
        self.terminal.log(f"🥽 [{node_name}] initialized. Waiting for Vision Pro at {self.avp_ip}.")

    def _connect(self):
        # VisionProStreamer() blocks until the headset streams, so it stays off the
        # node's init path. Imported here for the same reason: importing it on the
        # main thread would also swap in avp_stream's own SIGINT handler.
        try:
            from avp_stream import VisionProStreamer
            self.streamer = VisionProStreamer(ip=self.avp_ip, record=False)
            self.terminal.log(f"🥽 [{self.get_name()}] Vision Pro stream is up")
        except Exception as e:
            self.terminal.log(f"❌ [{self.get_name()}] Vision Pro stream failed: {e}")

    # ------------------------------------------------------------------
    def reset(self):
        self.anchors = {}       # side -> (wrist 4x4, eef pose) at engage
        self.commanded = {}     # arm -> last commanded eef pose
        self.seeds = {}         # arm -> full-DOF IK seed, carried between waypoints
        self.held = set()       # arms currently refused by IK, so the log fires once
        self.last_frame = None  # newest raw frame and when it arrived
        self.last_frame_time = 0.0
        self.last_wrist = {}    # side -> (wrist 4x4, when it last moved)

    def _solve(self, name: str, pose: list, seed):
        """Chain solves: one Gauss-Newton pass undershoots badly when the seed is
        far from the target, so re-seed it with its own result until the residual
        settles. Returns (full-DOF solution, residual in metres)."""
        indices = self.kin.dof_indices[name].tolist()
        q, residual = seed, float('inf')
        for _ in range(max(1, self.ik_iterations)):
            q = self.kin.solve(name, pose, q)
            residual = pose_distance(pose, self.kin.eef_pose(
                name, [float(q[i]) for i in indices]))[0]
            if residual <= self.ik_tolerance:
                break
        return q, residual

    def _tracked(self, side: str, wrist, now: float) -> bool:
        """Frames carry no validity flag, so a lost hand shows up only as a wrist
        that sits at the origin or stops changing while frames keep arriving."""
        last = self.last_wrist.get(side)
        if last is None or not np.array_equal(last[0], wrist):
            self.last_wrist[side] = last = (wrist, now)
        return bool(np.any(wrist[:3, 3])) and now - last[1] <= self.stale_timeout

    # ------------------------------------------------------------------
    def _format_for_policy(self, obs_history: list, inference_option: InferenceOption) -> dict:
        joints = obs_history[-1].get("joints", {})
        eef_poses = self.kin.forward_kinematics(joints)  # keeps current_eef_poses fresh for the checks

        if inference_option.restart:
            self.reset()

        return {"joints": joints, "eef_poses": eef_poses}

    def _communicate_with_policy(self, formatted_obs: dict) -> dict:
        data = self.streamer.get_latest() if self.streamer else None
        if data is None:
            return {}

        now = time.monotonic()
        if data.raw is not self.last_frame:
            self.last_frame, self.last_frame_time = data.raw, now
        elif now - self.last_frame_time > self.stale_timeout:
            if self.anchors:
                self.terminal.log(f"⚠️ [{self.get_name()}] Vision Pro stream is stale, holding")
            self.anchors.clear()  # re-engage from wherever the hands are when it returns
            return {}

        return {**formatted_obs, "frame": data.raw}

    def _format_for_muscle(self, raw_action: dict) -> list:
        frame = raw_action.get("frame")
        if frame is None:
            return []

        now = time.monotonic()
        step = {}
        for side in ('left', 'right'):
            wrist = frame[f"{side}_wrist"].reshape(4, 4)
            if not self._tracked(side, wrist, now):
                if self.anchors.pop(side, None):
                    self.terminal.log(f"⚠️ [{self.get_name()}] {side} hand lost, holding")
                continue

            arm = self.arm_map.get(side)
            if arm in self.kin.robots:
                positions = self._arm_positions(side, arm, wrist, raw_action)
                if positions is not None:
                    step[arm] = self._joint_state(arm, positions)

            hand = self.hand_map.get(side)
            if hand in self.robots_cfg:
                step[hand] = self._joint_state(
                    hand, self._hand_positions(side, hand, frame[f"{side}_fingers"]))

        return [step] if step else []

    # ------------------------------------------------------------------
    def _arm_positions(self, side: str, arm: str, wrist, obs: dict):
        if side not in self.anchors:
            observed = obs["joints"].get(arm)
            if not observed or not observed.position:
                return None
            self.anchors[side] = (wrist, obs["eef_poses"][arm])
            self.commanded[arm] = obs["eef_poses"][arm]
            self.seeds[arm] = self.kin.seed_from_positions(arm, observed.position)
            self.terminal.log(f"🥽 [{self.get_name()}] {side} hand engaged '{arm}'")

        wrist0, eef0 = self.anchors[side]
        # The wrist's motion since engage, turned into the arm's base frame and
        # applied on top of the eef pose at engage. Rotation is a world-frame
        # delta, so the hand never has to be aligned with the eef.
        position = np.array(eef0[:3]) + self.scale * self.r_map @ (wrist[:3, 3] - wrist0[:3, 3])
        delta = Rotation.from_matrix(self.r_map @ wrist[:3, :3] @ wrist0[:3, :3].T @ self.r_map.T)
        rpy = (delta * Rotation.from_euler('xyz', eef0[3:])).as_euler('xyz')
        target = [float(v) for v in (*position, *rpy)]

        # Never ask for more than one step's worth of travel, however far the
        # hand got ahead: a tracking glitch then costs a step, not a lunge.
        linear_d, angular_d = pose_distance(self.commanded[arm], target)
        alpha = min(1.0, self.max_linear_step / max(linear_d, 1e-9),
                    self.max_angular_step / max(angular_d, 1e-9))
        pose = interpolate_pose(self.commanded[arm], target, alpha)

        solved, residual = self._solve(arm, pose, self.seeds[arm])
        if residual > self.ik_tolerance:
            if arm not in self.held:
                self.terminal.log(
                    f"🛑 [{self.get_name()}] '{arm}' IK off by {residual * 1000:.1f} mm, holding")
                self.held.add(arm)
            return None

        self.held.discard(arm)
        self.seeds[arm], self.commanded[arm] = solved, pose
        return [float(solved[i]) for i in self.kin.dof_indices[arm].tolist()]

    def _hand_positions(self, side: str, hand: str, skeleton) -> list:
        p = skeleton[:, :3, 3]

        def bones(chain):
            return [(p[b] - p[a]) / (np.linalg.norm(p[b] - p[a]) + 1e-9) for a, b in zip(chain, chain[1:])]

        def bend(a, b):
            return float(np.arccos(np.clip(a @ b, -1.0, 1.0)))

        # Out of the palm, for either hand: wrist->index x wrist->little points
        # that way on a right hand and the opposite way on a left one.
        normal = np.cross(p[6] - p[0], p[21] - p[0]) * (1.0 if side == 'right' else -1.0)
        normal /= np.linalg.norm(normal) + 1e-9
        forward = bones((0, 11))[0]            # wrist -> middle knuckle
        lateral = np.cross(normal, forward)    # toward the thumb on a left hand, away on a right

        # Operator angles per finger X, in joint Y order.
        base, proximal, distal = bones(THUMB)
        radial = lateral if side == 'left' else -lateral
        angles = [(
            float(np.arcsin(np.clip(base @ normal, -1.0, 1.0))),   # whole thumb toward the palm
            float(np.arctan2(base @ radial, base @ forward)),      # swing out from the fingers
            bend(base, proximal),
            bend(proximal, distal),
        )]
        knuckles = [p[chain[1]] for chain in FINGERS]
        for i, chain in enumerate(FINGERS):
            metacarpal, b1, b2, b3 = bones(chain)
            # The knuckle's hinge runs along the line through its neighbours, index
            # to little. Taking it per finger follows the palm's arch, which one
            # flat palm plane reads as spread on every curled finger.
            ulnar = knuckles[min(i + 1, 3)] - knuckles[max(i - 1, 0)]
            ulnar -= (ulnar @ metacarpal) * metacarpal
            ulnar /= np.linalg.norm(ulnar) + 1e-9
            out = np.cross(metacarpal, ulnar) * (1.0 if side == 'right' else -1.0)
            spread = float(np.arcsin(np.clip(b1 @ ulnar, -1.0, 1.0)))
            angles.append((
                spread if side == 'left' else -spread,          # toward the little finger is + on a left hand
                float(np.arctan2(b1 @ out, b1 @ metacarpal)),   # knuckle
                bend(b1, b2),
                bend(b2, b3),
            ))

        cfg = self.robots_cfg[hand]
        limits = cfg.get('joint_limits', {})
        positions = []
        for name in cfg['joint_names']:
            x, y = int(name[-2]), int(name[-1])
            gain, offset = (self.thumb_gain, self.thumb_offset) if x == 0 \
                else (self.finger_gain, self.finger_offset)
            value = gain[y] * angles[x][y] + offset[y]
            lo, hi = limits.get(name, (-np.pi, np.pi))
            # Spread is signed already. Everything else only bends one way, toward
            # the far end of its range -- which is what mirrors left and right.
            if (x == 0 or y > 0) and -lo > hi:
                value = -value
            positions.append(float(np.clip(value, lo, hi)))
        return positions

    def _joint_state(self, name: str, positions) -> JointState:
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = self.robots_cfg.get(name, {}).get('joint_names', [])
        msg.position = list(positions)
        return msg
