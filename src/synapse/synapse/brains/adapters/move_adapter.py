import time

from ..base_brain_adapter import BaseBrainAdapter
from ..base_brain_adapter import InferenceOption
from sensor_msgs.msg import JointState
from synapse.utils.embodiment_parser import EmbodimentParser
from synapse.utils.kinematics import ArmKinematics, interpolate_pose, pose_distance
from synapse.utils.trajectory_builder import smoothstep


class MoveAdapter(BaseBrainAdapter):
    """Point-to-point Cartesian mover, driven by a scenario Action's params.

    A is wherever the arm is when the Action starts, so it is never written in
    the YAML; B comes from params (see configs/scn_mover_test.yaml). Components
    absent from params get no target at all and the muscle holds their last one.

    One waypoint leaves per infer call, never the whole path: JointState targets
    keep only a chunk's last waypoint (build_target_messages), so a full-path
    chunk would teleport in sim while interpolating on hardware. Progress is paced
    by wall clock, not by waypoint count -- the node needs two ticks per waypoint
    (submit, then collect), so counting waypoints made every move take about twice
    its planned duration and tripped the Actions' time-based failure guards.
    """

    def __init__(self, terminal, node_name="move_adapter", parameter_overrides=None):
        super().__init__(terminal, node_name, parameter_overrides)

        def _p(name, default):
            full = f"{self.get_name()}.{name}"
            return self.get_parameter(full).value if self.has_parameter(full) else default

        self.linear_speed = _p('linear_speed', 0.05)    # m/s
        self.angular_speed = _p('angular_speed', 0.5)   # rad/s
        self.joint_speed = _p('joint_speed', 1.0)       # rad/s, for joint_targets legs
        self.min_duration = _p('min_duration', 0.5)     # s, floor for very short moves
        self.ik_tolerance = _p('ik_tolerance', 0.005)   # m, residual that blocks the move
        self.ik_iterations = _p('ik_iterations', 4)      # re-seeded solves per waypoint

        self.robots_cfg = EmbodimentParser(self.embodiment_name).get_robots()
        self.kin = ArmKinematics(self.robots_cfg, terminal)
        self.current_eef_poses = self.kin.current_eef_poses  # the BT pose checks read this

        self.reset()
        self.terminal.log(
            f"🎯 [{node_name}] initialized. {self.linear_speed} m/s, {self.angular_speed} rad/s, "
            f"min {self.min_duration}s, IK tolerance {self.ik_tolerance * 1000:.0f} mm."
        )

    # ------------------------------------------------------------------
    def reset(self):
        self.plan = {}          # component -> {kind: 'pose'|'joints', start, target}
        self.seeds = {}         # component -> full-DOF IK seed, carried between waypoints
        self.duration = 0.0
        self.start_time = None
        self.progress = 0.0     # 0..1 of the planned duration
        self.last_done = False  # PolicyCompleteCheck reads this
        self.blocked = False
        self.planned = False

    def _plan(self, params, eef_poses, joints):
        """One leg per component, one duration for all of them, so a dual-arm
        move starts and finishes together however far each arm travels."""
        self.reset()
        self.planned = True
        components = (params or {}).get('components') or {}
        if not components:
            return

        linear = max(params.get('linear_speed') or self.linear_speed, 1e-6)
        angular = max(params.get('angular_speed') or self.angular_speed, 1e-6)
        duration = self.min_duration

        for name, spec in components.items():
            spec = spec or {}
            target_pose = spec.get('target_pose_values')  # named poses resolved at parse time
            target_joints = spec.get('joint_targets')
            observed = joints.get(name)

            if target_pose is not None:
                if name not in self.kin.robots:
                    self.terminal.log(f"⚠️ [{self.get_name()}] '{name}' is no manipulator, holding it")
                    continue
                start = eef_poses.get(name)
                self.plan[name] = {'kind': 'pose', 'start': start, 'target': list(target_pose)}
                self.seeds[name] = self.kin.seed_from_positions(
                    name, observed.position if observed else [])
                _, residual = self._solve(name, list(target_pose), self.seeds[name])
                if residual > self.ik_tolerance:
                    # Refuse the whole Action: moving one arm while another's
                    # target is bogus is worse than not moving at all.
                    self.terminal.log(
                        f"🛑 [{self.get_name()}] '{name}' target is unreachable "
                        f"(IK off by {residual * 1000:.1f} mm), not moving")
                    self.plan.clear()
                    self.blocked = True
                    return

                linear_d, angular_d = pose_distance(start, target_pose)
                duration = max(duration, linear_d / linear, angular_d / angular)

            elif target_joints is not None:
                if not observed or len(observed.position) != len(target_joints):
                    self.terminal.log(f"⚠️ [{self.get_name()}] no joint state for '{name}', holding it")
                    continue
                self.plan[name] = {'kind': 'joints', 'start': list(observed.position),
                                   'target': list(target_joints)}
                span = max((abs(t - s) for s, t in zip(observed.position, target_joints)), default=0.0)
                duration = max(duration, span / max(self.joint_speed, 1e-6))

            else:
                self.terminal.log(f"⚠️ [{self.get_name()}] '{name}' has no target, holding it")

        self.duration = max(duration, self.chunk_dt)
        self.start_time = time.monotonic()
        self.terminal.log(f"🎯 [{self.get_name()}] moving {list(self.plan)} over {self.duration:.2f}s")

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

    # ------------------------------------------------------------------
    def _format_for_policy(self, obs_history: list, inference_option: InferenceOption) -> dict:
        joints = obs_history[-1].get("joints", {})
        eef_poses = self.kin.forward_kinematics(joints)  # keeps current_eef_poses fresh for the checks

        if inference_option.restart or not self.planned:
            self._plan(inference_option.params, eef_poses, joints)

        return {}

    def _communicate_with_policy(self, formatted_obs: dict) -> dict:
        if not self.plan or self.blocked or self.last_done:
            return {}

        self.progress = min(1.0, (time.monotonic() - self.start_time) / self.duration)
        alpha = smoothstep(self.progress)

        waypoints = {}
        for name, leg in self.plan.items():
            if leg['kind'] == 'pose':
                waypoints[name] = ('pose', interpolate_pose(leg['start'], leg['target'], alpha))
            else:
                waypoints[name] = ('joints', [s + (t - s) * alpha
                                              for s, t in zip(leg['start'], leg['target'])])

        return {"waypoints": waypoints}

    def _format_for_muscle(self, raw_action: dict) -> list:
        waypoints = raw_action.get("waypoints")
        if not waypoints:
            return []

        step = {}
        for name, (kind, value) in waypoints.items():
            if kind == 'joints':
                step[name] = self._joint_state(name, value)
                continue

            solved, residual = self._solve(name, value, self.seeds[name])
            positions = [float(solved[i]) for i in self.kin.dof_indices[name].tolist()]

            # Holding beats driving the arm at a pose IK could not reach -- the
            # Action's failure condition takes it from here.
            if residual > self.ik_tolerance:
                self.terminal.log(
                    f"🛑 [{self.get_name()}] '{name}' IK off by {residual * 1000:.1f} mm, holding")
                self.blocked = True
                return []

            self.seeds[name] = solved
            step[name] = self._joint_state(name, positions)

        if self.progress >= 1.0:
            self.last_done = True
            self.terminal.log(f"✅ [{self.get_name()}] move complete")

        return [step]

    def _joint_state(self, name: str, positions) -> JointState:
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = self.robots_cfg.get(name, {}).get('joint_names', [])
        msg.position = list(positions)
        return msg
