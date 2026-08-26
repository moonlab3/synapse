import time
import yaml
import py_trees

class BaseCheck:
    def __init__(self, cfg: dict, node_ref):
        self.description = cfg.get('description', '')
        self.node = node_ref  # SynapseMainNode reference

    def evaluate(self, action_ctx: dict) -> bool:
        raise NotImplementedError


class TimeCheck(BaseCheck):
    """Elapsed wall-clock time since this Action node started running."""
    def __init__(self, cfg, node_ref):
        super().__init__(cfg, node_ref)
        self.duration = cfg['duration']  # seconds

    def evaluate(self, action_ctx):
        elapsed = time.monotonic() - action_ctx['start_time']
        return elapsed >= self.duration


class PolicyCompleteCheck(BaseCheck):
    """
    Adapter-reported completion (e.g. rosbag exhausted, VLA <STOP> token).

    ASSUMPTION: adapters optionally expose a `last_done` bool attribute,
    set during `_communicate_with_policy` / `_format_for_muscle`. Adapters
    that never terminate on their own (e.g. ManualAdapter) simply never
    set it, so this check safely defaults to False via getattr.
    """
    def evaluate(self, action_ctx):
        adapter = action_ctx['adapter']
        return bool(getattr(adapter, 'last_done', False))


class DetectCheck(BaseCheck):
    def __init__(self, cfg, node_ref):
        super().__init__(cfg, node_ref)
        self.target = cfg['target']
        self.threshold = cfg.get('threshold', None)

    def evaluate(self, action_ctx):
        detections = getattr(self.node, 'latest_detections', {})
        value = detections.get(self.target)
        if value is None:
            return False
        if self.threshold is None:
            return bool(value)
        return value >= self.threshold


class PoseBoundCheck(BaseCheck):
    """
    Single-axis safety bound, e.g. "z < 0.1 -> arm dipped too low."

    Reads from the adapter's own `current_eef_poses` dict (already
    maintained by ManualAdapter / GR00TAdapter via their FK routines) so
    this doesn't need to duplicate kinematics.
    """
    OPS = {
        '<':  lambda a, b: a < b,
        '<=': lambda a, b: a <= b,
        '>':  lambda a, b: a > b,
        '>=': lambda a, b: a >= b,
        '==': lambda a, b: abs(a - b) < 1e-6,
    }
    AXES = {'x': 0, 'y': 1, 'z': 2, 'roll': 3, 'pitch': 4, 'yaw': 5}

    def __init__(self, cfg, node_ref):
        super().__init__(cfg, node_ref)
        self.robot = cfg['robot']
        self.axis = cfg['axis']
        self.operator = cfg['operator']
        self.value = cfg['value']

    def evaluate(self, action_ctx):
        adapter = action_ctx['adapter']
        pose = getattr(adapter, 'current_eef_poses', {}).get(self.robot)
        if pose is None:
            return False
        current = pose[self.AXES[self.axis]]
        return self.OPS[self.operator](current, self.value)


class PoseReachedCheck(BaseCheck):
    """
    Full pose-with-tolerance check ("has the robot arrived somewhere").

    `robot` may be a single robot name or "ALL" to require every
    manipulator in the current adapter's embodiment to be within tolerance.

    ASSUMPTION: `target_pose_values` is an explicit [x, y, z, r, p, y] list
    for now. `target_pose` (a named pose like "home" / "safe_retract") is
    accepted in the schema but has no lookup table yet -- until a poses
    library exists, named targets fail safe (never report reached) rather
    than silently matching anything.
    """
    def __init__(self, cfg, node_ref):
        super().__init__(cfg, node_ref)
        self.robot = cfg['robot']
        self.target_pose_name = cfg.get('target_pose')
        self.target_pose_values = cfg.get('target_pose_values')
        self.tolerance = cfg.get('tolerance', 0.02)

    def _target_robots(self, adapter):
        eef_poses = getattr(adapter, 'current_eef_poses', {})
        return list(eef_poses.keys()) if self.robot == "ALL" else [self.robot]

    def evaluate(self, action_ctx):
        if self.target_pose_values is None:
            return False  # named-pose lookup not implemented yet

        adapter = action_ctx['adapter']
        eef_poses = getattr(adapter, 'current_eef_poses', {})

        for robot in self._target_robots(adapter):
            pose = eef_poses.get(robot)
            if pose is None:
                return False
            error = sum((a - b) ** 2 for a, b in zip(pose[:3], self.target_pose_values[:3])) ** 0.5
            if error > self.tolerance:
                return False
        return True


CHECK_REGISTRY = {
    'time': TimeCheck,
    'policy_complete': PolicyCompleteCheck,
    'detect': DetectCheck,
    'pose_bound': PoseBoundCheck,
    'pose_reached': PoseReachedCheck,
}


class ConditionGroup:
    """
    An AND/OR group of checks, built from a `conditions.success` or
    `conditions.failure` YAML block. Empty/absent groups never fire
    (an Action with no success block will just run until a parent
    times it out or a failure condition trips it).
    """
    def __init__(self, cfg: dict, node_ref):
        self.logic = cfg.get('logic', 'AND').upper()
        self.checks = [
            CHECK_REGISTRY[c['type']](c, node_ref)
            for c in cfg.get('checks', [])
        ]

    def evaluate(self, action_ctx: dict) -> bool:
        if not self.checks:
            return False
        results = [c.evaluate(action_ctx) for c in self.checks]
        return all(results) if self.logic == 'AND' else any(results)

    def pending_descriptions(self, action_ctx: dict) -> list:
        """Descriptions of not-yet-satisfied checks, for TUI/debug display."""
        return [c.description for c in self.checks if not c.evaluate(action_ctx)]


# ============================================================
# 🌲 BEHAVIOR TREE LEAF
# ============================================================

class RunAction(py_trees.behaviour.Behaviour):
    """
    Drives one brain adapter's inference loop (background thread, same
    pattern as before) and evaluates its success/failure ConditionGroups
    every tick to decide when to report SUCCESS / FAILURE / RUNNING.

    Failure is checked before success each tick, since failure conditions
    here tend to be safety-relevant (e.g. collision risk) and should
    preempt a success condition that happens to be true on the same tick.
    """
    def __init__(self, name, adapter_key, conditions_cfg, bt_node_reference, description=""):
        super().__init__(name)
        self.adapter_key = adapter_key
        self.node = bt_node_reference
        self.description = description

        self.success_group = ConditionGroup(conditions_cfg.get('success', {}), self.node) \
            if 'success' in conditions_cfg else None
        self.failure_group = ConditionGroup(conditions_cfg.get('failure', {}), self.node) \
            if 'failure' in conditions_cfg else None

        self._start_time = None

    def initialise(self):
        # py_trees calls this once when the node transitions INVALID -> RUNNING,
        # i.e. exactly when this Action becomes active. Good spot to (re)start
        # the clock used by TimeCheck.
        self._start_time = time.monotonic()
        self.node.terminal_ui.log(f"▶️  [{self.name}] {self.description}")

    def _action_ctx(self):
        return {
            'adapter': self.node.brain_adapters.get(self.adapter_key),
            'start_time': self._start_time,
        }

    def update(self):
        ctx = self._action_ctx()
        adapter = ctx['adapter']

        if adapter is None:
            self.node.terminal_ui.log(f"❌ [{self.name}] unknown adapter key '{self.adapter_key}'")
            return py_trees.common.Status.FAILURE

        # 1. Failure conditions first.
        if self.failure_group and self.failure_group.evaluate(ctx):
            self.node.terminal_ui.log(f"🛑 [{self.name}] failure condition met")
            return py_trees.common.Status.FAILURE

        # 2. Success conditions.
        if self.success_group and self.success_group.evaluate(ctx):
            self.node.terminal_ui.log(f"✅ [{self.name}] success condition met")
            return py_trees.common.Status.SUCCESS

        # 3. Neither fired -- keep driving inference in the background,
        #    same submit/collect pattern as the original RunAdapterBehavior.
        if self.node.inference_future is not None:
            if self.node.inference_future.done():
                new_action_chunk = self.node.inference_future.result()
                if new_action_chunk:
                    self.node.action_buffer.update_chunk(new_action_chunk)
                self.node.inference_future = None
            return py_trees.common.Status.RUNNING

        historical_obs = list(self.node.obs_buffer)
        self.node.inference_future = self.node.inference_executor.submit(
            adapter.infer, historical_obs, self.node.running_default
        )
        return py_trees.common.Status.RUNNING

    def terminate(self, new_status):
        # Reset per-run state so re-entry (e.g. a Selector retrying this
        # branch later) starts its clock and conditions clean.
        self._start_time = None

class RunCondition(py_trees.behaviour.Behaviour):
    def __init__(self, name, conditions_cfg, bt_node_reference, description=""):
        super().__init__(name)
        self.node = bt_node_reference
        self.description = description

        self.success_group = ConditionGroup(conditions_cfg.get('success', {}), self.node) \
            if 'success' in conditions_cfg else None
        self.failure_group = ConditionGroup(conditions_cfg.get('failure', {}), self.node) \
            if 'failure' in conditions_cfg else None

        self._start_time = None

    def initialise(self):
        self._start_time = time.monotonic()
        self.node.terminal_ui.log(f"👁️  [{self.name}] watching -- {self.description}")

    def _action_ctx(self):
        return {
            'adapter': None,
            'start_time': self._start_time,
        }

    def update(self):
        ctx = self._action_ctx()

        if self.failure_group and self.failure_group.evaluate(ctx):
            self.node.terminal_ui.log(f"🛑 [{self.name}] failure condition met")
            return py_trees.common.Status.FAILURE

        if self.success_group and self.success_group.evaluate(ctx):
            print(f"✅ [{self.name}] success condition met")
            # self.node.terminal_ui.log(f"✅ [{self.name}] success condition met")
            return py_trees.common.Status.SUCCESS

        return py_trees.common.Status.RUNNING

    def terminate(self, new_status):
        self._start_time = None


# ============================================================
# 📄 SCENARIO PARSER
# ============================================================

class ScenarioParser:
    def __init__(self, synapse_node):
        # Reference to the main ROS node so Action leaves can reach
        # brain_adapters, obs_buffer, inference_future, terminal_ui, etc.
        self.node = synapse_node

    def parse(self, file_path: str) -> py_trees.behaviour.Behaviour:
        with open(file_path, 'r') as f:
            config = yaml.safe_load(f)

        bt_root = self._build_node(config.get('behavior_tree', {}))

        print(f"===== 🌳🌳  Scenario Parsed from [{file_path}] =====")
        print(py_trees.display.unicode_tree(bt_root, show_only_visited=False))

        return bt_root

    def _build_node(self, node_config: dict) -> py_trees.behaviour.Behaviour:
        if not node_config:
            raise ValueError("Encountered empty node configuration in YAML.")

        node_type = node_config.get('type')
        node_name = node_config.get('name', 'unnamed_node')

        if node_type == "Sequence":
            bt_node = py_trees.composites.Sequence(name=node_name, memory=True)
            for child_cfg in node_config.get('children', []):
                bt_node.add_child(self._build_node(child_cfg))
            return bt_node
        elif node_type == "Selector":
            bt_node = py_trees.composites.Selector(name=node_name, memory=False)
            for child_cfg in node_config.get('children', []):
                bt_node.add_child(self._build_node(child_cfg))
            return bt_node
        elif node_type == "Action":
            return RunAction(
                name=node_name,
                adapter_key=node_config['adapter'],
                conditions_cfg=node_config.get('conditions', {}),
                bt_node_reference=self.node,
                description=node_config.get('description', ''),
            )
        elif node_type == "Condition":
            return RunCondition(
                name=node_name,
                conditions_cfg=node_config.get('conditions', {}),
                bt_node_reference=self.node,
                description=node_config.get('description', ''),
            )
        else:
            raise ValueError(f"Unknown node type: {node_type}")