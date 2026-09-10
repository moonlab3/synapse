import yaml
import os
from pathlib import Path

DEFAULT_MSG_TYPE = 'JointState'

class ResolvedTopic:
    __slots__ = ('topic', 'msg_type')
    def __init__(self, topic: str, msg_type: str = DEFAULT_MSG_TYPE):
        self.topic = topic
        self.msg_type = msg_type

    def __repr__(self):
        return f"ResolvedTopic(topic={self.topic!r}, msg_type={self.msg_type!r})"

class EmbodimentParser:
    def __init__(self, embodiment_name: str):
        self.embodiment_name = embodiment_name
        
        share_dir = "/home/rog-sf/ws/synapse_ws/install/synapse/share/synapse"
        current_file = Path(__file__).resolve()
        dynamic_share = os.path.join(current_file.parents[5], 'share', 'synapse')
        if os.path.exists(os.path.join(dynamic_share, 'configs', 'embodiment_configs.yaml')):
            share_dir = dynamic_share
            
        self.config_path = os.path.join(share_dir, 'configs', 'embodiment_configs.yaml')
        self.full_config = self._load_file()
        self.embodiment_config = self._extract_embodiment()

    def _load_file(self) -> dict:
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"❌ Embodiment config not found at: {self.config_path}")
        
        with open(self.config_path, 'r') as file:
            return yaml.safe_load(file) or {}

    def _extract_embodiment(self) -> dict:
        if self.embodiment_name not in self.full_config:
            available = list(self.full_config.keys())
            raise ValueError(
                f"❌ Embodiment '{self.embodiment_name}' not found in {self.config_path}! "
                f"Available options: {available}"
            )
        return self.full_config[self.embodiment_name]

    def get_config(self) -> dict:
        return self.embodiment_config

    def get_cameras(self) -> dict:
        return self.embodiment_config.get('cameras', {})

    def _is_articulation_group(self, entry: dict) -> bool:
        return isinstance(entry, dict) and 'components' in entry

    def get_articulations(self) -> dict:
        raw_robots = self.embodiment_config.get('robots', {})
        return {
            name: cfg for name, cfg in raw_robots.items()
            if self._is_articulation_group(cfg)
        }

    def get_robots(self) -> dict:
        raw_robots = self.embodiment_config.get('robots', {})
        flattened = {}

        for name, cfg in raw_robots.items():
            if self._is_articulation_group(cfg):
                group_name = name
                group_level_keys = {k: v for k, v in cfg.items() if k != 'components'}
                for comp_name, comp_cfg in cfg['components'].items():
                    merged = dict(comp_cfg)
                    merged['_group'] = group_name
                    for k, v in group_level_keys.items():
                        merged.setdefault(k, v)
                    flattened[comp_name] = merged
            else:
                flattened[name] = cfg
        self._validate_joint_indices(flattened)
        return flattened

    def _validate_joint_indices(self, flattened: dict):
        for name, cfg in flattened.items():
            if cfg.get('type') not in ('manipulator', 'end-effector'):
                continue
            joint_names = cfg.get('joint_names')
            joint_indices = cfg.get('joint_indices')
            if joint_indices is None:
                raise ValueError(
                    f"❌ '{name}' in embodiment '{self.embodiment_name}' is missing joint_indices"
                )
            if len(joint_indices) != len(joint_names):
                raise ValueError(
                    f"❌ '{name}': joint_names has {len(joint_names)}, joint_indices has {len(joint_indices)}"
                )
        
    def get_total_dofs(self) -> int:
        robots = self.get_robots()
        return sum(len(robot.get('joint_names', [])) for robot in robots.values())

    def resolve_target(self, component_cfg: dict, muscle_option: str) -> ResolvedTopic | None:
        """
        Resolves one component's 'target' field for a specific muscle_option.

        Accepted shapes for component_cfg['target']:
          - absent / None                 -> no target, ever. Returns None.
          - "topic/string"                -> same topic for every muscle_option, JointState assumed.
          - { MUSCLE_OPTION: value, ... } -> per-backend lookup, where `value` is:
                - None                    -> explicitly not wired for this backend. Returns None.
                - "topic/string"          -> JointState assumed.
                - {topic, msg_type}       -> explicit message type (e.g. JointTrajectory).

        Returns None whenever there's nothing to wire up. Callers MUST skip
        creating a publisher/subscription in that case -- never guess a
        fallback topic name.
        """
        raw = component_cfg.get('target')
        if raw is None:
            return None

        if isinstance(raw, str):
            return ResolvedTopic(topic=raw)

        if isinstance(raw, dict):
            # A bare {topic, msg_type} dict shared across all muscle_options
            # (no muscle_option keys present) vs. a per-muscle_option map.
            if 'topic' in raw and muscle_option.upper() not in raw:
                return ResolvedTopic(topic=raw['topic'], msg_type=raw.get('msg_type', DEFAULT_MSG_TYPE))

            entry = raw.get(muscle_option.upper())
            if entry is None:
                return None
            if isinstance(entry, str):
                return ResolvedTopic(topic=entry)
            if isinstance(entry, dict):
                if 'topic' not in entry:
                    raise ValueError(
                        f"❌ target entry for muscle_option '{muscle_option}' missing 'topic': {entry}"
                    )
                return ResolvedTopic(topic=entry['topic'], msg_type=entry.get('msg_type', DEFAULT_MSG_TYPE))
            raise ValueError(f"❌ Unrecognized target entry shape: {entry}")

        raise ValueError(f"❌ Unrecognized 'target' shape for component: {raw}")

    def resolve_states(self, group_cfg: dict, group_name: str) -> str | None:
        """
        Resolves a group's 'states' topic.
          - key absent               -> conventional default: /synapse/joint_states/{group_name}
          - key present but null     -> explicitly no states topic; skip wiring it.
          - key present as a string  -> use as-is.
        Independent of resolve_target -- a group can skip states while its
        components still have targets, or vice versa.
        """
        if 'states' not in group_cfg:
            return f'/synapse/joint_states/{group_name}'
        return group_cfg['states']

    def get_component_targets(self, muscle_option: str) -> dict:
        """component_name -> ResolvedTopic | None, for the given muscle_option."""
        return {
            name: self.resolve_target(cfg, muscle_option)
            for name, cfg in self.get_robots().items()
        }