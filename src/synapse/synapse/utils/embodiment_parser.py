import yaml
import os
from pathlib import Path

class EmbodimentParser:
    # Centralized parser for the Synapse architecture. 
    # Supplies hardware topology to both Brain Adapters and Muscle Nodes.

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