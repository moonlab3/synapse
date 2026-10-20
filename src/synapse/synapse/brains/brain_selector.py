import importlib


class BrainSelector:
    # brain_type -> (module under .adapters, class, default name). A new adapter is one line.
    # Modules are imported on first use, so only the types in brain_registry are ever loaded
    # (GR00T, RL etc. pull in heavy or path-dependent packages).
    BRAINS = {
        'MANUAL': ('manual_adapter', 'ManualAdapter', 'manual_adapter'),
        'GR00T': ('gr00t_adapter', 'GR00TAdapter', 'gr00t_adapter'),
        'RANDOM': ('random_brain_adapter', 'RandomBrainAdapter', 'random_brain_adapter'),
        'DUMMY': ('dummy_brain_adapter', 'DummyBrainAdapter', 'dummy_brain_adapter'),
        'ROSBAG': ('rosbag_adapter', 'RosbagAdapter', 'rosbag_adapter'),
        'MOVE': ('move_adapter', 'MoveAdapter', 'move_adapter'),
        'AVP': ('avp_adapter', 'AvpTeleopAdapter', 'avp_adapter'),
        'RL': ('rl_adapter', 'RLAdapter', 'rl_adapter'),
    }

    @staticmethod
    def get_brain(terminal, brain_type: str, ros_node, params: dict, name: str = None):
        entry = BrainSelector.BRAINS.get(brain_type.upper())
        if entry is None:
            return None

        module_name, class_name, default_name = entry
        try:
            module = importlib.import_module(f'.adapters.{module_name}', __package__)
            brain_cls = getattr(module, class_name)
            return brain_cls(terminal, name or default_name, params, ros_node)
        except Exception as e:
            raise ValueError(f"❌❌ Error initializing brain of type {brain_type}: {e}") from e
