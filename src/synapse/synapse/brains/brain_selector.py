from .adapters.dummy_brain_adapter import DummyBrainAdapter
from .adapters.gr00t_adapter import GR00TAdapter
from .adapters.manual_adapter import ManualAdapter
from .adapters.move_adapter import MoveAdapter
from .adapters.random_brain_adapter import RandomBrainAdapter
from .adapters.rosbag_adapter import RosbagAdapter


class BrainSelector:
    # brain_type -> (class, default node name). A new adapter is one line.
    BRAINS = {
        'MANUAL': (ManualAdapter, 'manual_adapter'),
        'GR00T': (GR00TAdapter, 'gr00t_adapter'),
        'RANDOM': (RandomBrainAdapter, 'random_brain_adapter'),
        'DUMMY': (DummyBrainAdapter, 'dummy_brain_adapter'),
        'ROSBAG': (RosbagAdapter, 'rosbag_adapter'),
        'MOVE': (MoveAdapter, 'move_adapter'),
    }

    @staticmethod
    def get_brain(terminal, brain_type: str, node_name: str = None, parameter_overrides: list = None):
        entry = BrainSelector.BRAINS.get(brain_type.upper())
        if entry is None:
            return None

        brain_cls, default_node_name = entry
        try:
            return brain_cls(
                terminal,
                node_name=node_name if node_name else default_node_name,
                parameter_overrides=parameter_overrides)
        except Exception as e:
            raise ValueError(f"❌❌ Error initializing brain of type {brain_type}: {e}")
