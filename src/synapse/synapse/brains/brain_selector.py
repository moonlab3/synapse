from .adapters.gr00t_adapter import GR00TAdapter
from .adapters.manual_adapter import ManualAdapter
from .adapters.dummy_brain_adapter import DummyBrainAdapter

class BrainSelector:

    @staticmethod
    def get_brain(terminal, brain_type: str, node_name: str=None, parameter_overrides: list = None):

        type = brain_type.upper()
        try:
            match type:
                case 'MANUAL':
                    _node = node_name if node_name else "manual_adapter"
                    return ManualAdapter(
                        terminal,
                        node_name=_node,
                        parameter_overrides=parameter_overrides)
                case 'GR00T':
                    _node = node_name if node_name else "gr00t_adapter"
                    return GR00TAdapter(
                        terminal,
                        node_name=_node,
                        parameter_overrides=parameter_overrides)
                case 'DUMMY':  # <-- Add the DUMMY case
                    _node = node_name if node_name else "dummy_brain_adapter"
                    return DummyBrainAdapter(
                        terminal,
                        node_name=_node,
                        parameter_overrides=parameter_overrides)
                case _:
                    return None
        except Exception as e:
            raise ValueError(f"❌❌ Error initializing brain of type {brain_type}: {e}")