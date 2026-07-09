from .adapters.gr00t_adapter import GR00TAdapter
from .adapters.manual_adapter import ManualAdapter

class BrainSelector:

    @staticmethod
    def get_brain(brain_type):
        try:
            match brain_type:
                case 'MANUAL':
                    return ManualAdapter()
                case 'GR00T':
                    return GR00TAdapter()
                case _:
                    return None
        except Exception as e:
            raise ValueError(f"❌❌ Error initializing brain of type {brain_type}: {e}")