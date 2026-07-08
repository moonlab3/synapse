from .adapters.vla_adapter import VLAAdapter
from .adapters.manual_adapter import ManualAdapter

class BrainSelector:

    @staticmethod
    def get_brain(brain_type):
        try:
            match brain_type:
                case 'MANUAL':
                    return ManualAdapter()
                case 'VLA':
                    return VLAAdapter()
                case _:
                    return None
        except Exception as e:
            print(f"❌❌ Error initializing brain of type {brain_type}: {e}")
            return None