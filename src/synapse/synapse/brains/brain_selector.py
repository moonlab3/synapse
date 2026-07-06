from .adapters.rl_adapter import RLAdapter
from .adapters.vla_adapter import VLAAdapter
from .adapters.manual_adapter import ManualAdapter

class BrainSelector:

    @staticmethod
    def get_brain(config):
        brain_type = config.get('brain_type')
        checkpoint = config.get('checkpoint_path')

        try:
            match brain_type:
                case 'MANUAL':
                    return ManualAdapter()
                case 'RL':
                    return RLAdapter(checkpoint)
                case 'VLA':
                    return VLAAdapter(checkpoint)
                case _:
                    return None
        except Exception as e:
            print(f"❌❌ Error initializing brain of type {brain_type}: {e}")
            return None