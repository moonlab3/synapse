import yaml
import py_trees

class RunAdapterBehavior(py_trees.behaviour.Behaviour):
    def __init__(self, name, adapter_key, bt_node_reference):
        super().__init__(name)
        self.adapter_key = adapter_key
        self.node = bt_node_reference # Reference to SynapseBTNode to access buffers

    def update(self):
        # This is called by the py_trees tick (~10Hz)

        # 1. Check if inference is already running in the background
        if self.node.inference_future is not None:
            if self.node.inference_future.done():
                # Inference finished! Push it to the 100Hz buffer
                action_chunk = self.node.inference_future.result()
                if action_chunk:
                    self.node.action_buffer.update_chunk(action_chunk)
                
                self.node.inference_future = None
                return py_trees.common.Status.RUNNING 
            else:
                # Still thinking... tell the tree we are busy
                return py_trees.common.Status.RUNNING

        # 2. If idle, trigger the specific adapter
        active_adapter = self.node.adapters[self.adapter_key]
        historical_obs = list(self.node.obs_buffer)
        
        self.node.inference_future = self.node.inference_executor.submit(
            active_adapter.infer, historical_obs
        )
        
        return py_trees.common.Status.RUNNING


class TreeBuilder:
    def __init__(self, synapse_node):
        # We pass the main ROS node so the BT actions can access the adapters and buffers
        self.node = synapse_node 

    def build_from_file(self, file_path):
        with open(file_path, 'r') as f:
            config = yaml.safe_load(f)
        
        return self._build_node(config['root'])

    def _build_node(self, node_config):
        node_type = node_config.get('type')
        node_name = node_config.get('name', 'unnamed_node')

        # 1. Build Composites (Branches)
        if node_type == "Sequence":
            bt_node = py_trees.composites.Sequence(name=node_name, memory=True)
            for child_config in node_config.get('children', []):
                bt_node.add_child(self._build_node(child_config)) # Recursive call
            return bt_node

        elif node_type == "Selector":
            bt_node = py_trees.composites.Selector(name=node_name, memory=False)
            for child_config in node_config.get('children', []):
                bt_node.add_child(self._build_node(child_config)) # Recursive call
            return bt_node

        # 2. Build Leaves (Actions)
        elif node_type == "Action":
            return RunAdapterBehavior(
                name=node_name,
                adapter_key=node_config['adapter'],
                success_cond=node_config['success_condition'],
                failure_cond=node_config.get('failure_condition', None),
                bt_node_reference=self.node
            )
        
        else:
            raise ValueError(f"Unknown node type: {node_type}")