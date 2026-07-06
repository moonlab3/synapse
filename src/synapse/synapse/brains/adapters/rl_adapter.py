from ..base_brain_adapter import BaseBrainAdapter
from sensor_msgs.msg import JointState
from std_msgs.msg import ByteMultiArray
import numpy as np
import pickle
import requests # Example HTTP client for the AI Policy Server

class RLAdapter(BaseBrainAdapter):
    def __init__(self, config, checkpoint_path):
        super().__init__(config, 'RL_Adapter')
        
        # 1. AI Server Configuration
        self.declare_parameter('ai_server_url', 'http://localhost:8000/infer')
        self.ai_server_url = self.get_parameter('ai_server_url').value
        
        # 2. Local Observation Buffer (The Memory Bank)
        self.window_size = 5
        self.obs_history = []
        
        # 3. Direct IPC Subscriptions from Muscle
        # Assuming you set QoS profiles for Zero-Copy IPC in your launch file
        self.sub_joints = self.create_subscription(
            JointState, '/joint_states', self.joint_callback, 10
        )
        
        # 4. Publisher to Synapse BT Node
        self.pub_brain_output = self.create_publisher(
            ByteMultiArray, '/synapse/brain_output', 10
        )
        
        self.get_logger().info(f"🔗 RL Brain Adapter connecting to AI at {self.ai_server_url}")

    def joint_callback(self, msg: JointState):
        """
        1. Maintains the sliding window buffer directly from the muscle.
        2. Triggers the formatting and inference pipeline.
        """
        # --- Update Buffer ---
        self.obs_history.append(msg)
        if len(self.obs_history) > self.window_size:
            self.obs_history.pop(0)
            
        # Only run inference if the buffer is full
        if len(self.obs_history) == self.window_size:
            self.run_inference_pipeline()

    def run_inference_pipeline(self):
        try:
            # 1. Format & Convert (Specialization & Kinematics)
            formatted_tensor = self.format_for_policy(self.obs_history)
            
            # 2. Connect to separate AI Policy (e.g., via REST, gRPC, or ZMQ)
            action_chunk = self.query_ai_policy(formatted_tensor)
            
            # 3. Publish back to Synapse BT
            output_msg = ByteMultiArray()
            output_msg.data = pickle.dumps(action_chunk)
            self.pub_brain_output.publish(output_msg)
            
        except Exception as e:
            self.get_logger().error(f"Bridge Pipeline Failed: {e}")

    def format_for_policy(self, history: list) -> np.ndarray:
        """Runs the Forward Kinematics and flattens the history into a tensor."""
        latest_msg = history[-1]
        joints = np.array(latest_msg.position, dtype=np.float32)
        
        # (Insert your Pinocchio/Kinematics math here to get EEF pose)
        eef_pose = np.array([0.5, 0.0, 0.3, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        
        # Flatten and add batch dimension
        rl_state = np.concatenate([joints, eef_pose])
        return np.expand_dims(rl_state, axis=0)

    def query_ai_policy(self, tensor: np.ndarray) -> np.ndarray:
        """Sends the tensor to the independent AI process."""
        # This example uses HTTP, but ZMQ or gRPC is faster for local IPC
        payload = {'tensor_bytes': tensor.tobytes(), 'shape': tensor.shape}
        
        # Send to the pure Python/PyTorch server
        response = requests.post(self.ai_server_url, data=payload)
        response.raise_for_status()
        
        # Reconstruct the returned action chunk
        result_bytes = response.content
        # Assuming the policy returns a (5, 7) chunk of float32
        action_chunk = np.frombuffer(result_bytes, dtype=np.float32).reshape(5, 7)
        
        return action_chunk