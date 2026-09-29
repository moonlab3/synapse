from ..base_brain_adapter import BaseBrainAdapter
from ..base_brain_adapter import InferenceOption
import numpy as np
import os
import sys
from sensor_msgs.msg import JointState
from synapse.utils.embodiment_parser import EmbodimentParser
from synapse.utils.kinematics import ArmKinematics

# ==========================================
# ⚡ GR00T Policy Client Setup
# ==========================================
groot_path = os.path.abspath("/home/rog-sf/vla/Isaac-GR00T")
if groot_path not in sys.path:
    sys.path.append(groot_path)

try:
    from gr00t.policy.server_client import PolicyClient
except ImportError:
    print("⚠️ WARNING: Could not import GR00T PolicyClient. Ensure the path is correct.")
    PolicyClient = None

# ==========================================
# 🧠 VLA Brain Adapter (GR00T)
# ==========================================
class GR00TAdapter(BaseBrainAdapter):
    def __init__(self, terminal, node_name="gr00t_adapter", parameter_overrides=None):
        super().__init__(terminal, node_name, parameter_overrides)
        
        # 1. Initialize GR00T Policy Client
        ip = self.get_parameter(f"{self.get_name()}.policy_ip").value
        port = self.get_parameter(f"{self.get_name()}.policy_port").value
        self.default_command = self.get_parameter(f"{self.get_name()}.default_command").value

        self.client = PolicyClient(ip, port)
        if not self.client.ping():
            print(f"❌ PolicyClient NOT connected to GR00T server at {ip}:{port}.")
        else:
            print(f"🔌 PolicyClient initialized and connected to GR00T server at {ip}:{port}.")
        
        parser = EmbodimentParser(self.embodiment_name)
        self.robots_cfg = parser.get_robots()

        self.kin = ArmKinematics(self.robots_cfg)
        self.current_eef_poses = self.kin.current_eef_poses  # the BT pose checks read this

        print("⚡ JAX IK Compiler ready for all manipulators.")
        self.terminal = terminal
        self.terminal.debug("gr00t adapter init done")

    def _communicate_with_policy(self, formatted_obs: dict) -> dict:
        # Extract internal context before passing to GR00T
        base_poses = formatted_obs.pop("_internal_base_poses", {})
        original_joints = formatted_obs.pop("_internal_original_joints", {})

        action_dict = {}
        if self.client:
            try:
                raw_result = self.client.get_action(formatted_obs)
                if isinstance(raw_result, (list, tuple)):
                    action_dict = raw_result[0]
                elif isinstance(raw_result, dict):
                    action_dict = raw_result
            except Exception as e:
                pass # Fail silently, let the chunk return empty
                
        return {
            "action_chunk": action_dict, 
            "base_poses": base_poses,
            "original_joints": original_joints
        }

    def _format_for_policy(self, obs_history: list, inference_option: InferenceOption) -> dict:
        latest_obs = obs_history[-1]
        joints_dict = latest_obs.get("joints", {})
        images_dict = latest_obs.get("images", {})
        
        if inference_option.default_command:
            command = self.default_command
        else:
            raw_command = latest_obs.get("command")
            command = raw_command if isinstance(raw_command, str) else self.default_command
        
        # 1. Get Cartesian Poses for all manipulators
        eef_poses = self.kin.forward_kinematics(joints_dict)
        
        state = {}
        is_primary_arm = True
        
        # 2. Iterate through manipulators to build state tensors
        for name, cfg in self.robots_cfg.items():
            if cfg.get('type') == 'manipulator' and name in eef_poses:
                pose = eef_poses[name]
                joint_msg = joints_dict.get(name)
                
                # Get Gripper State (Assuming index 7 is the finger, max width ~0.08m)
                q = joint_msg.position if joint_msg and joint_msg.position else [0]*8
                current_width = q[7] * 2 if len(q) > 7 else 0.0
                normalized_gripper = np.clip(current_width / 0.08, 0.0, 1.0)
                
                # ⚡ THE FIX: Guarantee 'state.x' exists for the primary arm.
                # Only namespace secondary arms to prevent overwriting the dictionary.
                prefix = "" if is_primary_arm else f"{name}."
                
                state[f"{prefix}x"] = np.array([[[pose[0]]]], dtype=np.float32)
                state[f"{prefix}y"] = np.array([[[pose[1]]]], dtype=np.float32)
                state[f"{prefix}z"] = np.array([[[pose[2]]]], dtype=np.float32)
                state[f"{prefix}roll"] = np.array([[[pose[3]]]], dtype=np.float32)
                state[f"{prefix}pitch"] = np.array([[[pose[4]]]], dtype=np.float32)
                state[f"{prefix}yaw"] = np.array([[[pose[5]]]], dtype=np.float32)
                state[f"{prefix}pad"] = np.array([[[0.0]]], dtype=np.float32)
                state[f"{prefix}gripper"] = np.array([[[normalized_gripper]]], dtype=np.float32)
                
                is_primary_arm = False

        # 3. Construct the GR00T Dictionary
        obs = {
            "annotation.human.action.task_description": [command],
        }

        # Process multiple camera feeds dynamically
        if images_dict:
            for idx, (cam_name, img) in enumerate(images_dict.items()):
                if img is not None:
                    obs[f"video.image_{idx}"] = img[np.newaxis, np.newaxis, :, :, :]
                else:
                    obs[f"video.image_{idx}"] = np.zeros((1, 1, 256, 256, 3), dtype=np.uint8)
        else:
            obs["video.image_0"] = np.zeros((1, 1, 256, 256, 3), dtype=np.uint8)

        for item in state:
            obs[f"state.{item}"] = state[item]

        # Stash original states so we can calculate deltas in IK safely
        obs["_internal_base_poses"] = eef_poses
        obs["_internal_original_joints"] = joints_dict

        return obs


    def _format_for_muscle(self, raw_data: dict) -> list:
        action_dict = raw_data.get("action_chunk", {})
        base_poses = raw_data.get("base_poses", {})
        original_joints = raw_data.get("original_joints", {})
        
        if not action_dict or not original_joints:
            return []

        # Find the chunk length by sampling the first tensor dimension
        num_chunks = 0
        for val in action_dict.values():
            if hasattr(val, 'shape') and len(val.shape) > 1:
                num_chunks = val.shape[1]
                break

        if num_chunks == 0:
            return []

        action_chunk = []
        
        # 1. Setup the warm-start seeds from current reality for all manipulators
        seed_qs = {}
        for name, cfg in self.robots_cfg.items():
            if cfg.get('type') == 'manipulator' and name in original_joints:
                seed_qs[name] = self.kin.seed_from_positions(name, original_joints[name].position)

        # 2. Iterate through the GR00T chunk sequence
        for step_index in range(num_chunks):
            step_dict = {}
            is_primary_arm = True
            
            for name, cfg in self.robots_cfg.items():
                original_js = original_joints.get(name, JointState())
                
                if cfg.get('type') == 'manipulator' and name in base_poses:
                    # ⚡ THE FIX: Match the prefixing logic used in observations
                    prefix = "action." if is_primary_arm else f"action.{name}."
                    is_primary_arm = False
                    
                    try:
                        dx = action_dict[f'{prefix}x'][0, step_index, 0].item()
                        dy = action_dict[f'{prefix}y'][0, step_index, 0].item()
                        dz = action_dict[f'{prefix}z'][0, step_index, 0].item()
                        droll = action_dict[f'{prefix}roll'][0, step_index, 0].item()
                        dpitch = action_dict[f'{prefix}pitch'][0, step_index, 0].item()
                        dyaw = action_dict[f'{prefix}yaw'][0, step_index, 0].item()
                        gripper_cmd = action_dict[f'{prefix}gripper'][0, step_index, 0].item()
                    except KeyError:
                        # Missing keys for this robot, hold position
                        step_dict[name] = original_js
                        continue

                    # Apply deltas to the base pose
                    base_pose = base_poses[name]
                    target_eef_pose = [
                        base_pose[0] + dx, base_pose[1] + dy, base_pose[2] + dz,
                        base_pose[3] + droll, base_pose[4] + dpitch, base_pose[5] + dyaw,
                    ]
                    
                    optimized_q = self.kin.solve(name, target_eef_pose, seed_qs[name])
                    seed_qs[name] = optimized_q
                    
                    # Map gripper back to hardware limits
                    finger_target = float(np.clip(gripper_cmd, 0.0, 1.0) * 0.04)

                    idx_list = self.kin.dof_indices[name].tolist()
                    original_length = len(original_js.position)
                    final_positions = list(original_js.position)

                    for local_i, global_i in enumerate(idx_list):
                        if local_i < original_length:
                            final_positions[local_i] = float(optimized_q[global_i])
                    
                    if len(final_positions) >= 8:
                        final_positions[7] = finger_target

                    # Package into ROS2 Message
                    out_msg = JointState()
                    out_msg.header = original_js.header
                    out_msg.name = original_js.name
                    out_msg.position = final_positions

                    step_dict[name] = out_msg
                else:
                    step_dict[name] = original_js

            action_chunk.append(step_dict)

        return action_chunk
