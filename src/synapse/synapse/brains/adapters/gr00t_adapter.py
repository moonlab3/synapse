from ..base_brain_adapter import BaseBrainAdapter
import numpy as np
import os
import sys
import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import jaxlie
import jaxls
import pyroki as pk
from sensor_msgs.msg import JointState
from loguru import logger

# Silence JAXLS and PyRoki info/debug logs
logger.disable("jaxls")
logger.disable("pyroki")
import yourdfpy
from robot_descriptions.loaders.yourdfpy import load_robot_description

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
# ⚡ JIT-Compiled Kinematics Solvers
# ==========================================
@jdc.jit
def solve_ik_jit(
    robot: pk.Robot,
    target_se3: jaxlie.SE3,
    target_link_idx: jax.Array,
    initial_q: jax.Array,
) -> jax.Array:
    """JIT-compiled IK solver for extreme performance."""
    joint_var = robot.joint_var_cls(0)
    joint_mask = jnp.ones(robot.joints.num_actuated_joints)

    costs = [
        pk.costs.pose_cost_analytic_jac(
            robot, 
            joint_var,
            target_se3,
            target_link_idx, 
            pos_weight=1.0,
            ori_weight=1.0,
            joint_mask=joint_mask
        ),
        pk.costs.limit_constraint(robot, joint_var)
    ]

    init_vals = jaxls.VarValues.make([joint_var.with_value(initial_q)])

    sol = (
        jaxls.LeastSquaresProblem(costs=costs, variables=[joint_var])
        .analyze()
        .solve(
            initial_vals=init_vals, 
            verbose=False,
            linear_solver="dense_cholesky",
            trust_region=jaxls.TrustRegionConfig(lambda_initial=1.0)
        )
    )
    return sol[joint_var]

# ==========================================
# 🧠 VLA Brain Adapter (GR00T)
# ==========================================
class GR00TAdapter(BaseBrainAdapter):
    def __init__(self):
        super().__init__('gr00t_adapter')
        self.current_eef_pose = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0] 
        
        # 1. Initialize GR00T Policy Client
        self.declare_parameter('policy_ip', "0.0.0.0")
        self.declare_parameter('policy_port', "8888")
        ip = self.get_parameter('policy_ip').value
        port = self.get_parameter('policy_port').value
        
        if PolicyClient:
            self.client = PolicyClient(ip, port)
            if not self.client.ping():
                print(f"❌ PolicyClient NOT connected to GR00T server at {ip}:{port}.")
            else:
                print(f"🔌 PolicyClient initialized and connected to GR00T server at {ip}:{port}.")
        else:
            raise ValueError("NO POLICY CLIENT")
        
        if self.robot_description:
            self.declare_parameter('description_name', "panda_description")
            self.description_name = self.get_parameter('description_name').value
            urdf = load_robot_description(self.description_name)
        else:
            self.declare_parameter('urdf_path', "/home/rog-sf/ws/synapse_ws/src/synapse/resources/fairino5_v6.urdf")
            urdf_path = self.get_parameter('urdf_path').value
            urdf = yourdfpy.URDF.load(urdf_path)


        self.robot = pk.Robot.from_urdf(urdf=urdf)
        self.declare_parameter('eef_frame', "panda_hand")
        self.eef_frame = self.get_parameter('eef_frame').value

        # 3. Warm up JAX Compiler
        dummy_se3 = jaxlie.SE3.identity()
        dummy_idx = jnp.array(self.robot.links.names.index(self.eef_frame), dtype=jnp.int32)
        dummy_q = jnp.zeros(self.robot.joints.num_actuated_joints)
        _ = solve_ik_jit(self.robot, dummy_se3, dummy_idx, dummy_q) 
        print("⚡ JAX IK Compiler ready.")

    def _pad_joints(self, q_array: jnp.ndarray) -> jnp.ndarray:
        expected_dofs = self.robot.joints.num_actuated_joints
        current_dofs = q_array.shape[0]
        
        if current_dofs < expected_dofs:
            padding = jnp.zeros(expected_dofs - current_dofs)
            return jnp.concatenate([q_array, padding])
        elif current_dofs > expected_dofs:
            return q_array[:expected_dofs]
        return q_array

    def _se3_to_list(self, se3: jaxlie.SE3) -> list:
        translation = se3.translation()
        rpy = se3.rotation().as_rpy_radians() 
        return [
            float(translation[0]), float(translation[1]), float(translation[2]), 
            float(rpy[0]), float(rpy[1]), float(rpy[2])
        ]

    def _list_to_se3(self, pose: list) -> jaxlie.SE3:
        translation = jnp.array(pose[:3])
        rotation = jaxlie.SO3.from_rpy_radians(pose[3], pose[4], pose[5])
        return jaxlie.SE3.from_rotation_and_translation(rotation, translation)

    def _forward_kinematics(self, joint_state: JointState) -> list:
        if not joint_state or not joint_state.position:
            return self.current_eef_pose.copy()

        q = jnp.array(joint_state.position)
        q_padded = self._pad_joints(q)
        
        all_link_poses = self.robot.forward_kinematics(q_padded)
        eef_idx = self.robot.links.names.index(self.eef_frame)
        eef_pose_wxyz_xyz = all_link_poses[eef_idx]
        eef_se3 = jaxlie.SE3(eef_pose_wxyz_xyz)
        
        self.current_eef_pose = self._se3_to_list(eef_se3)
        return self.current_eef_pose.copy()

    def _format_for_policy(self, obs_history: list) -> dict:
        latest_obs = obs_history[-1]
        joint_msg = latest_obs.get("joint")
        command = latest_obs.get("command")
        image = latest_obs.get("image") 
        
        # 1. Get Cartesian Pose
        x, y, z, roll, pitch, yaw = self._forward_kinematics(joint_msg)
        
        # 2. Get Gripper State (Assuming index 7 is the finger, max width ~0.08m)
        q = joint_msg.position if joint_msg.position else [0]*8
        current_width = q[7] * 2 if len(q) > 7 else 0.0
        normalized_gripper = np.clip(current_width / 0.08, 0.0, 1.0)

        # 3. Construct the GR00T Dictionary
        state = {
            "x": np.array([[[x]]], dtype=np.float32),
            "y": np.array([[[y]]], dtype=np.float32),
            "z": np.array([[[z]]], dtype=np.float32),
            "roll": np.array([[[roll]]], dtype=np.float32),
            "pitch": np.array([[[pitch]]], dtype=np.float32),
            "yaw": np.array([[[yaw]]], dtype=np.float32),
            "pad": np.array([[[0.0]]], dtype=np.float32),
            "gripper": np.array([[[normalized_gripper]]], dtype=np.float32)
        }

        # clean_command = command.replace("CMD:", "").strip() if command else ""
        if command is None:
            # print("GR00T Adapter: command is None")
            clean_command = ""
        else:
            clean_command = command

        if image is not None:
            formatted_image = image[np.newaxis, np.newaxis, :, :, :]
        else:
            formatted_image = np.zeros((1, 1, 256, 256, 3), dtype=np.uint8)

        obs = {
            "annotation.human.action.task_description": [clean_command],
            "video.image_0": formatted_image,
        }

        for item in state:
            obs[f"state.{item}"] = state[item]

        # Stash original state so we can calculate deltas in IK
        obs["_internal_base_pose"] = [x, y, z, roll, pitch, yaw]
        obs["_internal_original_joint"] = joint_msg

        return obs
    def _communicate_with_policy(self, formatted_obs: dict) -> dict:
        # Extract internal context before passing to GR00T
        base_pose = formatted_obs.pop("_internal_base_pose")
        original_joint = formatted_obs.pop("_internal_original_joint")

        if self.client:
            try:
                raw_result = self.client.get_action(formatted_obs)
            except Exception:
                pass
            
            # ⚡ THE FIX: Extract the actual dictionary from the GR00T return tuple
            if isinstance(raw_result, (list, tuple)):
                action_dict = raw_result[0]
            else:
                action_dict = raw_result
                
            # print(f"GR00T_ADAPTER: Extracted keys: {action_dict.keys()}")
        else:
            print("NO CLIENT")
            action_dict = {} # Fallback if client failed to load

        return {
            "action_chunk": action_dict, 
            "base_pose": base_pose,
            "original_joint": original_joint
        }

    def _format_for_muscle(self, raw_data: dict) -> list:
        action_dict = raw_data.get("action_chunk") 
        base_pose = raw_data.get("base_pose") # [x, y, z, r, p, y]
        original_joint_state = raw_data.get("original_joint")
        
        if not action_dict or 'action.x' not in action_dict or not original_joint_state:
            return []

        eef_idx = self.robot.links.names.index(self.eef_frame)
        target_link_idx_jax = jnp.array(eef_idx, dtype=jnp.int32)

        # 1. Setup the starting seed from current reality
        current_q = jnp.array(original_joint_state.position)
        seed_q_padded = self._pad_joints(current_q)
        original_length = len(original_joint_state.position)

        action_chunk = []
        num_chunks = action_dict['action.x'].shape[1]

        # 2. Iterate through the GR00T chunk
        for step_index in range(num_chunks):
            # Extract deltas
            dx = action_dict['action.x'][0, step_index, 0].item()
            dy = action_dict['action.y'][0, step_index, 0].item()
            dz = action_dict['action.z'][0, step_index, 0].item()
            droll = action_dict['action.roll'][0, step_index, 0].item()
            dpitch = action_dict['action.pitch'][0, step_index, 0].item()
            dyaw = action_dict['action.yaw'][0, step_index, 0].item()
            gripper_cmd = action_dict['action.gripper'][0, step_index, 0].item()

            # Apply deltas to the base pose
            target_eef_pose = [
                base_pose[0] + dx,
                base_pose[1] + dy,
                base_pose[2] + dz,
                base_pose[3] + droll,
                base_pose[4] + dpitch,
                base_pose[5] + dyaw,
            ]
            
            target_se3 = self._list_to_se3(target_eef_pose)
            
            # ⚡ JIT IK Solve using the PREVIOUS step's output as the seed
            optimized_q = solve_ik_jit(
                self.robot,
                target_se3,
                target_link_idx_jax,
                seed_q_padded # Warm start
            )
            
            seed_q_padded = optimized_q
            
            # Map gripper (0.0 to 1.0) back to Franka finger limits (~0.04m)
            finger_target = float(np.clip(gripper_cmd, 0.0, 1.0) * 0.04)
            
            # Convert JAX array to standard list
            final_positions = optimized_q.tolist()
            
            # If the robot has at least 8 joints, overwrite the finger joint
            if len(final_positions) >= 8:
                final_positions[7] = finger_target

            # Package into ROS2 Message
            out_msg = JointState()
            out_msg.name = original_joint_state.name
            out_msg.position = final_positions[:original_length]

            action_chunk.append(out_msg)

        return action_chunk