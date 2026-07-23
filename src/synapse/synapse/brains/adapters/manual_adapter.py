import jax

from ..base_brain_adapter import BaseBrainAdapter
from sensor_msgs.msg import JointState
import jax.numpy as jnp
import jaxlie
import jaxls
import jax_dataclasses as jdc
import pyroki as pk
import yourdfpy
from robot_descriptions.loaders.yourdfpy import load_robot_description
from loguru import logger

logger.disable("jaxls")

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
        pk.costs.limit_constraint(
            robot,
            joint_var,
        )
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

class ManualAdapter(BaseBrainAdapter):
    def __init__(self, terminal, node_name="manual_adapter", parameter_overrides=None):
        super().__init__(terminal, node_name, parameter_overrides)
        self.step_size = 0.05
        # Conceptual state holding for FK/IK solvers
        self.current_eef_pose = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0] 

        if self.robot_description:
            self.description_name = self.get_parameter('description_name').value
            urdf = load_robot_description(self.description_name)
        else:
            urdf_path = self.get_parameter('urdf_path').value
            urdf = yourdfpy.URDF.load(urdf_path)


        self.robot = pk.Robot.from_urdf(urdf=urdf)
        self.eef_frame = self.get_parameter('eef_frame').value

        dummy_se3 = jaxlie.SE3.identity()
        dummy_idx = jnp.array(self.robot.links.names.index(self.eef_frame), dtype=jnp.int32)
        dummy_q = jnp.zeros(self.robot.joints.num_actuated_joints)
        
        _ = solve_ik_jit(self.robot, dummy_se3, dummy_idx, dummy_q) 
        print("⚡ JAX IK Compiler ready. Solving at microseconds.")
        print("Manual Mode Key input: eef pose +x [s], +y [d], +z[f], +roll[w], +pitch[e], +yaw[r]")
        print("                       eef pose -x [S], -y [D], -z[F], -roll[W], -pitch[E], -yaw[R]")

    def _se3_to_list(self, se3: jaxlie.SE3) -> list:
        """Convert jaxlie.SE3 to a list of [x, y, z, roll, pitch, yaw]."""
        translation = se3.translation()
        rotation = se3.rotation().as_rpy_radians()
        return [float(translation[0]), float(translation[1]), float(translation[2]), float(rotation[0]), float(rotation[1]), float(rotation[2])]
    def _list_to_se3(self, pose_list: list) -> jaxlie.SE3:
        """Convert a list of [x, y, z, roll, pitch, yaw] to jaxlie.SE3."""
        translation = jnp.array(pose_list[:3])
        rotation = jaxlie.SO3.from_rpy_radians(pose_list[3], pose_list[4], pose_list[5])
        return jaxlie.SE3.from_rotation_and_translation(rotation, translation)

    def _forward_kinematics(self, joint_state: JointState) -> list:
        if not joint_state or not joint_state.position:
            return self.current_eef_pose.copy()

        # 1. Force exact shape matching for PyRoki (Panda = 8 actuated joints)
        expected_dofs = self.robot.joints.num_actuated_joints
        raw_q = jnp.array(joint_state.position)
        
        if raw_q.shape[0] > expected_dofs:
            q_fk = raw_q[:expected_dofs]
        elif raw_q.shape[0] < expected_dofs:
            q_fk = jnp.pad(raw_q, (0, expected_dofs - raw_q.shape[0]))
        else:
            q_fk = raw_q
        
        # 2. PyRoki FK query with the correctly sized array
        all_link_poses = self.robot.forward_kinematics(q_fk)
        
        # Find the index of the end-effector frame
        eef_idx = self.robot.links.names.index(self.eef_frame)
        
        # Extract the 7D pose (wxyz_xyz) for the specific link
        eef_pose_wxyz_xyz = all_link_poses[eef_idx]
        
        # Convert the 7D array back into a jaxlie.SE3 object
        eef_se3 = jaxlie.SE3(eef_pose_wxyz_xyz)
        
        self.current_eef_pose = self._se3_to_list(eef_se3)
        return self.current_eef_pose.copy()

    def _inverse_kinematics(self, eef_pose: list, original_joint_state: JointState) -> JointState:
        if not original_joint_state or not original_joint_state.position:
            # self.get_logger().warn("Original joint state is missing or empty. Returning default joint state.")
            return JointState(name=[], position=[])

        target_se3 = self._list_to_se3(eef_pose)


        expected_dofs = self.robot.joints.num_actuated_joints
        raw_q = jnp.array(original_joint_state.position)
        
        if raw_q.shape[0] > expected_dofs:
            ik_seed_q = raw_q[:expected_dofs]
        elif raw_q.shape[0] < expected_dofs:
            ik_seed_q = jnp.pad(raw_q, (0, expected_dofs - raw_q.shape[0]))
        else:
            ik_seed_q = raw_q

        # 2. Get numerical index of the End-Effector frame
        eef_idx = self.robot.links.names.index(self.eef_frame)
        target_link_idx_jax = jnp.array(eef_idx, dtype=jnp.int32)
        
        # 3. ⚡ Call the JIT-compiled binary safely (Guaranteed to be size 8)
        optimized_q = solve_ik_jit(
            self.robot,
            target_se3,
            target_link_idx_jax,
            ik_seed_q
        )

        # 4. Reconstruct the full message 
        # (Isaac sends 9 DOFs, we solved 8. We merge them so the ROS2 message is perfectly formatted)
        final_position = list(original_joint_state.position) 
        
        for i in range(min(len(optimized_q), len(final_position))):
            final_position[i] = float(optimized_q[i])
        
        out_msg = JointState()
        out_msg.name = original_joint_state.name
        out_msg.position = final_position
        
        return out_msg
        
    def _format_for_policy(self, obs_history: list) -> dict:
        # Extract the most recent observation from the buffer history
        latest_obs = obs_history[-1]
        joint_msg = latest_obs.get("joint")
        command = latest_obs.get("command")
        
        # Forward Kinematics: Convert joint angles to End-Effector Pose
        eef_pose = self._forward_kinematics(joint_msg)
        # eef_pose = self.current_eef_pose.copy()  # Use the last known EEF pose if FK is not applied
        
        return {
            "command": command, 
            "eef_pose": eef_pose, 
            "original_joint": joint_msg
        }

    def _communicate_with_policy(self, formatted_obs: dict) -> dict:
        # Manual adapter acts as its own policy (bypassing ZMQ)
        command = formatted_obs.get("command")
        eef_pose = formatted_obs.get("eef_pose")

        if command is not None:
            if len(command) > 2:
                pass
            else:
                match command:
                    case 's': eef_pose[0] += self.step_size
                    case 'S': eef_pose[0] -= self.step_size
                    case 'd': eef_pose[1] += self.step_size
                    case 'D': eef_pose[1] -= self.step_size
                    case 'f': eef_pose[2] += self.step_size
                    case 'F': eef_pose[2] -= self.step_size
                    case 'w': eef_pose[3] += self.step_size
                    case 'W': eef_pose[3] -= self.step_size
                    case 'e': eef_pose[4] += self.step_size
                    case 'E': eef_pose[4] -= self.step_size
                    case 'r': eef_pose[5] += self.step_size
                    case 'R': eef_pose[5] -= self.step_size
                    case _:
                        pass

        formatted_obs["eef_pose"] = eef_pose
        self.current_eef_pose = eef_pose
        return formatted_obs

    def _format_for_muscle(self, raw_action: dict) -> list:
        eef_pose = raw_action.get("eef_pose")
        original_joint = raw_action.get("original_joint")
        
        # Inverse Kinematics: Convert target EEF pose back to JointState
        target_joint_state = self._inverse_kinematics(eef_pose, original_joint)
        
        # Return as an Action Chunk (list) so ActionChunkBuffer can process it
        return [target_joint_state]