import jax

from ..base_brain_adapter import BaseBrainAdapter
from ..base_brain_adapter import InferenceOption
from sensor_msgs.msg import JointState
import jax.numpy as jnp
import jaxlie
import jaxls
import jax_dataclasses as jdc
import pyroki as pk
import yourdfpy
from robot_descriptions.loaders.yourdfpy import load_robot_description
from loguru import logger
from synapse.utils.embodiment_parser import EmbodimentParser

logger.disable("jaxls")

@jdc.jit
def solve_ik_jit(
    robot: pk.Robot,
    target_se3: jaxlie.SE3,
    target_link_idx: jax.Array,
    initial_q: jax.Array,
    joint_mask: jax.Array
    ) -> jax.Array:
    """JIT-compiled IK solver for extreme performance."""
    joint_var = robot.joint_var_cls(0)
    # joint_mask = jnp.ones(robot.joints.num_actuated_joints)

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
        self.current_eef_poses = {}
        self.current_hand_joints = {}
        self.terminal = terminal

        parser = EmbodimentParser(self.embodiment_name)
        self.manipulator_step_size = parser.get_config().get('manipulator_step_size')
        self.eef_step_size = parser.get_config().get('eef_step_size')
        self.robots_cfg = parser.get_robots()

        dummy_se3 = jaxlie.SE3.identity()
        self.robots = {}
        self.eef_frame = {}
        self.dof_indices = {}
        self.dof_mask = {}

        for name, cfg in self.robots_cfg.items():
            if cfg.get('type') == 'manipulator':
                if cfg.get('yourdfpy_description'):
                    urdf = load_robot_description(cfg.get('description_name'))
                else:
                    urdf = yourdfpy.URDF.load(cfg.get('urdf_path'))
                self.robots[name] = pk.Robot.from_urdf(urdf=urdf)

                self.eef_frame[name] = cfg.get('eef_frame')
                expected_dofs = self.robots[name].joints.num_actuated_joints

                joint_indices = cfg.get('joint_indices')

                self.dof_indices[name] = jnp.array(joint_indices, dtype = jnp.int32)
                self.dof_mask[name] = jnp.zeros(expected_dofs).at[self.dof_indices[name]].set(1.0)

                self.terminal.wait_debug(f"[{self.robots[name].links.names}]")
                dummy_idx = jnp.array(self.robots[name].links.names.index(self.eef_frame[name]), dtype=jnp.int32)
                self.terminal.log(f"[{name}] actuated joints: {self.robots[name].joints.actuated_names}")
                dummy_q = jnp.zeros(expected_dofs)
                _ = solve_ik_jit(self.robots[name], dummy_se3, dummy_idx, dummy_q, self.dof_mask[name]) 
            elif cfg.get('type') == 'end-effector':
                self.single_finger_dofs = cfg.get('single_finger_dofs', 4)

        print("⚡ JAX IK Compiler ready. Solving at microseconds.")
        print("Manual Mode Key input: eef pose +x [s], +y [d], +z[f], +roll[w], +pitch[e], +yaw[r]")
        print("                       eef pose -x [S], -y [D], -z[F], -roll[W], -pitch[E], -yaw[R]")
        print("Hand Selection:[i] Toggle Active Hand")
        print("Hand Fingers:  Bend [g, h, j, k, l] -> Thumb, Index, Middle, Ring, Little")
        print("               Unbend [G, H, J, K, L]")
        self.terminal.wait_debug("manual adapter loading complete")


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

    def _scatter_to_full(self, name, raw_q):
        idx = self.dof_indices[name]
        n = idx.shape[0]
        if raw_q.shape[0] > n:
            raw_q = raw_q[:n]
        elif raw_q.shape[0] < n:
            raw_q = jnp.pad(raw_q, (0, n - raw_q.shape[0]))
        expected_dofs = self.robots[name].joints.num_actuated_joints
        return jnp.zeros(expected_dofs).at[idx].set(raw_q)

    def _forward_kinematics(self, joint_states_dict: dict) -> dict:
        eef_poses = {}
        
        for name, cfg in self.robots_cfg.items():
            if cfg.get('type') == 'manipulator':
                joint_state = joint_states_dict.get(name)
                
                if not joint_state or not joint_state.position:
                    eef_poses[name] = self.current_eef_poses.get(name, [0.0] * 6).copy()
                    print(f"eef poses zero copy")
                    continue

                raw_q = jnp.array(joint_state.position)
                q_fk = self._scatter_to_full(name, raw_q)

                all_link_poses = self.robots[name].forward_kinematics(q_fk)
                eef_idx = self.robots[name].links.names.index(self.eef_frame[name])
                
                eef_se3 = jaxlie.SE3(all_link_poses[eef_idx])
                eef_poses[name] = self._se3_to_list(eef_se3)
                self.current_eef_poses[name] = eef_poses[name].copy()
        return eef_poses

    def _inverse_kinematics(self, eef_poses_dict: dict, original_joint_states: dict, arms_to_solve=None) -> dict:
        target_joints = {}
        
        for name, cfg in self.robots_cfg.items():
            if cfg.get('type') == 'manipulator':
                original_js = original_joint_states.get(name)

                if arms_to_solve is not None and name not in arms_to_solve:
                    target_joints[name] = original_js if original_js else JointState()
                    continue

                eef_pose = eef_poses_dict.get(name)
                
                if not original_js or not original_js.position or not eef_pose:
                    target_joints[name] = original_js if original_js else JointState()
                    continue

                target_se3 = self._list_to_se3(eef_pose)
                
                raw_q = jnp.array(original_js.position)
                ik_seed_q = self._scatter_to_full(name, raw_q)

                eef_idx = self.robots[name].links.names.index(self.eef_frame[name])
                target_link_idx_jax = jnp.array(eef_idx, dtype=jnp.int32)
                
                optimized_q = solve_ik_jit(
                    self.robots[name],
                    target_se3,
                    target_link_idx_jax,
                    ik_seed_q,
                    self.dof_mask[name],
                )

                idx_list = self.dof_indices[name].tolist()
                final_position = list(original_js.position)
                for local_i, global_i in enumerate(idx_list):
                    if local_i < len(final_position):
                        final_position[local_i] = float(optimized_q[global_i])
                
                out_msg = JointState()
                out_msg.header = original_js.header
                out_msg.name = original_js.name
                out_msg.position = final_position
                
                target_joints[name] = out_msg
                
        return target_joints

    def _format_for_policy(self, obs_history: list, inference_option: InferenceOption) -> dict:
        latest_obs = obs_history[-1]
        joints_dict = latest_obs.get("joints", {})
        images_dict = latest_obs.get("images", {})
        command = latest_obs.get("command")
        
        eef_poses = self._forward_kinematics(joints_dict)
        
        return {
            "command": command, 
            "eef_poses": eef_poses, 
            "original_joints": joints_dict,
            "images": images_dict
        }

    def _format_for_muscle(self, raw_action: dict) -> list:
        eef_poses = raw_action.get("eef_poses", {})
        original_joints = raw_action.get("original_joints", {})
        target_joints_from_policy = raw_action.get("target_joints", {})
        moved_arms = raw_action.get("moved_arms", set())

        if moved_arms:
            manipulator_targets = self._inverse_kinematics(eef_poses, original_joints)
        else:
            manipulator_targets = {}
        
        target_dict = {}
        for name, cfg in self.robots_cfg.items():
            fallback_joints = original_joints.get(name, JointState())
            if cfg.get('type') == 'manipulator':
                target_dict[name] = manipulator_targets.get(name, fallback_joints)
            else:
                target_dict[name] = target_joints_from_policy.get(name, fallback_joints)
        
        return [target_dict]

    def _communicate_with_policy(self, formatted_obs: dict) -> dict:
        command = formatted_obs.get("command")
        eef_poses = formatted_obs.get("eef_poses", {})
        original_joints = formatted_obs.get("original_joints", {})
        target_joints_out ={}

        moved_arms = set()

        if not hasattr(self, 'manipulator_names'):
            self.manipulator_names = [name for name, cfg in self.robots_cfg.items() if cfg.get('type') == 'manipulator']
            self.active_robot_idx = len(self.manipulator_names) if self.manipulator_names else -1

            self.hand_names = [name for name, cfg in self.robots_cfg.items() if cfg.get('type') == 'end-effector']
            self.active_hand_idx = len(self.hand_names) if self.hand_names else -1

        for h_name in self.hand_names:
            if h_name not in self.current_hand_joints and h_name in original_joints and original_joints[h_name].position:
                self.current_hand_joints[h_name] = list(original_joints[h_name].position)
                self.terminal.wait_debug(f"[{h_name}]position [{original_joints[h_name].position}]")

        if command is not None:
            active_arm_name = (
                self.manipulator_names[self.active_robot_idx]
                if self.manipulator_names and self.active_robot_idx < len(self.manipulator_names)
                else "ALL ARMS"
            )
            # self.terminal.wait_debug(f"Active Arm:[{active_arm_name}] with command [{command}]")

            if command == 'o' and self.manipulator_names:
                self.active_robot_idx = (self.active_robot_idx + 1) % (len(self.manipulator_names) + 1)
                active_arm_name = self.manipulator_names[self.active_robot_idx] if self.active_robot_idx < len(self.manipulator_names) else "ALL ARMS"
                print(f"🔄 Switched manual control to: {active_arm_name}")
                
            elif command == 'p' and self.manipulator_names:
                self.active_robot_idx = (self.active_robot_idx - 1) % len(self.manipulator_names)
                active_arm_name = self.manipulator_names[self.active_robot_idx]
                print(f"🔄 Switched manual control to: {active_arm_name}")

            elif command == 'i' and self.hand_names:
                self.active_hand_idx = (self.active_hand_idx + 1) % (len(self.hand_names) + 1)
                active_hand_name = self.hand_names[self.active_hand_idx] if self.active_hand_idx < len(self.hand_names) else "ALL HANDS"
                print(f"🖐️ Switched manual hand control to: {active_hand_name}")

            # Apply kinematics deltas to the active robot
            elif len(command) == 1:
                if self.manipulator_names and self.active_robot_idx >= 0:
                    active_arms = self.manipulator_names if self.active_robot_idx == len(self.manipulator_names) else [self.manipulator_names[self.active_robot_idx]]
                    for active_arm_name in active_arms:
                        pose = eef_poses[active_arm_name]
                        match command:
                            case 's': pose[0] += self.manipulator_step_size
                            case 'S': pose[0] -= self.manipulator_step_size
                            case 'd': pose[1] += self.manipulator_step_size
                            case 'D': pose[1] -= self.manipulator_step_size
                            case 'f': pose[2] += self.manipulator_step_size
                            case 'F': pose[2] -= self.manipulator_step_size
                            case 'w': pose[3] += self.manipulator_step_size
                            case 'W': pose[3] -= self.manipulator_step_size
                            case 'e': pose[4] += self.manipulator_step_size
                            case 'E': pose[4] -= self.manipulator_step_size
                            case 'r': pose[5] += self.manipulator_step_size
                            case 'R': pose[5] -= self.manipulator_step_size
                            case _: pass

                        eef_poses[active_arm_name] = pose
                        self.current_eef_poses[active_arm_name] = pose.copy()
                        moved_arms.add(active_arm_name)

                if self.hand_names and self.active_hand_idx >= 0:
                    active_hands = self.hand_names if self.active_hand_idx == len(self.hand_names) else [self.hand_names[self.active_hand_idx]]
                    for active_hand_name in active_hands:
                        if active_hand_name in self.current_hand_joints:
                            joints = self.current_hand_joints[active_hand_name]
                            dofs = len(joints)

                            def apply_bend(finger_index, sign):
                                start = finger_index * self.single_finger_dofs
                                end = min(start + self.single_finger_dofs, dofs)
                                for idx in range(start, end):
                                    joints[idx] += sign * self.eef_step_size

                            match command:
                                case 'g': apply_bend(0, 1)
                                case 'G': apply_bend(0, -1)
                                case 'h': apply_bend(1, 1)
                                case 'H': apply_bend(1, -1)
                                case 'j': apply_bend(2, 1)
                                case 'J': apply_bend(2, -1)
                                case 'k': apply_bend(3, 1)
                                case 'K': apply_bend(3, -1)
                                case 'l': apply_bend(4, 1)
                                case 'L': apply_bend(4, -1)
                                
                            self.current_hand_joints[active_hand_name] = joints

        # 4. Pack Hand Joint States for the Muscle Layer
        for h_name in self.hand_names:
            if h_name in self.current_hand_joints and h_name in original_joints:
                js = JointState()
                js.header = original_joints[h_name].header
                js.name = original_joints[h_name].name
                js.position = self.current_hand_joints[h_name].copy()
                target_joints_out[h_name] = js

        formatted_obs["eef_poses"] = eef_poses
        formatted_obs["target_joints"] = target_joints_out
        formatted_obs["moved_arms"] = moved_arms

        return formatted_obs