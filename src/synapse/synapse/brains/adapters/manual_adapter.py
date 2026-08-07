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
from synapse.utils.embodiment_parser import EmbodimentParser

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
        self.step_size = 0.1
        self.hand_step_size = 0.1
        self.current_eef_poses = {}
        self.currnet_hand_joints = {}
        self.terminal = terminal

        parser = EmbodimentParser(self.embodiment_name)
        self.robots_cfg = parser.get_robots()

        dummy_se3 = jaxlie.SE3.identity()
        self.robots = {}
        self.eef_frame = {}
        terminal.wait_debug("manual adapter robot load")
        for name, cfg in self.robots_cfg.items():
            if cfg.get('type') == 'manipulator':
                terminal.wait_debug(f"[{name}]: [{cfg}]")
                if cfg.get('yourdfpy_description'):
                    urdf = load_robot_description(cfg.get('description_name'))
                else:
                    urdf = yourdfpy.URDF.load(cfg.get('urdf_path'))
                self.robots[name] = pk.Robot.from_urdf(urdf=urdf)

                self.eef_frame[name] = cfg.get('eef_frame')
                dummy_idx = jnp.array(self.robots[name].links.names.index(self.eef_frame[name]), dtype=jnp.int32)
                dummy_q = jnp.zeros(self.robots[name].joints.num_actuated_joints)
                _ = solve_ik_jit(self.robots[name], dummy_se3, dummy_idx, dummy_q) 

        print("⚡ JAX IK Compiler ready. Solving at microseconds.")
        print("Manual Mode Key input: eef pose +x [s], +y [d], +z[f], +roll[w], +pitch[e], +yaw[r]")
        print("                       eef pose -x [S], -y [D], -z[F], -roll[W], -pitch[E], -yaw[R]")
        print("Hand Selection:[i] Toggle Active Hand")
        print("Hand Fingers:  Bend [g, h, j, k, l] -> Thumb, Index, Middle, Ring, Little")
        print("               Unbend [G, H, J, K, L]")
        terminal.wait_debug("manual adapter init done")


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

    def _forward_kinematics(self, joint_states_dict: dict) -> dict:
        """
        Iterates over all robots. If it's a manipulator, solves FK.
        Returns a dictionary: { "robot_name": [x, y, z, roll, pitch, yaw], ... }
        """
        eef_poses = {}
        
        for name, cfg in self.robots_cfg.items():
            if cfg.get('type') == 'manipulator':
                joint_state = joint_states_dict.get(name)
                
                if not joint_state or not joint_state.position:
                    # Fallback to the last known pose if the robot hasn't published yet
                    eef_poses[name] = self.current_eef_poses.get(name, [0.0] * 6).copy()
                    continue

                # 1. Pad or truncate the joints to match the URDF's actuated joints perfectly
                expected_dofs = self.robots[name].joints.num_actuated_joints
                raw_q = jnp.array(joint_state.position)
                
                if raw_q.shape[0] > expected_dofs:
                    q_fk = raw_q[:expected_dofs]
                elif raw_q.shape[0] < expected_dofs:
                    q_fk = jnp.pad(raw_q, (0, expected_dofs - raw_q.shape[0]))
                else:
                    q_fk = raw_q
                
                # 2. PyRoki FK query
                all_link_poses = self.robots[name].forward_kinematics(q_fk)
                eef_idx = self.robots[name].links.names.index(self.eef_frame[name])
                
                # 3. Convert 7D array back to SE3 and then to our [x, y, z, r, p, y] list
                eef_se3 = jaxlie.SE3(all_link_poses[eef_idx])
                eef_poses[name] = self._se3_to_list(eef_se3)
                
                # Cache for fallbacks
                self.current_eef_poses[name] = eef_poses[name].copy()
                
        return eef_poses
    def _inverse_kinematics(self, eef_poses_dict: dict, original_joint_states: dict) -> dict:
        """
        Iterates over manipulators, taking their target EEF poses and original joints,
        and solves IK. Returns a dictionary of Target JointStates.
        """
        target_joints = {}
        
        for name, cfg in self.robots_cfg.items():
            if cfg.get('type') == 'manipulator':
                original_js = original_joint_states.get(name)
                eef_pose = eef_poses_dict.get(name)
                
                if not original_js or not original_js.position or not eef_pose:
                    # Skip or hold position if data is missing
                    target_joints[name] = original_js if original_js else JointState()
                    continue

                target_se3 = self._list_to_se3(eef_pose)
                
                # Setup warm-start seed
                expected_dofs = self.robots[name].joints.num_actuated_joints
                raw_q = jnp.array(original_js.position)

                if raw_q.shape[0] > expected_dofs:
                    ik_seed_q = raw_q[:expected_dofs]
                elif raw_q.shape[0] < expected_dofs:
                    ik_seed_q = jnp.pad(raw_q, (0, expected_dofs - raw_q.shape[0]))
                else:
                    ik_seed_q = raw_q

                eef_idx = self.robots[name].links.names.index(self.eef_frame[name])
                target_link_idx_jax = jnp.array(eef_idx, dtype=jnp.int32)
                
                # ⚡ Execute JIT-Compiled IK
                optimized_q = solve_ik_jit(
                    self.robots[name],
                    target_se3,
                    target_link_idx_jax,
                    ik_seed_q
                )

                # Reconstruct full ROS 2 message 
                final_position = list(original_js.position) 
                for i in range(min(len(optimized_q), len(final_position))):
                    final_position[i] = float(optimized_q[i])
                
                out_msg = JointState()
                out_msg.header = original_js.header
                out_msg.name = original_js.name
                out_msg.position = final_position
                
                target_joints[name] = out_msg
                
        return target_joints

    def _format_for_policy(self, obs_history: list, get_default) -> dict:
        """
        Extracts the synchronized multi-robot dictionaries from the BT node's buffer,
        calculates FK for the arms, and hands a unified state to the policy.
        """
        latest_obs = obs_history[-1]
        
        # Plural dictionaries natively synced by synapse_main_node.py
        joints_dict = latest_obs.get("joints", {})
        images_dict = latest_obs.get("images", {})
        command = latest_obs.get("command")
        
        # Get EEF poses ONLY for the arms/manipulators
        eef_poses = self._forward_kinematics(joints_dict)
        
        return {
            "command": command, 
            "eef_poses": eef_poses, 
            "original_joints": joints_dict,
            "images": images_dict
        }

    def _format_for_muscle(self, raw_action: dict) -> list:
        """
        Takes the policy's multi-robot action dictionary, applies IK to the arms, 
        passes the hand targets through, and formats them into the dynamic topic map.
        """
        eef_poses = raw_action.get("eef_poses", {})
        original_joints = raw_action.get("original_joints", {})
        
        # Optional: AI policies might directly specify joint targets for hands
        target_joints_from_policy = raw_action.get("target_joints", {})
        
        # 1. Solve IK strictly for the manipulators
        manipulator_targets = self._inverse_kinematics(eef_poses, original_joints)
        
        # 2. Build the final dispatch dictionary for the BT Node
        target_dict = {}
        for name, cfg in self.robots_cfg.items():
            if cfg.get('type') == 'manipulator':
                # Grab the solved IK
                target_dict[name] = manipulator_targets.get(name, JointState())
            else:
                # Non-manipulators (Hands, Mobile Bases) bypass IK entirely.
                # If the policy provided a target joint state, use it. Otherwise, hold current.
                fallback_joints = original_joints.get(name, JointState())
                target_dict[name] = target_joints_from_policy.get(name, fallback_joints)
        
        return [target_dict]

    def _communicate_with_policy(self, formatted_obs: dict) -> dict:
        """
        Acts as the manual 'policy'. Reads keyboard commands to adjust the EEF pose 
        of the actively selected manipulator.
        """
        command = formatted_obs.get("command")
        eef_poses = formatted_obs.get("eef_poses", {})
        original_joints = formatted_obs.get("original_joints", {})
        target_joints_out ={}

        # 1. Initialize the Target Selector on the first run
        if not hasattr(self, 'manipulator_names'):
            # Filter out hands/mobile bases to only get IK-controllable arms
            self.manipulator_names = [name for name, cfg in self.robots_cfg.items() if cfg.get('type') == 'manipulator']
            self.active_robot_idx = 0 if self.manipulator_names else -1

            self.hand_names = [name for name, cfg in self.robots_cfg.items() if cfg.get('type') == 'end-effector']
            self.active_hand_idx = 0 if self.hand_names else -1

        for h_name in self.hand_names:
            if h_name not in self.current_hand_joints and h_name in original_joints and original_joints[h_name].position:
                self.current_hand_joints[h_name] = list(original_joints[h_name].position)

        # 2. Process Input

        if command is not None and self.manipulator_names:
            active_name = self.manipulator_names[self.active_robot_idx]
            
            # Switch controlled robot
            if command == 'o' and self.manipulator_names:
                self.active_robot_idx = (self.active_robot_idx + 1) % len(self.manipulator_names)
                active_name = self.manipulator_names[self.active_robot_idx]
                print(f"🔄 Switched manual control to: {active_name}")
                
            elif command == 'p' and self.manipulator_names:
                self.active_robot_idx = (self.active_robot_idx - 1) % len(self.manipulator_names)
                active_name = self.manipulator_names[self.active_robot_idx]
                print(f"🔄 Switched manual control to: {active_name}")

            elif command == 'i' and self.hand_names:
                self.active_hand_idx = (self.active_hand_idx + 1) % len(self.hand_names)
                print(f"🖐️ Switched manual hand control to: {self.hand_names[self.active_hand_idx]}")
            # Apply kinematics deltas to the active robot
            elif len(command) == 1:
                if self.manipulator_names:
                    active_name = self.manipulator_names[self.active_robot_idx]
                    if active_name in eef_poses:
                        pose = eef_poses[active_name]
                        match command:
                            case 's': pose[0] += self.step_size
                            case 'S': pose[0] -= self.step_size
                            case 'd': pose[1] += self.step_size
                            case 'D': pose[1] -= self.step_size
                            case 'f': pose[2] += self.step_size
                            case 'F': pose[2] -= self.step_size
                            case 'w': pose[3] += self.step_size
                            case 'W': pose[3] -= self.step_size
                            case 'e': pose[4] += self.step_size
                            case 'E': pose[4] -= self.step_size
                            case 'r': pose[5] += self.step_size
                            case 'R': pose[5] -= self.step_size

                        eef_poses[active_name] = pose
                        self.current_eef_poses[active_name] = pose.copy()

                if self.hand_names and self.active_hand_idx >= 0:
                    active_hand = self.hand_names[self.active_hand_idx]
                    if active_hand in self.currnet_hand_joints:
                        joints = self.currnet_hand_joints[active_hand]
                        dofs = len(joints)

                        def apply_bend(finger_index, sign):
                            start = finger_index * 4
                            end = min(start + 4, dofs)
                            for idx in range(start, end):
                                joints[idx] += sign * self.hand_step_size

                        match command:
                            case 'g': apply_bend(0, 1)
                            case 'G': apply_bend(0, -1)
                            case 'h': apply_bend(1, 1)   # Index
                            case 'H': apply_bend(1, -1)
                            case 'j': apply_bend(2, 1)   # Middle
                            case 'J': apply_bend(2, -1)
                            case 'k': apply_bend(3, 1)   # Ring
                            case 'K': apply_bend(3, -1)
                            case 'l': apply_bend(4, 1)   # Little 
                            case 'L': apply_bend(4, -1)
                            
                        self.current_hand_joints[active_hand] = joints

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

        return formatted_obs