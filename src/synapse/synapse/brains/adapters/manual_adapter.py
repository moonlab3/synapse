from ..base_brain_adapter import BaseBrainAdapter
from ..base_brain_adapter import InferenceOption
from sensor_msgs.msg import JointState
from synapse.utils.embodiment_parser import EmbodimentParser
from synapse.utils.kinematics import ArmKinematics

class ManualAdapter(BaseBrainAdapter):
    def __init__(self, terminal, node_name="manual_adapter", parameter_overrides=None):
        super().__init__(terminal, node_name, parameter_overrides)
        self.current_hand_joints = {}
        self.terminal = terminal

        parser = EmbodimentParser(self.embodiment_name)
        self.manipulator_step_size = parser.get_config().get('manipulator_step_size', 0.1)
        self.eef_step_size = parser.get_config().get('eef_step_size', 0.01)
        self.robots_cfg = parser.get_robots()

        self.kin = ArmKinematics(self.robots_cfg, terminal)
        self.current_eef_poses = self.kin.current_eef_poses  # the BT pose checks read this
        self.single_finger_dofs = next(
            (cfg.get('single_finger_dofs', 4) for cfg in self.robots_cfg.values()
             if cfg.get('type') == 'end-effector'), 4)

        print("⚡ JAX IK Compiler ready. Solving at microseconds.")
        print("Manual Mode Key input: eef pose +x [s], +y [d], +z[f], +roll[w], +pitch[e], +yaw[r]")
        print("                       eef pose -x [S], -y [D], -z[F], -roll[W], -pitch[E], -yaw[R]")
        print("Hand Selection:[i] Toggle Active Hand")
        print("Hand Fingers:  Bend [g, h, j, k, l] -> Thumb, Index, Middle, Ring, Little")
        print("               Unbend [G, H, J, K, L]")
        self.terminal.debug("manual adapter loading complete")


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

                seed_q = self.kin.seed_from_positions(name, original_js.position)
                optimized_q = self.kin.solve(name, eef_pose, seed_q)

                idx_list = self.kin.dof_indices[name].tolist()
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
        
        eef_poses = self.kin.forward_kinematics(joints_dict)
        
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
                self.terminal.debug(f"[{h_name}]position [{original_joints[h_name].position}]")

        if command is not None:
            active_arm_name = (
                self.manipulator_names[self.active_robot_idx]
                if self.manipulator_names and self.active_robot_idx < len(self.manipulator_names)
                else "ALL ARMS"
            )
            # self.terminal.debug(f"Active Arm:[{active_arm_name}] with command [{command}]")

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