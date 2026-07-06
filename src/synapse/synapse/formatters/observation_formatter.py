import numpy as np

class ObservationFormatter:
    def __init__(self, brain_option: str, embodiment: str = "LIBERO_PANDA"):
        self.brain_option = brain_option.upper()
        self.embodiment = embodiment.upper()

    def format_frame(self, command: str, image_msgs: dict, joint_msg) -> dict:
        """
        Translates single synchronized ROS messages into raw Numpy dictionaries.
        The 'command' is tied directly to this specific temporal frame.
        """
        match self.brain_option:
            case "VLA":             
                return self._format_vla_frame(command, image_msgs, joint_msg)
            case "RL":
                return self._format_rl_frame(command, image_msgs, joint_msg)
            case "MANUAL":
                return self._format_manual_frame(command, image_msgs, joint_msg)
            case _:
                raise ValueError(f"Unknown brain option: {self.brain_option}")

    def _format_vla_frame(self, command: str, image_msgs: dict, joint_msg) -> dict:
        """Formats the single-frame dictionary required by GR00T."""
        
        # Map ROS JointState to a dictionary by joint name for safe extraction
        joints = dict(zip(joint_msg.name, joint_msg.position))
        
        video_dict = {}
        state_dict = {}
        language_dict = {}

        if self.embodiment == "LIBERO_PANDA":
            # --- VIDEO ---
            video_dict["image"] = self._ros_img_to_np(image_msgs.get("front_camera"))
            video_dict["wrist_image"] = self._ros_img_to_np(image_msgs.get("wrist_camera"))
            
            # --- STATE ---
            state_dict = {
                "x": np.array([joints.get("x", 0.0)], dtype=np.float32),
                "y": np.array([joints.get("y", 0.0)], dtype=np.float32),
                "z": np.array([joints.get("z", 0.0)], dtype=np.float32),
                "roll": np.array([joints.get("roll", 0.0)], dtype=np.float32),
                "pitch": np.array([joints.get("pitch", 0.0)], dtype=np.float32),
                "yaw": np.array([joints.get("yaw", 0.0)], dtype=np.float32),
                "gripper": np.array([joints.get("gripper", 0.0)], dtype=np.float32)
            }
            
            # --- LANGUAGE ---
            language_dict["annotation.human.action.task_description"] = command

        elif self.embodiment == "BEHAVIOR_R1_PRO":
            # --- VIDEO ---
            video_dict["observation.images.rgb.head_256_256"] = self._ros_img_to_np(image_msgs.get("head"))
            video_dict["observation.images.rgb.left_wrist_256_256"] = self._ros_img_to_np(image_msgs.get("left_wrist"))
            video_dict["observation.images.rgb.right_wrist_256_256"] = self._ros_img_to_np(image_msgs.get("right_wrist"))
            
            # --- STATE ---
            def get_j_array(names):
                return np.array([joints.get(n, 0.0) for n in names], dtype=np.float32)
            
            left_arm_joints = ['l_j1', 'l_j2', 'l_j3', 'l_j4', 'l_j5', 'l_j6', 'l_j7']
            right_arm_joints = ['r_j1', 'r_j2', 'r_j3', 'r_j4', 'r_j5', 'r_j6', 'r_j7']
            
            state_dict = {
                "robot_pos": np.array([0.0, 0.0, 0.0], dtype=np.float32), 
                "arm_left_qpos": get_j_array(left_arm_joints),
                "arm_left_qpos_sin": np.sin(get_j_array(left_arm_joints)),
                "arm_left_qpos_cos": np.cos(get_j_array(left_arm_joints)),
                "gripper_left_qpos": get_j_array(['l_finger1', 'l_finger2']),
                
                "arm_right_qpos": get_j_array(right_arm_joints),
                "arm_right_qpos_sin": np.sin(get_j_array(right_arm_joints)),
                "arm_right_qpos_cos": np.cos(get_j_array(right_arm_joints)),
                "gripper_right_qpos": get_j_array(['r_finger1', 'r_finger2']),
                # ... other r1 pro state variables ...
            }
            
            # --- LANGUAGE ---
            language_dict["annotation.human.coarse_action"] = command

        else:
            raise ValueError(f"Unknown embodiment configuration: {self.embodiment}")

        return {
            "video": video_dict,
            "state": state_dict,
            "language": language_dict
        }


    def _format_rl_frame(self, command: str, image_msgs: dict, joint_msg) -> dict:
        # RL might still use the command as a conditional token
        joints = np.array(joint_msg.position, dtype=np.float32)
        return {
            "proprioception": joints,
            "command_token": command
        }

    def _format_manual_frame(self, command: str, image_msgs: dict, joint_msg) -> dict:
        if not joint_msg.position:
            return {"eef_pose": {"x": 0.0, "y": 0.0, "z": 0.0, "roll": 0.0, "pitch": 0.0, "yaw": 0.0}}
        return {"eef_pose": {"x": joint_msg.position[0],
                              "y": joint_msg.position[1], 
                              "z": joint_msg.position[2],
                              "roll": joint_msg.position[3],
                              "pitch": joint_msg.position[4],
                              "yaw": joint_msg.position[5]}}

    def _ros_img_to_np(self, image_msg):
        """Native conversion from ROS Image to Numpy (No CvBridge required)."""
        if image_msg is None:
            # Fallback to prevent crashes if a camera drops a frame
            return np.zeros((256, 256, 3), dtype=np.uint8)
        
        # ROS 2 image_msg.data is an array of bytes. We read it directly into numpy.
        img_np = np.frombuffer(image_msg.data, dtype=np.uint8)
        
        # Reshape based on the exact dimensions provided by the camera node
        try:
            # Assuming standard 3-channel encoding (e.g., 'rgb8' or 'bgr8')
            img_np = img_np.reshape((image_msg.height, image_msg.width, 3))
        except ValueError:
            # Failsafe if the camera outputs something unexpected (like RGBA or Mono)
            print(f"Warning: Unexpected image shape. Buffer size: {len(img_np)}")
            return np.zeros((256, 256, 3), dtype=np.uint8)
            
        return img_np