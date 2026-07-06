from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})
import carb.windowing
import omni.appwindow
# app_window=omni.appwindow.get_default_app_window()
# windowing_interface=carb.windowing.acquire_windowing_interface()
# windowing_interface.maximize_window(app_window.get_window())

from isaacsim.core.api.world import World
from isaacsim.core.prims import Articulation
from isaacsim.core.prims import XFormPrim
from isaacsim.core.utils.extensions import get_extension_path_from_name
from omni.isaac.motion_generation import LulaKinematicsSolver

from omni.isaac.sensor import Camera
from omni.isaac.core.objects import VisualCylinder
from omni.isaac.core.utils.rotations import euler_angles_to_quat
from isaacsim.core.api.physics_context import PhysicsContext
import numpy as np
import threading
from scipy.spatial.transform import Rotation as R
from pprint import pprint

import json
import omni.usd
import os
import sys
new_path = os.path.abspath("/home/rog-sf/vla/Isaac-GR00T")
if new_path not in sys.path:
    sys.path.append(new_path)

from gr00t.policy.server_client import PolicyClient
import math

class PolicyClientWrapper:
    def __init__(self, xform_prim, articulation, ik_solver):
        self.xform_prim = xform_prim
        self.articulation = articulation
        self.ik_solver = ik_solver
        self.client = PolicyClient("0.0.0.0", 8888)
        if not self.client.ping():
            print("\n\n\nPolicyClient NOT connected to GR00T server.\n\n\n")
        else:
            print("\n\n\nPolicyClient initialized and connected to GR00T server.\n\n\n")

    def mapper_to_gr00t(self):
        # 1. Get Cartesian Pose from the Simulator
        position, orientation = self.xform_prim.get_world_poses()
        # print(f"Current Position: {position}, Current Orientation (quat): {orientation}")
        try:
            x, y, z = position[0]
        except Exception as e:
            print(f"Error unpacking position: {e}")
            x, y, z = 0.0, 0.0, 0.0
        

        z -= 0.10
        # Isaac Sim returns quaternions as [w, x, y, z]
        try:
            w, qx, qy, qz = orientation[0]
        except Exception as e:
            print(f"Error unpacking orientation: {e}")
            w, qx, qy, qz = 1.0, 0.0, 0.0, 0.0

        # Scipy expects quaternions as [x, y, z, w]
        rot = R.from_quat([qx, qy, qz, w])
        
        # Note: You MUST verify the Euler sequence ('xyz', 'zyx', etc.) 
        # used by the original WidowX dataset and match it here.
        roll, pitch, yaw = rot.as_euler('xyz', degrees=False) 

        # 2. Get Gripper State
        # Assuming standard Franka setup where the last two joints (indices 7, 8) are the fingers
        joint_positions = self.articulation.get_joint_positions()
        left_finger = joint_positions[0, 7]
        right_finger = joint_positions[0, 8]
        
        # Franka max width is ~0.08m (0.04m per finger). 
        # Normalize this to a 0.0 to 1.0 scale (adjust if WidowX used -1 to 1)
        current_width = left_finger + right_finger
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
        # print(f"Mapped state for GR00T: {state}")
        return state


    def apply_groot_action_slice(
        self,
        action_dict: dict, 
        step_index: int
    ):
        """
        Extracts a specific time slice from the GR00T action dictionary,
        applies the deltas to the current pose, and sends targets to the Franka.
        """
        # 1. Extract the scalar values for the current step
        # The shape is (Batch, Time, Dimension) -> (1, 8, 1)
        dx = action_dict['action.x'][0, step_index, 0].item()
        dy = action_dict['action.y'][0, step_index, 0].item()
        dz = action_dict['action.z'][0, step_index, 0].item()
        droll = action_dict['action.roll'][0, step_index, 0].item()
        dpitch = action_dict['action.pitch'][0, step_index, 0].item()
        dyaw = action_dict['action.yaw'][0, step_index, 0].item()

        # droll = action_dict['action.pitch'][0, step_index, 0].item()
        # dpitch = action_dict['action.roll'][0, step_index, 0].item()
        
        # droll = dpitch = 0.0

        # Gripper is absolute (0.0 to 1.0)
        gripper_cmd = action_dict['action.gripper'][0, step_index, 0].item()

        # 2. Get the CURRENT pose of the end-effector to apply the deltas
        positions, orientations = self.xform_prim.get_world_poses()
        current_pos = positions[0]
        w, qx, qy, qz = orientations[0]

        # Calculate absolute target position
        target_pos = current_pos + np.array([dx, dy, dz])
        # print(f"Current Pos: {current_pos}, Target Pos: {target_pos}, Gripper Cmd: {gripper_cmd}")

        # Calculate absolute target orientation
        rot = R.from_quat([qx, qy, qz, w])
        current_euler = rot.as_euler('xyz', degrees=False)
        target_euler = current_euler + np.array([droll, dpitch, dyaw])
        
        target_rot = R.from_euler('xyz', target_euler, degrees=False)
        t_qx, t_qy, t_qz, t_w = target_rot.as_quat()
        target_quat = np.array([t_w, t_qx, t_qy, t_qz]) # Isaac Sim [w, x, y, z]

        # 3. Inverse Kinematics
        # Use the 0th environment's joints for warm start
        current_joints = self.articulation.get_joint_positions()[0, :7]
        
        target_arm_joints, success = self.ik_solver.compute_inverse_kinematics(
            frame_name="panda_hand",
            target_position=target_pos,
            target_orientation=target_quat,
            warm_start=current_joints
        )

        if not success:
            print(f"Warning: IK failed at chunk step {step_index}. Maintaining pose.")
            target_arm_joints = current_joints

        # 4. Map Gripper
        finger_target = np.clip(gripper_cmd, 0.0, 1.0) * 0.04

        # Combine arm and fingers
        full_target = np.append(target_arm_joints, [finger_target, finger_target])

        # 5. Apply to Articulation
        # Because franka_arm is batched, we must reshape the 1D array to 2D (1, 9)
        self.articulation.set_joint_position_targets(full_target.reshape(1, -1))
        # self.articulation.set_joint_positions(full_target.reshape(1, -1))

    def send_command(self, image, command):
        state = self.mapper_to_gr00t()
        obs = {
                "annotation.human.action.task_description": [command],
                "video.image_0": image,
            }
        for item, i in state.items():
            obs[f"state.{item}"] = state[item]

        result = self.client.get_action(obs)
        # pprint(f"Received action from GR00T: {result}")
        # target = self.apply_groot_action(result)
        return result

class CommandListener(threading.Thread):
    def __init__(self, initial_command):
        super().__init__()
        self.current_command = initial_command
        self.daemon = True  # Thread dies when main script exits

    def run(self):
        while True:
            # This waits for input in the terminal background
            new_input = input("\n[CLI] Enter new command: ")
            if new_input.strip():
                self.current_command = new_input
                
                match self.current_command.strip():
                    case "re":
                        command = "Reset the robot status"
                    case "p":
                        command = "Pause the simulation"
                    case "q":
                        command = "Quit the simulation"
                    case _:
                        command = f"Sending instruction to GR00T: '{self.current_command}'"
                print(f"🤖🤖 Simulation Instruction: {command}")

robot_name = "panda"
robot_path="/World/franka_set/Franka/panda"
usd_path = "/home/rog-sf/vla/OXE.usd"

omni.usd.get_context().open_stage(usd_path)

world = World(physics_dt =1.0 / 20.0, rendering_dt = 1.0 / 20.0, stage_units_in_meters=1.0)
print("Successfully connected to the running Isaac Sim instance!")
physics_context = PhysicsContext(prim_path="/World/PhysicsScene")
world._physics_context = physics_context
print(f"Physics timestep set to: {world.get_physics_dt()} seconds")

if not world.scene.object_exists(robot_name):
    articulation = Articulation(
        prim_paths_expr=robot_path, 
        name=robot_name
        )
    world.scene.add(articulation)
    print(f"Added articulation '{robot_name}' to the world.")
else:
    print(f"Articulation '{robot_name}' already exists in the world.")
    articulation = world.scene.get_object(robot_name)

ee_path = "/World/franka_set/Franka/panda/panda_hand"
# ee_path = "/World/franka_set/Franka/panda/panda_hand/tool_center"
xform_prim = XFormPrim(prim_paths_expr=ee_path)
print("resetting world...")
world.reset()
print("initializing articulation...")
articulation.initialize()
xform_prim.initialize()

num_dofs = articulation.num_dof
# kps = np.full(num_dofs, 1e7) # Stiffness
# kds = np.full(num_dofs, 1e5) # Damping
# articulation.set_gains(kps=kps, kds=kds)


t = 0
simulation_app.update()

camera_config = "wrist_cam"
# camera_config = "quarter_view_cam"

if camera_config == "wrist_cam":
    camera_prim_path = "/World/franka_set/Franka/panda/panda_hand/wrist_cam"
    pitch_angle_rad=np.deg2rad(-75)
    roll_angle_rad=np.deg2rad(180)
    camera_translation = [-0.10, 0.00, -0.05]
    focal_length = 1.5
    camera_quat=euler_angles_to_quat(np.array([roll_angle_rad, pitch_angle_rad, 0.0]))
    fixed_camera = Camera(
        prim_path=camera_prim_path,
        translation=camera_translation,
        orientation=camera_quat,
        resolution=(256, 256),
        frequency=20,
    )
elif camera_config == "quarter_view_cam":
    camera_prim_path = "/World/franka_set/Azure/Camera"
    focal_length = 1.5
    fixed_camera = Camera(
        prim_path=camera_prim_path,
        resolution=(256, 256),
        frequency=20,
    )

fixed_camera.initialize()
fixed_camera.set_focal_length(focal_length)
fixed_camera.set_clipping_range(near_distance=0.01, far_distance=10.0)

import omni.kit.viewport.utility as vp_utils
robot_viewport = vp_utils.create_viewport_window("gr00t feed")
robot_viewport.viewport_api.set_active_camera(camera_prim_path)

limits = articulation.get_dof_limits()

initial_positions = articulation.get_joint_positions()
# for i in range(num_dofs):
#     current_positions[0, i] = grlimits[0, i, 0]
# articulation.set_joint_position_targets(current_positions)
simulation_app.update()

print(f"Number of DOFs: {num_dofs}")
for i in range(num_dofs):
    print(f"{i:2d} {articulation.dof_names[i]:<30} {initial_positions[0, i]:.4f}")
ee_pose = xform_prim.get_world_poses()
print(f"End-effector initial position: {ee_pose[0][0]}, orientation (quat): {ee_pose[1][0]}")
mg_extension_path = get_extension_path_from_name("isaacsim.robot_motion.motion_generation")
kinematics_config_dir = os.path.join(mg_extension_path, "motion_policy_configs")
ik_solver = LulaKinematicsSolver(
    robot_description_path = kinematics_config_dir + "/franka/rmpflow/robot_descriptor.yaml",
    urdf_path = kinematics_config_dir + "/franka/lula_franka_gen.urdf"
    )
robot_base_prim = XFormPrim(prim_paths_expr=robot_path)
base_positions, base_orientations = robot_base_prim.get_world_poses()
ik_solver.set_robot_base_pose(base_positions[0], base_orientations[0])
# print("LULA FRAMES:", ik_solver.get_all_frame_names())
policy_client = PolicyClientWrapper(xform_prim, articulation, ik_solver)

command = "pick up the red box and place it on the brown tray"
cli = CommandListener(command)
cli.start()
num_chunks = 8
dx = dy = dz = droll = dpitch = dyaw = dgripper = 0.0
groot = False
while world.is_playing():
    instruction = cli.current_command
    if instruction.strip() == "re":
        # articulation.set_joint_position_targets(initial_positions)
        articulation.set_joint_positions(initial_positions)
        t = 0
        cli.current_command=""
        for _ in range (200):
            world.step(render=True)
            simulation_app.update()
        continue
    elif instruction.strip() == "p":
        continue
    elif instruction.strip() == "x":
        dx = 0.01
        dy = dz = droll = dpitch = dyaw = dgripper = 0.0
    elif instruction.strip() == "mx":
        dx = -0.01
        dy = dz = droll = dpitch = dyaw = dgripper = 0.0
    elif instruction.strip() == "y":
        dy = 0.01
        dx = dz = droll = dpitch = dyaw = dgripper = 0.0
    elif instruction.strip() == "my":
        dy = -0.01
        dx = dz = droll = dpitch = dyaw = dgripper = 0.0
    elif instruction.strip() == "z":
        dz = 0.01
        dx = dy = droll = dpitch = dyaw = dgripper = 0.0
    elif instruction.strip() == "mz":
        dz = -0.01
        dx = dy = droll = dpitch = dyaw = dgripper = 0.0
    elif instruction.strip() == "roll":
        droll = 0.03
        dx = dy = dz = dpitch = dyaw = dgripper = 0.0
    elif instruction.strip() == "mroll":
        droll = -0.03
        dx = dy = dz = dpitch = dyaw = dgripper = 0.0
    elif instruction.strip() == "pitch":
        dpitch = 0.03
        dx = dy = dz = droll = dyaw = dgripper = 0.0
    elif instruction.strip() == "mpitch":
        dpitch = -0.03
        dx = dy = dz = droll = dyaw = dgripper = 0.0
    elif instruction.strip() == "yaw":
        dyaw = 0.03
        dx = dy = dz = droll = dpitch = dgripper = 0.0
    elif instruction.strip() == "myaw":
        dyaw = -0.03
        dx = dy = dz = droll = dpitch = dgripper = 0.0
    elif instruction.strip() == "g":
        dgripper = 0.0
        dx = dy = dz = droll = dpitch = dyaw = 0.0
    elif instruction.strip() == "ug":
        dgripper = 1.0
        dx = dy = dz = droll = dpitch = dyaw = 0.0
    elif instruction.strip() == "gr":
        groot = not groot
        cli.current_command= ""
        if groot:
            print("Switched to GR00T control mode. Awaiting commands...")
        else:
            print("Switched to manual control mode. Use the command keys to control the robot.")
        continue
    elif instruction.strip() != "":
        # print(f"Sending instruction to GR00T: '{instruction}'")
        command = instruction
    if instruction.strip() == "q":
        break
    frame_rgba = fixed_camera.get_rgba()
    if frame_rgba is None or frame_rgba.shape != (256, 256, 4):
        world.step(render=True)
        continue
    frame_rgb = frame_rgba[:, :, :3]
    frame_vla = frame_rgb[np.newaxis, np.newaxis, :, :, :]

    current_positions = articulation.get_joint_positions()
    if groot:
        if t % num_chunks == 0:
            action_chunk = policy_client.send_command(frame_vla, command)
            # print(f" sending command to GR00T {command} ")

        try:
            target = action_chunk[0]
        except Exception as e:
            # print(f" sending command to GR00T {command} failed with error: {e}")
            action_chunk = policy_client.send_command(frame_vla, command)
            target = action_chunk[0]
    else:
        target = {
            'action.x': np.full((1, num_chunks, 1), dx, dtype=np.float32),
            'action.y': np.full((1, num_chunks, 1), dy, dtype=np.float32),
            'action.z': np.full((1, num_chunks, 1), dz, dtype=np.float32),
            'action.roll': np.full((1, num_chunks, 1), droll, dtype=np.float32),
            'action.pitch': np.full((1, num_chunks, 1), dpitch, dtype=np.float32),
            'action.yaw': np.full((1, num_chunks, 1), dyaw, dtype=np.float32),
            'action.gripper': np.full((1, num_chunks, 1), dgripper, dtype=np.float32),
        }

    policy_client.apply_groot_action_slice(
        target,
        t % num_chunks)

    t += 1

    simulation_app.update()
    world.step(render=True)
    


    
    
