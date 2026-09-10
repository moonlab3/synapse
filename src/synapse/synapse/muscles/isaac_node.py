#!/home/rog-sf/installs/isaacsim/python.sh

import sys
import os
# --- 0. INJECT WORKSPACE ---
# Force Isaac Sim's isolated Python to recognize your local ROS 2 workspace
workspace_path = "/home/rog-sf/ws/synapse_ws/install/synapse/lib/python3.12/site-packages"
if workspace_path not in sys.path:
    sys.path.insert(0, workspace_path)

from synapse.utils.embodiment_parser import EmbodimentParser

# current_domain = os.environ.get("ROS_DOMAIN_ID", "44")
current_domain = os.environ.get("ROS_DOMAIN_ID", "0")
os.environ["ROS_DOMAIN_ID"] = current_domain

# --- 1. THE SELF-RESTARTING SCRUBBER & REBUILDER ---
if os.environ.get('_ISAAC_ENV_CLEANED') != '1':
    for key in list(os.environ.keys()):
        if any(trigger in key for trigger in ['ROS', 'AMENT', 'RMW']):
            del os.environ[key]

    for path_var in ['LD_LIBRARY_PATH', 'PYTHONPATH']:
        if path_var in os.environ:
            old_path = os.environ[path_var]
            # Removed 'synapse_ws' from the deletion filter so it survives the reboot
            new_path = ':'.join([p for p in old_path.split(':') if 'ros/jazzy' not in p and 'python3.12' not in p])
            os.environ[path_var] = new_path

    isaac_ros_lib = "/home/rog-sf/installs/isaacsim/exts/isaacsim.ros2.bridge/jazzy/lib"
    os.environ['ROS_DISTRO'] = 'jazzy'
    os.environ['RMW_IMPLEMENTATION'] = 'rmw_fastrtps_cpp'
    
    current_ld = os.environ.get('LD_LIBRARY_PATH', '')
    os.environ['LD_LIBRARY_PATH'] = f"{isaac_ros_lib}:{current_ld}" if current_ld else isaac_ros_lib

    os.environ['_ISAAC_ENV_CLEANED'] = '1'
    print("🔄 System ROS 2 purged. Injecting Isaac Sim internal ROS 2 libs and rebooting process...")
    os.execv(sys.executable, [sys.executable] + sys.argv)

# Removed 'synapse_ws' from the sys.path deletion filter here as well
# sys.path = [p for p in sys.path if 'ros/jazzy' not in p and 'python3.12' not in p]
# ---------------------------------------------------

if "--disable" not in sys.argv:
    sys.argv.extend(["--disable", "omni.isaac.ros2_bridge", "--enable", "isaacsim.ros2.bridge"])

# 3. Boot Isaac Sim
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

injected_args = ["--disable", "omni.isaac.ros2_bridge", "--enable", "isaacsim.ros2.bridge"]
sys.argv = [arg for arg in sys.argv if arg not in injected_args]

# 4. ROS2 and Isaac Sim core imports
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, Image
from std_msgs.msg import String

from isaacsim.core.api.world import World
from isaacsim.core.prims import Articulation
from isaacsim.core.api.physics_context import PhysicsContext
from omni.isaac.sensor import Camera
import omni.usd
import functools
import torch

class IsaacNode(Node):
    def __init__(self, node_name="isaac_node", parameter_overrides=None):
        super().__init__(
            node_name,
            parameter_overrides=parameter_overrides,
            allow_undeclared_parameters=True,
            automatically_declare_parameters_from_overrides=True
        )

        self.publish_rate = self.get_parameter('publish_rate_hz').value
        embodiment_name = self.get_parameter('embodiment_name').value

        parser = EmbodimentParser(embodiment_name)
        self.usd_path = parser.get_config().get('usd_path')
        self.camera_cfg = parser.get_cameras()

        self.camera_publishers = {}
        for name, cfg in self.camera_cfg.items():
            topic = cfg.get('topic', f'/synapse/camera/{name}/image_raw')
            self.camera_publishers[name] = self.create_publisher(Image, topic, 10)
        self.cameras = {}

        self.articulation_groups = parser.get_articulations()  # new nested schema only
        flat_robots = parser.get_robots()                  # flattened, all components

        self.component_group = {}
        for group_name, group in self.articulation_groups.items():
            for component_name in group['components']:
                self.component_group[component_name] = group_name

        self.component_targets = parser.get_component_targets('ISAAC')
        self.pub_joint_states = {}
        self.sub_targets = {}
        for component_name, resolved in self.component_targets.items():
            if resolved is None:
                continue
            joint_indices = flat_robots[component_name].get('joint_indices')
            self.sub_targets[component_name] = self.create_subscription(
                JointState, resolved.topic,
                functools.partial(self.target_callback, component_name=component_name, joint_indices=joint_indices),
                10
            )

        self.articulations = {}
        self.target_joints = {}
        self.initial_positions = {}
        self.joint_names = {}
        _log_robot_names = ""

        for group_name, group in self.articulation_groups.items():
            self.pub_joint_states[group_name] = self.create_publisher(JointState, group['states'], 10)
            _log_robot_names += f"{group_name}[{', '.join(group['components'].keys())}], "

        self.sub_synapse_command = self.create_subscription(String, '/synapse/command', self.synapse_command_callback, 10)
        self._setup_isaac_sim()

        self.reset_process = 100
        self.get_logger().info(f"🦾 Isaac Muscle Node initialized for {_log_robot_names}")

    def synapse_command_callback(self, msg: String):
        """Receives commands from synapse_bt_node (e.g., start, stop)"""
        command = msg.data
        match command:
            case "START":
                self.get_logger().info("Received START command. Resuming simulation.")
                self.world.play()
            case "PAUSE":
                self.get_logger().info("Received PAUSE command. Pausing simulation.")
                self.world.pause()
            case "QUIT":
                self.get_logger().info("Received QUIT command. Shutting down Isaac Muscle Node.")
                simulation_app.close()
                rclpy.shutdown()
            case "RESET":
                self.get_logger().info("Received RESET command. Resetting robot to initial pose")
                self.target_joints = self.initial_positions.copy()
                self.reset_process = 0

            case _:
                self.get_logger().warn(f"Unknown command received: {command}")

    def _to_host_numpy(self, data):
        if hasattr(data, 'cpu'):
            return data.detach().cpu().numpy()
        return np.array(data)

    def _setup_isaac_sim(self):

        self.get_logger().info(f"Opening stage: {self.usd_path}")
        omni.usd.get_context().open_stage(self.usd_path)
        dt = 1.0 / self.publish_rate
        self.world = World(physics_dt=dt, rendering_dt=dt, stage_units_in_meters=1.0, backend="torch", device="cuda:0")
        self.world._physics_context = PhysicsContext(prim_path="/World/PhysicsScene")

        self.get_logger().info("Articulation Initializing =================")
        for group_name, group in self.articulation_groups.items():
            prim_path = group['prim_path']

            if not self.world.scene.object_exists(group_name):
                articulation = Articulation(prim_paths_expr=prim_path, name=group_name)
                self.world.scene.add(articulation)
                self.get_logger().info(f"[{group_name}] is not in the scene, added manually")
            else:
                articulation = self.world.scene.get_object(group_name)
                self.get_logger().info(f"[{group_name}] is in the scene.")

            self.articulations[group_name] = articulation

        self.world.reset()
        self.get_logger().info(f"Isaac Sim World reset")

        for group_name, articulation in self.articulations.items():
            articulation.initialize()
            self.get_logger().info(f"Articulation [{group_name}] initialized - {articulation.num_dof}, {articulation.dof_names}")

            init_pos = self._to_host_numpy(articulation.get_joint_positions()[0])

            self.initial_positions[group_name] = init_pos
            if hasattr(init_pos, 'cpu'):
                init_pos = init_pos.detach().cpu().numpy()
            self.target_joints[group_name] = np.array(init_pos, dtype=np.float32)

            self.joint_names[group_name] = articulation.dof_names

        # LOADING CAMERA LOADING CAMERA
        for name, cfg in self.camera_cfg.items():
            camera_kwargs= {
                "prim_path": cfg.get('prim_path'),
                "resolution": (cfg.get('height'), cfg.get('width')),
                "frequency": cfg.get('frequency', 20)
            }
             
            if 'translation' in cfg:
                self.get_logger().info(f"translation is in CFG and [{cfg['translation']}]")
                camera_kwargs['translation'] = np.array(cfg['translation'])
            if 'orientation' in cfg:
                camera_kwargs['orientation'] = np.array(cfg['orientation'])

            cam = Camera(**camera_kwargs)
            cam.initialize()

            if 'focal_length' in cfg:
                cam.set_focal_length(cfg['focal_length'])
            if 'clipping_near_distance' in cfg:
                self.get_logger().info(f"clipping is in CFG and [{cfg['clipping_near_distance']}, {cfg['clipping_far_distance']}]")
                cam.set_clipping_range(near_distance=cfg['clipping_near_distance'], far_distance=cfg['clipping_far_distance'])
            self.cameras[name] = cam
            self.get_logger().info(f"Camera initialized : {name}")
        # LOADING CAMERA LOADING CAMERA

        simulation_app.update()
    def target_callback(self, msg: JointState, component_name: str, joint_indices: list):
        if not msg.position or self.reset_process < 100 or not joint_indices:
            return
        group_name = self.component_group[component_name]
        n = min(len(msg.position), len(joint_indices))
        for i in range(n):
            self.target_joints[group_name][joint_indices[i]] = msg.position[i] 
    # def target_callback(self, msg: JointState, group_name: str):
    #     if msg.position and self.reset_process >= 100:
    #         if msg.name:
    #             for joint_name, joint_pos in zip(msg.name, msg.position):
    #                 if joint_name in self.joint_names[group_name]:
    #                     sim_idx = self.joint_names[group_name].index(joint_name)
    #                     self.target_joints[group_name][sim_idx] = joint_pos
    #         else:
    #             self.get_logger().info("no msg name")
    #             copy_len = min(len(msg.position), len(self.target_joints[group_name]))
    #             self.target_joints[group_name][:copy_len] = msg.position[:copy_len]

    #             if len(msg.position) == 8 and len(self.target_joints[group_name]) == 9:
    #                 self.target_joints[group_name][-1] = msg.position[-1]

    def spin_and_step(self):
        while simulation_app.is_running():
            rclpy.spin_once(self, timeout_sec=0.0)

            if self.world.is_playing():
                if self.reset_process < 100:
                    self.reset_process += 1

                for name, articulation in self.articulations.items():
                    target_tensor = torch.tensor(
                        self.target_joints[name],
                        dtype=torch.float32,
                        device=self.world.get_physics_context().device
                    )
                    articulation.set_joint_position_targets(target_tensor.reshape(1, -1))

                self.world.step(render=True)
                simulation_app.update()

                for group_name, articulation in self.articulations.items():
                    current_positions = self._to_host_numpy(articulation.get_joint_positions()[0])

                    msg = JointState()
                    msg.header.stamp = self.get_clock().now().to_msg()
                    msg.name = self.joint_names[group_name]
                    msg.position = current_positions.tolist()
                    self.pub_joint_states[group_name].publish(msg)

                for name, cam in self.cameras.items():
                    frame_rgba = cam.get_rgba()
                    if frame_rgba is not None and len(frame_rgba.shape) == 3 and frame_rgba.shape[2] == 4:
                        height, width = frame_rgba.shape[:2]
                        frame_rgb = frame_rgba[:, :, :3]
                        img_msg = Image()
                        img_msg.header.stamp = self.get_clock().now().to_msg()
                        img_msg.header.frame_id = name
                        img_msg.height = height
                        img_msg.width = width
                        img_msg.encoding = "rgb8"
                        img_msg.is_bigendian = 0
                        img_msg.step = width * 3
                        img_msg.data = frame_rgb.astype(np.uint8).tobytes()
                        if name in self.camera_publishers:
                            self.camera_publishers[name].publish(img_msg)

            else:
                self.world.render()
                simulation_app.update()
            

def main(args=None):
    rclpy.init(args=args)
    node = IsaacNode()
    
    try:
        node.spin_and_step()
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down Isaac Muscle Node...")
    except Exception as e:
        node.get_logger().error(f" FATAL PYTHON ERROR IN LOOP: {e}")
        import traceback
        traceback.print_exc()
    finally:
        node.destroy_node()
        rclpy.shutdown()
        simulation_app.close()

if __name__ == '__main__':
    main()