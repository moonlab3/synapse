#!/home/rog-sf/installs/isaacsim/python.sh
import sys
import os

# --- 1. THE SELF-RESTARTING SCRUBBER & REBUILDER ---
# Use a custom flag to prevent the infinite reboot loop
if os.environ.get('_ISAAC_ENV_CLEANED') != '1':
    
    # A. Purge System ROS 2 variables
    for key in list(os.environ.keys()):
        if any(trigger in key for trigger in ['ROS', 'AMENT', 'RMW']):
            del os.environ[key]

    # B. Scrub System LD_LIBRARY_PATH and PYTHONPATH
    for path_var in ['LD_LIBRARY_PATH', 'PYTHONPATH']:
        if path_var in os.environ:
            old_path = os.environ[path_var]
            new_path = ':'.join([p for p in old_path.split(':') if 'ros/jazzy' not in p and 'synapse_ws' not in p and 'python3.12' not in p])
            os.environ[path_var] = new_path

    # C. Inject Isaac Sim's INTERNAL ROS 2 libraries explicitly
    isaac_ros_lib = "/home/rog-sf/installs/isaacsim/exts/isaacsim.ros2.bridge/jazzy/lib"
    os.environ['ROS_DISTRO'] = 'jazzy'
    os.environ['RMW_IMPLEMENTATION'] = 'rmw_fastrtps_cpp'
    
    current_ld = os.environ.get('LD_LIBRARY_PATH', '')
    os.environ['LD_LIBRARY_PATH'] = f"{isaac_ros_lib}:{current_ld}" if current_ld else isaac_ros_lib

    # D. Set the safety flag and reboot into the sterile environment
    os.environ['_ISAAC_ENV_CLEANED'] = '1'
    print("🔄 System ROS 2 purged. Injecting Isaac Sim internal ROS 2 libs and rebooting process...")
    os.execv(sys.executable, [sys.executable] + sys.argv)

# E. Ensure current Python runtime's sys.path is also clean after reboot
sys.path = [p for p in sys.path if 'ros/jazzy' not in p and 'synapse_ws' not in p and 'python3.12' not in p]
# ---------------------------------------------------

import numpy as np

# 2. Block the Ghost Extension
if "--disable" not in sys.argv:
    sys.argv.extend(["--disable", "omni.isaac.ros2_bridge", "--enable", "isaacsim.ros2.bridge"])

# 3. Boot Isaac Sim
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

injected_args = ["--disable", "omni.isaac.ros2_bridge", "--enable", "isaacsim.ros2.bridge"]
sys.argv = [arg for arg in sys.argv if arg not in injected_args]

# 4. ROS2 and Isaac Sim core imports
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, Image
from std_msgs.msg import String
import numpy as np

from isaacsim.core.api.world import World
from isaacsim.core.prims import Articulation
from isaacsim.core.api.physics_context import PhysicsContext
from omni.isaac.sensor import Camera
from omni.isaac.core.utils.rotations import euler_angles_to_quat
import omni.usd

class IsaacMuscleNode(Node):
    def __init__(self):
        super().__init__('isaac_muscle_node')

        # Declare ROS2 Parameters for configuration
        self.declare_parameter('usd_path', '/home/rog-sf/vla/OXE.usd')
        self.declare_parameter('robot_name', 'panda')
        self.declare_parameter('robot_prim_path', '/World/franka_set/Franka/panda')
        self.declare_parameter('publish_rate_hz', 100)
        self.declare_parameter('camera_config', 'wrist_cam')

        self.usd_path = self.get_parameter('usd_path').value
        self.robot_name = self.get_parameter('robot_name').value
        self.robot_prim_path = self.get_parameter('robot_prim_path').value
        self.publish_rate = self.get_parameter('publish_rate_hz').value
        self.camera_config = self.get_parameter('camera_config').value

        # ROS2 Interfaces
        self.pub_joint_states = self.create_publisher(JointState, '/synapse/joint_states', 10)
        self.pub_camera = self.create_publisher(Image, "/synapse/camera/image_raw", 10)
        self.sub_synapse_command = self.create_subscription(String, '/synapse/command', self.synapse_command_callback, 10)
        self.sub_brain_output = self.create_subscription( JointState, '/synapse/brain_output', self.brain_output_callback, 10)

        # Isaac Sim Environment Setup
        self._setup_isaac_sim()

        # State
        self.num_dofs = self.articulation.num_dof
        self.joint_names = self.articulation.dof_names
        
        # Start with current positions to prevent sudden jumping
        initial_positions = self.articulation.get_joint_positions()[0]
        self.target_joints = np.array(initial_positions, dtype=np.float32)

        self.get_logger().info(f"🦾 Isaac Muscle Node initialized for {self.robot_name} with {self.num_dofs} DOFs.")

    def synapse_command_callback(self, msg: String):
        """Receives commands from synapse_bt_node (e.g., start, stop)"""
        command = msg.data
        if command == "START":
            self.get_logger().info("Received START command. Resuming simulation.")
            self.world.play()
        elif command == "PAUSE":
            self.get_logger().info("Received PAUSE command. Pausing simulation.")
            self.world.pause()
        elif command == "QUIT":
            self.get_logger().info("Received QUIT command. Shutting down Isaac Muscle Node.")
            simulation_app.close()
            rclpy.shutdown()
        else:
            self.get_logger().warn(f"Unknown command received: {command}")

    def _setup_isaac_sim(self):
        self.get_logger().info(f"Opening stage: {self.usd_path}")
        omni.usd.get_context().open_stage(self.usd_path)

        # Match BT Node frequency or physics frequency
        dt = 1.0 / self.publish_rate
        self.world = World(physics_dt=dt, rendering_dt=dt, stage_units_in_meters=1.0)
        
        physics_context = PhysicsContext(prim_path="/World/PhysicsScene")
        self.world._physics_context = physics_context
        
        if not self.world.scene.object_exists(self.robot_name):
            self.articulation = Articulation(
                prim_paths_expr=self.robot_prim_path, 
                name=self.robot_name
            )
            self.world.scene.add(self.articulation)
        else:
            self.articulation = self.world.scene.get_object(self.robot_name)

        self.world.reset()
        self.articulation.initialize()

        # Camera Setup
        if self.camera_config == "wrist_cam":
            camera_prim_path = f"{self.robot_prim_path}/panda_hand/wrist_cam"
            pitch_angle_rad = np.deg2rad(-75)
            roll_angle_rad = np.deg2rad(180)
            camera_translation = [-0.10, 0.00, -0.05]
            focal_length = 1.5
            camera_quat = euler_angles_to_quat(np.array([roll_angle_rad, pitch_angle_rad, 0.0]))

            self.camera = Camera(
                prim_path=camera_prim_path,
                translation=camera_translation,
                orientation=camera_quat,
                resolution=(256,256),
                frequency=20,
            )
        elif self.camera_config == "quarter_view_cam":
            camera_prim_path = "/World/franka_set/Azure/Camera"
            focal_length = 1.5
            self.camera = Camera(
                prim_path=camera_prim_path,
                resolution=(256, 256),
                frequenct=20,
            )
        
        if self.camera:
            self.camera.initialize()
            self.camera.set_focal_length(focal_length)
            self.camera.set_clipping_range(near_distance=0.01, far_distance=10.0)
            self.get_logger().info(f"📷 Camera initialized: {self.camera_config}")
        
        # Update once to populate internal physics buffers
        simulation_app.update()

    def brain_output_callback(self, msg: JointState):
        """Receives target joints from synapse_bt_node"""
        if msg.position:
            length = min(len(msg.position), self.num_dofs)
            self.target_joints[:length] = np.array(msg.position)[:length]

    def spin_and_step(self):
        """Manual loop to step both ROS2 and Isaac Sim concurrently"""
        while simulation_app.is_running() and self.world.is_playing():
            # 1. Process ROS2 callbacks (non-blocking)
            rclpy.spin_once(self, timeout_sec=0.0)

            # 2. Apply targets to Isaac Sim (reshape to 1xN for batched articulation API)
            self.articulation.set_joint_position_targets(self.target_joints.reshape(1, -1))

            # 3. Step simulation
            self.world.step(render=True)
            simulation_app.update()

            # 4. Read physical state and publish back to Synapse
            current_positions = self.articulation.get_joint_positions()[0]
            
            msg = JointState()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.name = self.joint_names
            msg.position = current_positions.tolist()
            
            self.pub_joint_states.publish(msg)

            if self.camera:
                frame_rgba = self.camera.get_rgba()
                if frame_rgba is not None and frame_rgba.shape == (256, 256, 4):
                    frame_rgb = frame_rgba[:, :, :3]
                    img_msg = Image()
                    img_msg.header.stamp = self.get_clock().now().to_msg()
                    img_msg.header.frame_id = self.camera_config
                    img_msg.height = 256
                    img_msg.width = 256
                    img_msg.encoding = "rgb8"
                    img_msg.is_bigendian = 0
                    img_msg.step = 256 * 3
                    img_msg.data = frame_rgb.astype(np.uint8).tobytes()
                    self.pub_camera.publish(img_msg)

def main(args=None):
    rclpy.init(args=args)
    node = IsaacMuscleNode()
    
    try:
        node.spin_and_step()
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down Isaac Muscle Node...")
    finally:
        node.destroy_node()
        rclpy.shutdown()
        simulation_app.close()

if __name__ == '__main__':
    main()