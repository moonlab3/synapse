import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory

class RealRobotNode(Node):
    def __init__(self):
        # Initialize the ROS 2 node with a specific name
        super().__init__('real_robot_node')
        
        # 2. Subscribe to the action chunks coming from the BT node
        self.trajectory_sub = self.create_subscription(
            JointTrajectory, '/joint_commands', self.trajectory_callback, 10
        )
        
        # (Hardware specific initialization would go here, like connecting to xArm API)

    def trajectory_callback(self, msg: JointTrajectory):
        # Pass the trajectory to the physical robot controller
        self.get_logger().wait_debug("Executing new trajectory chunk...")
        pass

    def hardware_estop(self):
        # e.g., xarm_api.emergency_stop()
        self.get_logger().info("xArm brakes engaged.")

    def hardware_shutdown(self):
        # e.g., xarm_api.disconnect()
        self.get_logger().info("xArm disconnected.")

def main(args=None):
    rclpy.init(args=args)
    node = RealRobotNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()