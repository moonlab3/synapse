#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration
from synapse.utils.embodiment_parser import EmbodimentParser

# Maps Synapse's component names (from embodiment_configs.yaml) to the real
# xarm_ros2 trajectory controller topics confirmed working via CLI testing.
# Only manipulators go through joint_trajectory_controller here; end-effectors
# (hands) would need their own real driver, not handled by this bridge yet.
CONTROLLER_TOPICS = {
    'left_arm': '/L_xarm7_traj_controller/joint_trajectory',
    'right_arm': '/R_xarm7_traj_controller/joint_trajectory',
}


class RealRobotNode(Node):
    def __init__(self, node_name="real_robot_node", parameter_overrides=None):
        super().__init__(
            node_name,
            parameter_overrides=parameter_overrides,
            allow_undeclared_parameters=True,
            automatically_declare_parameters_from_overrides=True
        )

        embodiment_name = self.get_parameter('embodiment_name').value

        if not self.has_parameter('move_time_sec'):
            self.declare_parameter('move_time_sec', 0.01)
        self.move_time_sec = self.get_parameter('move_time_sec').value

        parser = EmbodimentParser(embodiment_name)
        self.robots_cfg = parser.get_robots()  # flattened: left_arm, right_arm, left_hand, right_hand, ...

        # --- Command path: Synapse target -> per-arm JointTrajectory ---
        self.traj_pubs = {}
        for component_name, topic in CONTROLLER_TOPICS.items():
            if component_name in self.robots_cfg:
                self.traj_pubs[component_name] = self.create_publisher(JointTrajectory, topic, 10)
            else:
                self.get_logger().warn(f"⚠️ '{component_name}' not found in embodiment '{embodiment_name}', skipping its controller.")

        self.target_sub = self.create_subscription(
            JointState, '/synapse/target/dual_xarm_unit', self._target_callback, 10
        )

        # --- Feedback path: real controllers -> Synapse joint_states ---
        # joint_state_broadcaster publishes every joint from both arms in one
        # combined /joint_states message, which is exactly what Synapse's
        # obs_callback already expects to split per-component.
        self.feedback_sub = self.create_subscription(
            JointState, '/joint_states', self._feedback_callback, 10
        )
        self.feedback_pub = self.create_publisher(
            JointState, '/synapse/joint_states/dual_xarm_unit', 10
        )

        self.get_logger().info(
            f"🌉 RealRobotNode bridge ready for '{embodiment_name}': "
            f"controllers={list(self.traj_pubs.keys())}, move_time_sec={self.move_time_sec}"
        )

    def _target_callback(self, msg: JointState):
        """Split one combined JointState target into per-arm JointTrajectory commands."""
        name_to_pos = dict(zip(msg.name, msg.position))

        for component_name, pub in self.traj_pubs.items():
            joint_names = self.robots_cfg[component_name].get('joint_names', [])

            positions = []
            missing = []
            for jn in joint_names:
                if jn in name_to_pos:
                    positions.append(name_to_pos[jn])
                else:
                    missing.append(jn)

            if missing:
                # Incomplete target for this arm on this tick -- skip rather
                # than send a partial/garbage trajectory to real hardware.
                self.get_logger().warn(f"⚠️ [{component_name}] missing joints in target: {missing}, skipping tick.")
                continue

            traj = JointTrajectory()
            traj.joint_names = joint_names
            point = JointTrajectoryPoint()
            point.positions = positions
            point.time_from_start = Duration(sec=0, nanosec=int(self.move_time_sec * 1e9))
            traj.points = [point]

            pub.publish(traj)

    def _feedback_callback(self, msg: JointState):
        """Pass the broadcaster's combined feedback through under Synapse's topic name."""
        self.feedback_pub.publish(msg)


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