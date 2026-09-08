#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration
from synapse.utils.embodiment_parser import EmbodimentParser

# Maps Synapse's component names (from embodiment_configs.yaml) to the real
# xarm_ros2 trajectory controller topics confirmed working via CLI testing.
# Only manipulators go through joint_trajectory_controller here; end-effectors
# (hands) would need their own real driver, not handled by this bridge yet.
CONTROLLER_TOPICS = {
    'left_arm': '/left_xarm7_traj_controller/joint_trajectory',
    'right_arm': '/right_xarm7_traj_controller/joint_trajectory',
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
        self.move_time_sec = self.get_parameter('move_time_sec').value
        self.min_move_time_sec = self.get_parameter('min_move_time_sec').value

        self.last_positions = {}
        self.last_command_time = None

        parser = EmbodimentParser(embodiment_name)
        self.robots_cfg = parser.get_robots()  # flattened: left_arm, right_arm, left_hand, right_hand, ...

        self.last_known_position = {name: None for name in self.robots_cfg}

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

        # obs_callback already expects to split per-component.
        self.feedback_sub = self.create_subscription(
            JointState, '/joint_states', self._feedback_callback, 10
        )
        self.feedback_pub = self.create_publisher(
            JointState, '/synapse/joint_states/dual_xarm_unit', 10
        )

        self.sub_synapse_command = self.create_subscription(String, '/synapse/command', self.synapse_command_callback, 10)

        self.get_logger().info(
            f"🌉 RealRobotNode bridge ready for '{embodiment_name}': "
            f"controllers={list(self.traj_pubs.keys())}, move_time_sec={self.move_time_sec}"
        )

    def _target_callback(self, msg: JointState):
        now = self.get_clock().now()
        if self.last_command_time is not None:
            dt = (now - self.last_command_time).nanoseconds / 1e9
            dt = max(dt, self.min_move_time_sec)
        else:
            dt = self.move_time_sec
        self.last_command_time = now

        name_to_pos = dict(zip(msg.name, msg.position))

        for component_name, cfg in self.robots_cfg.items():
            joint_names = cfg.get('joint_names', [])
            positions = [name_to_pos[jn] for jn in joint_names if jn in name_to_pos]
            if len(positions) == len(joint_names) and joint_names:
                self.last_known_position[component_name] = positions

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
                self.get_logger().warn(f"⚠️ [{component_name}] missing joints in target: {missing}, skipping tick.")
                continue

            prev_positions = self.last_positions.get(component_name)
            if prev_positions is not None and len(prev_positions) == len(positions):
                velocities = [(p - c) / dt for p, c in zip(positions, prev_positions)]
            else:
                velocities = [0.0] * len(positions)


            traj = JointTrajectory()
            traj.joint_names = joint_names
            point = JointTrajectoryPoint()
            point.positions = positions
            point.velocities = velocities
            point.time_from_start = Duration(sec=0, nanosec=int(self.move_time_sec * 1e9))
            traj.points = [point]

            pub.publish(traj)

            self.last_positions[component_name] = positions

    def _feedback_callback(self, msg: JointState):
        """Pass the broadcaster's combined feedback through under Synapse's topic name."""

        combined = JointState()
        combined.header = msg.header
        combined.name = list(msg.name)
        combined.position = list(msg.position)

        for component_name, cfg in self.robots_cfg.items():
            if component_name in CONTROLLER_TOPICS:
                continue
            joint_names = cfg.get('joint_names', [])
            positions = self.last_known_position.get(component_name)
            if positions is None or len(positions) != len(joint_names):
                positions = [0.0] * len(joint_names)

            combined.name.extend(joint_names)
            combined.position.extend(positions)

        # self.feedback_pub.publish(msg)
        self.feedback_pub.publish(combined)

    def synapse_command_callback(self, msg: String):
        """Receives commands from synapse_bt_node (e.g., start, stop)"""
        command = msg.data
        match command:
            case "START":
                self.get_logger().info("Received START command. Resuming simulation.")
            case "PAUSE":
                self.get_logger().info("Received PAUSE command. Pausing simulation.")
            case "QUIT":
                self.get_logger().info("Received QUIT command. Shutting down Isaac Muscle Node.")
                rclpy.shutdown()
            case "RESET":
                self.get_logger().info("Received RESET command. Resetting robot to initial pose")
                # self.target_joints = self.initial_positions.copy()
                # self.reset_process = 0

            case _:
                self.get_logger().warn(f"Unknown command received: {command}")


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