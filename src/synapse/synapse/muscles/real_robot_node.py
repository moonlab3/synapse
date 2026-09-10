#!/usr/bin/env python3
import rclpy
from sensor_msgs.msg import JointState
from synapse.utils.embodiment_parser import EmbodimentParser
from synapse.muscles.base_muscle import BaseMuscle

class RealRobotNode(BaseMuscle):
    """Targets now go straight from SynapseMainNode to each component's real
    controller topic (see EmbodimentParser.resolve_target). This node only:
      1. Relays the hardware's combined /joint_states feed back onto
         Synapse's expected states topic.
      2. Handles /synapse/command QUIT/ESTOP via BaseMuscle.
    """
    def __init__(self, node_name="real_robot_node", parameter_overrides=None):
        super().__init__(node_name, parameter_overrides)

        embodiment_name = self.get_parameter('embodiment_name').value
        parser = EmbodimentParser(embodiment_name)
        self.robots_cfg = parser.get_robots()

        articulation_groups = parser.get_articulations()
        group_name, group_cfg = next(iter(articulation_groups.items()))
        self.group_states_topic = parser.resolve_states(group_cfg, group_name)

        self.last_known_position = {name: None for name in self.robots_cfg}

        self.feedback_sub = self.create_subscription(JointState, '/joint_states', self._feedback_callback, 10)
        self.feedback_pub = (
            self.create_publisher(JointState, self.group_states_topic, 10)
            if self.group_states_topic else None
        )

        self.get_logger().info(
            f"🌉 RealRobotNode feedback bridge ready for '{embodiment_name}' "
            f"-> {self.group_states_topic or '(no states topic configured)'}"
        )

    def _feedback_callback(self, msg: JointState):
        if self.feedback_pub is None:
            return

        name_to_pos = dict(zip(msg.name, msg.position))
        for component_name, cfg in self.robots_cfg.items():
            joint_names = cfg.get('joint_names', [])
            positions = [name_to_pos[jn] for jn in joint_names if jn in name_to_pos]
            if len(positions) == len(joint_names) and joint_names:
                self.last_known_position[component_name] = positions

        combined = JointState()
        combined.header = msg.header
        combined.name = list(msg.name)
        combined.position = list(msg.position)

        for component_name, cfg in self.robots_cfg.items():
            joint_names = cfg.get('joint_names', [])
            if all(jn in msg.name for jn in joint_names):
                continue  # already in the raw feedback, don't duplicate
            positions = self.last_known_position.get(component_name)
            if positions is None or len(positions) != len(joint_names):
                positions = [0.0] * len(joint_names)
            combined.name.extend(joint_names)
            combined.position.extend(positions)

        self.feedback_pub.publish(combined)

    def publish_trajectory(self, action_chunk):
        self.get_logger().warning("publish_trajectory() called, but targets bypass this node now.")

    def hardware_estop(self):
        self.get_logger().error("🚨 ESTOP requested — no automatic real-arm stop wired yet.")

    def hardware_shutdown(self):
        self.get_logger().info("Real robot bridge shutting down.")

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