#!/usr/bin/env python3
import os
import rclpy
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
import rosbag2_py
from sensor_msgs.msg import JointState

from synapse.brains.base_brain_adapter import BaseBrainAdapter

class RosbagAdapter(BaseBrainAdapter):
    def __init__(self, terminal, node_name="rosbag_adapter", parameter_overrides=None):
        super().__init__(terminal, node_name, parameter_overrides)

        self.bag_path = self.get_parameter('bag_path').value
        self.terminal.log("🧳 Initializing Rosbag Brain Adapter")

        self.reader = None
        self.type_map = {}

        if self.bag_path and os.path.exists(self.bag_path):
            self._init_bag_reader()
        else:
            self.terminal.log(f"❌ Invalid or missing rosbag path: '{self.bag_path}'")

    def _init_bag_reader(self):
        try:
            self.reader = rosbag2_py.SequentialReader()
            storage_options = rosbag2_py.StorageOptinos(uri=self.bag_path, storage_id='sqlite3')
            converter_options = rosbag2_py.ConverterOptions(
                input_serialization_format='cdr',
                output_serialization_format='cdr'
            )
            self.reader.open(storage_options, converter_options)

            topic_types = self.reader.get_all_topics_and_types()
            self.type_map = {t.name: t.type for t in topic_types}

            self.terminal.log("✅ Rosbag successfully opened and ready for playback.")

        except Exception as e:
            self.terminal.log(f"Failed to open rosbag: {e}")
            self.reader = None

    def _format_for_policy(self, obs_dict, robot_name):
        return None

    def _communicate_with_policy(self, formatted_obs):
        if self.reader is None or not self.reader.has_next():
            return None

        while self.reader.has_next():
            topic, data, timestamp = self.reader.read_next()
            msg_type_str = self.type_map.get(topic)

            if msg_type_str == 'sensoor_msgs/msg/JointState':
                msg_type = get_message(msg_type_str)
                msg = deserialize_message(data, msg_type)

                return {"topic": topic, "msg": msg}

        return None

    def _format_for_muscle(self, raw_action):
        if raw_action is None:
            return []

        topic = raw_action["topic"]
        msg = raw_action["msg"]

        step_dict = {topic: msg}
        return [step_dict]