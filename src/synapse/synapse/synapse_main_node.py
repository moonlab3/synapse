#!/usr/bin/env python3
import rclpy
import numpy as np
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import JointState, Image
from collections import deque
from synapse.utils.terminal_manager import BackgroundTUI
from concurrent.futures import ThreadPoolExecutor
from synapse.brains.brain_selector import BrainSelector
from synapse.utils.scenario_parser import ScenarioParser
from synapse.utils.embodiment_parser import EmbodimentParser
import functools
import py_trees
import os
from ament_index_python.packages import get_package_share_directory

class ActionChunkBuffer:
    def __init__(self):
        self.action_queue = []
        self.last_valid_action = None

    def update_chunk(self, new_chunk: list):
        self.action_queue = list(new_chunk)
        
    def get_length(self):
        return len(self.action_queue)

    def pop_next_action(self):
        if self.action_queue:
            self.last_valid_action = self.action_queue.pop(0)
            return self.last_valid_action, "EXECUTING_CHUNK"
        
        if self.last_valid_action is not None:
            return self.last_valid_action, "BUFFER_STARVATION_HOLD"
        
        return None, "NO_DATA"

class SynapseMainNode(Node):
    def __init__(self):
        super().__init__(
            'synapse_bt_node',
            allow_undeclared_parameters=True,
            automatically_declare_parameters_from_overrides=True
            )

        self.terminal_ui = BackgroundTUI(self.get_parameter('debug_mode').value)

        embodiment_name = self.get_parameter('embodiment_name').value
        tick_freq = self.get_parameter('bt_tick_frequency_hz').value
        obs_buffer_size = self.get_parameter('obs_buffer_window_size').value
        muscle_option = self.get_parameter('muscle_option').value
        registry_list = self.get_parameter('brain_registry').value
        scenario_filename = self.get_parameter('scenario_filename').value
        if scenario_filename is None:
            scenario_filename = "scn_pilot.yaml"

        all_params = self.get_parameters_by_prefix('')
        param_overrides = list(all_params.values())

        self.brain_adapters = {}
        for entry in registry_list:
            node_name, adapter_type = entry.split(':')
            self.brain_adapters[node_name] = BrainSelector.get_brain(
                self.terminal_ui,
                brain_type=adapter_type,
                node_name=node_name,
                parameter_overrides=param_overrides
            )

        self.brain_node_list = list(self.brain_adapters.keys())
        self.brain_node_num = len(self.brain_adapters)
        self.brain_node_map = "Brain Adapters "
        self.running_brain = self.brain_node_list[0]
        for i, name in enumerate(self.brain_node_list):
            self.brain_node_map += f"[{i+1}: {name}]  "
        self.terminal_ui.log(f"⚙️ ros2 brains{self.brain_node_map} initialized")

        scenario_parser = ScenarioParser(synapse_node=self)
        scenario_path = os.path.join(get_package_share_directory('synapse'), 'configs', scenario_filename)
        self.bt_root = scenario_parser.parse(scenario_path)
        self.bt_manager = py_trees.trees.BehaviourTree(self.bt_root)
        self.terminal_ui.log(f"🌲🌲 Behaviour Tree loaded from {scenario_filename}")

        embodiment_parser = EmbodimentParser(embodiment_name)
        cameras = embodiment_parser.get_cameras()
        robots = embodiment_parser.get_robots()

        self.camera_subscribers = {}
        for name, cfg in cameras.items():
            topic = cfg.get('topic', f'/synapse/camera/{name}/image_raw')
            self.camera_subscribers[name] = self.create_subscription(
                Image, topic, 
                functools.partial(self.image_callback, topic_name=name), 10)

        self.target_publishers = {}
        self.robot_subscribers = {}
        for name, cfg in robots.items():
            joint_states = cfg.get('states', f'/synapse/joint_states/{name}')
            self.robot_subscribers[name] = self.create_subscription(
                JointState, joint_states,
                functools.partial(self.obs_callback, topic_name=name), 10)

            target = cfg.get('target', f'/synapse/target/{name}')
            self.target_publishers[name] = self.create_publisher(JointState, target, 10)

        self.terminal_ui.log(f"⚙️ ros2 topics initialized")
        self.latest_images = {}
        self.latest_joints = {}
        self.updated_joints = set()
        self.expected_robots = set(robots.keys())

        self.action_buffer = ActionChunkBuffer()  # Manage action chunks from the brain
        self.obs_buffer = deque(maxlen=obs_buffer_size)
        
        # State
        self.inference_future = None
        self.inference_executor = ThreadPoolExecutor(max_workers=1)  # Dedicated thread for inference
        self.is_ticking = False

        self.pub_synapse_command = self.create_publisher(String, '/synapse/command', 10)  # For future use (e.g., start/stop signals)
        
        self.terminal_ui.log(f"⚙️  BT Tick Frequency: {tick_freq} Hz, Observation Buffer Size: {obs_buffer_size}")
        self.terminal_ui.log(f"⚙️️  Muscle: {muscle_option}, Embodiment Config: {embodiment_name}")
        self.terminal_ui.log("🎉 Synapse BT Node Ready.")

        self.to_brain = None
        self.last_command = self.last_command_to_show = ""
        self.running_default = True
        self.status = "Idle"
        self.timer = self.create_timer(1.0 / tick_freq, self.tick)

        
    def image_callback(self, msg: Image, topic_name: str):
        frame = np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.width, 3))
        self.latest_images[topic_name] = frame

    def obs_callback(self, msg: JointState, topic_name: str):
        self.latest_joints[topic_name] = msg
        self.updated_joints.add(topic_name)
        if self.to_brain is None and len(self.last_command) == 1:
            self.last_command = ""
        if self.to_brain is not None:
            self.last_command = self.to_brain
        if len(self.updated_joints) == len (self.expected_robots):
            obs_dict = {
                "images": self.latest_images.copy(), 
                "joints": self.latest_joints.copy(),
                "command": self.last_command
                }
            self.obs_buffer.append(obs_dict)
            self.updated_joints.clear()

    def tick(self):
        key = self.terminal_ui.get_command()
        
        if key and key.startswith("CMD:"):
            command_sentence = key[4:]
            self.to_brain = self.last_command_to_show = command_sentence
            self.running_default = False
            self.terminal_ui.log(f"entered: {command_sentence}")
        else:
            match key:
                case 'v':
                    self.pub_synapse_command.publish(String(data="QUIT"))
                    raise KeyboardInterrupt
                case 'x' if not self.is_ticking:
                    self.is_ticking = True
                    self.status = "Running"
                    self.pub_synapse_command.publish(String(data="START"))
                    self.terminal_ui.log(f"🌲🌲BT Ticking Started.▶️ Status: {self.status}")
                case 'z' if self.is_ticking:
                    self.is_ticking = False
                    self.status = "Paused"
                    self.pub_synapse_command.publish(String(data="PAUSE"))
                    self.terminal_ui.log(f"🌲🌲BT Freezed. ⏸️ Status: {self.status}")
                case 'q':
                    self.pub_synapse_command.publish(String(data="RESET"))
                case '1' | '2' | '3' | '4' | '5' | '6' | '7' | '8' | '9' | '0':
                    idx = int(key) - 1
                    if idx + 1 <= self.brain_node_num:
                        self.running_brain = self.brain_node_list[idx]
                        self.terminal_ui.log(f"idx:{idx} node: {self.running_brain}")
                        self.running_default = True
                case None:
                    self.to_brain = None
                    pass
                case _:
                    self.to_brain = self.last_command_to_show = key

        if not self.is_ticking:
            return

        # 2. Behavior Tree Execution Logic
        if self.inference_future is not None and self.inference_future.done():
            new_action_chunk = self.inference_future.result()
            if new_action_chunk:
                self.action_buffer.update_chunk(new_action_chunk)
            self.inference_future = None
        
        if self.inference_future is None and len(self.obs_buffer) > 0:
            # Run inference in a separate thread to avoid blocking the BT tick
            historical_obs = list(self.obs_buffer)
            # self.inference_future = self.inference_executor.submit(self.brain_adapter.infer, historical_obs)
            self.inference_future = self.inference_executor.submit(
                self.brain_adapters[self.running_brain].infer, 
                historical_obs, 
                self.running_default
                )

        action, status = self.action_buffer.pop_next_action()

        self.terminal_ui.update_status(
            self.status, 
            len(self.obs_buffer), 
            status, 
            self.action_buffer.get_length(), 
            self.last_command_to_show, 
            self.brain_node_map, 
            self.running_brain
        )

        if action is not None:
            for robot, msg in action.items():
                if robot in self.target_publishers:
                    self.target_publishers[robot].publish(msg)



def main(args=None):

    rclpy.init(args=args)
    node = SynapseMainNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()