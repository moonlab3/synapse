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
from bt_parser import ScenarioParser
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

        self.muscle_embodiment = self.get_parameter('muscle_embodiment').value
        self.tick_freq = self.get_parameter('bt_tick_frequency_hz').value
        self.obs_buffer_size = self.get_parameter('obs_buffer_window_size').value
        self.brain_option = self.get_parameter('brain_option').value
        self.muscle_option = self.get_parameter('muscle_option').value
        self.camera_topic = self.get_parameter('camera_topic').value
        registry_list = self.get_parameter('brain_registry').value
        scenario_filename = self.get_parameter('scenario_filename').value

        all_params = self.get_parameters_by_prefix('')
        param_overrides = list(all_params.values())

        self.brain_adapters = {}
        # self.terminal_ui.wait_debug("BEFORE brain_adapters")

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

        if not self.brain_adapters:
            raise ValueError(f"Invalid brain option: {self.brain_option}")

        parser = ScenarioParser(synapse_node=self, terminal=self.terminal_ui)
        self.terminal_ui.wait_debug("before tree parsing")
        scenario_path = os.path.join(get_package_share_directory('synapse'), 'configs', scenario_filename)
        self.bt_root = parser.parse(scenario_path)
        self.terminal_ui.wait_debug("after tree parsing")
        self.bt_manager = py_trees.trees.BehaviourTree(self.bt_root)
        self.terminal_ui.log(f"🌲🌲 Behaviour Tree loaded from {scenario_filename}")

        self.action_buffer = ActionChunkBuffer()  # Manage action chunks from the brain
        self.obs_buffer = deque(maxlen=self.obs_buffer_size)
        
        # State
        self.inference_future = None
        self.inference_executor = ThreadPoolExecutor(max_workers=1)  # Dedicated thread for inference
        self.is_ticking = False
        self.latest_image = None

        self.terminal_ui.wait_debug("before ROS2 Interface setting")
        
        # ROS2 Interfaces
        self.sub_joint_states = self.create_subscription(JointState, '/synapse/joint_states', self.obs_callback, 10)
        if self.camera_topic is not None:
            self.sub_camera = self.create_subscription(Image, self.camera_topic, self.image_callback, 10)
        self.pub_brain_output = self.create_publisher(JointState, '/synapse/brain_output', 10)
        self.pub_synapse_command = self.create_publisher(String, '/synapse/command', 10)  # For future use (e.g., start/stop signals)
        
        self.terminal_ui.log(f"⚙️  BT Tick Frequency: {self.tick_freq} Hz")
        self.terminal_ui.log(f"⚙️  Observation Buffer Window Size: {self.obs_buffer_size}")
        self.terminal_ui.log(f"⚙️️  Brain Option: {self.brain_option}")
        self.terminal_ui.log(f"⚙️️  Hardware Setup: {self.muscle_embodiment}")
        self.terminal_ui.log(f"⚙️️  Muscle Option: {self.muscle_option}")
        self.terminal_ui.log("🎉 Synapse BT Node Ready.")
        self.terminal_ui.wait_debug("DONE DONE DONE")

        self.to_brain = "pick up the box"
        self.running_default = False
        self.status = "Idle"
        self.timer = self.create_timer(1.0 / self.tick_freq, self.tick)

        
    def image_callback(self, msg: Image):
        frame = np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.width, 3))
        self.latest_image = frame

    def obs_callback(self, msg):
        obs_dict = {"image": self.latest_image, "joint": msg, "command": self.to_brain}  # Placeholder for actual image_msgs
        self.obs_buffer.append(obs_dict)

    def tick(self):
        key = self.terminal_ui.get_command()
        
        if key and key.startswith("CMD:"):
            command_sentence = key[4:]
            self.to_brain = command_sentence
            self.running_default = False
            self.terminal_ui.log(f"entered: {command_sentence}")
        else:
            match key:
                case 'v':
                    self.terminal_ui.log("Quitting Synapse.")
                    self.pub_synapse_command.publish(String(data="QUIT"))
                    raise KeyboardInterrupt
                case 'x' if not self.is_ticking:
                    self.is_ticking = True
                    self.status = "Running"
                    self.pub_synapse_command.publish(String(data="START"))
                    self.terminal_ui.log(f"🌲🌲BT Ticking Started ({self.tick_freq}Hz).▶️ Status: {self.status}")
                case 'z' if self.is_ticking:
                    self.is_ticking = False
                    self.status = "Paused"
                    self.pub_synapse_command.publish(String(data="PAUSE"))
                    self.terminal_ui.log(f"🌲🌲BT Freezed. ⏸️ Status: {self.status}")
                case 'q':
                    self.pub_synapse_command.publish(String(data="RESET"))
                case '1' | '2' | '3' | '4' | '5' | '6' | '7' | '8' | '9' | '0':
                    idx = int(key) - 1
                    self.running_brain = self.brain_node_list[idx]
                    self.terminal_ui.log(f"idx:{idx} node: {self.running_brain}")
                    self.running_default = True
                    
                case None:
                    # if len(self.to_brain) < 2:
                        # self.to_brain = None
                    pass
                case _:
                    self.to_brain = key
            
        self.terminal_ui.update_status(
            self.status, 
            len(self.obs_buffer), 
            self.action_buffer.get_length(), 
            self.to_brain, 
            self.brain_node_map, 
            self.running_brain
        )
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
            self.inference_future = self.inference_executor.submit(self.brain_adapters[self.running_brain].infer, historical_obs, self.running_default)

        action, status = self.action_buffer.pop_next_action()

        # if status == "BUFFER_STARVATION_HOLD":
        #     self.get_logger().warning("Action buffer starvation! Holding last valid action.")

        if action is not None:
            self.pub_brain_output.publish(action)
        else:
            self.terminal_ui.log("Action buffer is empty")


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