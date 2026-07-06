#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import JointState
from synapse.utils.terminal_manager import KeyboardListener, BackgroundTUI
import atexit
from concurrent.futures import ThreadPoolExecutor
from synapse.brains.brain_selector import BrainSelector
from collections import deque

class ActionChunkBuffer:
    def __init__(self):
        self.action_queue = []
        self.last_valid_action = None

    def update_chunk(self, new_chunk: list):
        self.action_queue = list(new_chunk)
        
    def pop_next_action(self):
        if self.action_queue:
            self.last_valid_action = self.action_queue.pop(0)
            return self.last_valid_action, "EXECUTING_CHUNK"
        
        if self.last_valid_action is not None:
            return self.last_valid_action, "BUFFER_STARVATION_HOLD"
        
        return None, "NO_DATA"

class SynapseBTNode(Node):
    def __init__(self):
        super().__init__('synapse_bt_node')

        self.declare_parameter('bt_tick_frequency_hz', 100)
        self.declare_parameter('obs_buffer_window_size', 10)
        self.declare_parameter('brain_option', "manual")  # Placeholder for future brain options
        self.declare_parameter('muscle_option', "isaac")  # Placeholder for future muscle options
        self.declare_parameter('muscle_embodiment', "LIBERO_PANDA")  # Placeholder for future muscle embodiment options
        self.declare_parameter('checkpoint_path', "/checkpoint")

        self.tick_freq = self.get_parameter('bt_tick_frequency_hz').value
        self.obs_buffer_size = self.get_parameter('obs_buffer_window_size').value
        self.brain_option = self.get_parameter('brain_option').value
        self.muscle_option = self.get_parameter('muscle_option').value
        self.muscle_embodiment = self.get_parameter('muscle_embodiment').value
        self.checkpoint_path = self.get_parameter('checkpoint_path').value

        # Components
        adapter_config = {
            'brain_type': self.brain_option,
            'checkpoint_path': self.checkpoint_path,
            'muscle_embodiment': self.muscle_embodiment
        }
        
        self.brain_adapter = BrainSelector.get_brain(adapter_config)
        if self.brain_adapter is None:
            self.get_logger().error(f"Invalid brain option: {self.brain_option}. Please check your configuration.")
            raise ValueError(f"Invalid brain option: {self.brain_option}")

        self.action_buffer = ActionChunkBuffer()  # Manage action chunks from the brain
        self.obs_buffer = deque(maxlen=self.obs_buffer_size)
        
        # State
        self.inference_future = None
        self.inference_executor = ThreadPoolExecutor(max_workers=1)  # Dedicated thread for inference
        self.is_ticking = False
        
        # ROS2 Interfaces
        self.sub_joint_states = self.create_subscription(JointState, '/synapse/joint_states', self.obs_callback, 10)
        self.pub_brain_output = self.create_publisher(JointState, '/synapse/brain_output', 10)
        self.pub_synapse_command = self.create_publisher(String, '/synapse/command', 10)  # For future use (e.g., start/stop signals)
        
        # Behavior Tree Tick (tick_freq Hz = 1/tick_freq seconds)
        self.timer = self.create_timer(1.0 / self.tick_freq, self.tick)
        
        self.get_logger().info(f"⚙️  BT Tick Frequency: {self.tick_freq} Hz")
        self.get_logger().info(f"⚙️  Observation Buffer Window Size: {self.obs_buffer_size}")
        self.get_logger().info(f"⚙️️  Brain Option: {self.brain_option}, Checkpoint Path: {self.checkpoint_path}")
        self.get_logger().info("🎉 Synapse BT Node Ready.")
        self.get_logger().info("Controls: [s] Start | [p] Pause | [q] Quit | Manual Commands: [z/x/y/r/t/w] (for manual mode)")
        self.count = 0
        self.last_command = None

        self.terminal_ui = BackgroundTUI()
        self.key_listener = KeyboardListener()
        self.status = "Idle"

    def obs_callback(self, msg):
        self.count += 1
        # Native asynchronous buffering. Always holds the freshest data.
        # if self.count % 1000 == 0:  # Log every 1000th message to avoid spamming
        #     self.get_logger().info(f"Received obs {msg}")
        obs_dict = {"image": None, "joint": msg, "command": self.last_command}  # Placeholder for actual image_msgs
        self.obs_buffer.append(obs_dict)

    def tick(self):

        # 1. System Input Checking
        key = self.key_listener.get_key_and_clear()
        
        if key and key.startswith("CMD:"):
            command_sentence = key[4:]  # Extract the command after "CMD:"
            # self.get_logger().info(f"📝 Received command sentence: {command_sentence}")
            self.last_command = command_sentence
        else:
            match key:
                case 'q':
                    self.get_logger().info("Quitting Synapse.")
                    self.pub_synapse_command.publish(String(data="QUIT"))
                    self.last_command = None
                    raise KeyboardInterrupt
                case 's' if not self.is_ticking:
                    # self.get_logger().info(f"🌲🌲BT Ticking Started ({self.tick_freq}Hz).▶️")
                    self.is_ticking = True
                    self.last_command = None
                    self.status = "Running"
                    self.pub_synapse_command.publish(String(data="START"))
                    self.terminal_ui.update_status(self.status)
                    self.terminal_ui.log(f"Status: {self.status}")
                case 'p' if self.is_ticking:
                    # self.get_logger().info("🌲🌲BT Paused. ⏸️")
                    self.is_ticking = False
                    self.last_command = None
                    self.status = "Paused"
                    self.pub_synapse_command.publish(String(data="PAUSE"))
                    self.terminal_ui.update_status(self.status)
                    self.terminal_ui.log(f"Status: {self.status}")
                case 'z' | 'Z' | 'x' | 'X' | 'y' | 'Y' | 'r' | 'R' | 't' | 'T' | 'w' | 'W':
                    # This is for manual mode only
                    self.last_command = key
                    self.terminal_ui.log(f"Manual command: {key}")
                case _:
                    self.last_command = None
                    pass
                
            
        if not self.is_ticking:
            return
        # 2. Behavior Tree Execution Logic
        # Leaf: Format Observation
        if self.inference_future is not None and self.inference_future.done():
            new_action_chunk = self.inference_future.result()
            # self.get_logger().info(f"🌲🌲 Inference completed")
            if new_action_chunk:
                self.action_buffer.update_chunk(new_action_chunk)
            self.inference_future = None
        
        if self.inference_future is None and len(self.obs_buffer) > 0:
             # Run inference in a separate thread to avoid blocking the BT tick
            historical_obs = list(self.obs_buffer)
            self.inference_future = self.inference_executor.submit(self.brain_adapter.infer, historical_obs)
            # self.get_logger().info(f"🌲🌲 Inference submitted with obs_buffer length: {len(historical_obs)}")

        action, status = self.action_buffer.pop_next_action()

        # if status == "BUFFER_STARVATION_HOLD":
        #     self.get_logger().warning("Action buffer starvation! Holding last valid action.")

        if action is not None:
            self.pub_brain_output.publish(action)


def main(args=None):

    rclpy.init(args=args)
    node = SynapseBTNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()