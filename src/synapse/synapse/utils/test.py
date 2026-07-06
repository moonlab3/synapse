#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import curses
from collections import deque
import time

class MyNode(Node):
    def __init__(self):
        super().__init__('my_tui_node')
        self.status = "Idle"
        
        # We store logs in a scrolling queue to display them safely in the UI
        self.log_buffer = deque(maxlen=30) 
        
        # Core node logic runs on a timer
        self.timer = self.create_timer(0.1, self.tick)

    def tick(self):
        """Your main ROS 2 logic goes here."""
        if self.status == "Running":
            # Example of doing work
            pass

    def tui_log(self, msg):
        """Use this instead of self.get_logger().info() for UI logs"""
        timestamp = time.strftime("%H:%M:%S")
        self.log_buffer.append(f"[{timestamp}] {msg}")
        # You can still call self.get_logger().info(msg) here if you want it saved to disk

def run_tui(stdscr, node):
    """The main UI and Event Loop"""
    # curses setup
    curses.curs_set(0)      # Hide cursor
    stdscr.nodelay(True)    # Make getch() non-blocking so ROS keeps spinning
    
    # Define screen layout
    split_line = 8

    while rclpy.ok():
        stdscr.erase()
        max_y, max_x = stdscr.getmaxyx()

        # --- 1. Draw Top Menu ---
        top_bar = " (Q)uit | (S)tart | (P)ause | (C)ommand "
        stdscr.addstr(0, 0, top_bar.center(max_x), curses.A_REVERSE)
        
        stdscr.addstr(2, 2, "[S]tart - Start the process")
        stdscr.addstr(3, 2, "[P]ause - Pause the process")
        stdscr.addstr(4, 2, "[C]     - Enter custom sentence")
        stdscr.addstr(6, 2, f"Status: {node.status}", curses.A_BOLD)

        # Draw dividing line
        stdscr.hline(split_line, 0, curses.ACS_HLINE, max_x)

        # --- 2. Draw Logs ---
        log_start_row = split_line + 1
        log_lines_available = max_y - log_start_row - 1
        
        # Get only the logs that fit on screen
        visible_logs = list(node.log_buffer)[-log_lines_available:]
        for i, log_msg in enumerate(visible_logs):
            # Truncate log to terminal width to prevent wrap-around visual bugs
            stdscr.addstr(log_start_row + i, 2, log_msg[:max_x - 3])

        stdscr.refresh()

        # --- 3. Handle Keyboard Input ---
        try:
            key = stdscr.getch()
            if key != -1:
                char = chr(key).lower()
                
                if char == 'q':
                    break
                elif char == 's':
                    node.status = "Running"
                    node.tui_log("Process started.")
                elif char == 'p':
                    node.status = "Paused"
                    node.tui_log("Process paused.")
                elif char == 'c':
                    # Switch to blocking mode for sentence input
                    stdscr.nodelay(False)
                    curses.curs_set(1)
                    
                    # Draw prompt
                    stdscr.addstr(split_line, 2, " 📝 Enter command: ")
                    curses.echo()
                    
                    # Wait for user to type sentence and press Enter
                    # 50 is the max string length
                    sentence_bytes = stdscr.getstr(split_line, 21, 50) 
                    sentence = sentence_bytes.decode('utf-8')
                    
                    # Put UI back to non-blocking mode
                    curses.noecho()
                    curses.curs_set(0)
                    stdscr.nodelay(True)
                    
                    node.tui_log(f"Received sentence: {sentence}")
        except ValueError:
            pass

        # --- 4. Spin ROS 2 ---
        # Instead of rclpy.spin(), we spin once per loop to keep the UI responsive
        rclpy.spin_once(node, timeout_sec=0.05)

def main(args=None):
    rclpy.init(args=args)
    node = MyNode()
    
    try:
        # curses.wrapper safely initializes the terminal and restores it on crash/exit
        curses.wrapper(run_tui, node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
