import termios
import threading
import atexit
import sys
import os
import curses
import time
from collections import deque

class StdoutRedirector:
    """Catches all print() and ROS logs and pipes them to the TUI."""
    def __init__(self, tui):
        self.tui = tui

    def write(self, msg):
        clean_msg = msg
        if clean_msg:
            self.tui.log(clean_msg)

    def flush(self):
        pass

class BackgroundTUI:
    def __init__(self):
        self.status = "Idle"
        self.obs_buffer_length = self.action_buffer_length = 0
        self.current_command = ""
        self.log_buffer = deque(maxlen=50)
        self.commands_queue = deque()
        self._running = True
        
        # Lock to prevent ROS 2 main thread and UI background thread from colliding
        self._lock = threading.Lock()
        
        # ⚡ CRITICAL FIX: Hijack standard output so print() doesn't destroy curses
        self.old_stdout = sys.stdout
        self.old_stderr = sys.stderr
        sys.stdout = StdoutRedirector(self)
        sys.stderr = StdoutRedirector(self)
        
        self.thread = threading.Thread(target=self._start_curses, daemon=True)
        self.thread.start()
        
        atexit.register(self._cleanup)

    def _cleanup(self):
        self._running = False
        # Restore standard output before closing
        sys.stdout = self.old_stdout
        sys.stderr = self.old_stderr
        try:
            curses.endwin()
        except:
            pass

    def _start_curses(self):
        try:
            curses.wrapper(self._ui_loop)
        except Exception as e:
            # Safely print post-crash since we restored stdout in wrapper exit
            print(f"TUI Thread Crashed: {e}")

    def _ui_loop(self, stdscr):
        curses.curs_set(0)
        stdscr.nodelay(True)
        stdscr.timeout(50)

        # Fixed split line for controls (dynamic based on terminal size is safer, but 8 is reliable)
        split_line = 10

        while self._running:
            stdscr.erase()
            max_y, max_x = stdscr.getmaxyx()

            with self._lock:
                current_status = self.status
                obs_buffer_length = self.obs_buffer_length
                action_buffer_length = self.action_buffer_length
                current_command = self.current_command
                visible_logs = list(self.log_buffer)

            # --- 1. Draw UI ---
            try:
                top_bar = " Free(Z)e | E(X)ecute | Lea(V)e | (C)ustom Sentence ".center(max_x - 1)
                stdscr.addstr(0, 0, top_bar[:max_x - 1], curses.A_REVERSE)
                
                stdscr.addstr(2, 2, f"Status: {current_status}"[:max_x - 3], curses.A_BOLD)
                stdscr.addstr(3, 2, f"Current Command: {self.current_command}"[:max_x - 3], curses.A_BOLD)
                stdscr.addstr(5, 2, f"Action Buffer: {action_buffer_length}"[:max_x - 3])
                stdscr.addstr(6, 2, f"Observation Queue: {obs_buffer_length}"[:max_x - 3])

                stdscr.hline(split_line, 0, curses.ACS_HLINE, max_x - 1)
            except curses.error:
                pass # Terminal too small, skip drawing frame

            # --- 2. Draw Logs ---
            log_start_row = split_line + 1
            log_lines_available = max_y - log_start_row - 1
            
            logs_to_draw = visible_logs[-log_lines_available:] if log_lines_available > 0 else []
            
            for i, log_msg in enumerate(logs_to_draw):
                try:
                    stdscr.addstr(log_start_row + i, 2, log_msg[:max_x - 4])
                except curses.error:
                    pass

            stdscr.refresh()

            # --- 3. Handle Input ---
            try:
                key = stdscr.getch()
                if key != -1:
                    # Catch terminal resize keys and non-ASCII to prevent chr() crash
                    if 0 <= key <= 255:
                        char = chr(key)
                        
                        if char == 'c':
                            stdscr.nodelay(False)
                            curses.curs_set(1)
                            # ⚡ CRITICAL FIX: Removed the emoji. It breaks curses getstr offsets.
                            stdscr.addstr(split_line - 1, 2, "[CMD] Enter command: ")
                            curses.echo()
                            
                            sentence_bytes = stdscr.getstr(split_line - 1, 23, 50)
                            
                            curses.noecho()
                            curses.curs_set(0)
                            stdscr.nodelay(True)
                            
                            with self._lock:
                                self.commands_queue.append(f"CMD:{sentence_bytes.decode('utf-8')}")
                        elif char.isalpha():
                            with self._lock:
                                self.commands_queue.append(char)
            except Exception:
                pass

    # --- Thread-Safe API for ROS Code ---
    def log(self, msg):
        timestamp = time.strftime("%H:%M:%S")
        with self._lock:
            self.log_buffer.append(f"[{timestamp}] {msg}")

    def update_status(self, new_status, obs_buffer_length, action_buffer_length, command):
        with self._lock:
            self.status = new_status
            self.obs_buffer_length = obs_buffer_length
            self.action_buffer_length = action_buffer_length
            self.current_command = command

    def get_command(self):
        with self._lock:
            if self.commands_queue:
                return self.commands_queue.popleft()
            return None
