import threading
import atexit
import sys
import curses
import time
from collections import deque

class StdoutRedirector:
    """Catches all print() and ROS logs and pipes them to the TUI."""
    def __init__(self, tui):
        self.tui = tui

    def write(self, msg):
        if not msg:
            return
        for line in msg.splitlines():
            if line.strip():
                self.tui.log(line)

    def flush(self):
        pass

class BackgroundTUI:
    def __init__(self, debug_mode=False):
        self.status = "Idle"
        self.obs_buffer_length = self.action_buffer_length = 0
        self.action_buffer_status = self.current_command = self.brain_node_map = self.running_brain = ""
        self.log_buffer = deque(maxlen=50)
        self.commands_queue = deque()
        self._running = True
        self.debug_mode = debug_mode
        
        self._lock = threading.Lock()
        
        self.old_stdout = sys.stdout
        self.old_stderr = sys.stderr
        sys.stdout = StdoutRedirector(self)
        sys.stderr = StdoutRedirector(self)
        
        self.thread = threading.Thread(target=self._start_curses, daemon=True)
        self.thread.start()
        
        atexit.register(self._cleanup)

    def wait_debug(self, msg):
        if self.debug_mode:
            mm = f"[DEBUG] Press any key to continue. [{msg}]"
            self.log(mm)
            while self._running:
                key = self.get_command()
                if key is not None:
                    return key
                time.sleep(0.05)

    def _cleanup(self):
        self._running = False
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

        split_line = 10

        while self._running:
            stdscr.erase()
            max_y, max_x = stdscr.getmaxyx()

            with self._lock:
                current_status = self.status
                obs_buffer_length = self.obs_buffer_length
                action_buffer_status = self.action_buffer_status
                action_buffer_length = self.action_buffer_length
                current_command = self.current_command
                node_map = self.brain_node_map
                running_brain = self.running_brain
                visible_logs = list(self.log_buffer)

            # --- 1. Draw UI ---
            try:
                top_bar = " Free(Z)e | E(X)ecute | S(C)enario Run | Lea(V)e | (B)ackup command | (Q)uick Reset".center(max_x - 1)
                stdscr.addstr(0, 0, top_bar[:max_x - 1], curses.A_REVERSE)
                
                stdscr.addstr(1, 2, f"{node_map}"[:max_x - 3], curses.A_BOLD)
                stdscr.addstr(3, 2, f"Status: {current_status}"[:max_x - 3], curses.A_BOLD)
                stdscr.addstr(4, 2, f"Running: {running_brain}"[:max_x - 3], curses.A_BOLD)
                stdscr.addstr(5, 2, f"Current Command: {current_command}"[:max_x - 3], curses.A_BOLD)
                stdscr.addstr(6, 2, f"Action Buffer Status: {action_buffer_status} ({action_buffer_length})"[:max_x - 3])
                stdscr.addstr(7, 2, f"Observation Queue: {obs_buffer_length}"[:max_x - 3])

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
                        # self.log(f"key:[{char}]")
                        
                        if char == 'b':
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
                        elif char.isalpha() or char.isdigit():
                            with self._lock:
                                self.commands_queue.append(char)
            except Exception:
                pass

    # --- Thread-Safe API for ROS Code ---
    def log(self, msg):
        timestamp = time.strftime("%H:%M:%S")
        with self._lock:
            self.log_buffer.append(f"[{timestamp}] {msg}")

    def update_status(self, new_status, obs_buffer_length, action_buffer_status, action_buffer_length, command, node_map, running_brain):
        with self._lock:
            self.status = new_status
            self.obs_buffer_length = obs_buffer_length
            self.action_buffer_status = action_buffer_status
            self.action_buffer_length = action_buffer_length
            self.current_command = command
            self.brain_node_map = node_map
            self.running_brain = running_brain

    def get_command(self):
        with self._lock:
            if self.commands_queue:
                return self.commands_queue.popleft()
            return None
