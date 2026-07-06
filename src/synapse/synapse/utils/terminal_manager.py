import termios
import threading
import atexit
import sys
import os
import curses
import time
from collections import deque

import curses
import threading
import time
import atexit
from collections import deque

class BackgroundTUI:
    def __init__(self):
        self.status = "Idle"
        self.log_buffer = deque(maxlen=50)
        self.commands_queue = deque()
        self._running = True
        
        # Lock to prevent ROS 2 main thread and UI background thread from colliding
        self._lock = threading.Lock()
        
        self.thread = threading.Thread(target=self._start_curses, daemon=True)
        self.thread.start()
        
        atexit.register(self._cleanup)

    def _cleanup(self):
        self._running = False
        try:
            curses.endwin()
        except:
            pass

    def _start_curses(self):
        try:
            curses.wrapper(self._ui_loop)
        except Exception as e:
            # If it crashes, print the error to terminal after curses closes
            print(f"TUI Thread Crashed: {e}")

    def _ui_loop(self, stdscr):
        curses.curs_set(0)
        stdscr.nodelay(True)
        stdscr.timeout(100)  # Refresh 10 Hz

        split_line = 18

        while self._running:
            stdscr.erase()
            max_y, max_x = stdscr.getmaxyx()

            # Safely grab a snapshot of the data so we don't hold the lock while drawing
            with self._lock:
                current_status = self.status
                visible_logs = list(self.log_buffer)

            # --- 1. Draw UI (with strict bounds checking) ---
            try:
                # Top bar (subtract 1 from max_x to prevent line-wrap crashes)
                top_bar = " (Q)uit | (S)tart | (P)ause | (C)ommand ".center(max_x - 1)
                stdscr.addstr(0, 0, top_bar[:max_x - 1], curses.A_REVERSE)
                
                stdscr.addstr(2, 2, "[S]tart - Start the process"[:max_x - 3])
                stdscr.addstr(3, 2, "[P]ause - Pause the process"[:max_x - 3])
                stdscr.addstr(4, 2, "[C]     - Enter custom sentence"[:max_x - 3])
                stdscr.addstr(6, 2, f"Status: {current_status}"[:max_x - 3], curses.A_BOLD)

                stdscr.hline(split_line, 0, curses.ACS_HLINE, max_x - 1)
            except curses.error:
                # Terminal might be too small, ignore drawing error for this frame
                pass

            # --- 2. Draw Logs ---
            log_start_row = split_line + 1
            log_lines_available = max_y - log_start_row - 1
            
            # Slice the list to only get what fits on screen
            logs_to_draw = visible_logs[-log_lines_available:] if log_lines_available > 0 else []
            
            for i, log_msg in enumerate(logs_to_draw):
                try:
                    # Cut string short to avoid wrap-around exceptions
                    stdscr.addstr(log_start_row + i, 2, log_msg[:max_x - 4])
                except curses.error:
                    pass

            stdscr.refresh()

            # --- 3. Handle Input ---
            try:
                key = stdscr.getch()
                if key != -1:
                    char = chr(key).lower()
                    
                    if char in ['q', 's', 'p']:
                        with self._lock:
                            self.commands_queue.append(char)
                    elif char == 'c':
                        # Pause UI, show cursor, get string
                        stdscr.nodelay(False)
                        curses.curs_set(1)
                        stdscr.addstr(split_line, 2, " 📝 Enter command: ")
                        curses.echo()
                        
                        sentence_bytes = stdscr.getstr(split_line, 21, 50)
                        
                        curses.noecho()
                        curses.curs_set(0)
                        stdscr.nodelay(True)
                        
                        with self._lock:
                            self.commands_queue.append(f"CMD:{sentence_bytes.decode('utf-8')}")
            except ValueError:
                pass

    # --- Thread-Safe API for ROS Code ---
    
    def log(self, msg):
        """Thread-safe logging"""
        timestamp = time.strftime("%H:%M:%S")
        with self._lock:
            self.log_buffer.append(f"[{timestamp}] {msg}")

    def update_status(self, new_status):
        with self._lock:
            self.status = new_status

    def get_command(self):
        with self._lock:
            if self.commands_queue:
                return self.commands_queue.popleft()
            return None

class KeyboardListener:
    def __init__(self):
        self.current_key = None
        self._lock = threading.Lock()
        
        # Bypassing ROS 2 IO wrappers using your original approach
        self.fd = os.open('/dev/tty', os.O_RDWR)
        
        self.old_settings = termios.tcgetattr(self.fd)
        self.new_settings = termios.tcgetattr(self.fd)
        # Disable canonical mode (require Enter) and echo
        self.new_settings[3] = self.new_settings[3] & ~(termios.ICANON | termios.ECHO)
        
        termios.tcsetattr(self.fd, termios.TCSADRAIN, self.new_settings)
        
        atexit.register(self._restore)
        
        self.thread = threading.Thread(target=self._listen, daemon=True)
        self.thread.start()

    def _restore(self):
        termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old_settings)
        os.close(self.fd)

    def _listen(self):
        while True:
            try:
                # Blocks until a key is pressed, but does not block ROS tick() 
                # because this is running in a daemon thread.
                char_bytes = os.read(self.fd, 1)
                if not char_bytes:
                    continue
                
                key = char_bytes.decode('utf-8', errors='ignore')
                
                # Ctrl+C
                if key == '\x03': 
                    with self._lock:
                        self.current_key = 'q'
                    break
                    
                # Sentence Input Mode
                if key in ['c', 'C']:
                    # Temporarily restore normal terminal settings
                    termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old_settings)
                    
                    sys.stdout.write("\033[s\033[8;1H\033[2K")
                    sys.stdout.write(" 📝 Enter command sentence: ")
                    sys.stdout.flush()
                    
                    # Read character by character until Enter is pressed
                    sentence = ""
                    while True:
                        c = os.read(self.fd, 1).decode('utf-8', errors='ignore')
                        if c == '\n' or c == '\r':
                            break
                        # Handle backspace
                        if c == '\x7f':
                            sentence = sentence[:-1]
                            sys.stdout.write("\b \b")
                            sys.stdout.flush()
                        else:
                            sentence += c
                            sys.stdout.write(c)
                            sys.stdout.flush()
                        
                    # Clear input line and restore cursor
                    sys.stdout.write("\033[8;1H\033[2K\033[u")
                    sys.stdout.flush()
                    
                    # Re-apply strict terminal settings
                    termios.tcsetattr(self.fd, termios.TCSADRAIN, self.new_settings)
                    
                    with self._lock:
                        self.current_key = f"CMD:{sentence.strip()}"
                    continue
                    
                # Single Key Commands
                if key.lower() in ['s', 'p', 'q', 'z', 'x', 'y', 'r', 't', 'w']:
                    with self._lock:
                        self.current_key = key
                        
            except Exception as e:
                # If it fails, print immediately so it doesn't fail silently
                sys.stdout.write(f"\033[1;1H Listener Error: {e} \n")
                sys.stdout.flush()
                break

    def get_key_and_clear(self):
        with self._lock:
            key = self.current_key
            self.current_key = None
            return key
