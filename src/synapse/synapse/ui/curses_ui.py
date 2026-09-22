"""Curses front-end -- the original BackgroundTUI, now behind BaseUI.

Behaviour is unchanged from the terminal_manager version: a background thread
owns the screen, ROS code pushes state in from whatever thread it likes, and
stdout/stderr are captured into the log pane. The differences are that the
stream capture now happens in start() instead of as a constructor side effect,
and that keypresses leave here as Intents rather than raw characters.
"""
import atexit
import curses
import sys
import threading
import time
from collections import deque
from typing import Optional

from .base import (BaseUI, HELP_LINE, Intent, LogLevel, UIEvent, UIStatus,
                   format_brain_map, intent_from_key)


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


class CursesUI(BaseUI):
    SPLIT_LINE = 10

    def __init__(self, min_level: LogLevel = LogLevel.INFO):
        super().__init__(min_level)
        self._status = UIStatus()
        self.log_buffer = deque(maxlen=200)
        self._events = deque()
        self._running = False

        self._lock = threading.Lock()
        self.old_stdout = self.old_stderr = None
        self.thread = None

    # --- lifecycle -------------------------------------------------------
    def start(self):
        if self._running:
            return
        self._running = True

        self.old_stdout, self.old_stderr = sys.stdout, sys.stderr
        sys.stdout = StdoutRedirector(self)
        sys.stderr = StdoutRedirector(self)

        self.thread = threading.Thread(target=self._start_curses, daemon=True)
        self.thread.start()
        atexit.register(self.shutdown)

    def shutdown(self):
        self._running = False
        if self.old_stdout is not None:
            sys.stdout, sys.stderr = self.old_stdout, self.old_stderr
            self.old_stdout = self.old_stderr = None
        try:
            curses.endwin()
        except Exception:
            pass

    # --- BaseUI ----------------------------------------------------------
    def log(self, msg, level: LogLevel = LogLevel.INFO):
        if level < self.min_level:
            return
        timestamp = time.strftime("%H:%M:%S")
        with self._lock:
            self.log_buffer.append(f"[{timestamp}] {msg}")

    def update_status(self, status: UIStatus):
        with self._lock:
            self._status = status

    def poll(self) -> Optional[UIEvent]:
        with self._lock:
            return self._events.popleft() if self._events else None

    # --- curses thread ---------------------------------------------------
    def _start_curses(self):
        try:
            curses.wrapper(self._ui_loop)
        except Exception as e:
            # Safely print post-crash since wrapper's exit restored the screen.
            print(f"TUI Thread Crashed: {e}")

    def _ui_loop(self, stdscr):
        curses.curs_set(0)
        stdscr.nodelay(True)
        stdscr.timeout(50)

        while self._running:
            stdscr.erase()
            max_y, max_x = stdscr.getmaxyx()

            with self._lock:
                st = self._status
                visible_logs = list(self.log_buffer)

            self._draw_header(stdscr, st, max_x)
            self._draw_logs(stdscr, visible_logs, max_y, max_x)
            stdscr.refresh()
            self._handle_input(stdscr)

    def _draw_header(self, stdscr, st: UIStatus, max_x):
        try:
            stdscr.addstr(0, 0, HELP_LINE.center(max_x - 1)[:max_x - 1], curses.A_REVERSE)
            stdscr.addstr(1, 2, format_brain_map(st.brain_names)[:max_x - 3], curses.A_BOLD)
            stdscr.addstr(3, 2, f"Status: {st.status}"[:max_x - 3], curses.A_BOLD)
            stdscr.addstr(4, 2, f"Running: {st.running_brain}"[:max_x - 3], curses.A_BOLD)
            stdscr.addstr(5, 2, f"Current Command: {st.command}"[:max_x - 3], curses.A_BOLD)
            stdscr.addstr(6, 2, f"Action Buffer Status: {st.action_buffer_status} "
                                f"({st.action_buffer_length})"[:max_x - 3])
            stdscr.addstr(7, 2, f"Observation Queue: {st.obs_buffer_length}"[:max_x - 3])
            stdscr.hline(self.SPLIT_LINE, 0, curses.ACS_HLINE, max_x - 1)
        except curses.error:
            pass  # Terminal too small, skip drawing frame

    def _draw_logs(self, stdscr, visible_logs, max_y, max_x):
        log_start_row = self.SPLIT_LINE + 1
        log_lines_available = max_y - log_start_row - 1
        logs_to_draw = visible_logs[-log_lines_available:] if log_lines_available > 0 else []

        for i, log_msg in enumerate(logs_to_draw):
            try:
                stdscr.addstr(log_start_row + i, 2, log_msg[:max_x - 4])
            except curses.error:
                pass

    def _handle_input(self, stdscr):
        try:
            key = stdscr.getch()
            if key == -1:
                return
            # Catch terminal resize keys and non-ASCII to prevent chr() crash
            if not 0 <= key <= 255:
                return
            char = chr(key)

            if char == 'b':
                event = UIEvent(Intent.SEND_COMMAND, self._prompt_sentence(stdscr))
            else:
                event = intent_from_key(char)

            if event is not None:
                with self._lock:
                    self._events.append(event)
        except Exception:
            pass

    def _prompt_sentence(self, stdscr) -> str:
        stdscr.nodelay(False)
        curses.curs_set(1)
        # NOTE: no emoji here -- it breaks curses getstr offsets.
        stdscr.addstr(self.SPLIT_LINE - 1, 2, "[CMD] Enter command: ")
        curses.echo()
        sentence_bytes = stdscr.getstr(self.SPLIT_LINE - 1, 23, 50)
        curses.noecho()
        curses.curs_set(0)
        stdscr.nodelay(True)
        return sentence_bytes.decode('utf-8')
