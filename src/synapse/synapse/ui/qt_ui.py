"""Qt front-end. Same contract and same threading shape as CursesUI.

Threading rule: Qt objects are touched only on the Qt (main) thread. ROS
threads call log()/update_status()/poll(), which touch plain Python
containers behind a lock. A QTimer on the Qt thread drains them and repaints,
so no signal or widget call ever crosses threads.

Qt must own the main thread, so run() moves the node -- construction and
spin -- onto a worker. Building the node there is deliberate: brain loading
(GR00T/JAX) takes seconds, and the window stays live and shows the init log
instead of freezing.
"""
import html
import sys
import threading
import time
import traceback
from collections import deque
from typing import Optional

from PyQt5 import QtCore, QtGui, QtWidgets

from .base import BaseUI, Intent, LogLevel, UIEvent, UIStatus, intent_from_key

MAX_LOG_LINES = 2000


class _StreamTee:
    """Copies print()/stderr into the log pane and still writes it to the
    original stream, so the launch console keeps the full record."""
    def __init__(self, ui, original, level):
        self.ui, self.original, self.level = ui, original, level

    def write(self, msg):
        self.original.write(msg)
        for line in msg.splitlines():
            if line.strip():
                self.ui._enqueue_log(line, self.level)
        return len(msg)

    def flush(self):
        self.original.flush()


class QtUI(BaseUI):
    REFRESH_HZ = 25
    QUIT_GRACE_MS = 3000
    WORKER_JOIN_SEC = 2.0

    def __init__(self, min_level: LogLevel = LogLevel.INFO):
        super().__init__(min_level)
        self._lock = threading.Lock()
        self._status = UIStatus()
        self._new_logs = deque(maxlen=MAX_LOG_LINES)  # (level, line), drained by the timer
        self._events = deque()

        self._app = self._window = self._timer = None
        self._worker = None
        self._worker_exc = None
        self._orig_stdout = self._orig_stderr = None
        self._orig_excepthook = None

    # --- lifecycle -------------------------------------------------------
    def start(self):
        # argv[:1] so Qt never tries to parse --ros-args.
        self._app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv[:1])
        # Closing the window is a QUIT intent (see closeEvent), not a hard exit.
        self._app.setQuitOnLastWindowClosed(False)

        self._window = _MainWindow(self)
        self._window.show()

        self._timer = QtCore.QTimer(self._window)
        self._timer.timeout.connect(self._on_timer)
        self._timer.start(int(1000 / self.REFRESH_HZ))

        self._orig_stdout, self._orig_stderr = sys.stdout, sys.stderr
        sys.stdout = _StreamTee(self, self._orig_stdout, LogLevel.INFO)
        sys.stderr = _StreamTee(self, self._orig_stderr, LogLevel.WARN)

        self._orig_excepthook = sys.excepthook
        sys.excepthook = self._excepthook

    def run(self, spin_fn):
        self._worker = threading.Thread(
            target=self._run_worker, args=(spin_fn,), name="synapse-ros", daemon=True)
        self._worker.start()
        self._app.exec_()
        # Ctrl-C quits Qt at once, while rclpy is still stopping the spin on
        # the worker. Give it a moment so main() never destroys a node that
        # is still spinning.
        self._worker.join(self.WORKER_JOIN_SEC)
        if self._worker_exc is not None:
            raise self._worker_exc

    def shutdown(self):
        if self._orig_stdout is not None:
            sys.stdout, sys.stderr = self._orig_stdout, self._orig_stderr
            self._orig_stdout = self._orig_stderr = None
        if self._orig_excepthook is not None:
            sys.excepthook = self._orig_excepthook
            self._orig_excepthook = None

    # --- BaseUI (any thread) ---------------------------------------------
    def log(self, msg, level: LogLevel = LogLevel.INFO):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        # The pane keeps every level; its dropdown decides what is shown.
        self._enqueue_log(line, level)
        if level >= self.min_level and self._orig_stdout is not None:
            self._orig_stdout.write(line + "\n")
            self._orig_stdout.flush()

    def update_status(self, status: UIStatus):
        with self._lock:
            self._status = status

    def poll(self) -> Optional[UIEvent]:
        with self._lock:
            return self._events.popleft() if self._events else None

    # --- plumbing ----------------------------------------------------------
    def emit(self, event: UIEvent):
        """Queue an operator event from the Qt thread; tick() picks it up."""
        with self._lock:
            self._events.append(event)

    def worker_alive(self) -> bool:
        return self._worker is not None and self._worker.is_alive()

    def _enqueue_log(self, line, level):
        with self._lock:
            self._new_logs.append((level, line))

    def _run_worker(self, spin_fn):
        try:
            spin_fn()
        except BaseException as e:  # re-raised on the main thread by run()
            self._worker_exc = e
            self.log("".join(traceback.format_exception(e)).rstrip(), LogLevel.ERROR)

    def _excepthook(self, exc_type, exc, tb):
        # PyQt hands exceptions that escape a slot to sys.excepthook. Ctrl-C
        # is raised on the Qt thread at the *entry* of whichever slot runs
        # next -- before any try block in it -- so this is the one place it
        # can be caught.
        if issubclass(exc_type, KeyboardInterrupt) and self._app is not None:
            self._app.quit()
        else:
            self._orig_excepthook(exc_type, exc, tb)

    def _on_timer(self):
        # Qt thread. Exceptions must not escape a slot: PyQt5 aborts the
        # process on an unhandled one, which would take the robot link with it.
        try:
            with self._lock:
                status = self._status
                logs = list(self._new_logs)
                self._new_logs.clear()

            if not self._window.stopped:
                self._window.render_status(status)
            self._window.append_logs(logs)

            if self._worker is not None and not self._worker.is_alive():
                if self._worker_exc is None:
                    self._app.quit()            # QUIT intent, Ctrl-C, rclpy shutdown
                elif not self._window.stopped:
                    self._window.show_stopped()  # crash: keep the traceback on screen
        except KeyboardInterrupt:  # Ctrl-C mid-body; at slot entry see _excepthook
            self._app.quit()
        except Exception:
            traceback.print_exc(file=self._orig_stderr or sys.__stderr__)


class _MainWindow(QtWidgets.QWidget):
    CONTROLS = (
        (Intent.START, "E(X)ecute"),
        (Intent.PAUSE, "Free(Z)e"),
        (Intent.RUN_SCENARIO, "S(C)enario Run"),
        (Intent.RESET, "(Q)uick Reset"),
        (Intent.QUIT, "Lea(V)e"),
    )
    LEVEL_COLORS = {
        LogLevel.DEBUG: "#888888",
        LogLevel.WARN: "#d08000",
        LogLevel.ERROR: "#e04040",
    }

    def __init__(self, ui: QtUI):
        super().__init__()
        self.ui = ui
        self.stopped = False
        self._brain_names = None
        self._log_store = deque(maxlen=MAX_LOG_LINES)  # lets the level filter re-render

        self.setWindowTitle("Synapse")
        self.resize(960, 640)
        # Only the command box may take keyboard focus; everything else is
        # NoFocus so keys always reach keyPressEvent below.
        self.setFocusPolicy(QtCore.Qt.StrongFocus)

        root = QtWidgets.QVBoxLayout(self)

        # 1. Controls -- one button per control intent.
        row = QtWidgets.QHBoxLayout()
        self.control_buttons = {}
        for intent, label in self.CONTROLS:
            btn = self._button(label, lambda i=intent: ui.emit(UIEvent(i)))
            self.control_buttons[intent] = btn
            row.addWidget(btn)
        root.addLayout(row)

        # 2. Brains -- rebuilt whenever the node reports a different list.
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("Brain:"))
        self.brain_buttons = QtWidgets.QHBoxLayout()
        row.addLayout(self.brain_buttons)
        row.addStretch()
        self.brain_group = QtWidgets.QButtonGroup(self)
        self.brain_group.setExclusive(True)
        root.addLayout(row)

        # 3. Status line.
        row = QtWidgets.QHBoxLayout()
        self.lbl_status = QtWidgets.QLabel()
        self.lbl_status.setStyleSheet("font-weight: bold")
        self.lbl_chunk = QtWidgets.QLabel()
        self.lbl_obs = QtWidgets.QLabel()
        for w in (self.lbl_status, self.lbl_chunk, self.lbl_obs):
            row.addWidget(w)
            row.addSpacing(24)
        row.addStretch()
        root.addLayout(row)

        # 4. Command box -- the GUI's version of the TUI's 'b' prompt.
        row = QtWidgets.QHBoxLayout()
        self.cmd_edit = QtWidgets.QLineEdit()
        self.cmd_edit.setPlaceholderText("Command to brain   (B focus · Enter send · Esc back to keys)")
        self.cmd_edit.returnPressed.connect(self._send_command)
        row.addWidget(self.cmd_edit, stretch=1)
        row.addWidget(self._button("Send", self._send_command))
        self.lbl_current = QtWidgets.QLabel()
        row.addWidget(self.lbl_current)
        root.addLayout(row)

        # 5. Log pane with a live level filter.
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("Log"))
        row.addStretch()
        self.level_combo = QtWidgets.QComboBox()
        self.level_combo.setFocusPolicy(QtCore.Qt.NoFocus)
        for lvl in LogLevel:
            self.level_combo.addItem(lvl.name, lvl)
        self.level_combo.setCurrentIndex(self.level_combo.findData(ui.min_level))
        self.level_combo.currentIndexChanged.connect(self._rerender_logs)
        row.addWidget(self.level_combo)
        root.addLayout(row)

        self.log_view = QtWidgets.QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(MAX_LOG_LINES)
        self.log_view.setFocusPolicy(QtCore.Qt.NoFocus)  # right-click still offers Copy
        self.log_view.setFont(QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.FixedFont))
        root.addWidget(self.log_view, stretch=1)

        self.render_status(UIStatus())

    # --- rendering (Qt thread, called from QtUI._on_timer) -----------------
    def render_status(self, st: UIStatus):
        if st.brain_names != self._brain_names:
            self._rebuild_brains(st.brain_names)
        if st.running_brain in self._brain_names:
            self.brain_group.button(self._brain_names.index(st.running_brain)).setChecked(True)

        # Mirrors the guards in SynapseMainNode.handle_ui_event.
        self.control_buttons[Intent.START].setEnabled(not st.is_ticking or st.scenario_running)
        self.control_buttons[Intent.PAUSE].setEnabled(st.is_ticking)
        self.control_buttons[Intent.RUN_SCENARIO].setEnabled(not st.scenario_running)

        self.lbl_status.setText(f"Status: {st.status}")
        self.lbl_chunk.setText(f"Chunk: {st.action_buffer_status} ({st.action_buffer_length})")
        self.lbl_obs.setText(f"Obs: {st.obs_buffer_length}")
        self.lbl_current.setText(f"now: {st.command!r}" if st.command else "")

    def append_logs(self, logs):
        threshold = self.level_combo.currentData()
        for level, line in logs:
            self._log_store.append((level, line))
            if level >= threshold:
                self._append_line(level, line)

    def show_stopped(self):
        self.stopped = True
        self.setWindowTitle("Synapse — stopped (see log, close to exit)")
        for b in list(self.control_buttons.values()) + self.brain_group.buttons():
            b.setEnabled(False)
        self.cmd_edit.setEnabled(False)

    # --- input ---------------------------------------------------------------
    def keyPressEvent(self, e):
        # Same keymap as the TUI: every key works identically in both.
        if e.key() == QtCore.Qt.Key_Escape:
            self.setFocus()
            return
        text = e.text()
        if text == 'b':
            self.cmd_edit.setFocus()
            return
        event = intent_from_key(text)
        if event is not None:
            self.ui.emit(event)
        else:
            super().keyPressEvent(e)

    def closeEvent(self, e):
        if self.ui.worker_alive():
            # Same path as pressing 'v', so the muscle still receives QUIT.
            # The grace timer covers a node that never ticks (still loading).
            self.ui.emit(UIEvent(Intent.QUIT))
            QtCore.QTimer.singleShot(QtUI.QUIT_GRACE_MS, self.ui._app.quit)
            e.ignore()
        else:
            self.ui._app.quit()
            e.accept()

    # --- helpers ---------------------------------------------------------------
    def _button(self, label, on_click, checkable=False):
        btn = QtWidgets.QPushButton(label)
        btn.setFocusPolicy(QtCore.Qt.NoFocus)  # keep keyboard focus on the window
        btn.setCheckable(checkable)
        btn.clicked.connect(lambda _checked=False: on_click())
        return btn

    def _rebuild_brains(self, names):
        for btn in self.brain_group.buttons():
            self.brain_group.removeButton(btn)
            btn.deleteLater()
        for i, name in enumerate(names):
            hint = f"{i + 1}   " if i < 9 else ""  # keys 1-9 map to the first nine
            btn = self._button(f"{hint}{name}",
                               lambda i=i: self.ui.emit(UIEvent(Intent.SELECT_BRAIN, i)),
                               checkable=True)
            self.brain_group.addButton(btn, i)
            self.brain_buttons.addWidget(btn)
        self._brain_names = list(names)

    def _send_command(self):
        text = self.cmd_edit.text().strip()
        if text:
            self.ui.emit(UIEvent(Intent.SEND_COMMAND, text))
        self.cmd_edit.clear()
        self.setFocus()

    def _append_line(self, level, line):
        color = self.LEVEL_COLORS.get(level)
        style = "white-space:pre-wrap;" + (f"color:{color};" if color else "")
        text = html.escape(line).replace("\n", "<br>")
        self.log_view.appendHtml(f'<span style="{style}">{text}</span>')

    def _rerender_logs(self):
        self.log_view.clear()
        threshold = self.level_combo.currentData()
        for level, line in self._log_store:
            if level >= threshold:
                self._append_line(level, line)
