"""UI contract shared by every front-end (curses TUI, Qt GUI).

The node never talks to a concrete UI. It pushes a UIStatus snapshot and pulls
UIEvents; whatever is on the other end is irrelevant to it. Keyboard-to-intent
translation lives here (intent_from_key) so char-driven front-ends share one
keymap and a GUI can emit the same intents from buttons without faking
keystrokes.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, IntEnum, auto
from typing import Optional


class LogLevel(IntEnum):
    DEBUG = 10
    INFO = 20
    WARN = 30
    ERROR = 40


class Intent(Enum):
    """What the operator asked for, independent of how they asked."""
    START = auto()          # begin ticking the brain
    PAUSE = auto()          # freeze ticking
    RUN_SCENARIO = auto()   # begin ticking the behavior tree
    RESET = auto()          # broadcast RESET to the muscle
    QUIT = auto()           # shut the node down
    SELECT_BRAIN = auto()   # payload: int, index into brain_node_list
    SEND_COMMAND = auto()   # payload: str, free-text instruction for the brain
    TELEOP = auto()         # payload: str, single char consumed by ManualAdapter


@dataclass(frozen=True)
class UIEvent:
    intent: Intent
    payload: Optional[object] = None


@dataclass
class UIStatus:
    """One frame of node state. Replaces the old 7-positional-arg call."""
    status: str = "Idle"
    obs_buffer_length: int = 0
    action_buffer_status: str = ""
    action_buffer_length: int = 0
    command: str = ""
    brain_names: list = field(default_factory=list)
    running_brain: str = ""
    is_ticking: bool = False
    scenario_running: bool = False


# Reserved control keys. Everything else alphabetic falls through to the brain
# as a teleop character (ManualAdapter reads s/d/f/w/e/r, g/h/j/k/l, o/p/i).
_CONTROL_KEYS = {
    'x': Intent.START,
    'z': Intent.PAUSE,
    'c': Intent.RUN_SCENARIO,
    'q': Intent.RESET,
    'v': Intent.QUIT,
}

HELP_LINE = (" Free(Z)e | E(X)ecute | S(C)enario Run | Lea(V)e | "
             "(B)ackup command | (Q)uick Reset")


def format_brain_map(brain_names) -> str:
    """'Brain Adapters [1: a]  [2: b]' -- the number is the key that selects it."""
    return "Brain Adapters " + "".join(
        f"[{i + 1}: {name}]  " for i, name in enumerate(brain_names))


def intent_from_key(char: str) -> Optional[UIEvent]:
    """Map one keypress to a UIEvent. None if the key means nothing.

    'b' is deliberately absent: it opens a text prompt, which is a front-end
    concern. Whoever handles it emits SEND_COMMAND once the text is complete.
    """
    if not char:
        return None
    if char in _CONTROL_KEYS:
        return UIEvent(_CONTROL_KEYS[char])
    if char.isdigit():
        # Preserves the original mapping, including '0' -> -1 (last brain).
        return UIEvent(Intent.SELECT_BRAIN, int(char) - 1)
    if char.isalpha():
        return UIEvent(Intent.TELEOP, char)
    return None


class BaseUI(ABC):
    """Every front-end implements exactly these four things."""

    def __init__(self, min_level: LogLevel = LogLevel.INFO):
        self.min_level = min_level

    def debug(self, msg):
        self.log(msg, LogLevel.DEBUG)

    @abstractmethod
    def log(self, msg, level: LogLevel = LogLevel.INFO):
        """Append one line to the log view."""

    @abstractmethod
    def update_status(self, status: UIStatus):
        """Replace the status snapshot shown to the operator."""

    @abstractmethod
    def poll(self) -> Optional[UIEvent]:
        """Pop the next operator event, or None. Must never block."""

    def start(self):
        """Bring the front-end up. Called once, before the node is built."""

    def run(self, spin_fn):
        """Drive spin_fn (build node + spin) until it returns.

        Default: run it right here on the calling (main) thread. A front-end
        that must own the main thread itself (Qt) overrides this and moves
        spin_fn onto a worker.
        """
        spin_fn()

    def shutdown(self):
        """Tear the front-end down and restore the terminal."""
