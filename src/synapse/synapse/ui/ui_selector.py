from .base import LogLevel
from .curses_ui import CursesUI


class UISelector:

    @staticmethod
    def get_ui(ui_mode: str, debug_mode: bool = False):
        """ui_mode: TUI (curses, own terminal) | GUI (Qt window)."""
        mode = (ui_mode or "TUI").upper()
        min_level = LogLevel.DEBUG if debug_mode else LogLevel.INFO

        match mode:
            case 'TUI':
                return CursesUI(min_level=min_level)
            case 'GUI':
                # Imported here so TUI-only machines never need PyQt5.
                from .qt_ui import QtUI
                return QtUI(min_level=min_level)
            case _:
                raise ValueError(
                    f"❌ Unknown ui mode '{ui_mode}'. Expected tui | gui.")
