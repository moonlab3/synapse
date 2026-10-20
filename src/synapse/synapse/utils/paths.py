"""Runtime path resolution, so no machine-specific absolute path lives in code.

Paths come from the shared `paths:` block in config_main.yaml (merged into every
profile by the launch file) and accept `~` and `${ENV}`.
"""
import os
import re

_UNRESOLVED = re.compile(r'\$\{?\w+\}?')


def resolve_path(value: str) -> str:
    """Expand `~` and `${ENV}`. An unset variable is an error, not a silent literal."""
    if not value:
        raise ValueError("empty path")
    expanded = os.path.expandvars(os.path.expanduser(str(value)))
    leftover = _UNRESOLVED.search(expanded)
    if leftover:
        raise ValueError(f"path '{value}' uses unset environment variable {leftover.group(0)}")
    return expanded


def synapse_share_dir() -> str:
    """share/synapse of the installed package.

    SYNAPSE_SHARE_DIR comes first: isaac_node runs in Isaac Sim's Python with the
    AMENT_* variables scrubbed, so the ament index is unreachable there. The launch
    file sets it for every node.
    """
    share = os.environ.get('SYNAPSE_SHARE_DIR')
    if share:
        return share
    try:
        from ament_index_python.packages import get_package_share_directory
        return get_package_share_directory('synapse')
    except Exception as e:
        raise FileNotFoundError(
            "synapse share directory not found: set SYNAPSE_SHARE_DIR or source the workspace") from e
