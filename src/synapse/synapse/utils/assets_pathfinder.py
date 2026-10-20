import os
from pathlib import Path

from synapse.utils.paths import resolve_path


def _assets_dir() -> Path:
    """SYNAPSE_ASSETS_DIR first (the launch file sets it from paths.assets), then
    the first parent of this file that has an 'assets' directory:

    /workspace_root
    ├── build/
    ├── install/
    ├── src/
    └── assets/          <-- Assets live here
    """
    override = os.environ.get('SYNAPSE_ASSETS_DIR')
    if override:
        return Path(resolve_path(override))

    current_file = Path(__file__).resolve()
    for parent in current_file.parents:
        if (parent / 'assets').is_dir():
            return parent / 'assets'

    raise FileNotFoundError(
        f"Could not find the workspace root containing the 'assets' directory. "
        f"Searched upwards from: {current_file}. Set paths.assets in config_main.yaml "
        f"or SYNAPSE_ASSETS_DIR.")


def assets_get_path(filename: str) -> str:
    """Absolute path of `filename` inside the assets directory."""
    asset_path = _assets_dir() / filename

    # Optional: Warn if the specific file isn't found inside the assets folder
    if not asset_path.exists():
        print(f"⚠️ Warning: Asset '{filename}' does not exist at {asset_path}")

    return str(asset_path)
