import os
from pathlib import Path

def assets_get_path(filename: str) -> str:
    """
    Resolves the absolute path to an asset located in the workspace root's 'assets' directory.
    
    Expected Workspace Structure:
    /workspace_root
    ├── build/
    ├── install/
    ├── src/
    └── assets/          <-- Assets live here
        └── filename     <-- Target file
    """
    # Resolve the absolute path of this exact script
    current_file = Path(__file__).resolve()
    workspace_root = None
    
    # Traverse up the directory tree to find the workspace root
    for parent in current_file.parents:
        # We identify the workspace root by looking for the 'assets' sibling directory
        if (parent / 'assets').is_dir():
            workspace_root = parent
            break

    if not workspace_root:
        raise FileNotFoundError(
            f"Could not find the workspace root containing the 'assets' directory. "
            f"Searched upwards from: {current_file}"
        )

    asset_path = workspace_root / 'assets' / filename
    
    # Optional: Warn if the specific file isn't found inside the assets folder
    if not asset_path.exists():
        print(f"⚠️ Warning: Asset '{filename}' does not exist at {asset_path}")
        
    return str(asset_path)