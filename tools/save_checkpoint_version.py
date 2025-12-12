"""Save a checkpoint with a version number for future training resumes."""

import shutil
import sys
from pathlib import Path
from typing import Optional

from shallow_fake.config import VoiceConfig
from shallow_fake.utils import ensure_dir, setup_logging
from tools.export_onnx import find_best_checkpoint

logger = setup_logging()


def save_checkpoint_version(
    config: VoiceConfig,
    version: str,
    checkpoint_path: Optional[Path] = None,
    overwrite: bool = False,
):
    """
    Save a checkpoint with a version number.
    
    Args:
        config: Voice configuration
        version: Version number (e.g., "1", "2")
        checkpoint_path: Path to checkpoint to save. If None, finds best checkpoint.
        overwrite: If True, overwrite existing versioned checkpoint. Default False.
    """
    workspace_dir = config.paths.workspace_dir
    checkpoints_dir = workspace_dir / "training" / "checkpoints"
    ensure_dir(checkpoints_dir)

    # Find checkpoint if not provided
    if checkpoint_path is None:
        checkpoint_path = find_best_checkpoint(workspace_dir, config.voice_id)
        if checkpoint_path is None:
            raise FileNotFoundError("No checkpoint found. Please specify checkpoint path or ensure training has completed.")

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Copy to versioned location
    versioned_path = checkpoints_dir / f"v{version}.ckpt"
    
    # Protect v1 from accidental overwrite
    if versioned_path.exists() and version == "1" and not overwrite:
        logger.warning(f"v1 checkpoint already exists at {versioned_path}")
        logger.warning("v1 is protected from overwrite. Use --overwrite flag if you really want to replace it.")
        raise FileExistsError(
            f"v1 checkpoint already exists at {versioned_path}. "
            "v1 is protected from accidental overwrite. Use --overwrite flag if needed."
        )
    
    if versioned_path.exists() and not overwrite:
        logger.warning(f"Version {version} checkpoint already exists at {versioned_path}")
        response = input(f"Overwrite existing v{version} checkpoint? (yes/no): ")
        if response.lower() != "yes":
            logger.info("Aborted.")
            return

    shutil.copy2(checkpoint_path, versioned_path)
    logger.info(f"Saved checkpoint as version {version}: {versioned_path}")
    logger.info(f"  Source: {checkpoint_path}")
    logger.info(f"  Destination: {versioned_path}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python save_checkpoint_version.py <config_path> <version> [--checkpoint <path>] [--overwrite]")
        print("Example: python save_checkpoint_version.py config/claudia.yaml 1")
        sys.exit(1)

    config_path = Path(sys.argv[1])
    version = sys.argv[2]
    
    config = VoiceConfig.from_yaml(config_path)
    config.ensure_directories()

    checkpoint_path = None
    overwrite = False
    
    args = sys.argv[3:]
    i = 0
    while i < len(args):
        if args[i] == "--checkpoint" and i + 1 < len(args):
            checkpoint_path = Path(args[i + 1])
            i += 2
        elif args[i] == "--overwrite":
            overwrite = True
            i += 1
        else:
            i += 1

    try:
        save_checkpoint_version(config, version, checkpoint_path, overwrite)
    except Exception as e:
        logger.error(f"Failed to save checkpoint version: {e}")
        sys.exit(1)

