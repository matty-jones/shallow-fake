"""Export trained model to ONNX format."""

import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Optional

from shallow_fake.config import VoiceConfig
from shallow_fake.language_utils import format_model_name, get_language_metadata, get_espeak_voice
from shallow_fake.utils import ensure_dir, setup_logging

logger = setup_logging()


def find_best_checkpoint(workspace_dir: Path, voice_id: str) -> Optional[Path]:
    """Find the best checkpoint (prefers last.ckpt, otherwise highest epoch)."""
    # Check prepared dataset directory for lightning_logs
    # Pattern: workspace/{voice_id}/datasets/prepared/lightning_logs/version_X/checkpoints/
    prepared_dir = workspace_dir / "datasets" / "prepared"
    
    if prepared_dir.exists():
        lightning_logs = prepared_dir / "lightning_logs"
        if lightning_logs.exists():
            version_dirs = [d for d in lightning_logs.iterdir() if d.is_dir() and d.name.startswith("version_")]
            if version_dirs:
                latest_version = max(version_dirs, key=lambda d: int(d.name.split("_")[1]) if d.name.split("_")[1].isdigit() else 0)
                version_checkpoints_dir = latest_version / "checkpoints"
                if version_checkpoints_dir.exists():
                    checkpoints = list(version_checkpoints_dir.glob("*.ckpt"))
                    if checkpoints:
                        # Prefer last.ckpt if it exists (final checkpoint from training)
                        last_ckpt = [c for c in checkpoints if 'last' in c.name.lower()]
                        if last_ckpt:
                            logger.info(f"Found final checkpoint: {last_ckpt[0]}")
                            return last_ckpt[0]
                        
                        # Otherwise, use highest epoch checkpoint
                        def get_epoch(ckpt_path: Path) -> int:
                            match = re.search(r"epoch[=](\d+)", ckpt_path.name)
                            return int(match.group(1)) if match else 0
                        best_checkpoint = max(checkpoints, key=get_epoch)
                        logger.info(f"Found best checkpoint: {best_checkpoint}")
                        return best_checkpoint
    
    # Location 2: Training checkpoints directory (exported checkpoints)
    training_checkpoints = workspace_dir / "training" / "checkpoints"
    if training_checkpoints.exists():
        checkpoints = list(training_checkpoints.glob("*.ckpt"))
        if checkpoints:
            last_ckpt = [c for c in checkpoints if 'last' in c.name.lower()]
            if last_ckpt:
                logger.info(f"Found checkpoint in training directory: {last_ckpt[0]}")
                return last_ckpt[0]
            latest = max(checkpoints, key=lambda p: p.stat().st_mtime)
            logger.info(f"Found checkpoint in training directory: {latest}")
            return latest
    
    # Location 3: Legacy location in logs directory (for backward compatibility)
    lightning_logs = workspace_dir / "logs" / "lightning_logs"
    if lightning_logs.exists():
        version_dirs = [d for d in lightning_logs.iterdir() if d.is_dir() and d.name.startswith("version_")]
        if version_dirs:
            latest_version = max(version_dirs, key=lambda d: int(d.name.split("_")[1]) if d.name.split("_")[1].isdigit() else 0)
            version_checkpoints_dir = latest_version / "checkpoints"
            if version_checkpoints_dir.exists():
                checkpoints = list(version_checkpoints_dir.glob("*.ckpt"))
                if checkpoints:
                    def get_epoch(ckpt_path: Path) -> int:
                        match = re.search(r"epoch[=](\d+)", ckpt_path.name)
                        return int(match.group(1)) if match else 0
                    best_checkpoint = max(checkpoints, key=get_epoch)
                    logger.info(f"Found best checkpoint: {best_checkpoint}")
                    return best_checkpoint
    
    return None


def export_onnx_from_container(
    workspace_dir: Path,
    checkpoint_path: str,
    output_path: str,
):
    """Export ONNX model using piper_train.export_onnx in a temporary container."""
    # Use docker run since training container is torn down after completion
    # checkpoint_path and output_path are relative to /workspace inside container
    
    # Use stable CPU export image to avoid torch.export nightly dynamic-shape issues
    image = "textymcspeechy-piper:export-cpu"
    workspace_abs = workspace_dir.resolve()
    
    # Check if image exists, build if it doesn't
    logger.info(f"Checking for export image: {image}")
    check_result = subprocess.run(
        ["docker", "images", "-q", image],
        capture_output=True,
        text=True,
    )
    if not check_result.stdout.strip():
        logger.info(f"Export image not found. Building {image}...")
        dockerfile_path = Path(__file__).parent.parent / "docker" / "Dockerfile.piper_export_cpu"
        if not dockerfile_path.exists():
            raise FileNotFoundError(
                f"Dockerfile not found: {dockerfile_path}. "
                "Cannot build export image automatically."
            )
        build_cmd = [
            "docker",
            "build",
            "-f", str(dockerfile_path),
            "-t", image,
            str(dockerfile_path.parent.parent),  # Build context (project root)
        ]
        build_result = subprocess.run(
            build_cmd,
            check=True,
            capture_output=True,
            text=True,
        )
        logger.info("Export image built successfully")
    else:
        logger.info(f"Export image found: {image}")
    
    # Create a wrapper script that patches torch.load and handles export issues
    # This handles PyTorch 2.6+ weights_only=True default issue with pathlib.PosixPath
    # Also attempts to work around torch.export dynamic shape issues
    wrapper_script = f"""import sys
import os
import pathlib
import torch

# Set environment variables for more lenient export behavior
os.environ.setdefault('TORCH_LOGS', '+dynamo')
os.environ.setdefault('TORCHDYNAMO_VERBOSE', '0')

# Patch torch.load to handle pathlib.PosixPath in checkpoints (PyTorch 2.6+ compatibility)
_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    # If weights_only is not explicitly set, default to False for compatibility
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    # If weights_only=True is set, add pathlib.PosixPath to safe globals
    elif kwargs.get('weights_only', False):
        try:
            torch.serialization.add_safe_globals([pathlib.PosixPath])
        except (AttributeError, TypeError):
            # Fallback: use weights_only=False if safe_globals not available
            kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

# Try to patch torch.export to be more lenient with dynamic shapes
# This is a workaround for the GuardOnDataDependentSymNode error
try:
    import torch._dynamo
    # Disable strict checks for data-dependent operations
    torch._dynamo.config.suppress_errors = True
    torch._dynamo.config.assume_static_by_default = False
except (ImportError, AttributeError):
    pass

# Now run the export
from piper_train.export_onnx import main
sys.argv = ['export_onnx', '/workspace/{checkpoint_path}', '/workspace/{output_path}']
main()
"""
    
    # Build docker run command (CPU export; no GPU needed)
    cmd = [
        "docker",
        "run",
        "--rm",  # Automatically remove container after it exits
        "-v", f"{workspace_abs}:/workspace",  # Mount workspace
        "-w", "/app",  # Working directory
        image,
        "python3",
        "-c",
        wrapper_script,
    ]

    logger.info(f"Exporting ONNX model from checkpoint: {checkpoint_path}")
    logger.info(f"Command: docker run ... python3 -c '...'")

    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
        )
        logger.info("ONNX export completed successfully")
        if result.stdout:
            logger.info(result.stdout)
    except subprocess.CalledProcessError as e:
        logger.error(f"ONNX export failed: {e.stderr}")
        if e.stdout:
            logger.error(f"stdout: {e.stdout}")
        raise


def find_config_json(checkpoint_path: Path, voice_id: str, workspace_dir: Path) -> Optional[Path]:
    """Find config.json associated with the model."""
    # The config.json is created during preprocessing in the prepared directory
    # This is the primary location where it should be found
    primary_location = workspace_dir / "datasets" / "prepared" / "config.json"
    if primary_location.exists():
        return primary_location
    
    # Fallback: Look for config.json in various locations relative to checkpoint
    possible_locations = [
        checkpoint_path.parent / "config.json",
        checkpoint_path.parent.parent / "config.json",
        checkpoint_path.parent.parent.parent / "config.json",
    ]

    for loc in possible_locations:
        if loc.exists():
            return loc

    return None


def export_onnx(config: VoiceConfig, checkpoint_path: Optional[Path] = None):
    """Export trained model to ONNX format."""
    workspace_dir = config.paths.workspace_dir
    output_dir = config.paths.output_models_dir
    ensure_dir(output_dir)

    # Find checkpoint if not provided
    if checkpoint_path is None:
        checkpoint_path = find_best_checkpoint(workspace_dir, config.voice_id)
        if checkpoint_path is None:
            logger.error("No checkpoint found. Please specify checkpoint path or ensure training has completed.")
            raise FileNotFoundError("No checkpoint found")

    # Determine output paths using Piper naming convention
    # Format: <language>_<REGION>-<name>-<quality>
    model_name = format_model_name(
        config.language_code,
        config.region,
        config.voice_id,
        config.training.quality
    )
    onnx_path = output_dir / f"{model_name}.onnx"
    json_path = output_dir / f"{model_name}.onnx.json"

    # Ensure training directory exists in workspace (where container will write)
    training_dir = config.paths.training_dir
    ensure_dir(training_dir)

    # Export ONNX
    # Note: Paths inside container should be relative to /workspace
    container_checkpoint = checkpoint_path.relative_to(workspace_dir)
    container_output = f"training/{onnx_path.name}"  # Relative to /workspace

    export_onnx_from_container(
        workspace_dir,
        str(container_checkpoint),
        container_output
    )

    # Copy ONNX file from container to host
    container_onnx = training_dir / onnx_path.name
    if container_onnx.exists():
        shutil.copy2(container_onnx, onnx_path)
        logger.info(f"ONNX model saved: {onnx_path}")
    else:
        logger.warning(f"ONNX file not found in expected location: {container_onnx}")

    # Find and load config.json from the preprocessing directory
    config_json = find_config_json(checkpoint_path, config.voice_id, workspace_dir)
    if config_json and config_json.exists():
        # Load the original config.json
        with open(config_json, "r", encoding="utf-8") as f:
            onnx_data = json.load(f)
        
        # Update with Piper-compliant fields
        # The "dataset" field MUST match the file name (without .onnx extension)
        onnx_data["dataset"] = model_name
        
        # Add language metadata
        language_metadata = get_language_metadata(config.language_code, config.region)
        onnx_data["language"] = {
            "code": f"{config.language_code}_{config.region}",
            "family": config.language_code,
            "region": config.region,
            "name_native": language_metadata["name_native"],
            "name_english": language_metadata["name_english"],
            "country_english": language_metadata["country_english"],
        }
        
        # Add espeak voice mapping
        espeak_voice = get_espeak_voice(config.language_code, config.region)
        if "espeak" not in onnx_data:
            onnx_data["espeak"] = {}
        onnx_data["espeak"]["voice"] = espeak_voice
        logger.info(f"Set espeak voice to: {espeak_voice}")
        
        # Write the updated onnx.json
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(onnx_data, f, indent=2, ensure_ascii=False)
        logger.info(f"ONNX JSON saved: {json_path}")
    else:
        logger.error(f"config.json not found. Expected location: {workspace_dir / 'datasets' / 'prepared' / 'config.json'}")
        logger.error("The config.json should have been created during preprocessing. Please ensure training has completed successfully.")
        raise FileNotFoundError(
            f"config.json not found. This file is required for model evaluation. "
            f"It should be at: {config.paths.workspace_dir / 'datasets' / 'prepared' / 'config.json'}"
        )

    logger.info(f"Export complete: {onnx_path} and {json_path}")


if __name__ == "__main__":
    import sys
    from pathlib import Path

    if len(sys.argv) < 2:
        print("Usage: python export_onnx.py <config_path> [--checkpoint <path>]")
        sys.exit(1)

    config_path = Path(sys.argv[1])
    config = VoiceConfig.from_yaml(config_path)

    checkpoint_path = None
    args = sys.argv[2:]
    i = 0
    while i < len(args):
        if args[i] == "--checkpoint" and i + 1 < len(args):
            checkpoint_path = Path(args[i + 1])
            i += 2
        else:
            i += 1

    export_onnx(config, checkpoint_path)

