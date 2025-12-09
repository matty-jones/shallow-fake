"""Export trained model to ONNX format."""

import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Optional

from shallow_fake.config import VoiceConfig
from shallow_fake.utils import ensure_dir, setup_logging

logger = setup_logging()


def find_best_checkpoint(checkpoints_dir: Path, voice_id: str) -> Optional[Path]:
    """Find the best checkpoint (prefers last.ckpt, otherwise highest epoch)."""
    # Check multiple possible locations for checkpoints
    
    # Location 1: lightning_logs structure in datasets (where training actually saves)
    # Pattern: tms_workspace/datasets/{voice}/combined_prepared/lightning_logs/version_X/checkpoints/
    datasets_dir = checkpoints_dir.parent / "datasets"  # Fixed: was parent.parent
    voice_dir = datasets_dir / voice_id  # Use specific voice_id instead of iterating
    prepared_dir = voice_dir / "combined_prepared"
    
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
    
    # Location 2: lightning_logs structure in logs directory (legacy)
    lightning_logs = checkpoints_dir.parent / "logs" / "lightning_logs"
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
    
    # Location 3: Direct checkpoints directory
    checkpoints = list(checkpoints_dir.glob("*.ckpt"))
    if checkpoints:
        return max(checkpoints, key=lambda p: p.stat().st_mtime)
    
    return None


def export_onnx_from_container(
    tms_workspace_dir: Path,
    checkpoint_path: str,
    output_path: str,
):
    """Export ONNX model using piper_train.export_onnx in a temporary container."""
    # Use docker run since training container is torn down after completion
    # checkpoint_path and output_path are relative to /workspace inside container
    # GPU passthrough configuration matches docker-compose.training.yml (CDI format)
    
    image = "textymcspeechy-piper:5080-cu128"
    workspace_abs = tms_workspace_dir.resolve()
    
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
    
    # Build docker run command with CDI GPU passthrough (matches training container)
    cmd = [
        "docker",
        "run",
        "--rm",  # Automatically remove container after it exits
        "--device", "nvidia.com/gpu=all",  # CDI GPU passthrough (matches docker-compose.training.yml)
        "-v", f"{workspace_abs}:/workspace",  # Mount workspace
        "-w", "/app",  # Working directory
        # Environment variables matching docker-compose.training.yml
        "-e", "CUDA_VISIBLE_DEVICES=0",
        "-e", "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True",
        "-e", "NVIDIA_VISIBLE_DEVICES=all",
        "-e", "NVIDIA_DRIVER_CAPABILITIES=compute,utility",
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


def find_config_json(checkpoint_dir: Path) -> Optional[Path]:
    """Find config.json associated with checkpoint."""
    # Look for config.json in various locations
    possible_locations = [
        checkpoint_dir / "config.json",
        checkpoint_dir.parent / "config.json",
        checkpoint_dir.parent.parent / "config.json",
    ]

    for loc in possible_locations:
        if loc.exists():
            return loc

    return None


def export_onnx(config: VoiceConfig, checkpoint_path: Optional[Path] = None):
    """Export trained model to ONNX format."""
    checkpoints_dir = config.paths.tms_workspace_dir / "checkpoints"
    output_dir = config.paths.output_models_dir
    ensure_dir(output_dir)

    # Find checkpoint if not provided
    if checkpoint_path is None:
        checkpoint_path = find_best_checkpoint(checkpoints_dir, config.voice_id)
        if checkpoint_path is None:
            logger.error("No checkpoint found. Please specify checkpoint path or ensure training has completed.")
            raise FileNotFoundError("No checkpoint found")

    # Determine output paths
    model_name = f"{config.voice_id}-{config.training.quality}"
    onnx_path = output_dir / f"{model_name}.onnx"
    json_path = output_dir / f"{model_name}.onnx.json"

    # Ensure models directory exists in workspace (where container will write)
    workspace_models_dir = config.paths.tms_workspace_dir / "models"
    ensure_dir(workspace_models_dir)

    # Export ONNX
    # Note: Paths inside container should be relative to /workspace
    container_checkpoint = checkpoint_path.relative_to(config.paths.tms_workspace_dir)
    container_output = f"models/{onnx_path.name}"  # Relative to /workspace

    export_onnx_from_container(
        config.paths.tms_workspace_dir,
        str(container_checkpoint),
        container_output
    )

    # Copy ONNX file from container to host
    container_onnx = config.paths.tms_workspace_dir / "models" / onnx_path.name
    if container_onnx.exists():
        shutil.copy2(container_onnx, onnx_path)
        logger.info(f"ONNX model saved: {onnx_path}")
    else:
        logger.warning(f"ONNX file not found in expected location: {container_onnx}")

    # Find and copy config.json
    config_json = find_config_json(checkpoint_path.parent)
    if config_json and config_json.exists():
        shutil.copy2(config_json, json_path)
        logger.info(f"Config JSON saved: {json_path}")
    else:
        logger.warning("config.json not found. You may need to create it manually.")
        # Create a minimal config.json
        minimal_config = {
            "audio": {
                "sample_rate": 22050,
            },
            "espeak": {
                "voice": config.phoneme_check.language,
            },
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(minimal_config, f, indent=2)
        logger.info(f"Created minimal config JSON: {json_path}")

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

