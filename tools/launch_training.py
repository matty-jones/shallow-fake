"""Launch TMS training container."""

import os
import subprocess
from pathlib import Path
from typing import Optional

import torch

from shallow_fake.config import VoiceConfig
from shallow_fake.utils import ensure_dir, setup_logging

logger = setup_logging()


def prepare_workspace(config: VoiceConfig):
    """Verify workspace is ready for training (no copying needed - files are already in workspace)."""
    combined_dataset = config.paths.combined_dataset_dir
    combined_metadata = combined_dataset / "metadata.csv"
    combined_wavs = combined_dataset / "wavs"

    if not combined_metadata.exists():
        logger.error(f"Combined dataset metadata not found: {combined_metadata}")
        raise FileNotFoundError(f"Combined dataset metadata not found: {combined_metadata}")

    if not combined_wavs.exists() or len(list(combined_wavs.glob("*.wav"))) == 0:
        logger.error(f"No WAV files found in {combined_wavs}")
        raise FileNotFoundError(f"Combined dataset WAVs not found: {combined_wavs}")

    logger.info(f"Workspace ready: {config.paths.workspace_dir}")
    logger.info(f"  Combined dataset: {combined_dataset}")
    logger.info(f"  Metadata: {combined_metadata}")
    logger.info(f"  WAV files: {len(list(combined_wavs.glob('*.wav')))} files")

    # Check for base checkpoint
    base_checkpoint = config.training.base_checkpoint
    checkpoint_path = config.paths.base_checkpoints_dir / base_checkpoint
    if not checkpoint_path.exists():
        logger.warning(f"Base checkpoint not found: {checkpoint_path}")
        logger.warning("Please download the base checkpoint and place it in:")
        logger.warning(f"  {config.paths.base_checkpoints_dir}")
        logger.warning(f"Expected filename: {base_checkpoint}")
    else:
        logger.info(f"Base checkpoint found: {checkpoint_path}")


def launch_training(config: VoiceConfig, resume_from_version: Optional[str] = None):
    """Launch TMS training container.
    
    Args:
        config: Voice configuration
        resume_from_version: Optional version number (e.g., "1") to resume from a versioned checkpoint.
                           If provided, loads from workspace/{voice_id}/training/checkpoints/v{version}.ckpt
                           instead of the base checkpoint.
    """
    compose_file = config.tms.docker_compose_file
    if not compose_file.exists():
        logger.error(f"Docker compose file not found: {compose_file}")
        raise FileNotFoundError(f"Docker compose file not found: {compose_file}")

    # Prepare workspace
    prepare_workspace(config)

    # Determine checkpoint to use
    if resume_from_version:
        # Load from versioned checkpoint
        checkpoint_path = config.paths.training_checkpoints_dir / f"v{resume_from_version}.ckpt"
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Version {resume_from_version} checkpoint not found at {checkpoint_path}. "
                f"Please save a checkpoint as v{resume_from_version} first using: "
                f"python tools/save_checkpoint_version.py config/{config.voice_id}.yaml {resume_from_version}"
            )
        base_checkpoint_name = f"v{resume_from_version}.ckpt"
        checkpoint_source = "workspace"  # Indicates it's in workspace, not shared
        logger.info(f"Resuming training from version {resume_from_version} checkpoint: {checkpoint_path}")
    else:
        # Use base checkpoint from config
        checkpoint_path = config.paths.base_checkpoints_dir / config.training.base_checkpoint
        base_checkpoint_name = config.training.base_checkpoint
        checkpoint_source = "shared"  # Indicates it's in shared base_checkpoints
        if checkpoint_path.exists():
            logger.info(f"Base checkpoint found: {checkpoint_path}")
        else:
            logger.warning(f"Checkpoint not found at {checkpoint_path}; training from scratch.")

    # Determine effective max epochs: just use the configured value
    # Since we're doing weights-only loading (not Lightning resume),
    # PyTorch Lightning will start from epoch 0, but the weights are already loaded
    max_epochs = config.training.max_epochs
    logger.info(f"Training for {max_epochs} epochs with pre-loaded weights")

    # Set environment variables for docker-compose
    env = os.environ.copy()
    env.update({
        "PROJECT_NAME": config.voice_id,  # Use voice_id as project name
        "BASE_CHECKPOINT": base_checkpoint_name,
        "CHECKPOINT_SOURCE": checkpoint_source,  # "shared" or "workspace"
        "BATCH_SIZE": str(config.training.batch_size),
        "MAX_EPOCHS": str(max_epochs),
        "QUALITY": config.training.quality,
        "ACCELERATOR": config.training.accelerator,
        "DEVICES": str(config.training.devices),
    })

    # Build docker-compose command
    compose_dir = compose_file.parent
    compose_file_name = compose_file.name

    cmd = [
        "docker",
        "compose",
        "-f", str(compose_file),
        "up",
        "-d",
    ]

    logger.info("Launching training container...")
    logger.info(f"Command: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd,
            cwd=compose_dir,
            env=env,
            check=True,
            capture_output=True,
            text=True,
        )
        logger.info("Training container launched successfully")
        logger.info(f"Container name: tms-{config.voice_id}-trainer")
        logger.info("Use 'docker logs -f tms-{config.voice_id}-trainer' to view logs")
        if config.tms.expose_tensorboard:
            logger.info(f"TensorBoard available at http://localhost:{config.tms.tensorboard_port}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to launch training container: {e.stderr}")
        raise
    except FileNotFoundError:
        logger.error("docker command not found. Please install Docker.")
        raise


if __name__ == "__main__":
    import sys
    from pathlib import Path

    if len(sys.argv) < 2:
        print("Usage: python launch_training.py <config_path>")
        sys.exit(1)

    config_path = Path(sys.argv[1])
    config = VoiceConfig.from_yaml(config_path)
    config.ensure_directories()
    launch_training(config)

