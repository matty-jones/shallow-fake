"""Launch TMS training container."""

import os
import subprocess
from pathlib import Path

from shallow_fake.config import VoiceConfig
from shallow_fake.utils import ensure_dir, setup_logging

logger = setup_logging()


def prepare_tms_workspace(config: VoiceConfig):
    """Prepare TMS workspace with dataset and checkpoint."""
    tms_workspace = config.paths.tms_workspace_dir
    combined_dataset = config.paths.combined_dataset_dir
    tms_datasets_dir = tms_workspace / "datasets" / config.voice_id
    tms_checkpoints_dir = tms_workspace / "checkpoints" / "base_checkpoints"

    ensure_dir(tms_datasets_dir)
    ensure_dir(tms_checkpoints_dir)

    # Copy combined dataset to TMS workspace
    combined_metadata = combined_dataset / "metadata.csv"
    combined_wavs = combined_dataset / "wavs"

    if not combined_metadata.exists():
        logger.error(f"Combined dataset metadata not found: {combined_metadata}")
        raise FileNotFoundError(f"Combined dataset metadata not found: {combined_metadata}")

    # Create dataset structure in TMS workspace
    tms_dataset_dir = tms_datasets_dir / "combined"
    tms_wavs_dir = tms_dataset_dir / "wavs"
    ensure_dir(tms_wavs_dir)

    # Copy metadata
    import shutil
    shutil.copy2(combined_metadata, tms_dataset_dir / "metadata.csv")

    # Copy WAVs
    logger.info(f"Copying WAVs from {combined_wavs} to {tms_wavs_dir}")
    if combined_wavs.exists():
        for wav_file in combined_wavs.glob("*.wav"):
            shutil.copy2(wav_file, tms_wavs_dir / wav_file.name)

    logger.info(f"TMS workspace prepared: {tms_workspace}")

    # Check for base checkpoint
    base_checkpoint = config.training.base_checkpoint
    checkpoint_path = tms_checkpoints_dir / base_checkpoint
    if not checkpoint_path.exists():
        logger.warning(f"Base checkpoint not found: {checkpoint_path}")
        logger.warning("Please download the base checkpoint and place it in:")
        logger.warning(f"  {tms_checkpoints_dir}")
        logger.warning(f"Expected filename: {base_checkpoint}")
    else:
        logger.info(f"Base checkpoint found: {checkpoint_path}")


def launch_training(config: VoiceConfig):
    """Launch TMS training container."""
    compose_file = config.tms.docker_compose_file
    if not compose_file.exists():
        logger.error(f"Docker compose file not found: {compose_file}")
        raise FileNotFoundError(f"Docker compose file not found: {compose_file}")

    # Prepare workspace
    prepare_tms_workspace(config)

    # Set environment variables for docker-compose
    env = os.environ.copy()
    env.update({
        "PROJECT_NAME": config.voice_id,  # Use voice_id as project name
        "BASE_CHECKPOINT": config.training.base_checkpoint,
        "BATCH_SIZE": str(config.training.batch_size),
        "MAX_EPOCHS": str(config.training.max_epochs),
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

