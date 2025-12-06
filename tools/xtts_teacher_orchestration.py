"""Orchestration helpers for XTTS teacher service."""

import random
import subprocess
import time
from pathlib import Path
from typing import List

import requests

from shallow_fake.config import VoiceConfig
from shallow_fake.utils import ensure_dir, setup_logging

logger = setup_logging()


def select_reference_audio(reference_dir: Path, num_clips: int) -> List[Path]:
    """
    Randomly select N WAV files from the reference directory.

    Args:
        reference_dir: Directory containing reference WAV files
        num_clips: Number of clips to select

    Returns:
        List of selected WAV file paths
    """
    if not reference_dir.exists():
        raise ValueError(f"Reference audio directory does not exist: {reference_dir}")

    wav_files = list(reference_dir.glob("*.wav"))
    if not wav_files:
        raise ValueError(f"No WAV files found in reference directory: {reference_dir}")

    if len(wav_files) <= num_clips:
        return wav_files
    else:
        return random.sample(wav_files, num_clips)


def start_xtts_teacher(config: VoiceConfig) -> None:
    """
    Start the XTTS teacher service using Docker Compose.

    Args:
        config: Voice configuration containing teacher settings
    """
    if not config.synthetic.teacher:
        raise ValueError("XTTS teacher not configured in synthetic.teacher")

    teacher = config.synthetic.teacher
    if teacher.kind != "xtts":
        raise ValueError(f"Unsupported teacher kind: {teacher.kind}")

    # Validate reference audio directory
    reference_dir = Path(teacher.reference_audio_dir)
    if not reference_dir.exists():
        raise ValueError(
            f"Reference audio directory does not exist: {reference_dir}. "
            "Run 'build-dataset' and 'verify' first to create real_clean dataset."
        )

    wav_files = list(reference_dir.glob("*.wav"))
    if not wav_files:
        raise ValueError(
            f"No WAV files found in reference directory: {reference_dir}. "
            "Run 'build-dataset' and 'verify' first to create cleaned audio segments."
        )

    logger.info(f"Starting XTTS teacher service for {config.voice_id}...")
    logger.info(f"Reference audio directory: {reference_dir}")
    logger.info(f"Found {len(wav_files)} reference WAV files")

    # Ensure XTTS model cache directory exists
    project_root = Path.cwd()
    model_cache_dir = project_root / "models" / "xtts_baseline"
    ensure_dir(model_cache_dir)
    
    # Check if model is already cached, and download if needed
    tts_cache = model_cache_dir / "tts"
    model_path = tts_cache / "tts_models--multilingual--multi-dataset--xtts_v2"
    model_file = model_path / "model.pth"
    
    if model_file.exists():
        file_size_mb = model_file.stat().st_size / (1024 * 1024)
        logger.info(f"Found cached XTTS model at {model_path} ({file_size_mb:.0f} MB)")
    else:
        logger.info(f"XTTS model not found in cache at {model_path}")
        logger.info("Pre-downloading model to cache (~1.7GB, one-time download)...")
        
        try:
            import sys
            tools_dir = Path(__file__).parent
            if str(tools_dir) not in sys.path:
                sys.path.insert(0, str(tools_dir))
            
            from download_xtts_model import download_xtts_model
            
            download_xtts_model(
                model_name=teacher.model_name,
                cache_dir=model_cache_dir,
                device="cpu",  # Use CPU for download to avoid GPU requirements
            )
            logger.info("Model pre-download complete")
        except Exception as e:
            logger.warning(f"Pre-download failed: {e}. Container will download on first use.")

    # Generate .env file with environment variables
    env_file = project_root / f".env.{config.voice_id}.xtts"
    
    # Resolve reference audio directory to absolute path for Docker volume mount
    reference_dir_abs = reference_dir.resolve()
    model_cache_dir_abs = model_cache_dir.resolve()

    with open(env_file, "w") as f:
        f.write(f"VOICE_ID={config.voice_id}\n")
        f.write(f"XTTS_PORT={teacher.port}\n")
        f.write(f"XTTS_MODEL_NAME={teacher.model_name}\n")
        f.write(f"XTTS_LANGUAGE={teacher.language}\n")
        f.write(f"XTTS_DEVICE={teacher.device}\n")
        f.write(f"XTTS_NUM_REFERENCE_CLIPS={teacher.num_reference_clips}\n")
        f.write(f"REFERENCE_AUDIO_DIR={reference_dir_abs}\n")
        f.write(f"MODEL_CACHE_DIR={model_cache_dir_abs}\n")
        # XTTS_SPEAKER_WAVS will be set to the directory path, server will find WAVs
        f.write(f"XTTS_SPEAKER_WAVS=/speakers\n")

    # Get docker-compose file path
    compose_file = project_root / "docker" / "docker-compose.xtts.yml"
    if not compose_file.exists():
        raise FileNotFoundError(f"Docker compose file not found: {compose_file}")

    # Start the service
    try:
        logger.info("Building and starting XTTS teacher container (this may take a few minutes on first build)...")
        result = subprocess.run(
            [
                "docker",
                "compose",
                "-f",
                str(compose_file),
                "--env-file",
                str(env_file),
                "up",
                "-d",
                "--build",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        # Log output if available (usually empty for -d mode, but useful for debugging)
        if result.stdout and result.stdout.strip():
            logger.debug(f"Docker compose output: {result.stdout}")
        if result.stderr and result.stderr.strip():
            logger.debug(f"Docker compose stderr: {result.stderr}")
        logger.info("XTTS teacher container started")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to start XTTS teacher: {e.stderr}")
        raise

    # Wait for service to be ready
    base_url = f"http://localhost:{teacher.port}"
    max_retries = 60  # Increased to 2 minutes for model download
    retry_delay = 2

    logger.info("Waiting for XTTS teacher service to be ready (this may take a few minutes on first run while model downloads)...")
    
    for attempt in range(max_retries):
        try:
            response = requests.get(f"{base_url}/health", timeout=5)
            if response.status_code == 200:
                logger.info("XTTS teacher service is ready")
                return
        except requests.exceptions.RequestException as e:
            pass

        if attempt < max_retries - 1:
            if attempt % 10 == 0:  # Log every 10 attempts (20 seconds)
                logger.info(f"Waiting for XTTS teacher service... ({attempt * retry_delay}s elapsed)")
            time.sleep(retry_delay)
        else:
            # Check container status before raising error
            try:
                result = subprocess.run(
                    ["docker", "ps", "-a", "--filter", f"name={config.voice_id}_xtts_teacher", "--format", "{{.Status}}"],
                    capture_output=True,
                    text=True,
                    check=True,
                )
                container_status = result.stdout.strip()
                logger.error(f"Container status: {container_status}")
            except Exception:
                pass
            
            raise RuntimeError(
                f"XTTS teacher service did not become ready after {max_retries * retry_delay} seconds. "
                f"Check container logs: docker logs {config.voice_id}_xtts_teacher"
            )


def stop_xtts_teacher(config: VoiceConfig) -> None:
    """
    Stop the XTTS teacher service.

    Args:
        config: Voice configuration containing teacher settings
    """
    if not config.synthetic.teacher:
        return

    teacher = config.synthetic.teacher
    logger.info(f"Stopping XTTS teacher service for {config.voice_id}...")

    project_root = Path.cwd()
    env_file = project_root / f".env.{config.voice_id}.xtts"
    compose_file = project_root / "docker" / "docker-compose.xtts.yml"

    if not compose_file.exists():
        logger.warning(f"Docker compose file not found: {compose_file}")
        return

    try:
        subprocess.run(
            [
                "docker",
                "compose",
                "-f",
                str(compose_file),
                "--env-file",
                str(env_file),
                "down",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        logger.info("XTTS teacher container stopped")
    except subprocess.CalledProcessError as e:
        logger.warning(f"Failed to stop XTTS teacher: {e.stderr}")

    # Clean up .env file
    if env_file.exists():
        env_file.unlink()
        logger.debug(f"Cleaned up {env_file}")


def is_xtts_teacher_running(config: VoiceConfig) -> bool:
    """
    Check if the XTTS teacher service is running.

    Args:
        config: Voice configuration containing teacher settings

    Returns:
        True if service is running and responding, False otherwise
    """
    if not config.synthetic.teacher:
        return False

    teacher = config.synthetic.teacher
    base_url = f"http://localhost:{teacher.port}"

    try:
        response = requests.get(f"{base_url}/health", timeout=2)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False

