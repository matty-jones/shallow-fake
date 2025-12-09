"""Orchestration helpers for teacher model service."""

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
    Start the teacher model service using Docker Compose.

    Args:
        config: Voice configuration containing teacher settings
    """
    if not config.synthetic.teacher:
        raise ValueError("Teacher model not configured in synthetic.teacher")

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

    logger.info(f"Starting teacher model service for {config.voice_id}...")
    logger.info(f"Reference audio directory: {reference_dir}")
    logger.info(f"Found {len(wav_files)} reference WAV files")

    # Ensure teacher model cache directory exists
    project_root = Path.cwd()
    model_cache_dir = project_root / "models" / "shared" / "xtts_baseline"
    ensure_dir(model_cache_dir)
    
    # Check if model is already cached, and download if needed
    tts_cache = model_cache_dir / "tts"
    model_path = tts_cache / "tts_models--multilingual--multi-dataset--xtts_v2"
    model_file = model_path / "model.pth"
    
    if model_file.exists():
        file_size_mb = model_file.stat().st_size / (1024 * 1024)
        logger.info(f"Found cached teacher model at {model_path} ({file_size_mb:.0f} MB)")
    else:
        logger.info(f"Teacher model not found in cache at {model_path}")
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
        f.write(f"UVICORN_WORKERS={teacher.workers}\n")
        f.write(f"REFERENCE_AUDIO_DIR={reference_dir_abs}\n")
        f.write(f"MODEL_CACHE_DIR={model_cache_dir_abs}\n")
        # XTTS_SPEAKER_WAVS will be set to the directory path, server will find WAVs
        # (Note: Environment variable names remain XTTS-specific for compatibility)
        f.write(f"XTTS_SPEAKER_WAVS=/speakers\n")

    # Get docker-compose file path
    compose_file = project_root / "docker" / "docker-compose.xtts.yml"
    if not compose_file.exists():
        raise FileNotFoundError(f"Docker compose file not found: {compose_file}")

    # Start the service
    # Define compose args that will be reused
    compose_base_args = [
        "docker",
        "compose",
        "-f",
        str(compose_file),
        "--env-file",
        str(env_file),
    ]
    
    try:
        logger.info("Building and starting teacher model container (using cache if available)...")
        # Build with cache - Docker will automatically detect changes and rebuild only what's necessary
        # Multi-stage Dockerfile ensures expensive dependency layers are cached
        build_args = compose_base_args + ["build"]
        
        # Build first
        logger.info("Building Docker image (will use cache if nothing changed)...")
        build_result = subprocess.run(
            build_args,
            check=True,
            capture_output=True,
            text=True,
        )
        if build_result.stdout and build_result.stdout.strip():
            logger.debug(f"Build output: {build_result.stdout}")
        if build_result.stderr and build_result.stderr.strip():
            logger.debug(f"Build stderr: {build_result.stderr}")
        logger.info("Docker image built successfully")
        
        # Then start the container
        start_args = compose_base_args + ["up", "-d"]
        
        logger.info("Starting container...")
        start_result = subprocess.run(
            start_args,
            check=True,
            capture_output=True,
            text=True,
        )
        
        # Log output if available (usually empty for -d mode, but useful for debugging)
        if start_result.stdout and start_result.stdout.strip():
            logger.debug(f"Docker compose output: {start_result.stdout}")
        if start_result.stderr and start_result.stderr.strip():
            logger.debug(f"Docker compose stderr: {start_result.stderr}")
        logger.info("Teacher model container started")
        
        # Clean up old/unused images after successful build to free disk space
        # This removes dangling images and images not associated with any container
        logger.debug("Cleaning up old Docker images...")
        try:
            subprocess.run(
                ["docker", "image", "prune", "-f"],
                check=False,  # Don't fail if cleanup fails
                capture_output=True,
                text=True,
            )
        except Exception:
            pass  # Non-critical cleanup
        
    except subprocess.CalledProcessError as e:
        # Check if error is related to cache corruption (missing parent snapshot)
        error_msg = e.stderr or ""
        if "parent snapshot" in error_msg or "does not exist: not found" in error_msg:
            logger.warning("Docker build cache appears corrupted. Cleaning cache and retrying...")
            # Clean cache and retry with --no-cache
            try:
                subprocess.run(
                    ["docker", "builder", "prune", "-f"],
                    check=False,
                    capture_output=True,
                    text=True,
                )
                logger.info("Retrying build with --no-cache after cache cleanup...")
                # Retry with --no-cache
                build_args_retry = compose_base_args + ["build", "--no-cache"]
                subprocess.run(
                    build_args_retry,
                    check=True,
                    capture_output=True,
                    text=True,
                )
                # Start container after successful retry
                start_result = subprocess.run(
                    start_args,
                    check=True,
                    capture_output=True,
                    text=True,
                )
                logger.info("Teacher model container started after cache cleanup")
            except subprocess.CalledProcessError as retry_error:
                logger.error(f"Failed to start teacher model service even after cache cleanup: {retry_error.stderr}")
                raise
        else:
            logger.error(f"Failed to start teacher model service: {e.stderr}")
            if e.stdout:
                logger.error(f"Build output: {e.stdout}")
            raise

    # Wait for service to be ready
    base_url = f"http://localhost:{teacher.port}"
    max_retries = 60  # Increased to 2 minutes for model download
    retry_delay = 2

    logger.info("Waiting for teacher model service to be ready (this may take a few minutes on first run while model downloads)...")
    
    for attempt in range(max_retries):
        try:
            response = requests.get(f"{base_url}/health", timeout=5)
            if response.status_code == 200:
                health_data = response.json()
                logger.info("Teacher model service is ready")
                # Log device and GPU information
                device = health_data.get("device", "unknown")
                cuda_available = health_data.get("cuda_available", False)
                gpu_name = health_data.get("gpu_name")
                num_clips = health_data.get("num_reference_clips", "unknown")
                speaker_files = health_data.get("speaker_files", 0)
                
                if device == "cuda" and cuda_available:
                    logger.info(f"GPU acceleration: ENABLED ({gpu_name})")
                    # Log GPU memory info if available
                    gpu_memory = health_data.get("gpu_memory")
                    if gpu_memory:
                        allocated = gpu_memory.get("allocated_gb", 0)
                        total = gpu_memory.get("total_memory_gb", 0)
                        utilization = gpu_memory.get("utilization_percent", 0)
                        logger.info(f"GPU memory: {allocated:.2f} GB / {total:.2f} GB ({utilization:.1f}% utilized)")
                elif device == "cuda" and not cuda_available:
                    logger.warning(f"GPU acceleration: REQUESTED but CUDA not available, using CPU")
                else:
                    logger.info(f"GPU acceleration: DISABLED (using CPU)")
                
                if num_clips == 0:
                    logger.info(f"Reference audio: Using all {speaker_files} available files per request")
                else:
                    logger.info(f"Reference audio: Using {num_clips} files per request (from {speaker_files} available)")
                return
        except requests.exceptions.RequestException as e:
            pass

        if attempt < max_retries - 1:
            if attempt % 10 == 0:  # Log every 10 attempts (20 seconds)
                logger.info(f"Waiting for teacher model service... ({attempt * retry_delay}s elapsed)")
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
                f"Teacher model service did not become ready after {max_retries * retry_delay} seconds. "
                f"Check container logs: docker logs {config.voice_id}_xtts_teacher"
            )


def stop_xtts_teacher(config: VoiceConfig) -> None:
    """
    Stop the teacher model service.

    Args:
        config: Voice configuration containing teacher settings
    """
    if not config.synthetic.teacher:
        return

    teacher = config.synthetic.teacher
    logger.info(f"Stopping teacher model service for {config.voice_id}...")

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
        logger.info("Teacher model container stopped")
    except subprocess.CalledProcessError as e:
        logger.warning(f"Failed to stop teacher model service: {e.stderr}")

    # Clean up .env file
    if env_file.exists():
        env_file.unlink()
        logger.debug(f"Cleaned up {env_file}")


def is_xtts_teacher_running(config: VoiceConfig) -> bool:
    """
    Check if the teacher model service is running.

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

