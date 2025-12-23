"""Orchestration helpers for MetaVoice teacher model service."""

import subprocess
import time
from pathlib import Path
from typing import Optional

import requests

from shallow_fake.config import MetaVoiceTeacherConfig, VoiceConfig
from shallow_fake.utils import ensure_dir, setup_logging
from tools.reference_audio_utils import concatenate_reference_audio

logger = setup_logging()


def prepare_reference_audio(config: VoiceConfig) -> Path:
    """
    Prepare reference audio for MetaVoice by selecting best clip from verified dataset.

    Args:
        config: Voice configuration containing MetaVoice teacher settings

    Returns:
        Path to the prepared reference audio file
    """
    teacher = config.synthetic.teacher
    if not isinstance(teacher, MetaVoiceTeacherConfig):
        raise ValueError("Teacher config must be MetaVoiceTeacherConfig")

    # Validate reference audio directory
    reference_dir = Path(teacher.reference_audio_dir)
    if not reference_dir.exists():
        raise ValueError(
            f"Reference audio directory does not exist: {reference_dir}. "
            "Run 'build-dataset' and 'verify' first to create real dataset."
        )

    wav_files = list(reference_dir.glob("*.wav"))
    if not wav_files:
        raise ValueError(
            f"No WAV files found in reference directory: {reference_dir}. "
            "Run 'build-dataset' and 'verify' first to create cleaned audio segments."
        )

    # Create reference directory in workspace
    reference_output_dir = config.paths.workspace_dir / "reference"
    ensure_dir(reference_output_dir)
    reference_output_path = reference_output_dir / "voice_ref.wav"

    logger.info(f"Concatenating all {len(wav_files)} reference clips for MetaVoice (requires >=30s total)...")
    concatenate_reference_audio(
        reference_dir=reference_dir,
        output_path=reference_output_path,
        gap_duration=1.0,  # 1 second gap between files
    )

    return reference_output_path


def start_metavoice_teacher(config: VoiceConfig) -> None:
    """
    Start the MetaVoice teacher model service using Docker Compose.

    Args:
        config: Voice configuration containing MetaVoice teacher settings
    """
    if not config.synthetic.teacher:
        raise ValueError("Teacher model not configured in synthetic.teacher")

    teacher = config.synthetic.teacher
    if not isinstance(teacher, MetaVoiceTeacherConfig):
        raise ValueError(f"Expected MetaVoiceTeacherConfig, got {type(teacher).__name__}")

    logger.info(f"Starting MetaVoice teacher model service for {config.voice_id}...")

    # Prepare reference audio
    reference_audio_path = prepare_reference_audio(config)
    logger.info(f"Reference audio prepared: {reference_audio_path}")

    # Generate .env file with environment variables
    project_root = Path.cwd()
    env_file = project_root / f".env.{config.voice_id}.metavoice"

    # Resolve paths to absolute for Docker volume mount
    reference_audio_path_abs = reference_audio_path.resolve()
    reference_audio_dir_abs = reference_audio_path_abs.parent.resolve()

    with open(env_file, "w") as f:
        f.write(f"VOICE_ID={config.voice_id}\n")
        f.write(f"METAVOICE_PORT={teacher.port}\n")
        f.write(f"METAVOICE_REPO_ID={teacher.huggingface_repo_id}\n")
        f.write(f"REFERENCE_AUDIO_PATH={reference_audio_path_abs}\n")
        f.write(f"REFERENCE_AUDIO_DIR={reference_audio_dir_abs}\n")
        f.write(f"SPEAKER_REF_PATH={teacher.speaker_ref_path}\n")

    # Get docker-compose file path
    compose_file = project_root / "docker" / "docker-compose.metavoice.yml"
    if not compose_file.exists():
        raise FileNotFoundError(f"Docker compose file not found: {compose_file}")

    # Start the service
    compose_base_args = [
        "docker",
        "compose",
        "-f",
        str(compose_file),
        "--env-file",
        str(env_file),
    ]

    try:
        logger.info("Building and starting MetaVoice teacher model container...")
        # Build first
        build_args = compose_base_args + ["build"]
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

        # Then start the container (force recreate to ensure new GPU config is applied)
        start_args = compose_base_args + ["up", "-d", "--force-recreate"]
        logger.info("Starting container (forcing recreate to apply GPU configuration)...")
        start_result = subprocess.run(
            start_args,
            check=True,
            capture_output=True,
            text=True,
        )
        if start_result.stdout and start_result.stdout.strip():
            logger.debug(f"Docker compose output: {start_result.stdout}")
        if start_result.stderr and start_result.stderr.strip():
            logger.debug(f"Docker compose stderr: {start_result.stderr}")
        logger.info("MetaVoice teacher model container started")

    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to start MetaVoice teacher model service: {e.stderr}")
        if e.stdout:
            logger.error(f"Build output: {e.stdout}")
        raise

    # Wait for service to be ready
    base_url = f"http://localhost:{teacher.port}"
    max_retries = 300  # 10 minutes (300 * 2 seconds = 600 seconds)
    retry_delay = 2

    logger.info(
        "Waiting for MetaVoice teacher model service to be ready "
        "(this may take several minutes on first run while model compiles and initializes)..."
    )

    for attempt in range(max_retries):
        try:
            # Try /docs endpoint (FastAPI Swagger UI) - same as XTTS uses /health
            response = requests.get(f"{base_url}/docs", timeout=5)
            if response.status_code == 200:
                logger.info("MetaVoice teacher model service is ready")
                logger.info(f"Service URL: {base_url}")
                logger.info(f"Guidance: {teacher.guidance}, Top-p: {teacher.top_p}, Top-k: {teacher.top_k}")
                return
        except requests.exceptions.RequestException:
            pass

        if attempt < max_retries - 1:
            if attempt % 10 == 0:  # Log every 10 attempts (20 seconds)
                logger.info(
                    f"Waiting for MetaVoice teacher model service... ({attempt * retry_delay}s elapsed)"
                )
            time.sleep(retry_delay)
        else:
            # Check container status before raising error
            container_name = f"{config.voice_id}_metavoice_teacher"
            try:
                result = subprocess.run(
                    [
                        "docker",
                        "ps",
                        "-a",
                        "--filter",
                        f"name={container_name}",
                        "--format",
                        "{{.Status}}",
                    ],
                    capture_output=True,
                    text=True,
                    check=True,
                )
                container_status = result.stdout.strip()
                logger.error(f"Container status: {container_status}")
            except Exception:
                pass

            raise RuntimeError(
                f"MetaVoice teacher model service did not become ready after {max_retries * retry_delay} seconds. "
                f"Check container logs: docker logs {container_name}"
            )


def stop_metavoice_teacher(config: VoiceConfig) -> None:
    """
    Stop the MetaVoice teacher model service.

    Args:
        config: Voice configuration containing MetaVoice teacher settings
    """
    if not config.synthetic.teacher:
        return

    teacher = config.synthetic.teacher
    if not isinstance(teacher, MetaVoiceTeacherConfig):
        return

    logger.info(f"Stopping MetaVoice teacher model service for {config.voice_id}...")

    project_root = Path.cwd()
    env_file = project_root / f".env.{config.voice_id}.metavoice"
    compose_file = project_root / "docker" / "docker-compose.metavoice.yml"

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
        logger.info("MetaVoice teacher model container stopped")
    except subprocess.CalledProcessError as e:
        logger.warning(f"Failed to stop MetaVoice teacher model service: {e.stderr}")

    # Don't clean up .env file immediately - keep it for debugging/manual testing
    # It will be overwritten on next start anyway
    if env_file.exists():
        logger.debug(f"Env file kept for debugging: {env_file}")


def is_metavoice_teacher_running(config: VoiceConfig) -> bool:
    """
    Check if the MetaVoice teacher model service is running.

    Args:
        config: Voice configuration containing MetaVoice teacher settings

    Returns:
        True if service is running and responding, False otherwise
    """
    if not config.synthetic.teacher:
        return False

    teacher = config.synthetic.teacher
    if not isinstance(teacher, MetaVoiceTeacherConfig):
        return False

    base_url = f"http://localhost:{teacher.port}"

    try:
        response = requests.get(f"{base_url}/docs", timeout=2)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False





