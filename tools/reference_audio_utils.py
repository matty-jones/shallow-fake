"""Utilities for selecting and preparing reference audio for teacher models."""

import subprocess
from pathlib import Path
from typing import Optional

from shallow_fake.utils import ensure_dir, setup_logging

logger = setup_logging()


def get_audio_duration(audio_path: Path) -> Optional[float]:
    """
    Get duration of audio file in seconds using ffprobe.

    Args:
        audio_path: Path to audio file

    Returns:
        Duration in seconds, or None if unable to determine
    """
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(audio_path),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        return float(result.stdout.strip())
    except (subprocess.CalledProcessError, ValueError, FileNotFoundError) as e:
        logger.warning(f"Could not determine duration for {audio_path}: {e}")
        return None


def select_best_reference_clip(
    reference_dir: Path,
    output_path: Path,
    min_duration: float = 5.0,
    max_duration: float = 30.0,
) -> Path:
    """
    Select the best reference audio clip from a directory for MetaVoice.

    Selects the longest clip within the preferred duration range (5-30 seconds).
    If no clips are in range, selects the longest available clip.

    Args:
        reference_dir: Directory containing reference WAV files
        output_path: Path where selected clip should be copied
        min_duration: Minimum preferred duration in seconds (default: 5.0)
        max_duration: Maximum preferred duration in seconds (default: 30.0)

    Returns:
        Path to the selected reference clip

    Raises:
        ValueError: If no valid WAV files are found in reference_dir
    """
    if not reference_dir.exists():
        raise ValueError(f"Reference audio directory does not exist: {reference_dir}")

    wav_files = list(reference_dir.glob("*.wav"))
    if not wav_files:
        raise ValueError(f"No WAV files found in reference directory: {reference_dir}")

    logger.info(f"Evaluating {len(wav_files)} reference audio files...")

    # Score each file by duration
    scored_files = []
    for wav_file in wav_files:
        duration = get_audio_duration(wav_file)
        if duration is None:
            logger.debug(f"Skipping {wav_file.name}: could not determine duration")
            continue

        # Prefer files in the 5-30 second range
        if min_duration <= duration <= max_duration:
            # Score: duration (prefer longer within range)
            score = duration
            in_range = True
        else:
            # Score: negative duration (prefer shorter if outside range)
            # This ensures we only use out-of-range files if nothing is in range
            score = -duration
            in_range = False

        scored_files.append((score, duration, in_range, wav_file))

    if not scored_files:
        raise ValueError(
            f"No valid WAV files with determinable duration found in {reference_dir}"
        )

    # Sort by score (highest first), then by in_range status
    scored_files.sort(key=lambda x: (x[2], x[0]), reverse=True)

    # Select the best file
    best_score, best_duration, best_in_range, best_file = scored_files[0]

    if best_in_range:
        logger.info(
            f"Selected {best_file.name} ({best_duration:.2f}s) - optimal duration for MetaVoice"
        )
    else:
        logger.warning(
            f"Selected {best_file.name} ({best_duration:.2f}s) - outside preferred range "
            f"({min_duration}-{max_duration}s), but best available option"
        )

    # Ensure output directory exists
    ensure_dir(output_path.parent)

    # Copy selected file to output path
    import shutil

    shutil.copy2(best_file, output_path)
    logger.info(f"Copied reference clip to {output_path}")

    return output_path





