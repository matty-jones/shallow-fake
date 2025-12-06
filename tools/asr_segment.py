"""ASR segmentation tool using Whisper."""

import json
import subprocess
from pathlib import Path
from typing import List

import ffmpeg
from faster_whisper import WhisperModel

from shallow_fake.config import VoiceConfig
from shallow_fake.utils import ensure_dir, setup_logging

logger = setup_logging()


def normalize_audio(input_path: Path, output_path: Path) -> Path:
    """Normalize audio to 22.05 kHz mono 16-bit PCM WAV."""
    logger.info(f"Normalizing audio: {input_path.name} -> {output_path.name}")

    try:
        (
            ffmpeg.input(str(input_path))
            .output(
                str(output_path),
                acodec="pcm_s16le",
                ac=1,
                ar=22050,
                loglevel="error",
            )
            .overwrite_output()
            .run(capture_output=True, check=True)
        )
        return output_path
    except ffmpeg.Error as e:
        logger.error(f"FFmpeg error: {e.stderr.decode() if e.stderr else 'Unknown error'}")
        raise
    except Exception as e:
        logger.error(f"Error normalizing audio: {e}")
        raise


def find_audio_files(directory: Path) -> List[Path]:
    """Find all audio files in a directory."""
    audio_extensions = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".webm", ".mp4"}
    audio_files = []
    for ext in audio_extensions:
        audio_files.extend(directory.glob(f"*{ext}"))
        audio_files.extend(directory.glob(f"*{ext.upper()}"))
    return sorted(audio_files)


def segment_audio_with_whisper(
    audio_path: Path,
    model_size: str,
    device: str,
    beam_size: int,
    min_segment_seconds: float,
    max_segment_seconds: float,
    min_confidence: float,
) -> List[dict]:
    """Segment audio using Whisper and return segment metadata."""
    logger.info(f"Loading Whisper model: {model_size} on {device}")
    model = WhisperModel(model_size, device=device, compute_type="float16" if device == "cuda" else "int8")

    logger.info(f"Transcribing: {audio_path.name}")
    segments, info = model.transcribe(
        str(audio_path),
        beam_size=beam_size,
        language="en",
        word_timestamps=True,
    )

    logger.info(f"Detected language: {info.language} (probability: {info.language_probability:.2f})")

    segment_list = []
    for segment in segments:
        duration = segment.end - segment.start

        # Filter by duration
        if duration < min_segment_seconds or duration > max_segment_seconds:
            continue

        # Filter by confidence (if available)
        # Note: faster-whisper doesn't provide segment-level confidence directly
        # We'll use average word confidence as a proxy
        if segment.words:
            avg_confidence = sum(w.probability for w in segment.words) / len(segment.words)
            if avg_confidence < min_confidence:
                logger.debug(f"Skipping segment with low confidence: {avg_confidence:.2f}")
                continue

        segment_list.append({
            "start": segment.start,
            "end": segment.end,
            "duration": duration,
            "text": segment.text.strip(),
            "confidence": avg_confidence if segment.words else 1.0,
        })

    logger.info(f"Extracted {len(segment_list)} segments from {audio_path.name}")
    return segment_list


def extract_segment_audio(
    source_audio: Path,
    start: float,
    end: float,
    output_path: Path,
) -> Path:
    """Extract a segment from source audio."""
    try:
        (
            ffmpeg.input(str(source_audio), ss=start, t=end - start)
            .output(
                str(output_path),
                acodec="pcm_s16le",
                ac=1,
                ar=22050,
                loglevel="error",
            )
            .overwrite_output()
            .run(capture_output=True, check=True)
        )
        return output_path
    except ffmpeg.Error as e:
        logger.error(f"FFmpeg error extracting segment: {e.stderr.decode() if e.stderr else 'Unknown error'}")
        raise
    except Exception as e:
        logger.error(f"Error extracting segment: {e}")
        raise


def process_audio_files(config: VoiceConfig):
    """Process all audio files in the raw audio directory."""
    raw_dir = config.paths.raw_audio_dir
    normalized_dir = config.paths.normalized_dir
    segments_dir = config.paths.segments_dir
    metadata_path = config.paths.asr_metadata

    ensure_dir(normalized_dir)
    ensure_dir(segments_dir)

    audio_files = find_audio_files(raw_dir)
    if not audio_files:
        logger.warning(f"No audio files found in {raw_dir}")
        return

    logger.info(f"Found {len(audio_files)} audio file(s) to process")

    all_segments = []
    segment_counter = 0

    for audio_file in audio_files:
        logger.info(f"Processing: {audio_file.name}")

        # Normalize audio
        normalized_path = normalized_dir / f"{audio_file.stem}.wav"
        normalize_audio(audio_file, normalized_path)

        # Segment with Whisper
        segments = segment_audio_with_whisper(
            normalized_path,
            config.asr.model_size,
            config.asr.device,
            config.asr.beam_size,
            config.asr.min_segment_seconds,
            config.asr.max_segment_seconds,
            config.asr.min_confidence,
        )

        # Extract segment audio files
        for seg in segments:
            segment_id = f"seg_{segment_counter:04d}"
            segment_wav = segments_dir / f"{segment_id}.wav"

            extract_segment_audio(
                normalized_path,
                seg["start"],
                seg["end"],
                segment_wav,
            )

            # Add metadata
            seg["id"] = segment_id
            seg["audio_path"] = str(segment_wav.relative_to(segments_dir.parent))
            all_segments.append(seg)
            segment_counter += 1

    # Write JSONL metadata
    logger.info(f"Writing {len(all_segments)} segments to {metadata_path}")
    with open(metadata_path, "w", encoding="utf-8") as f:
        for seg in all_segments:
            f.write(json.dumps(seg, ensure_ascii=False) + "\n")

    logger.info(f"ASR segmentation complete. {len(all_segments)} segments created.")


if __name__ == "__main__":
    import sys
    from pathlib import Path

    if len(sys.argv) < 2:
        print("Usage: python asr_segment.py <config_path>")
        sys.exit(1)

    config_path = Path(sys.argv[1])
    config = VoiceConfig.from_yaml(config_path)
    config.ensure_directories()
    process_audio_files(config)

