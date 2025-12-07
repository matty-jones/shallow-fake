"""Build dataset metadata in LJSpeech format."""

import json
import shutil
from pathlib import Path

from shallow_fake.config import VoiceConfig
from shallow_fake.utils import ensure_dir, setup_logging

logger = setup_logging()


def filter_segment(segment: dict, min_words: int = 3, max_words: int = 50) -> bool:
    """Filter segments based on text quality."""
    text = segment.get("text", "")
    words = text.split()

    # Check word count
    if len(words) < min_words or len(words) > max_words:
        return False

    # Check for too many non-ASCII characters (basic heuristic)
    non_ascii_ratio = sum(1 for c in text if ord(c) > 127) / len(text) if text else 0
    if non_ascii_ratio > 0.1:  # More than 10% non-ASCII
        return False

    return True


def build_metadata(config: VoiceConfig, min_words: int = 3, max_words: int = 50, max_segments: int = None):
    """Build LJSpeech format metadata.csv from ASR segments."""
    asr_metadata_path = config.paths.asr_metadata
    segments_dir = config.paths.segments_dir
    real_dataset_dir = config.paths.real_dataset_dir
    wavs_dir = real_dataset_dir / "wavs"
    metadata_csv = real_dataset_dir / "metadata.csv"

    ensure_dir(wavs_dir)

    if not asr_metadata_path.exists():
        logger.error(f"ASR metadata file not found: {asr_metadata_path}")
        raise FileNotFoundError(f"ASR metadata file not found: {asr_metadata_path}")

    # Read segments
    segments = []
    with open(asr_metadata_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                segments.append(json.loads(line))

    logger.info(f"Loaded {len(segments)} segments from ASR metadata")

    # Filter segments
    filtered_segments = [s for s in segments if filter_segment(s, min_words, max_words)]
    logger.info(f"After filtering: {len(filtered_segments)} segments")

    # Limit number of segments if specified
    if max_segments and len(filtered_segments) > max_segments:
        logger.info(f"Limiting to {max_segments} segments")
        filtered_segments = filtered_segments[:max_segments]

    # Copy WAVs and build metadata
    metadata_lines = []
    for i, segment in enumerate(filtered_segments):
        segment_id = segment["id"]
        source_wav = segments_dir / f"{segment_id}.wav"

        if not source_wav.exists():
            logger.warning(f"Segment WAV not found: {source_wav}")
            continue

        # Copy to dataset directory
        dest_wav = wavs_dir / f"{segment_id}.wav"
        shutil.copy2(source_wav, dest_wav)

        # Build metadata line (LJSpeech format: wavs/filename.wav|transcript)
        text = segment["text"].strip()
        # Escape pipe characters in text
        text = text.replace("|", " ")
        metadata_line = f"wavs/{segment_id}.wav|{text}"
        metadata_lines.append(metadata_line)

    # Write metadata.csv
    logger.info(f"Writing metadata.csv with {len(metadata_lines)} entries")
    with open(metadata_csv, "w", encoding="utf-8") as f:
        for line in metadata_lines:
            f.write(line + "\n")

    logger.info(f"Dataset built: {len(metadata_lines)} entries in {real_dataset_dir}")


if __name__ == "__main__":
    import sys
    from pathlib import Path

    if len(sys.argv) < 2:
        print("Usage: python build_metadata.py <config_path> [--min-words N] [--max-words N] [--max-segments N]")
        sys.exit(1)

    config_path = Path(sys.argv[1])
    config = VoiceConfig.from_yaml(config_path)
    config.ensure_directories()

    # Parse optional arguments
    min_words = 3
    max_words = 50
    max_segments = None

    args = sys.argv[2:]
    i = 0
    while i < len(args):
        if args[i] == "--min-words" and i + 1 < len(args):
            min_words = int(args[i + 1])
            i += 2
        elif args[i] == "--max-words" and i + 1 < len(args):
            max_words = int(args[i + 1])
            i += 2
        elif args[i] == "--max-segments" and i + 1 < len(args):
            max_segments = int(args[i + 1])
            i += 2
        else:
            i += 1

    build_metadata(config, min_words, max_words, max_segments)




