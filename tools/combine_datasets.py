"""Combine real and synthetic datasets."""

import shutil
from pathlib import Path
from typing import Optional

from shallow_fake.config import VoiceConfig
from shallow_fake.utils import ensure_dir, setup_logging

logger = setup_logging()


def combine_datasets(config: VoiceConfig, subsample_synth: Optional[float] = None):
    """
    Combine real and synthetic datasets.

    Args:
        config: Voice configuration
        subsample_synth: If provided, subsample synthetic data by this ratio (0.0-1.0)
    """
    real_dir = config.paths.real_dataset_dir
    synth_dir = config.paths.synth_dataset_dir
    combined_dir = config.paths.combined_dataset_dir
    combined_wavs_dir = combined_dir / "wavs"
    combined_metadata = combined_dir / "metadata.csv"

    ensure_dir(combined_wavs_dir)

    # Use cleaned datasets if they exist
    real_clean_dir = real_dir.parent / f"{real_dir.name}_clean"
    synth_clean_dir = synth_dir.parent / f"{synth_dir.name}_clean"

    real_metadata = real_clean_dir / "metadata.csv" if (real_clean_dir / "metadata.csv").exists() else real_dir / "metadata.csv"
    synth_metadata = synth_clean_dir / "metadata.csv" if (synth_clean_dir / "metadata.csv").exists() else synth_dir / "metadata.csv"

    all_entries = []

    # Read real dataset
    if real_metadata.exists():
        real_wavs_dir = real_clean_dir / "wavs" if (real_clean_dir / "wavs").exists() else real_dir / "wavs"
        logger.info(f"Reading real dataset from {real_metadata}")
        with open(real_metadata, "r", encoding="utf-8") as f:
            for line in f:
                if "|" in line:
                    parts = line.strip().split("|", 1)
                    if len(parts) == 2:
                        wav_path, text = parts
                        source_wav = real_wavs_dir / Path(wav_path).name
                        if source_wav.exists():
                            all_entries.append(("real", wav_path, text, source_wav))
        logger.info(f"Loaded {len([e for e in all_entries if e[0] == 'real'])} real entries")
    else:
        logger.warning(f"Real dataset metadata not found: {real_metadata}")

    # Read synthetic dataset
    synth_entries = []
    if synth_metadata.exists():
        synth_wavs_dir = synth_clean_dir / "wavs" if (synth_clean_dir / "wavs").exists() else synth_dir / "wavs"
        logger.info(f"Reading synthetic dataset from {synth_metadata}")
        with open(synth_metadata, "r", encoding="utf-8") as f:
            for line in f:
                if "|" in line:
                    parts = line.strip().split("|", 1)
                    if len(parts) == 2:
                        wav_path, text = parts
                        source_wav = synth_wavs_dir / Path(wav_path).name
                        if source_wav.exists():
                            synth_entries.append((wav_path, text, source_wav))

        # Subsample if requested
        if subsample_synth and 0 < subsample_synth < 1.0:
            import random
            random.seed(42)  # For reproducibility
            num_to_keep = int(len(synth_entries) * subsample_synth)
            synth_entries = random.sample(synth_entries, num_to_keep)
            logger.info(f"Subsampled synthetic data: {num_to_keep} entries")

        for wav_path, text, source_wav in synth_entries:
            all_entries.append(("synth", wav_path, text, source_wav))
        logger.info(f"Loaded {len(synth_entries)} synthetic entries")
    else:
        logger.warning(f"Synthetic dataset metadata not found: {synth_metadata}")

    # Copy WAVs and build combined metadata
    logger.info(f"Combining {len(all_entries)} total entries")
    metadata_lines = []

    for source_type, wav_path, text, source_wav in all_entries:
        # Copy WAV to combined directory
        dest_wav = combined_wavs_dir / source_wav.name
        shutil.copy2(source_wav, dest_wav)

        # Build metadata line
        combined_wav_path = f"wavs/{source_wav.name}"
        metadata_lines.append(f"{combined_wav_path}|{text}")

    # Write combined metadata
    logger.info(f"Writing combined metadata with {len(metadata_lines)} entries")
    with open(combined_metadata, "w", encoding="utf-8") as f:
        for line in metadata_lines:
            f.write(line + "\n")

    logger.info(f"Combined dataset created: {combined_dir}")
    logger.info(f"  Real entries: {len([e for e in all_entries if e[0] == 'real'])}")
    logger.info(f"  Synthetic entries: {len([e for e in all_entries if e[0] == 'synth'])}")


if __name__ == "__main__":
    import sys
    from pathlib import Path
    from typing import Optional

    if len(sys.argv) < 2:
        print("Usage: python combine_datasets.py <config_path> [--subsample-synth RATIO]")
        sys.exit(1)

    config_path = Path(sys.argv[1])
    config = VoiceConfig.from_yaml(config_path)
    config.ensure_directories()

    subsample_ratio = None
    args = sys.argv[2:]
    i = 0
    while i < len(args):
        if args[i] == "--subsample-synth" and i + 1 < len(args):
            subsample_ratio = float(args[i + 1])
            i += 2
        else:
            i += 1

    combine_datasets(config, subsample_ratio)

