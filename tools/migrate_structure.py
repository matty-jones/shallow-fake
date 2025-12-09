"""Migration script to move from old directory structure to new unified structure."""

import os
import shutil
from pathlib import Path
from typing import Optional

from shallow_fake.config import VoiceConfig
from shallow_fake.utils import ensure_dir, setup_logging

logger = setup_logging()


def migrate_project(config: VoiceConfig, dry_run: bool = False):
    """Migrate a project from old structure to new unified structure."""
    voice_id = config.voice_id
    project_root = Path.cwd()
    
    logger.info(f"Migrating project '{voice_id}' to new structure...")
    if dry_run:
        logger.info("DRY RUN MODE - No files will be moved")
    
    # Create new directory structure
    new_input_dir = config.paths.input_audio_dir
    new_workspace_dir = config.paths.workspace_dir
    new_models_dir = config.paths.models_dir
    new_shared_dir = config.paths.shared_models_dir
    
    if not dry_run:
        ensure_dir(new_input_dir)
        ensure_dir(new_workspace_dir)
        ensure_dir(new_models_dir)
        ensure_dir(new_shared_dir)
    
    # Migration mappings: (old_path, new_path, use_hard_link)
    migrations = []
    
    # 1. Input audio
    old_input = project_root / "data_raw" / voice_id / "input_audio"
    if old_input.exists():
        for file in old_input.glob("*"):
            if file.is_file():
                new_file = new_input_dir / file.name
                migrations.append((file, new_file, False))
    
    # 2. Segments
    old_segments = project_root / "data_processed" / voice_id / "segments"
    new_segments = new_workspace_dir / "segments"
    if old_segments.exists():
        if not dry_run:
            ensure_dir(new_segments)
        for file in old_segments.glob("*.wav"):
            new_file = new_segments / file.name
            migrations.append((file, new_file, True))  # Use hard link
    
    # 3. ASR metadata
    old_metadata = project_root / "data_processed" / voice_id / "asr_segments.jsonl"
    new_metadata = new_segments / "asr_metadata.jsonl"
    if old_metadata.exists():
        migrations.append((old_metadata, new_metadata, False))
    
    # 4. Real dataset - move to workspace, use hard links for wavs
    old_real = project_root / "datasets" / voice_id / "real"
    new_real = new_workspace_dir / "datasets" / "real"
    if old_real.exists():
        if not dry_run:
            ensure_dir(new_real)
            ensure_dir(new_real / "wavs")
        # Metadata
        old_meta = old_real / "metadata.csv"
        if old_meta.exists():
            migrations.append((old_meta, new_real / "metadata.csv", False))
        # WAVs - use hard links
        old_wavs = old_real / "wavs"
        if old_wavs.exists():
            for file in old_wavs.glob("*.wav"):
                new_file = new_real / "wavs" / file.name
                migrations.append((file, new_file, True))
    
    # 5. Real clean dataset - merge into real (update metadata, use hard links)
    old_real_clean = project_root / "datasets" / voice_id / "real_clean"
    if old_real_clean.exists():
        old_clean_meta = old_real_clean / "metadata.csv"
        if old_clean_meta.exists() and not (new_real / "metadata.csv").exists():
            # Use cleaned metadata if original doesn't exist
            migrations.append((old_clean_meta, new_real / "metadata.csv", False))
        # WAVs - use hard links (will overwrite if duplicates)
        old_clean_wavs = old_real_clean / "wavs"
        if old_clean_wavs.exists():
            for file in old_clean_wavs.glob("*.wav"):
                new_file = new_real / "wavs" / file.name
                migrations.append((file, new_file, True))
    
    # 6. Synthetic dataset
    old_synth = project_root / "datasets" / voice_id / "synth"
    new_synth = new_workspace_dir / "datasets" / "synth"
    if old_synth.exists():
        if not dry_run:
            ensure_dir(new_synth)
            ensure_dir(new_synth / "wavs")
        old_meta = old_synth / "metadata.csv"
        if old_meta.exists():
            migrations.append((old_meta, new_synth / "metadata.csv", False))
        old_wavs = old_synth / "wavs"
        if old_wavs.exists():
            for file in old_wavs.glob("*.wav"):
                new_file = new_synth / "wavs" / file.name
                migrations.append((file, new_file, False))  # Copy synth files
    
    # 7. Synthetic clean dataset - merge into synth
    old_synth_clean = project_root / "datasets" / voice_id / "synth_clean"
    if old_synth_clean.exists():
        old_clean_meta = old_synth_clean / "metadata.csv"
        if old_clean_meta.exists() and not (new_synth / "metadata.csv").exists():
            migrations.append((old_clean_meta, new_synth / "metadata.csv", False))
        old_clean_wavs = old_synth_clean / "wavs"
        if old_clean_wavs.exists():
            for file in old_clean_wavs.glob("*.wav"):
                new_file = new_synth / "wavs" / file.name
                migrations.append((file, new_file, False))
    
    # 8. Combined dataset
    old_combined = project_root / "datasets" / voice_id / "combined"
    new_combined = new_workspace_dir / "datasets" / "combined"
    if old_combined.exists():
        if not dry_run:
            ensure_dir(new_combined)
            ensure_dir(new_combined / "wavs")
        old_meta = old_combined / "metadata.csv"
        if old_meta.exists():
            migrations.append((old_meta, new_combined / "metadata.csv", False))
        old_wavs = old_combined / "wavs"
        if old_wavs.exists():
            for file in old_wavs.glob("*.wav"):
                new_file = new_combined / "wavs" / file.name
                migrations.append((file, new_file, True))  # Use hard links
    
    # 9. TMS workspace datasets
    old_tms_combined = project_root / "tms_workspace" / "datasets" / voice_id / "combined"
    if old_tms_combined.exists():
        old_meta = old_tms_combined / "metadata.csv"
        if old_meta.exists() and not (new_combined / "metadata.csv").exists():
            migrations.append((old_meta, new_combined / "metadata.csv", False))
        old_wavs = old_tms_combined / "wavs"
        if old_wavs.exists():
            for file in old_wavs.glob("*.wav"):
                new_file = new_combined / "wavs" / file.name
                migrations.append((file, new_file, True))
    
    # 10. TMS prepared dataset
    old_prepared = project_root / "tms_workspace" / "datasets" / voice_id / "combined_prepared"
    new_prepared = new_workspace_dir / "datasets" / "prepared"
    if old_prepared.exists():
        if not dry_run:
            ensure_dir(new_prepared)
        # Copy entire prepared directory
        for item in old_prepared.iterdir():
            if item.is_file():
                migrations.append((item, new_prepared / item.name, False))
            elif item.is_dir():
                # Recursively copy directories
                new_item_dir = new_prepared / item.name
                if not dry_run:
                    ensure_dir(new_item_dir)
                for subitem in item.rglob("*"):
                    if subitem.is_file():
                        rel_path = subitem.relative_to(old_prepared)
                        new_subitem = new_prepared / rel_path
                        migrations.append((subitem, new_subitem, False))
    
    # 11. Models
    old_models = project_root / "models" / voice_id
    if old_models.exists():
        if not dry_run:
            ensure_dir(new_models_dir)
        for file in old_models.glob("*"):
            if file.is_file():
                new_file = new_models_dir / file.name
                migrations.append((file, new_file, False))
    
    # 12. Samples (move from models/samples to workspace/training/samples)
    old_samples = project_root / "models" / "samples" / voice_id
    new_samples = new_workspace_dir / "training" / "samples"
    if old_samples.exists():
        if not dry_run:
            ensure_dir(new_samples)
        for file in old_samples.glob("*"):
            if file.is_file():
                new_file = new_samples / file.name
                migrations.append((file, new_file, False))
    
    # 13. Shared resources
    old_base_checkpoints = project_root / "tms_workspace" / "checkpoints" / "base_checkpoints"
    new_base_checkpoints = new_shared_dir / "base_checkpoints"
    if old_base_checkpoints.exists():
        if not dry_run:
            ensure_dir(new_base_checkpoints)
        for file in old_base_checkpoints.glob("*.ckpt"):
            new_file = new_base_checkpoints / file.name
            migrations.append((file, new_file, False))
    
    # 14. Corpus
    old_corpus = project_root / "data_raw" / "external_corpus" / "corpus.txt"
    new_corpus = config.paths.corpus_path
    if old_corpus.exists() and not new_corpus.exists():
        migrations.append((old_corpus, new_corpus, False))
    
    # Execute migrations
    logger.info(f"Found {len(migrations)} files/directories to migrate")
    for old_path, new_path, use_hard_link in migrations:
        if not old_path.exists():
            continue
        
        if dry_run:
            link_type = "hard link" if use_hard_link else "copy"
            logger.info(f"Would {link_type}: {old_path} -> {new_path}")
        else:
            # Ensure parent directory exists
            new_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Skip if destination exists and is same file
            if new_path.exists():
                if use_hard_link and new_path.samefile(old_path):
                    continue
                elif not use_hard_link and new_path.stat().st_size == old_path.stat().st_size:
                    # Check if files are identical
                    if new_path.read_bytes() == old_path.read_bytes():
                        continue
            
            try:
                if use_hard_link:
                    # Try hard link first, fall back to copy if cross-filesystem
                    try:
                        os.link(old_path, new_path)
                        logger.debug(f"Hard linked: {old_path} -> {new_path}")
                    except OSError:
                        # Cross-filesystem or other error, use copy
                        shutil.copy2(old_path, new_path)
                        logger.debug(f"Copied (hard link failed): {old_path} -> {new_path}")
                else:
                    shutil.copy2(old_path, new_path)
                    logger.debug(f"Copied: {old_path} -> {new_path}")
            except Exception as e:
                logger.error(f"Failed to migrate {old_path}: {e}")
    
    if not dry_run:
        logger.info(f"Migration complete for project '{voice_id}'")
    else:
        logger.info("Dry run complete - no files were moved")


if __name__ == "__main__":
    import sys
    from pathlib import Path
    
    if len(sys.argv) < 2:
        print("Usage: python migrate_structure.py <config_path> [--dry-run]")
        sys.exit(1)
    
    config_path = Path(sys.argv[1])
    dry_run = "--dry-run" in sys.argv
    
    config = VoiceConfig.from_yaml(config_path)
    migrate_project(config, dry_run=dry_run)

