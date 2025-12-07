"""Initialize a new project with directory structure and config file."""

from pathlib import Path

import yaml
from rich.console import Console

from shallow_fake.utils import ensure_dir, setup_logging

console = Console()
logger = setup_logging()


def initialize_project(project_name: str, base_dir: Path = None):
    """
    Initialize a new project with directory structure and config file.

    Args:
        project_name: Name of the project (will be used as voice_id)
        base_dir: Base directory for the project (defaults to current directory)
    """
    if base_dir is None:
        base_dir = Path.cwd()

    # Validate project name (basic validation)
    if not project_name or not project_name.replace("_", "").replace("-", "").isalnum():
        raise ValueError(
            f"Invalid project name: {project_name}. "
            "Project name should contain only alphanumeric characters, hyphens, and underscores."
        )

    console.print(f"[bold blue]Initializing project: {project_name}[/bold blue]")

    # Create directory structure (organized by project)
    directories = [
        base_dir / "data_raw" / project_name / "input_audio",
        base_dir / "data_processed" / project_name / "normalized",
        base_dir / "data_processed" / project_name / "segments",
        base_dir / "datasets" / project_name / "real" / "wavs",
        base_dir / "datasets" / project_name / "synth" / "wavs",
        base_dir / "datasets" / project_name / "combined" / "wavs",
        base_dir / "tms_workspace" / "datasets",
        base_dir / "tms_workspace" / "checkpoints" / "base_checkpoints",
        base_dir / "tms_workspace" / "logs",
        base_dir / "tms_workspace" / "audio_samples",
        base_dir / "models" / project_name,
        base_dir / "samples" / project_name,
    ]

    for directory in directories:
        ensure_dir(directory)
        console.print(f"  Created: {directory}")

    # Create project-agnostic external corpus directory (shared across all projects)
    external_corpus_dir = base_dir / "data_raw" / "external_corpus"
    was_new = not external_corpus_dir.exists()
    ensure_dir(external_corpus_dir)
    if was_new:
        console.print(f"  Created: {external_corpus_dir}")

    # Create teacher model baseline directory (universal across all projects)
    # Note: Directory name remains xtts_baseline for compatibility with existing setups
    xtts_baseline_dir = base_dir / "models" / "xtts_baseline"
    was_new_xtts = not xtts_baseline_dir.exists()
    ensure_dir(xtts_baseline_dir)
    if was_new_xtts:
        console.print(f"  Created: {xtts_baseline_dir}")

    # Create config file
    config_dir = base_dir / "config"
    ensure_dir(config_dir)
    config_file = config_dir / f"{project_name}.yaml"

    config_data = {
        "voice_id": project_name,
        "language": "en_GB",
        "paths": {
            "raw_audio_dir": f"data_raw/{project_name}/input_audio",
            "normalized_dir": f"data_processed/{project_name}/normalized",
            "segments_dir": f"data_processed/{project_name}/segments",
            "asr_metadata": f"data_processed/{project_name}/asr_segments.jsonl",
            "real_dataset_dir": f"datasets/{project_name}/real",
            "synth_dataset_dir": f"datasets/{project_name}/synth",
            "combined_dataset_dir": f"datasets/{project_name}/combined",
            "tms_workspace_dir": "tms_workspace",
            "output_models_dir": f"models/{project_name}",
        },
        "asr": {
            "model_size": "medium.en",
            "device": "cuda",
            "beam_size": 5,
            "max_segment_seconds": 15,
            "min_segment_seconds": 1.0,
            "min_confidence": 0.7,
        },
        "phoneme_check": {
            "language": "en-gb",
            "max_phoneme_distance": 0.1,
            "use_tts_roundtrip": True,
            "parallel_workers": 4,
        },
        "synthetic": {
            "enabled": True,
            "corpus_text_path": "data_raw/external_corpus/corpus.txt",
            "max_sentences": 2000,
            "tts_backend": "http",
            "tts_http": {
                "base_url": "http://localhost:9010/tts",
                "voice_id": f"{project_name}_clone",
            },
            "teacher": {
                "kind": "xtts",
                "port": 9010,
                "model_name": "tts_models/multilingual/multi-dataset/xtts_v2",
                "language": "en",
                "device": "cuda",
                "reference_audio_dir": f"datasets/{project_name}/real_clean/wavs",
                "num_reference_clips": 3,
                "workers": 3,
            },
            "max_parallel_jobs": 4,
        },
        "training": {
            "base_checkpoint": "en_GB-base-medium.ckpt",
            "batch_size": 32,
            "max_epochs": 1000,
            "quality": "medium",
            "accelerator": "gpu",
            "devices": 1,
        },
        "tms": {
            "enable_tts_dojo": True,
            "docker_compose_file": "docker/docker-compose.training.yml",
            "project_name": f"{project_name}-voice",
            "expose_tensorboard": True,
            "tensorboard_port": 6006,
        },
    }

    with open(config_file, "w", encoding="utf-8") as f:
        yaml.dump(config_data, f, default_flow_style=False, sort_keys=False, indent=2)

    console.print(f"  Created: {config_file}")

    # Create placeholder corpus file (project-agnostic, shared across all projects)
    corpus_file = base_dir / "data_raw" / "external_corpus" / "corpus.txt"
    if not corpus_file.exists():
        corpus_file.write_text(
            "# Add your text corpus here, one sentence per line.\n"
            "# This will be used for synthetic data generation.\n"
            "# This corpus is shared across all projects.\n"
        )
        console.print(f"  Created: {corpus_file}")

    console.print(f"\n[green]Project '{project_name}' initialized successfully![/green]")
    console.print(f"\nNext steps:")
    console.print(f"  1. Place your raw audio files in: [cyan]data_raw/{project_name}/input_audio/[/cyan]")
    console.print(f"  2. (Optional) Add text corpus to: [cyan]data_raw/external_corpus/corpus.txt[/cyan]")
    console.print(f"  3. Review and adjust: [cyan]config/{project_name}.yaml[/cyan]")
    console.print(f"  4. Run: [cyan]shallow-fake asr-segment --config {project_name}.yaml[/cyan]")

