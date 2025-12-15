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

    # Create new unified directory structure
    # Note: synth dataset directories are created on-demand based on teacher kind
    # (synth-xtts or synth-metavoice) when build-synth runs
    directories = [
        base_dir / "input" / project_name / "audio",
        base_dir / "workspace" / project_name / "segments",
        base_dir / "workspace" / project_name / "datasets" / "real" / "wavs",
        base_dir / "workspace" / project_name / "datasets" / "combined" / "wavs",
        base_dir / "workspace" / project_name / "datasets" / "prepared",
        base_dir / "workspace" / project_name / "training" / "checkpoints",
        base_dir / "workspace" / project_name / "training" / "samples",
        base_dir / "models" / project_name,
        base_dir / "models" / "shared" / "base_checkpoints",
        base_dir / "models" / "shared" / "xtts_baseline",
        base_dir / "input" / "shared",
    ]

    for directory in directories:
        ensure_dir(directory)
        console.print(f"  Created: {directory}")

    # Create teacher model baseline directory (shared across all projects)
    xtts_baseline_dir = base_dir / "models" / "shared" / "xtts_baseline"
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
            "input_audio_dir": f"input/{project_name}/audio",
            "workspace_dir": f"workspace/{project_name}",
            "models_dir": f"models/{project_name}",
            "shared_models_dir": "models/shared",
            "corpus_path": "input/shared/corpus.txt",
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
            "max_phoneme_distance": 0.1,
            "use_tts_roundtrip": True,
            "parallel_workers": 4,
        },
        "synthetic": {
            "enabled": True,
            "corpus_text_path": "input/shared/corpus.txt",
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
                "reference_audio_dir": f"workspace/{project_name}/datasets/real/wavs",
                "num_reference_clips": 3,
                "workers": 3,
            },
            # Alternative: Use MetaVoice-1B as teacher model
            # "teacher": {
            #     "kind": "metavoice",
            #     "base_url": "http://localhost:58003",
            #     "huggingface_repo_id": "metavoiceio/metavoice-1B-v0.1",
            #     "speaker_ref_path": "/speakers/voice_ref.wav",
            #     "guidance": 3.0,
            #     "top_p": 0.95,
            #     "top_k": 200,
            #     "port": 58003,
            #     "reference_audio_dir": f"workspace/{project_name}/datasets/real/wavs",
            # },
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

    # Create placeholder corpus file (shared across all projects)
    corpus_file = base_dir / "input" / "shared" / "corpus.txt"
    if not corpus_file.exists():
        corpus_file.write_text(
            "# Add your text corpus here, one sentence per line.\n"
            "# This will be used for synthetic data generation.\n"
            "# This corpus is shared across all projects.\n"
        )
        console.print(f"  Created: {corpus_file}")
    
    # Create placeholder evaluation file (shared across all projects)
    evaluation_file = base_dir / "input" / "shared" / "evaluation.txt"
    if not evaluation_file.exists():
        evaluation_file.write_text(
            "# Add your evaluation phrases here, one per line.\n"
            "# These phrases will be used to generate audio samples for model evaluation.\n"
            "# This file is shared across all projects.\n"
            "# Lines starting with # are ignored.\n"
            "\n"
            "# Example phrases:\n"
            "# This is a test of the voice synthesis system.\n"
            "# The quick brown fox jumps over the lazy dog.\n"
            "# Hello, this is my custom voice model speaking.\n"
        )
        console.print(f"  Created: {evaluation_file}")

    console.print(f"\n[green]Project '{project_name}' initialized successfully![/green]")
    console.print(f"\nNext steps:")
    console.print(f"  1. Place your raw audio files in: [cyan]input/{project_name}/audio/[/cyan]")
    console.print(f"  2. (Optional) Add text corpus to: [cyan]input/shared/corpus.txt[/cyan]")
    console.print(f"  3. Review and adjust: [cyan]config/{project_name}.yaml[/cyan]")
    console.print(f"  4. Run: [cyan]shallow-fake asr-segment --config {project_name}.yaml[/cyan]")

