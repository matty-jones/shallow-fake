"""Status checking for pipeline progress."""

from pathlib import Path
from typing import Dict, List, Tuple, Optional

from rich.console import Console
from rich.table import Table

from shallow_fake.config import VoiceConfig
from shallow_fake.utils import setup_logging

console = Console()
logger = setup_logging()


def discover_projects(base_dir: Path = None) -> List[Tuple[str, Path]]:
    """
    Discover all projects by scanning config files and directory structure.
    
    Returns a list of (project_name, config_path) tuples.
    """
    if base_dir is None:
        base_dir = Path.cwd()
    
    projects = []
    
    # Method 1: Find all config files
    config_dir = base_dir / "config"
    if config_dir.exists():
        for config_file in config_dir.glob("*.yaml"):
            try:
                config = VoiceConfig.from_yaml(config_file)
                projects.append((config.voice_id, config_file))
            except Exception as e:
                logger.debug(f"Failed to load config {config_file}: {e}")
                continue
    
    # Method 2: Find projects from workspace directory structure (new structure)
    workspace_dir = base_dir / "workspace"
    if workspace_dir.exists():
        for project_dir in workspace_dir.iterdir():
            if project_dir.is_dir():
                project_name = project_dir.name
                # Check if we already found this project via config
                if not any(name == project_name for name, _ in projects):
                    # Look for a matching config file
                    config_file = config_dir / f"{project_name}.yaml"
                    if config_file.exists():
                        try:
                            config = VoiceConfig.from_yaml(config_file)
                            projects.append((config.voice_id, config_file))
                        except Exception:
                            # If config doesn't load, still add it as discovered
                            projects.append((project_name, config_file))
                    else:
                        # Project exists but no config - still report it
                        projects.append((project_name, None))
    
    # Method 3: Fallback - Find projects from old datasets directory structure (for migration)
    datasets_dir = base_dir / "datasets"
    if datasets_dir.exists():
        for project_dir in datasets_dir.iterdir():
            if project_dir.is_dir():
                project_name = project_dir.name
                # Check if we already found this project
                if not any(name == project_name for name, _ in projects):
                    # Look for a matching config file
                    config_file = config_dir / f"{project_name}.yaml"
                    if config_file.exists():
                        try:
                            config = VoiceConfig.from_yaml(config_file)
                            projects.append((config.voice_id, config_file))
                        except Exception:
                            projects.append((project_name, config_file))
                    else:
                        projects.append((project_name, None))
    
    # Remove duplicates (keep first occurrence)
    seen = set()
    unique_projects = []
    for name, path in projects:
        if name not in seen:
            seen.add(name)
            unique_projects.append((name, path))
    
    return unique_projects


def check_stage_status(config: VoiceConfig) -> Dict[str, Tuple[bool, str]]:
    """
    Check the status of each pipeline stage.
    
    Returns a dict mapping stage names to (completed, description) tuples.
    """
    status = {}
    
    # Stage 1: Initialization
    workspace_exists = config.paths.workspace_dir.exists()
    status["init"] = (
        workspace_exists,
        "Project initialized" if workspace_exists else "Project not initialized"
    )
    
    # Stage 2: ASR Segmentation
    asr_metadata = config.paths.asr_metadata
    segments_dir = config.paths.segments_dir
    has_segments = asr_metadata.exists() and segments_dir.exists()
    segment_count = len(list(segments_dir.glob("*.wav"))) if segments_dir.exists() else 0
    status["asr-segment"] = (
        has_segments and segment_count > 0,
        f"ASR segmentation complete ({segment_count} segments)" if has_segments and segment_count > 0
        else "ASR segmentation not run"
    )
    
    # Stage 3: Build Dataset
    real_metadata = config.paths.real_dataset_dir / "metadata.csv"
    real_wavs = config.paths.real_dataset_dir / "wavs"
    has_real_dataset = real_metadata.exists() and real_wavs.exists()
    real_count = len(list(real_wavs.glob("*.wav"))) if real_wavs.exists() else 0
    status["build-dataset"] = (
        has_real_dataset and real_count > 0,
        f"Real dataset built ({real_count} entries)" if has_real_dataset and real_count > 0
        else "Real dataset not built"
    )
    
    # Stage 4: Verify (now in-place, check if real dataset has been verified)
    real_metadata = config.paths.real_dataset_dir / "metadata.csv"
    real_wavs = config.paths.real_dataset_dir / "wavs"
    has_real = real_metadata.exists() and real_wavs.exists()
    real_count = len(list(real_wavs.glob("*.wav"))) if real_wavs.exists() else 0
    # Verification is now in-place, so if real dataset exists, it's been verified
    status["verify"] = (
        has_real and real_count > 0,
        f"Phoneme verification complete ({real_count} valid entries)" if has_real and real_count > 0
        else "Phoneme verification not run"
    )
    
    # Stage 5: Build Synthetic (verification is now in-place)
    synth_metadata = config.paths.synth_dataset_dir / "metadata.csv"
    synth_wavs = config.paths.synth_dataset_dir / "wavs"
    has_synth = synth_metadata.exists() and synth_wavs.exists()
    synth_count = len(list(synth_wavs.glob("*.wav"))) if synth_wavs.exists() else 0
    status["build-synth"] = (
        has_synth and synth_count > 0,
        f"Synthetic dataset built ({synth_count} entries)" if has_synth and synth_count > 0
        else "Synthetic dataset not built"
    )
    
    # Stage 6: Combine
    combined_metadata = config.paths.combined_dataset_dir / "metadata.csv"
    combined_wavs = config.paths.combined_dataset_dir / "wavs"
    has_combined = combined_metadata.exists() and combined_wavs.exists()
    combined_count = len(list(combined_wavs.glob("*.wav"))) if combined_wavs.exists() else 0
    status["combine"] = (
        has_combined and combined_count > 0,
        f"Datasets combined ({combined_count} total entries)" if has_combined and combined_count > 0
        else "Datasets not combined"
    )
    
    # Stage 7: Train
    # Check for training output checkpoints (not base checkpoints)
    # Training checkpoints are saved to: workspace/{voice_id}/datasets/prepared/lightning_logs/version_X/checkpoints/
    training_checkpoints_dir = config.paths.prepared_dataset_dir / "lightning_logs"
    has_training_checkpoints = False
    checkpoint_count = 0
    checkpoint_info = "Training not started"
    
    # Look for training output checkpoints in the correct location
    if training_checkpoints_dir.exists():
        version_dirs = [d for d in training_checkpoints_dir.iterdir() if d.is_dir() and d.name.startswith("version_")]
        if version_dirs:
            latest_version = max(version_dirs, key=lambda d: int(d.name.split("_")[1]) if d.name.split("_")[1].isdigit() else 0)
            version_checkpoints = latest_version / "checkpoints"
            if version_checkpoints.exists():
                checkpoints = list(version_checkpoints.glob("*.ckpt"))
                if checkpoints:
                    has_training_checkpoints = True
                    checkpoint_count = len(checkpoints)
                    checkpoint_info = f"Training completed ({checkpoint_count} checkpoint(s) found)"
    
    # Check if container is running
    import subprocess
    container_name = f"tms-{config.voice_id}-trainer"
    container_running = False
    try:
        result = subprocess.run(
            ["docker", "ps", "--filter", f"name={container_name}", "--format", "{{.Names}}"],
            capture_output=True,
            text=True,
            timeout=2,
        )
        container_running = container_name in result.stdout
    except (FileNotFoundError, subprocess.TimeoutExpired):
        container_running = False
    
    # Determine training status:
    # - If container is running: "In Progress" (not complete)
    # - If container not running but checkpoints exist: "Complete"
    # - If container not running and no checkpoints: "Not Started" or "Pending"
    if container_running:
        status["train"] = (False, "Training in progress (container running)")
    elif has_training_checkpoints:
        status["train"] = (True, checkpoint_info)
    else:
        status["train"] = (False, checkpoint_info)
    
    # Stage 8: Export
    models_dir = config.paths.output_models_dir
    onnx_files = list(models_dir.glob("*.onnx")) if models_dir.exists() else []
    status["export"] = (
        len(onnx_files) > 0,
        f"Model exported ({len(onnx_files)} ONNX file(s))" if len(onnx_files) > 0
        else "Model not exported"
    )
    
    # Stage 9: Eval
    samples_dir = config.paths.training_samples_dir
    sample_files = list(samples_dir.glob("*.wav")) if samples_dir.exists() else []
    status["eval"] = (
        len(sample_files) > 0,
        f"Evaluation samples generated ({len(sample_files)} samples)" if len(sample_files) > 0
        else "Evaluation samples not generated"
    )
    
    return status


def get_next_step(status: Dict[str, Tuple[bool, str]]) -> str:
    """Determine the next step in the pipeline."""
    stages = [
        "init",
        "asr-segment",
        "build-dataset",
        "verify",
        "build-synth",
        "combine",
        "train",
        "export",
        "eval",
    ]
    
    for stage in stages:
        if not status[stage][0]:
            return stage
    
    return "complete"


def show_status(config: VoiceConfig, project_name: str = None):
    """Display pipeline status for a single project."""
    status = check_stage_status(config)
    next_step = get_next_step(status)
    
    project_display = project_name or config.voice_id
    
    # Create status table
    table = Table(title=f"Pipeline Status: {project_display}", show_header=True, header_style="bold blue")
    table.add_column("Stage", style="cyan", no_wrap=True)
    table.add_column("Status", style="magenta")
    table.add_column("Details", style="white")
    table.add_column("Command", style="dim", no_wrap=True)
    
    stages = [
        ("init", "Initialize", "init"),
        ("asr-segment", "ASR Segment", "asr-segment"),
        ("build-dataset", "Build Dataset", "build-dataset"),
        ("verify", "Verify", "verify"),
        ("build-synth", "Build Synthetic", "build-synth"),
        ("combine", "Combine", "combine"),
        ("train", "Train", "train"),
        ("export", "Export", "export"),
        ("eval", "Evaluate", "eval"),
    ]
    
    for stage_key, stage_name, command in stages:
        completed, details = status[stage_key]
        status_icon = "✓" if completed else "○"
        
        # Special handling for "in progress" status
        if "in progress" in details.lower() or "container running" in details.lower():
            status_text = "[bold cyan]In Progress[/bold cyan]"
            status_icon = "⟳"
        elif completed:
            status_text = "[green]Complete[/green]"
        else:
            status_text = "[yellow]Pending[/yellow]"
        
        # Highlight next step
        if stage_key == next_step and not completed:
            status_text = "[bold yellow]Next Step[/bold yellow]"
            status_icon = "→"
        
        table.add_row(
            f"{status_icon} {stage_name}",
            status_text,
            details,
            command
        )
    
    console.print(table)
    console.print()
    
    if next_step == "complete":
        console.print(f"[green]✓ Pipeline complete for {project_display}! All stages finished.[/green]")
    else:
        config_name = f"{config.voice_id}.yaml" if project_name else "config.yaml"
        console.print(f"[bold yellow]Next step:[/bold yellow] [cyan]shallow-fake {next_step} --config {config_name}[/cyan]")


def show_all_status(base_dir: Path = None):
    """Display pipeline status for all discovered projects."""
    projects = discover_projects(base_dir)
    
    if not projects:
        console.print("[yellow]No projects found. Run 'shallow-fake init <project_name>' to create a new project.[/yellow]")
        return
    
    console.print(f"[bold blue]Found {len(projects)} project(s):[/bold blue]\n")
    
    for project_name, config_path in projects:
        if config_path is None or not config_path.exists():
            # Project exists but no valid config
            console.print(f"[yellow]Project '{project_name}': Config file not found or invalid[/yellow]")
            console.print()
            continue
        
        try:
            config = VoiceConfig.from_yaml(config_path)
            show_status(config, project_name)
            console.print()  # Add spacing between projects
        except Exception as e:
            console.print(f"[red]Error loading config for '{project_name}': {e}[/red]")
            console.print()
            continue

