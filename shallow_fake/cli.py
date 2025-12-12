"""Typer-based CLI for ShallowFaker pipeline."""

import sys
from pathlib import Path

import typer
from rich.console import Console

from shallow_fake.config import VoiceConfig
from shallow_fake.utils import setup_logging

# Add tools directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

app = typer.Typer(help="ShallowFaker - Voice Piper Clone Pipeline")
console = Console()
logger = setup_logging()


def load_config(config_path: Path) -> VoiceConfig:
    """Load and validate configuration."""
    # Convert to Path if it's a string
    if isinstance(config_path, str):
        config_path = Path(config_path)
    
    # If path is just a filename (no directory separators), assume it's in config/
    config_str = str(config_path)
    if "/" not in config_str and "\\" not in config_str:
        # Just a filename, prepend config/
        config_path = Path("config") / config_path
    
    # If no extension provided, add .yaml
    if not config_path.suffix:
        config_path = config_path.with_suffix(".yaml")
    
    # If still no directory separators after adding extension, prepend config/
    config_str = str(config_path)
    if "/" not in config_str and "\\" not in config_str:
        config_path = Path("config") / config_path
    
    if not config_path.exists():
        console.print(f"[red]Error: Config file not found: {config_path}[/red]")
        raise typer.Exit(1)

    try:
        config = VoiceConfig.from_yaml(config_path)
        config.ensure_directories()
        return config
    except Exception as e:
        console.print(f"[red]Error loading config: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def status(
    config: Path = typer.Option(None, "--config", "-c", help="Show status for specific project (optional)"),
):
    """Show pipeline status and progress for all projects (or a specific project if --config is provided)."""
    from shallow_fake.status import show_all_status, show_status
    
    try:
        if config is not None:
            # Show status for specific project
            cfg = load_config(config)
            show_status(cfg)
        else:
            # Show status for all projects
            show_all_status()
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def init(
    project_name: str = typer.Argument(..., help="Name of the project to initialize"),
    base_dir: Path = typer.Option(None, "--base-dir", "-d", help="Base directory (defaults to current directory)"),
):
    """Initialize a new project with directory structure and config file."""
    from shallow_fake.init import initialize_project

    try:
        initialize_project(project_name, base_dir)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def asr_segment(
    config: Path = typer.Option("voice1.yaml", "--config", "-c", help="Config file name (in config/ directory)"),
    cpu: bool = typer.Option(False, "--cpu", help="Force CPU mode (overrides config device setting)"),
):
    """Run ASR segmentation on raw audio files."""
    console.print("[bold blue]Running ASR segmentation...[/bold blue]")
    cfg = load_config(config)
    
    # Override device if --cpu flag is set
    if cpu:
        cfg.asr.device = "cpu"
        console.print("[yellow]Using CPU mode (--cpu flag overrides config)[/yellow]")

    from tools.asr_segment import process_audio_files

    try:
        process_audio_files(cfg)
        console.print("[green]ASR segmentation complete![/green]")
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def build_dataset(
    config: Path = typer.Option("voice1.yaml", "--config", "-c", help="Config file name (in config/ directory)"),
    min_words: int = typer.Option(3, "--min-words", help="Minimum words per segment"),
    max_words: int = typer.Option(50, "--max-words", help="Maximum words per segment"),
    max_segments: int = typer.Option(None, "--max-segments", help="Maximum segments to include"),
):
    """Build dataset metadata from ASR segments."""
    console.print("[bold blue]Building dataset...[/bold blue]")
    cfg = load_config(config)

    from tools.build_metadata import build_metadata

    try:
        build_metadata(cfg, min_words, max_words, max_segments)
        console.print("[green]Dataset built successfully![/green]")
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def verify(
    config: Path = typer.Option("voice1.yaml", "--config", "-c", help="Config file name (in config/ directory)"),
    baseline_model: Path = typer.Option(None, "--baseline-model", help="Path to baseline Piper model"),
    cpu: bool = typer.Option(False, "--cpu", help="Force CPU mode for Whisper transcription"),
):
    """Run phoneme-based verification on dataset."""
    console.print("[bold blue]Running phoneme verification...[/bold blue]")
    cfg = load_config(config)
    
    # Note: verify_phonemes uses CPU by default, but flag is available for consistency
    if cpu:
        console.print("[yellow]Using CPU mode for verification[/yellow]")

    from tools.verify_phonemes import verify_dataset

    try:
        verify_dataset(cfg, str(baseline_model) if baseline_model else None)
        console.print("[green]Phoneme verification complete![/green]")
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def build_synth(
    config: Path = typer.Option("voice1.yaml", "--config", "-c", help="Config file name (in config/ directory)"),
    cpu: bool = typer.Option(False, "--cpu", help="Force CPU mode for Whisper verification"),
):
    """Generate synthetic dataset from text corpus."""
    console.print("[bold blue]Building synthetic dataset...[/bold blue]")
    cfg = load_config(config)
    
    # Note: build_synthetic_dataset uses CPU by default for verification, but flag is available for consistency
    if cpu:
        console.print("[yellow]Using CPU mode for verification[/yellow]")

    # Check if teacher model is configured
    if cfg.synthetic.teacher and cfg.synthetic.teacher.kind == "xtts":
        console.print("[cyan]Teacher model service will be started automatically[/cyan]")

    from tools.build_synthetic_dataset import build_synthetic_dataset

    try:
        build_synthetic_dataset(cfg)
        console.print("[green]Synthetic dataset built successfully![/green]")
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def combine(
    config: Path = typer.Option("voice1.yaml", "--config", "-c", help="Config file name (in config/ directory)"),
    subsample_synth: float = typer.Option(None, "--subsample-synth", help="Subsample synthetic data by ratio (0.0-1.0)"),
):
    """Combine real and synthetic datasets."""
    console.print("[bold blue]Combining datasets...[/bold blue]")
    cfg = load_config(config)

    from tools.combine_datasets import combine_datasets

    try:
        combine_datasets(cfg, subsample_synth)
        console.print("[green]Datasets combined successfully![/green]")
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def train(
    config: Path = typer.Option("voice1.yaml", "--config", "-c", help="Config file name (in config/ directory)"),
    resume_from_version: str = typer.Option(None, "--resume-from-version", help="Resume training from a versioned checkpoint (e.g., '1')"),
):
    """Launch TMS training container."""
    console.print("[bold blue]Launching training...[/bold blue]")
    cfg = load_config(config)

    from tools.launch_training import launch_training

    try:
        launch_training(cfg, resume_from_version=resume_from_version)
        console.print("[green]Training container launched![/green]")
        # Show just the filename for monitor command
        config_name = config.name if config.parent.name == "config" else str(config)
        console.print(f"Monitor with: [cyan]shallow-fake monitor --config {config_name}[/cyan]")
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def monitor(
    config: Path = typer.Option("voice1.yaml", "--config", "-c", help="Config file name (in config/ directory)"),
    no_follow: bool = typer.Option(False, "--no-follow", help="Don't follow logs, just show recent"),
    tensorboard: bool = typer.Option(False, "--tensorboard", help="Show TensorBoard information"),
):
    """Monitor training progress."""
    cfg = load_config(config)

    from tools.monitor_training import monitor_training

    try:
        monitor_training(cfg, follow_logs=not no_follow, tensorboard=tensorboard)
    except KeyboardInterrupt:
        console.print("\n[yellow]Monitoring stopped[/yellow]")
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def save_checkpoint(
    config: Path = typer.Option("voice1.yaml", "--config", "-c", help="Config file name (in config/ directory)"),
    version: str = typer.Argument(..., help="Version number (e.g., '1', '2')"),
    checkpoint: Path = typer.Option(None, "--checkpoint", help="Path to checkpoint file (default: finds best checkpoint)"),
    overwrite: bool = typer.Option(False, "--overwrite", help="Overwrite existing versioned checkpoint (v1 is always protected)"),
):
    """Save a checkpoint with a version number for future training resumes."""
    console.print(f"[bold blue]Saving checkpoint as version {version}...[/bold blue]")
    cfg = load_config(config)

    from tools.save_checkpoint_version import save_checkpoint_version

    try:
        save_checkpoint_version(cfg, version, checkpoint, overwrite)
        console.print(f"[green]Checkpoint saved as v{version}![/green]")
        console.print(f"Resume training with: [cyan]shallow-fake train --config {config.name if config.parent.name == 'config' else str(config)} --resume-from-version {version}[/cyan]")
    except FileExistsError as e:
        console.print(f"[yellow]{e}[/yellow]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def export(
    config: Path = typer.Option("voice1.yaml", "--config", "-c", help="Config file name (in config/ directory)"),
    checkpoint: Path = typer.Option(None, "--checkpoint", help="Path to checkpoint file"),
    version: str = typer.Option(None, "--version", help="Version number for the exported model (e.g., '1', '2'). v1 is protected from overwrite."),
):
    """Export trained model to ONNX format."""
    console.print("[bold blue]Exporting ONNX model...[/bold blue]")
    cfg = load_config(config)

    from tools.export_onnx import export_onnx

    try:
        export_onnx(cfg, checkpoint, version)
        console.print("[green]ONNX export complete![/green]")
    except FileExistsError as e:
        console.print(f"[yellow]{e}[/yellow]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def eval(
    config: Path = typer.Option("voice1.yaml", "--config", "-c", help="Config file name (in config/ directory)"),
    phrases_file: Path = typer.Option(None, "--phrases-file", help="File with phrases to generate (one per line)"),
    model_name: str = typer.Option(None, "--model-name", help="Model name (default: <language>_<REGION>-<voice_id>-<quality>)"),
):
    """Generate evaluation samples from exported model."""
    console.print("[bold blue]Generating evaluation samples...[/bold blue]")
    cfg = load_config(config)

    phrases = None
    if phrases_file and phrases_file.exists():
        with open(phrases_file, "r", encoding="utf-8") as f:
            phrases = [line.strip() for line in f if line.strip()]

    from tools.eval_model import eval_model

    try:
        eval_model(cfg, phrases, model_name)
        console.print("[green]Evaluation samples generated![/green]")
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def cleanup_docker(
    all_resources: bool = typer.Option(False, "--all", "-a", help="Also remove unused images, containers, and volumes"),
):
    """Clean up Docker build cache and unused resources to free disk space."""
    import subprocess
    
    console.print("[bold blue]Cleaning up Docker resources...[/bold blue]")
    
    try:
        # Always clean build cache
        console.print("[cyan]Cleaning Docker build cache...[/cyan]")
        result = subprocess.run(
            ["docker", "builder", "prune", "-f"],
            capture_output=True,
            text=True,
            check=True,
        )
        if result.stdout:
            console.print(result.stdout)
        
        if all_resources:
            # Also clean unused images, containers, networks, and volumes
            console.print("[cyan]Cleaning unused Docker resources (images, containers, volumes)...[/cyan]")
            result = subprocess.run(
                ["docker", "system", "prune", "-af", "--volumes"],
                capture_output=True,
                text=True,
                check=True,
            )
            if result.stdout:
                console.print(result.stdout)
        else:
            console.print("[yellow]Use --all to also remove unused images, containers, and volumes[/yellow]")
        
        console.print("[green]Docker cleanup complete![/green]")
    except subprocess.CalledProcessError as e:
        console.print(f"[red]Error cleaning up Docker: {e.stderr}[/red]")
        raise typer.Exit(1)
    except FileNotFoundError:
        console.print("[red]Error: Docker command not found. Please install Docker.[/red]")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()

