"""Monitor training progress with rich terminal UI."""

import re
import subprocess
from collections import deque
from pathlib import Path
from typing import Optional, Deque

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.text import Text
from rich.layout import Layout
from rich.align import Align

from shallow_fake.config import VoiceConfig
from shallow_fake.utils import setup_logging

logger = setup_logging()
console = Console()


def is_progress_line(line: str) -> bool:
    """Detect if line is a progress bar that should be overwritten."""
    # PyTorch Lightning progress bars contain patterns like:
    # - "Epoch X: Y%|..." 
    # - Progress bar characters (█, ▓, etc.)
    # - Contains percentage and "it/s"
    progress_patterns = [
        r'Epoch\s+\d+:\s+\d+%',  # "Epoch 78: 17%"
        r'\|\s*\d+/\d+',          # "| 31/186"
        r'\d+\.\d+it/s',          # "3.75it/s"
        r'loss=[\d.]+',           # "loss=16.8"
    ]
    return any(re.search(pattern, line) for pattern in progress_patterns)


def is_error_or_warning(line: str) -> bool:
    """Detect errors and warnings that should be preserved."""
    line_lower = line.lower()
    return any(keyword in line_lower for keyword in [
        'error', 'warning', 'exception', 'traceback', 
        'failed', 'critical', 'fatal', 'warn'
    ])


def parse_training_metrics(line: str) -> Optional[dict]:
    """Parse training metrics from log line."""
    metrics = {}
    
    # Look for patterns like "loss: 0.123" or "loss=16.8" or "epoch: 10"
    loss_match = re.search(r"loss[:\s=]+([\d.]+)", line, re.IGNORECASE)
    epoch_match = re.search(r"epoch[:\s]+(\d+)", line, re.IGNORECASE)
    step_match = re.search(r"step[:\s=]+(\d+)", line, re.IGNORECASE)
    
    # Extract percentage from progress bar
    percent_match = re.search(r"(\d+)%", line)
    
    if loss_match:
        metrics['loss'] = float(loss_match.group(1))
    if epoch_match:
        metrics['epoch'] = int(epoch_match.group(1))
    if step_match:
        metrics['step'] = int(step_match.group(1))
    if percent_match:
        metrics['percent'] = int(percent_match.group(1))
    
    return metrics if metrics else None


def create_display_layout(status_text: str, error_log: Deque[str]) -> Layout:
    """Create a rich layout with status and error panels."""
    layout = Layout()
    
    # Split into two sections: status (top) and errors (bottom)
    layout.split_column(
        Layout(name="status", size=3),
        Layout(name="errors", ratio=1),
    )
    
    # Status panel
    status_panel = Panel(
        Align.left(Text(status_text, style="bold green")),
        title="[bold cyan]Training Status[/bold cyan]",
        border_style="green",
    )
    layout["status"].update(status_panel)
    
    # Error log panel
    if error_log:
        error_content = "\n".join(error_log)
        error_panel = Panel(
            Text(error_content, style="yellow"),
            title=f"[bold yellow]Recent Errors/Warnings ({len(error_log)})[/bold yellow]",
            border_style="yellow",
        )
    else:
        error_panel = Panel(
            Text("No errors or warnings", style="dim"),
            title="[bold]Recent Errors/Warnings[/bold]",
            border_style="dim",
        )
    layout["errors"].update(error_panel)
    
    return layout


def tail_docker_logs(container_name: str, follow: bool = True):
    """Tail logs from Docker container with rich display."""
    cmd = ["docker", "logs"]
    if follow:
        cmd.append("-f")
    cmd.append(container_name)

    status_text = "Waiting for training to start..."
    error_log: Deque[str] = deque(maxlen=10)  # Keep last 10 errors/warnings
    last_metrics = {}

    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        with Live(
            create_display_layout(status_text, error_log),
            refresh_per_second=4,
            screen=True,  # Clear screen for clean overwriting of progress lines
        ) as live:
            for line in process.stdout:
                line = line.rstrip('\n\r')
                
                # Skip empty lines
                if not line.strip():
                    continue
                
                # Parse metrics
                metrics = parse_training_metrics(line)
                if metrics:
                    last_metrics.update(metrics)
                
                if is_error_or_warning(line):
                    # Add to error log
                    error_log.append(line)
                    # Also print to console for immediate visibility
                    console.print(f"[yellow]⚠️  {line}[/yellow]")
                elif is_progress_line(line):
                    # Update status with progress line
                    status_text = line
                elif "INFO" in line and "Epoch:" in line:
                    # Skip the duplicate epoch logging from parse_training_metrics
                    continue
                else:
                    # Regular informational lines - show in status if interesting
                    if any(keyword in line.lower() for keyword in [
                        'checkpoint', 'saved', 'completed', 'starting', 'preprocessing'
                    ]):
                        status_text = line
                
                # Build enhanced status with metrics
                enhanced_status = status_text
                if last_metrics:
                    metric_parts = []
                    if 'epoch' in last_metrics:
                        metric_parts.append(f"Epoch: {last_metrics['epoch']}")
                    if 'loss' in last_metrics:
                        metric_parts.append(f"Loss: {last_metrics['loss']:.4f}")
                    if 'percent' in last_metrics:
                        metric_parts.append(f"Progress: {last_metrics['percent']}%")
                    if metric_parts:
                        enhanced_status = f"{status_text} | {' | '.join(metric_parts)}"
                
                # Update display
                live.update(create_display_layout(enhanced_status, error_log))

        process.wait()
    except KeyboardInterrupt:
        console.print("\n[yellow]Stopping log monitoring...[/yellow]")
        if 'process' in locals():
            process.terminate()
    except FileNotFoundError:
        console.print("[red]Error: docker command not found[/red]")
    except Exception as e:
        console.print(f"[red]Error monitoring logs: {e}[/red]")


def check_tensorboard(logs_dir: Path, port: int = 6006):
    """Check if TensorBoard logs are available."""
    # logs_dir is already the lightning_logs directory
    if logs_dir.exists():
        logger.info(f"TensorBoard logs found: {logs_dir}")
        logger.info(f"Start TensorBoard with: tensorboard --logdir {logs_dir} --port {port}")
        return True
    return False


def monitor_training(config: VoiceConfig, follow_logs: bool = True, tensorboard: bool = False):
    """Monitor training progress."""
    container_name = f"tms-{config.voice_id}-trainer"
    # Training logs are in the prepared dataset directory
    logs_dir = config.paths.prepared_dataset_dir / "lightning_logs"
    audio_samples_dir = config.paths.training_samples_dir

    logger.info(f"Monitoring training container: {container_name}")
    logger.info(f"Logs directory: {logs_dir}")
    logger.info(f"Audio samples: {audio_samples_dir}")

    # Check TensorBoard
    if tensorboard and config.tms.expose_tensorboard:
        check_tensorboard(logs_dir, config.tms.tensorboard_port)

    # Monitor logs
    if follow_logs:
        tail_docker_logs(container_name, follow=True)
    else:
        # Just show recent logs
        tail_docker_logs(container_name, follow=False)


if __name__ == "__main__":
    import sys
    from pathlib import Path

    if len(sys.argv) < 2:
        print("Usage: python monitor_training.py <config_path> [--no-follow] [--tensorboard]")
        sys.exit(1)

    config_path = Path(sys.argv[1])
    config = VoiceConfig.from_yaml(config_path)

    follow = True
    tensorboard = False

    args = sys.argv[2:]
    if "--no-follow" in args:
        follow = False
    if "--tensorboard" in args:
        tensorboard = True

    monitor_training(config, follow_logs=follow, tensorboard=tensorboard)






