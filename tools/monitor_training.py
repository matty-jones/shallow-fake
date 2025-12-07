"""Monitor training progress."""

import re
import subprocess
import time
from pathlib import Path
from typing import Optional

from shallow_fake.config import VoiceConfig
from shallow_fake.utils import setup_logging

logger = setup_logging()


def tail_docker_logs(container_name: str, follow: bool = True):
    """Tail logs from Docker container."""
    cmd = ["docker", "logs"]
    if follow:
        cmd.append("-f")
    cmd.append(container_name)

    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        for line in process.stdout:
            print(line, end="")
            # Parse loss if present
            parse_training_metrics(line)

        process.wait()
    except KeyboardInterrupt:
        logger.info("Stopping log monitoring...")
        process.terminate()
    except FileNotFoundError:
        logger.error("docker command not found")
    except Exception as e:
        logger.error(f"Error monitoring logs: {e}")


def parse_training_metrics(line: str):
    """Parse training metrics from log line."""
    # Look for patterns like "loss: 0.123" or "epoch: 10"
    loss_match = re.search(r"loss[:\s]+([\d.]+)", line, re.IGNORECASE)
    epoch_match = re.search(r"epoch[:\s]+(\d+)", line, re.IGNORECASE)
    step_match = re.search(r"step[:\s]+(\d+)", line, re.IGNORECASE)

    if loss_match:
        loss = float(loss_match.group(1))
        logger.info(f"Loss: {loss:.4f}")

    if epoch_match:
        epoch = int(epoch_match.group(1))
        logger.info(f"Epoch: {epoch}")


def check_tensorboard(logs_dir: Path, port: int = 6006):
    """Check if TensorBoard logs are available."""
    lightning_logs = logs_dir / "lightning_logs"
    if lightning_logs.exists():
        logger.info(f"TensorBoard logs found: {lightning_logs}")
        logger.info(f"Start TensorBoard with: tensorboard --logdir {lightning_logs} --port {port}")
        return True
    return False


def monitor_training(config: VoiceConfig, follow_logs: bool = True, tensorboard: bool = False):
    """Monitor training progress."""
    container_name = f"tms-{config.voice_id}-trainer"
    logs_dir = config.paths.tms_workspace_dir / "logs"
    audio_samples_dir = config.paths.tms_workspace_dir / "audio_samples"

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




