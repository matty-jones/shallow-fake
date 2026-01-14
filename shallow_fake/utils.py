"""Utility functions for the pipeline."""

import logging
from pathlib import Path

from rich.console import Console
from rich.logging import RichHandler

console = Console()


def setup_logging(level: str = "INFO") -> logging.Logger:
    """Set up rich logging."""
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True)],
    )
    return logging.getLogger("shallow_fake")


def ensure_dir(path: Path):
    """Ensure a directory exists, creating it if necessary."""
    path.mkdir(parents=True, exist_ok=True)
    return path









































