"""Download XTTS model to local cache directory."""

import os
import sys
from pathlib import Path

from TTS.api import TTS

from shallow_fake.utils import ensure_dir, setup_logging

logger = setup_logging()


def download_xtts_model(
    model_name: str = "tts_models/multilingual/multi-dataset/xtts_v2",
    cache_dir: Path = None,
    device: str = "cpu",  # Use CPU for download to avoid GPU requirements
) -> Path:
    """
    Download XTTS model to local cache directory.
    
    Args:
        model_name: Name of the TTS model to download
        cache_dir: Directory to cache the model (default: models/xtts_baseline)
        device: Device to use for download (default: cpu)
    
    Returns:
        Path to the cache directory
    """
    if cache_dir is None:
        project_root = Path.cwd()
        cache_dir = project_root / "models" / "xtts_baseline"
    
    cache_dir = Path(cache_dir)
    ensure_dir(cache_dir)
    
    # TTS library uses ~/.local/share/tts by default
    # We'll set TTS_HOME to point to a subdirectory, then mount that to the container
    # The actual cache will be at cache_dir/tts (TTS creates a 'tts' subdir when TTS_HOME is set)
    tts_cache = cache_dir / "tts"
    ensure_dir(tts_cache)
    
    # Set TTS_HOME to the cache directory (TTS will create 'tts' subdirectory)
    os.environ["TTS_HOME"] = str(cache_dir)
    
    # Create TOS acceptance file in the tts cache directory
    tos_file = tts_cache / ".coqui_tos"
    if not tos_file.exists():
        tos_file.write_text("accepted\n")
        logger.info(f"Created TOS acceptance file: {tos_file}")
    
    # Check if model already exists
    # TTS creates structure: TTS_HOME/tts/tts_models--multilingual--multi-dataset--xtts_v2/
    model_path = tts_cache / "tts_models--multilingual--multi-dataset--xtts_v2"
    if model_path.exists() and any(model_path.iterdir()):
        # Check for key model files
        required_files = ["model.pth", "config.json"]
        if all((model_path / f).exists() for f in required_files):
            logger.info(f"Model already cached at {model_path}, skipping download")
            return cache_dir
        else:
            logger.info(f"Model directory exists but incomplete, will re-download")
    
    logger.info(f"Downloading XTTS model '{model_name}' to {cache_dir}...")
    logger.info("This may take several minutes on first run (~1.7GB download)...")
    
    try:
        # Redirect stdin to auto-accept any prompts
        from io import StringIO
        original_stdin = sys.stdin
        sys.stdin = StringIO("y\n")
        
        try:
            # Initialize TTS - this will download the model if not present
            use_gpu = device == "cuda"
            tts = TTS(model_name, gpu=use_gpu)
            logger.info("Model downloaded successfully!")
            
            return cache_dir
        finally:
            sys.stdin = original_stdin
    except Exception as e:
        logger.error(f"Failed to download model: {e}")
        # Don't raise - let the container download it if this fails
        logger.warning("Model download failed, container will download on first use")
        return cache_dir


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Download XTTS model to local cache")
    parser.add_argument(
        "--model-name",
        default="tts_models/multilingual/multi-dataset/xtts_v2",
        help="Name of the TTS model to download",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Directory to cache the model (default: models/xtts_baseline)",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cpu",
        help="Device to use for download",
    )
    
    args = parser.parse_args()
    
    try:
        download_xtts_model(args.model_name, args.cache_dir, args.device)
        print("Model download complete!")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

