"""Evaluate exported ONNX model by generating sample phrases."""

import subprocess
from pathlib import Path
from typing import List

from shallow_fake.config import VoiceConfig
from shallow_fake.language_utils import format_model_name
from shallow_fake.utils import ensure_dir, setup_logging

logger = setup_logging()


def load_evaluation_phrases(evaluation_path: Path) -> List[str]:
    """Load evaluation phrases from file, one per line."""
    if not evaluation_path.exists():
        logger.warning(f"Evaluation file not found: {evaluation_path}")
        logger.warning("Using default phrases. Create the file to customize evaluation phrases.")
        # Fallback to default phrases if file doesn't exist
        return [
            "This is a test of the voice synthesis system.",
            "The quick brown fox jumps over the lazy dog.",
            "Hello, this is my custom voice model speaking.",
            "I can generate natural sounding speech from text.",
            "This voice was trained using the ShallowFaker pipeline.",
        ]
    
    phrases = []
    with open(evaluation_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                phrases.append(line)
    
    if not phrases:
        logger.warning(f"No phrases found in {evaluation_path}")
        return [
            "This is a test of the voice synthesis system.",
            "The quick brown fox jumps over the lazy dog.",
            "Hello, this is my custom voice model speaking.",
            "I can generate natural sounding speech from text.",
            "This voice was trained using the ShallowFaker pipeline.",
        ]
    
    logger.info(f"Loaded {len(phrases)} evaluation phrases from {evaluation_path}")
    return phrases


def generate_samples(
    model_path: Path,
    json_path: Path,
    phrases: List[str],
    output_dir: Path,
):
    """Generate audio samples from ONNX model."""
    ensure_dir(output_dir)

    if not model_path.exists():
        logger.error(f"Model file not found: {model_path}")
        raise FileNotFoundError(f"Model file not found: {model_path}")

    if not json_path.exists():
        logger.warning(f"Config JSON not found: {json_path}")

    logger.info(f"Generating {len(phrases)} sample phrases...")

    for i, phrase in enumerate(phrases, 1):
        output_file = output_dir / f"sample_{i:02d}.wav"

        cmd = [
            "piper",
            "--model", str(model_path),
            "--output_file", str(output_file),
        ]

        if json_path.exists():
            cmd.extend(["--config", str(json_path)])

        try:
            process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            stdout, stderr = process.communicate(input=phrase)

            if process.returncode == 0:
                logger.info(f"Generated: {output_file.name} - '{phrase[:50]}...'")
            else:
                logger.error(f"Failed to generate {output_file.name}: {stderr}")
        except FileNotFoundError:
            logger.error("piper command not found. Please install piper-tts.")
            raise
        except Exception as e:
            logger.error(f"Error generating sample {i}: {e}")

    logger.info(f"Sample generation complete. Output directory: {output_dir}")


def eval_model(config: VoiceConfig, phrases: List[str] = None, model_name: str = None):
    """Evaluate exported model."""
    if phrases is None:
        # Load from input/shared/evaluation.txt
        evaluation_path = config.paths.corpus_path.parent / "evaluation.txt"
        phrases = load_evaluation_phrases(evaluation_path)

    if model_name is None:
        # Use Piper naming convention: <language>_<REGION>-<name>-<quality>
        model_name = format_model_name(
            config.language_code,
            config.region,
            config.voice_id,
            config.training.quality
        )

    model_path = config.paths.output_models_dir / f"{model_name}.onnx"
    json_path = config.paths.output_models_dir / f"{model_name}.onnx.json"
    
    # Extract version from model name if present (e.g., "en_GB-claudia-high-v1" -> "v1")
    # This allows separate output directories for different versions
    import re
    version_match = re.search(r'-v(\d+)$', model_name)
    if version_match:
        version = version_match.group(1)
        output_dir = config.paths.output_models_dir.parent / "samples" / f"{config.voice_id}-v{version}"
    else:
        output_dir = config.paths.output_models_dir.parent / "samples" / config.voice_id

    generate_samples(model_path, json_path, phrases, output_dir)


if __name__ == "__main__":
    import sys
    from pathlib import Path

    if len(sys.argv) < 2:
        print("Usage: python eval_model.py <config_path> [--phrases-file <path>] [--model-name <name>]")
        sys.exit(1)

    config_path = Path(sys.argv[1])
    config = VoiceConfig.from_yaml(config_path)

    phrases = None
    model_name = None

    args = sys.argv[2:]
    i = 0
    while i < len(args):
        if args[i] == "--phrases-file" and i + 1 < len(args):
            phrases_path = Path(args[i + 1])
            if phrases_path.exists():
                with open(phrases_path, "r", encoding="utf-8") as f:
                    phrases = [line.strip() for line in f if line.strip()]
            i += 2
        elif args[i] == "--model-name" and i + 1 < len(args):
            model_name = args[i + 1]
            i += 2
        else:
            i += 1

    eval_model(config, phrases, model_name)






