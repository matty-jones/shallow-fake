"""XTTS Teacher FastAPI server for synthetic data generation."""

import logging
import os
import random
import sys
import tempfile
import traceback
from functools import lru_cache
from io import StringIO
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel

# Patch torch.load to allow loading XTTS checkpoints (PyTorch 2.6+ requires weights_only=False)
# This is safe since we're loading trusted Coqui TTS models
try:
    import torch
    # Monkey-patch torch.load to use weights_only=False for TTS model loading
    _original_torch_load = torch.load
    def _patched_torch_load(*args, **kwargs):
        # If weights_only is not explicitly set, default to False for compatibility
        if 'weights_only' not in kwargs:
            kwargs['weights_only'] = False
        return _original_torch_load(*args, **kwargs)
    torch.load = _patched_torch_load
except ImportError:
    pass  # torch not available yet

from TTS.api import TTS

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------------------------------------------------------
# Environment-driven config (wired via docker compose)
# -------------------------------------------------------------------
MODEL_NAME = os.environ.get(
    "XTTS_MODEL_NAME",
    "tts_models/multilingual/multi-dataset/xtts_v2",
)
SPEAKER_WAVS = os.environ.get("XTTS_SPEAKER_WAVS", "/speakers")
LANGUAGE = os.environ.get("XTTS_LANGUAGE", "en")
DEVICE = os.environ.get("XTTS_DEVICE", "cuda")  # "cuda" or "cpu"
NUM_REFERENCE_CLIPS = int(os.environ.get("XTTS_NUM_REFERENCE_CLIPS", "3"))

# Parse speaker WAVs - can be a directory or comma-separated list of files
SPEAKER_DIR = Path(SPEAKER_WAVS)
if SPEAKER_DIR.is_dir():
    # If it's a directory, find all WAV files
    SPEAKER_WAV_LIST: List[str] = [
        str(p) for p in SPEAKER_DIR.glob("*.wav")
    ]
else:
    # Otherwise, treat as comma-separated list
    SPEAKER_WAV_LIST = [
        w.strip() for w in SPEAKER_WAVS.split(",") if w.strip()
    ]

app = FastAPI(title="XTTS Teacher Server")


class TTSRequest(BaseModel):
    text: str
    # Optional overrides per request (not used currently, but kept for API compatibility)
    voice: Optional[str] = None
    language: Optional[str] = None
    num_reference_clips: Optional[int] = None


@lru_cache(maxsize=1)
def get_tts_model() -> TTS:
    """Load the XTTS model once per process."""
    use_gpu = DEVICE == "cuda"
    
    # Model cache is mounted at ~/tts from host's models/xtts_baseline/tts
    # When TTS_HOME=/root, TTS uses /root/tts/ for models
    # Ensure TOS acceptance file exists (should be created on host, but create if missing)
    # TTS looks for TOS file in the tts directory
    model_cache_dir = os.path.expanduser("~/tts")
    os.makedirs(model_cache_dir, exist_ok=True)
    tos_file = os.path.join(model_cache_dir, ".coqui_tos")
    if not os.path.exists(tos_file):
        try:
            with open(tos_file, "w") as f:
                f.write("accepted\n")
            logger.info(f"Created TOS acceptance file: {tos_file}")
        except OSError as e:
            # If we can't write (read-only mount), try to continue anyway
            # TOS file might already exist or TTS might handle it differently
            logger.warning(f"Could not create TOS file (may be read-only mount): {e}")
            # Check if file exists (might have been created by host)
            if os.path.exists(tos_file):
                logger.info("TOS file exists, continuing")
            else:
                # Try alternative location
                alt_tos = os.path.join(os.path.expanduser("~"), ".coqui_tos")
                try:
                    with open(alt_tos, "w") as f:
                        f.write("accepted\n")
                    logger.info(f"Created TOS file at alternative location: {alt_tos}")
                except Exception:
                    logger.warning("Could not create TOS file, TTS may prompt for acceptance")
    
    # Check if model exists in cache before loading
    model_path = os.path.join(model_cache_dir, "tts_models--multilingual--multi-dataset--xtts_v2")
    model_file = os.path.join(model_path, "model.pth")
    
    if os.path.exists(model_file):
        file_size = os.path.getsize(model_file) / (1024 * 1024 * 1024)  # Size in GB
        logger.info(f"Found cached model at {model_path} ({file_size:.2f} GB)")
    else:
        logger.warning(f"Model not found at {model_path}, will download on first use")
        logger.warning(f"Cache directory contents: {os.listdir(model_cache_dir) if os.path.exists(model_cache_dir) else 'does not exist'}")
    
    # TTS library will use ~/.local/share/tts by default, which is where we mounted the cache
    # Set TTS_HOME to ensure it uses the mounted cache (if not already set by Docker)
    if "TTS_HOME" not in os.environ:
        # TTS_HOME should point to the parent of .local/share/tts
        # Since we mount to ~/.local/share/tts, we need to set TTS_HOME to ~
        os.environ["TTS_HOME"] = os.path.expanduser("~")
    
    logger.info(f"TTS_HOME={os.environ.get('TTS_HOME')}")
    logger.info(f"Model cache directory: {model_cache_dir}")
    logger.info(f"Model file exists: {os.path.exists(model_file)}")
    
    if os.path.exists(model_file):
        logger.info(f"Using cached model at {model_path}")
    else:
        logger.warning(f"Model not found in cache, TTS will download to {model_cache_dir}")
    
    logger.info(f"Loading TTS model '{MODEL_NAME}' (this may take a moment if verifying cache)...")
    
    # Redirect stdin to auto-accept TOS prompt if it still appears
    original_stdin = sys.stdin
    try:
        # Create a StringIO object that will provide "y" when read
        fake_input = StringIO("y\n")
        sys.stdin = fake_input
        
        # Initialize TTS - it should use the cached model if available
        # Use device parameter instead of deprecated gpu parameter
        device = "cuda" if use_gpu else "cpu"
        logger.info(f"Initializing TTS with model_name='{MODEL_NAME}', device={device}")
        tts = TTS(MODEL_NAME)
        # Move to device after initialization
        if use_gpu:
            try:
                tts.to(device)
                logger.info(f"Model moved to {device}")
            except Exception as e:
                logger.warning(f"Failed to move model to {device}, using CPU: {e}")
        logger.info("TTS model loaded successfully")
        return tts
    except EOFError as e:
        logger.error(f"EOFError during TTS initialization: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        # If EOFError still occurs, the TOS file might not be in the right location
        # Try creating it in multiple possible locations
        for cache_dir in [
            os.path.expanduser("~/.local/share/tts"),
            "/root/.local/share/tts",
            os.path.join(os.path.expanduser("~"), ".coqui"),
        ]:
            try:
                os.makedirs(cache_dir, exist_ok=True)
                tos_file = os.path.join(cache_dir, ".coqui_tos")
                with open(tos_file, "w") as f:
                    f.write("accepted\n")
            except Exception:
                pass
        
        # Retry with stdin still redirected
        sys.stdin = StringIO("y\n")
        try:
            device = "cuda" if use_gpu else "cpu"
            tts = TTS(MODEL_NAME)
            if use_gpu:
                try:
                    tts.to(device)
                except Exception:
                    logger.warning("Failed to move model to GPU, using CPU")
            return tts
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to load XTTS model after TOS setup: {str(e)}. "
                "Please check container logs for more details."
            )
    except Exception as e:
        error_trace = traceback.format_exc()
        logger.error(f"Failed to load XTTS model: {str(e)}")
        logger.error(f"Traceback: {error_trace}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load XTTS model: {str(e)}. Check container logs for details."
        )
    finally:
        # Restore original stdin
        sys.stdin = original_stdin


def get_reference_audio_files(num_clips: int = 3) -> List[str]:
    """Select reference audio files for synthesis."""
    if not SPEAKER_WAV_LIST:
        raise HTTPException(
            status_code=500,
            detail="No reference audio files found. Check XTTS_SPEAKER_WAVS environment variable.",
        )

    # If we have fewer files than requested, use all available
    if len(SPEAKER_WAV_LIST) <= num_clips:
        selected = SPEAKER_WAV_LIST
    else:
        # Randomly select N files
        selected = random.sample(SPEAKER_WAV_LIST, num_clips)

    # Verify files exist
    for path in selected:
        if not os.path.isfile(path):
            raise HTTPException(
                status_code=500,
                detail=f"Reference audio file not found: {path}",
            )

    return selected


@app.post("/tts")
def tts_endpoint(req: TTSRequest):
    """Generate speech from text using XTTS with reference audio."""
    if not req.text or not req.text.strip():
        raise HTTPException(status_code=400, detail="Text is empty")

    language = req.language or LANGUAGE
    output_path = None

    try:
        # Get reference audio files (use config default, can be overridden per request)
        num_reference_clips = req.num_reference_clips or NUM_REFERENCE_CLIPS
        logger.info(f"Getting {num_reference_clips} reference audio files...")
        speaker_wavs = get_reference_audio_files(num_reference_clips)
        logger.info(f"Using {len(speaker_wavs)} reference audio files: {speaker_wavs}")

        logger.info("Loading XTTS model...")
        tts = get_tts_model()
        logger.info("XTTS model loaded successfully")

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            output_path = tmp.name

        logger.info(f"Generating speech for text: {req.text[:50]}...")
        
        # XTTS can accept a list of speaker WAVs or a single file
        # Pass the list directly - XTTS will use them for conditioning
        speaker_wav = speaker_wavs if len(speaker_wavs) > 1 else speaker_wavs[0]

        tts.tts_to_file(
            text=req.text,
            file_path=output_path,
            speaker_wav=speaker_wav,
            language=language,
            split_sentences=True,
        )

        logger.info(f"Speech generated successfully, reading audio file...")
        
        # Read the generated audio
        with open(output_path, "rb") as f:
            audio = f.read()

        # Clean up temp file
        os.unlink(output_path)
        output_path = None

        logger.info(f"Returning audio response ({len(audio)} bytes)")
        # XTTS outputs 24 kHz mono WAV by default
        return Response(content=audio, media_type="audio/wav")
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        # Log the full error with traceback
        import traceback
        error_trace = traceback.format_exc()
        logger.error(f"Error generating speech: {str(e)}")
        logger.error(f"Traceback: {error_trace}")
        
        # Clean up on error
        if output_path and os.path.exists(output_path):
            try:
                os.unlink(output_path)
            except Exception:
                pass
        
        raise HTTPException(
            status_code=500,
            detail=f"Error generating speech: {str(e)}. Check container logs for details.",
        )


@app.get("/health")
def health_check():
    """Health check endpoint - doesn't load model, just confirms server is running."""
    return {
        "status": "ok",
        "model": MODEL_NAME,
        "device": DEVICE,
        "speaker_files": len(SPEAKER_WAV_LIST),
    }

