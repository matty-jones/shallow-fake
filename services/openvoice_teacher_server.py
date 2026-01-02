"""Teacher model FastAPI server for OpenVoice synthetic data generation."""

import logging
import os
import uuid
import base64
import io
import subprocess
from pathlib import Path
from typing import Optional, Tuple

import torch
import soundfile as sf
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel

from melo.api import TTS
from openvoice.api import ToneColorConverter
from openvoice import se_extractor

# Enable PyTorch memory optimization to reduce fragmentation
# This helps prevent CUDA OOM errors by allowing more efficient memory allocation
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------------------------------------------------------
# Environment-driven config (wired via docker compose)
# -------------------------------------------------------------------
OPENVOICE_CHECKPOINT_DIR = os.environ.get(
    "OPENVOICE_CHECKPOINT_DIR", "/models/openvoice/checkpoints_v2"
)
REFERENCE_AUDIO_DIR = os.environ.get("REFERENCE_AUDIO_DIR", "/speakers")
OPENVOICE_DEVICE = os.environ.get("OPENVOICE_DEVICE", "cuda").lower()
OPENVOICE_LANGUAGE = os.environ.get("OPENVOICE_LANGUAGE", "EN_NEWEST")
OPENVOICE_BASE_SPEAKER_KEY = os.environ.get("OPENVOICE_BASE_SPEAKER_KEY", "EN_BR")

# Determine device
DEVICE = "cuda" if OPENVOICE_DEVICE.startswith("cuda") and torch.cuda.is_available() else "cpu"

# Preferred speakers in order of preference
PREFERRED_SPEAKERS = [
    OPENVOICE_BASE_SPEAKER_KEY,
    "EN_DEFAULT",
    "EN_NEWEST",
    "EN_US",
]

# -------------------------------------------------------------------
# Helper functions
# -------------------------------------------------------------------

# OpenVoice expects 22.05 kHz mono audio
OPENVOICE_TARGET_SR = 22050
OPENVOICE_TARGET_CHANNELS = 1


def resample_audio_ffmpeg(
    input_path: str,
    output_path: str,
    target_sr: int = OPENVOICE_TARGET_SR,
    target_channels: int = OPENVOICE_TARGET_CHANNELS,
) -> bool:
    """
    Resample audio file to target sample rate and channels using ffmpeg.
    
    Args:
        input_path: Path to input audio file
        output_path: Path to output audio file
        target_sr: Target sample rate in Hz (default: 22050)
        target_channels: Target number of channels (default: 1 for mono)
    
    Returns:
        True if successful, False otherwise
    """
    try:
        result = subprocess.run(
            [
                "ffmpeg",
                "-i", input_path,
                "-ar", str(target_sr),
                "-ac", str(target_channels),
                "-y",  # Overwrite output file
                "-loglevel", "error",  # Suppress ffmpeg output
                output_path,
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg resampling failed: {e.stderr}")
        return False
    except Exception as e:
        logger.error(f"Error during audio resampling: {e}")
        return False


def get_audio_info(file_path: str) -> Optional[Tuple[int, int]]:
    """
    Get sample rate and channel count from audio file.
    
    Args:
        file_path: Path to audio file
    
    Returns:
        Tuple of (sample_rate, channels) or None if error
    """
    try:
        info = sf.info(file_path)
        return (info.samplerate, info.channels)
    except Exception as e:
        logger.error(f"Error getting audio info from {file_path}: {e}")
        return None


def ensure_audio_format(
    input_path: str,
    output_path: str,
    target_sr: int = OPENVOICE_TARGET_SR,
    target_channels: int = OPENVOICE_TARGET_CHANNELS,
) -> str:
    """
    Ensure audio file is in the correct format for OpenVoice.
    If resampling/conversion is needed, create a temporary resampled file.
    Otherwise, return the original path.
    
    Args:
        input_path: Path to input audio file
        output_path: Path to output resampled file (if needed)
        target_sr: Target sample rate in Hz
        target_channels: Target number of channels
    
    Returns:
        Path to audio file in correct format (original or resampled)
    """
    audio_info = get_audio_info(input_path)
    if audio_info is None:
        logger.warning(f"Could not read audio info, assuming resampling needed")
        if resample_audio_ffmpeg(input_path, output_path, target_sr, target_channels):
            return output_path
        else:
            raise RuntimeError(f"Failed to resample audio: {input_path}")
    
    sr, channels = audio_info
    
    # Check if resampling/conversion is needed
    if sr != target_sr or channels != target_channels:
        logger.info(
            f"Resampling audio: {sr} Hz, {channels} ch -> {target_sr} Hz, {target_channels} ch"
        )
        if resample_audio_ffmpeg(input_path, output_path, target_sr, target_channels):
            return output_path
        else:
            raise RuntimeError(f"Failed to resample audio: {input_path}")
    else:
        # Already in correct format
        logger.debug(f"Audio already in correct format: {sr} Hz, {channels} ch")
        return input_path


# -------------------------------------------------------------------
# Model initialization
# -------------------------------------------------------------------

logger.info(f"[openvoice-teacher] Using device: {DEVICE}")
logger.info(f"[openvoice-teacher] Checkpoint directory: {OPENVOICE_CHECKPOINT_DIR}")
logger.info(f"[openvoice-teacher] Reference audio directory: {REFERENCE_AUDIO_DIR}")

# 1) ToneColorConverter
ckpt_converter = os.path.join(OPENVOICE_CHECKPOINT_DIR, "converter")
converter_cfg = os.path.join(ckpt_converter, "config.json")
converter_ckpt = os.path.join(ckpt_converter, "checkpoint.pth")

if not os.path.exists(converter_cfg) or not os.path.exists(converter_ckpt):
    raise RuntimeError(
        f"OpenVoice V2 converter checkpoint not found under {ckpt_converter}"
    )

logger.info("Loading ToneColorConverter...")
tone_color_converter = ToneColorConverter(converter_cfg, device=DEVICE)
tone_color_converter.load_ckpt(converter_ckpt)
logger.info("ToneColorConverter loaded")

# 2) MeloTTS base model
logger.info(f"Loading MeloTTS model (language={OPENVOICE_LANGUAGE})...")
tts_model = TTS(language=OPENVOICE_LANGUAGE, device=DEVICE)
speaker_ids = tts_model.hps.data.spk2id
logger.info("MeloTTS model loaded")

# Pick a base speaker, preferring configured speaker if available
speaker_key = None
for candidate in PREFERRED_SPEAKERS:
    if candidate in speaker_ids:
        speaker_key = candidate
        break

if speaker_key is None:
    # Fall back to the first available
    speaker_key = list(speaker_ids.keys())[0]
    logger.warning(f"Configured base speaker not found, using first available: {speaker_key}")

speaker_id = speaker_ids[speaker_key]
speaker_key_file = speaker_key.lower().replace("_", "-")

src_se_path = os.path.join(
    OPENVOICE_CHECKPOINT_DIR, "base_speakers", "ses", f"{speaker_key_file}.pth"
)
if not os.path.exists(src_se_path):
    raise RuntimeError(f"Base speaker embedding not found: {src_se_path}")

logger.info(f"Loading base speaker embedding: {speaker_key}")
src_se = torch.load(src_se_path, map_location=DEVICE)
logger.info(f"Base speaker: {speaker_key}")

# 3) Target speaker embedding from reference WAV files
# Extract embeddings from all reference files and average them
reference_dir = Path(REFERENCE_AUDIO_DIR)
if not reference_dir.exists():
    raise RuntimeError(f"Reference audio directory does not exist: {REFERENCE_AUDIO_DIR}")

wav_files = sorted(reference_dir.glob("*.wav"))
if not wav_files:
    raise RuntimeError(f"No WAV files found in reference directory: {REFERENCE_AUDIO_DIR}")

logger.info(f"Found {len(wav_files)} reference WAV files, extracting embeddings from all...")

# Extract embeddings from each reference file and collect them
embeddings = []
temp_files_to_cleanup = []
successful_files = []
failed_files = []

for i, wav_file in enumerate(wav_files):
    wav_path = str(wav_file)
    if not os.path.exists(wav_path):
        logger.error(f"Reference WAV file not found, skipping: {wav_path}")
        failed_files.append((wav_file.name, "File not found"))
        continue
    
    try:
        # Ensure reference audio is in correct format for OpenVoice (22.05 kHz mono)
        # Resample at runtime if needed, using a temporary file
        reference_wav_resampled = f"/tmp/ref_{uuid.uuid4().hex}.wav"
        reference_wav_for_se = ensure_audio_format(
            wav_path,
            reference_wav_resampled,
            target_sr=OPENVOICE_TARGET_SR,
            target_channels=OPENVOICE_TARGET_CHANNELS,
        )
        
        # Track temp files for cleanup
        if reference_wav_for_se != wav_path:
            temp_files_to_cleanup.append(reference_wav_for_se)
        
        # Extract embedding from this file
        logger.info(f"Extracting embedding from {wav_file.name} ({i+1}/{len(wav_files)})...")
        se, _ = se_extractor.get_se(reference_wav_for_se, tone_color_converter, vad=False)
        embeddings.append(se.to(DEVICE))
        successful_files.append(wav_file.name)
        logger.info(f"Successfully extracted embedding from {wav_file.name}")
        
    except Exception as e:
        error_type = type(e).__name__
        error_msg = str(e)
        logger.error(
            f"Failed to extract embedding from {wav_file.name} ({i+1}/{len(wav_files)}): "
            f"{error_type}: {error_msg}"
        )
        logger.debug(f"Full exception for {wav_file.name}:", exc_info=True)
        failed_files.append((wav_file.name, f"{error_type}: {error_msg}"))
        continue

if not embeddings:
    raise RuntimeError(
        f"Failed to extract embeddings from any reference files in {REFERENCE_AUDIO_DIR}"
    )

# Log summary of successful and failed files
logger.info("=" * 60)
logger.info(f"Embedding extraction summary:")
logger.info(f"  Successful: {len(successful_files)}/{len(wav_files)} files")
if failed_files:
    logger.warning(f"  Failed: {len(failed_files)}/{len(wav_files)} files")
    for failed_name, error_reason in failed_files:
        logger.warning(f"    - {failed_name}: {error_reason}")
else:
    logger.info(f"  All files processed successfully")
logger.info("=" * 60)

# Average all embeddings
logger.info(f"Averaging {len(embeddings)} speaker embeddings...")
target_se = torch.stack(embeddings).mean(dim=0)
num_reference_files_used = len(embeddings)
logger.info(f"Target speaker embedding computed from {num_reference_files_used} reference files")

# Clean up temporary reference audio files
for temp_file in temp_files_to_cleanup:
    if os.path.exists(temp_file):
        try:
            os.remove(temp_file)
        except Exception:
            pass  # Non-critical cleanup

logger.info(f"[openvoice-teacher] Language={OPENVOICE_LANGUAGE}, base speaker={speaker_key}")

# -------------------------------------------------------------------
# FastAPI app
# -------------------------------------------------------------------

app = FastAPI(title="OpenVoice Teacher Model Server")


class TTSRequest(BaseModel):
    text: str
    speed: float = 1.0
    message: str = "@MyShell"  # OpenVoice uses this as an encode tag


class TTSResponse(BaseModel):
    audio_base64: str
    sample_rate: int


@app.get("/health")
def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "device": DEVICE,
        "cuda_available": torch.cuda.is_available() if DEVICE == "cuda" else False,
        "gpu_name": torch.cuda.get_device_name(0) if DEVICE == "cuda" and torch.cuda.is_available() else None,
        "language": OPENVOICE_LANGUAGE,
        "base_speaker": speaker_key,
        "num_reference_files": num_reference_files_used,
    }


@app.post("/tts", response_model=TTSResponse)
def tts(req: TTSRequest):
    """Generate speech using OpenVoice."""
    tmp_src = None
    tmp_src_resampled = None
    tmp_out = None
    
    try:
        # 1) Generate base speech with MeloTTS (outputs at 44.1 kHz)
        tmp_src = f"/tmp/src_{uuid.uuid4().hex}.wav"
        tmp_src_resampled = f"/tmp/src_resampled_{uuid.uuid4().hex}.wav"
        tmp_out = f"/tmp/out_{uuid.uuid4().hex}.wav"

        # NOTE: TTS.tts_to_file(text, speaker_id, output_path, speed=...)
        # MeloTTS outputs at 44.1 kHz, but OpenVoice expects 22.05 kHz
        tts_model.tts_to_file(
            req.text,
            speaker_id,
            tmp_src,
            speed=req.speed,
        )

        # Resample MeloTTS output (44.1 kHz) to OpenVoice input format (22.05 kHz mono)
        logger.debug("Resampling MeloTTS output to OpenVoice format...")
        melotts_sr, melotts_channels = get_audio_info(tmp_src) or (None, None)
        if melotts_sr:
            logger.debug(f"MeloTTS output: {melotts_sr} Hz, {melotts_channels} channel(s)")
        
        audio_src_for_converter = ensure_audio_format(
            tmp_src,
            tmp_src_resampled,
            target_sr=OPENVOICE_TARGET_SR,
            target_channels=OPENVOICE_TARGET_CHANNELS,
        )

        # 2) Apply ToneColorConverter to match your voice
        tone_color_converter.convert(
            audio_src_path=audio_src_for_converter,
            src_se=src_se,
            tgt_se=target_se,
            output_path=tmp_out,
            message=req.message,
        )

        # 3) Return as base64 WAV
        audio, sr = sf.read(tmp_out)
        buf = io.BytesIO()
        sf.write(buf, audio, sr, format="WAV")
        audio_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

        return TTSResponse(audio_base64=audio_b64, sample_rate=sr)
    except Exception as e:
        logger.error(f"Error generating speech: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up temp files
        for tmp_file in [tmp_src, tmp_src_resampled, tmp_out]:
            if tmp_file and os.path.exists(tmp_file):
                try:
                    os.remove(tmp_file)
                except Exception:
                    pass  # Non-critical cleanup

