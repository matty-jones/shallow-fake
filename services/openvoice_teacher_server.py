"""Teacher model FastAPI server for OpenVoice synthetic data generation."""

import logging
import os
import uuid
import base64
import io
from pathlib import Path

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

# 3) Target speaker embedding from reference WAV
# Find the first WAV file in the reference directory
reference_wav_path = None
reference_dir = Path(REFERENCE_AUDIO_DIR)
if reference_dir.exists():
    wav_files = list(reference_dir.glob("*.wav"))
    if wav_files:
        reference_wav_path = str(wav_files[0])
        logger.info(f"Using reference WAV: {reference_wav_path}")
    else:
        raise RuntimeError(f"No WAV files found in reference directory: {REFERENCE_AUDIO_DIR}")
else:
    raise RuntimeError(f"Reference audio directory does not exist: {REFERENCE_AUDIO_DIR}")

if not os.path.exists(reference_wav_path):
    raise RuntimeError(f"Reference wav not found at {reference_wav_path}")

logger.info("Extracting target speaker embedding from reference WAV...")
target_se, _ = se_extractor.get_se(reference_wav_path, tone_color_converter, vad=False)
target_se = target_se.to(DEVICE)
logger.info("Target speaker embedding extracted")

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
    }


@app.post("/tts", response_model=TTSResponse)
def tts(req: TTSRequest):
    """Generate speech using OpenVoice."""
    try:
        # 1) Generate base speech with MeloTTS
        tmp_src = f"/tmp/src_{uuid.uuid4().hex}.wav"
        tmp_out = f"/tmp/out_{uuid.uuid4().hex}.wav"

        # NOTE: TTS.tts_to_file(text, speaker_id, output_path, speed=...)
        tts_model.tts_to_file(
            req.text,
            speaker_id,
            tmp_src,
            speed=req.speed,
        )

        # 2) Apply ToneColorConverter to match your voice
        tone_color_converter.convert(
            audio_src_path=tmp_src,
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

        # Clean up temp files
        try:
            os.remove(tmp_src)
            os.remove(tmp_out)
        except Exception:
            pass  # Non-critical cleanup

        return TTSResponse(audio_base64=audio_b64, sample_rate=sr)
    except Exception as e:
        logger.error(f"Error generating speech: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

