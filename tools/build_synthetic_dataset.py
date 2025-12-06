"""Build synthetic dataset using TTS backend."""

import json
import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional

import ffmpeg
import requests
from faster_whisper import WhisperModel
from piper_phonemize import phonemize_espeak

from shallow_fake.config import VoiceConfig
from shallow_fake.utils import ensure_dir, setup_logging

logger = setup_logging()


class TTSBackend:
    """Abstract TTS backend interface."""

    def generate(self, text: str, output_path: Path) -> bool:
        """Generate audio from text. Returns True if successful."""
        raise NotImplementedError


class HTTPTTSBackend(TTSBackend):
    """HTTP-based TTS backend."""

    def __init__(self, base_url: str, voice_id: str):
        self.base_url = base_url.rstrip("/")
        self.voice_id = voice_id

    def generate(self, text: str, output_path: Path) -> bool:
        """Generate audio via HTTP API."""
        try:
            url = f"{self.base_url}"
            params = {"text": text, "voice": self.voice_id}
            response = requests.post(url, json=params, timeout=30)
            response.raise_for_status()

            # Save audio (assuming response is audio data)
            with open(output_path, "wb") as f:
                f.write(response.content)

            return output_path.exists()
        except Exception as e:
            logger.error(f"HTTP TTS error for text '{text[:50]}...': {e}")
            return False


def normalize_audio_to_piper(input_path: Path, output_path: Path) -> bool:
    """Normalize audio to Piper format (22.05 kHz mono 16-bit PCM)."""
    try:
        (
            ffmpeg.input(str(input_path))
            .output(
                str(output_path),
                acodec="pcm_s16le",
                ac=1,
                ar=22050,
                loglevel="error",
            )
            .overwrite_output()
            .run(quiet=True)
        )
        return True
    except ffmpeg.Error as e:
        error_msg = e.stderr.decode() if e.stderr else str(e)
        logger.error(f"FFmpeg error normalizing audio: {error_msg}")
        return False
    except Exception as e:
        logger.error(f"Error normalizing audio: {e}")
        return False


def load_corpus(corpus_path: Path, max_sentences: Optional[int] = None) -> List[str]:
    """Load text corpus, one sentence per line."""
    if not corpus_path.exists():
        logger.error(f"Corpus file not found: {corpus_path}")
        return []

    sentences = []
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                sentences.append(line)
                if max_sentences and len(sentences) >= max_sentences:
                    break

    logger.info(f"Loaded {len(sentences)} sentences from corpus")
    return sentences


def phonemize_text(text: str, language: str) -> str:
    """Convert text to phoneme sequence."""
    try:
        # piper-phonemize uses 'voice' parameter, not 'language'
        # Convert language code to voice format (e.g., 'en-gb' -> 'en')
        voice = language.split('-')[0] if '-' in language else language
        
        # phonemize_espeak returns List[List[str]] (list of lists, one per sentence)
        phonemes_nested = phonemize_espeak(text, voice=voice)
        
        # Flatten the nested list
        phonemes = [p for sublist in phonemes_nested for p in sublist]
        return " ".join(phonemes)
    except Exception as e:
        logger.error(f"Error phonemizing text: {e}")
        return ""


def normalized_edit_distance(s1: str, s2: str) -> float:
    """Compute normalized Levenshtein distance between phoneme sequences."""
    if not s1 and not s2:
        return 0.0
    if not s1 or not s2:
        return 1.0

    tokens1 = s1.split()
    tokens2 = s2.split()

    m, n = len(tokens1), len(tokens2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if tokens1[i - 1] == tokens2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = min(
                    dp[i - 1][j] + 1,
                    dp[i][j - 1] + 1,
                    dp[i - 1][j - 1] + 1,
                )

    max_len = max(m, n)
    return dp[m][n] / max_len if max_len > 0 else 0.0


def verify_synthetic_entry(
    text: str,
    audio_path: Path,
    config: VoiceConfig,
) -> tuple[bool, float]:
    """Verify synthetic entry using phoneme comparison."""
    # Transcribe with Whisper (always use CPU for verification)
    try:
        model = WhisperModel("base.en", device="cpu", compute_type="int8")
        segments, _ = model.transcribe(str(audio_path), language="en")
        whisper_text = " ".join(segment.text for segment in segments).strip()
    except Exception as e:
        logger.warning(f"Whisper transcription failed: {e}")
        return False, 1.0

    if not whisper_text:
        return False, 1.0

    # Phonemize both
    canonical_phonemes = phonemize_text(text, config.phoneme_check.language)
    whisper_phonemes = phonemize_text(whisper_text, config.phoneme_check.language)

    if not canonical_phonemes or not whisper_phonemes:
        return False, 1.0

    # Compute distance
    distance = normalized_edit_distance(canonical_phonemes, whisper_phonemes)
    is_valid = distance <= config.phoneme_check.max_phoneme_distance

    return is_valid, distance


def generate_synthetic_entry(
    text: str,
    index: int,
    tts_backend: TTSBackend,
    synth_dir: Path,
    config: VoiceConfig,
) -> Optional[tuple[str, str]]:
    """Generate a single synthetic dataset entry. Returns (wav_path, text) or None."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        raw_audio_path = Path(tmp.name)

    # Generate audio
    if not tts_backend.generate(text, raw_audio_path):
        raw_audio_path.unlink(missing_ok=True)
        return None

    # Normalize to Piper format
    synth_id = f"synth_{index:04d}"
    normalized_path = synth_dir / f"{synth_id}.wav"
    if not normalize_audio_to_piper(raw_audio_path, normalized_path):
        raw_audio_path.unlink(missing_ok=True)
        return None

    raw_audio_path.unlink()

    # Verify quality
    is_valid, distance = verify_synthetic_entry(text, normalized_path, config)
    if not is_valid:
        logger.debug(f"Rejected synthetic entry {synth_id}: distance={distance:.4f}")
        normalized_path.unlink()
        return None

    return (f"wavs/{synth_id}.wav", text)


def build_synthetic_dataset(config: VoiceConfig):
    """Build synthetic dataset from text corpus."""
    if not config.synthetic.enabled:
        logger.info("Synthetic data expansion is disabled")
        return

    corpus_path = config.synthetic.corpus_text_path
    synth_dataset_dir = config.paths.synth_dataset_dir
    wavs_dir = synth_dataset_dir / "wavs"
    metadata_csv = synth_dataset_dir / "metadata.csv"

    ensure_dir(wavs_dir)

    # Load corpus
    sentences = load_corpus(corpus_path, config.synthetic.max_sentences)
    if not sentences:
        logger.error("No sentences loaded from corpus")
        return

    # Create TTS backend
    if config.synthetic.tts_backend == "http":
        tts_backend = HTTPTTSBackend(
            config.synthetic.tts_http.base_url,
            config.synthetic.tts_http.voice_id,
        )
    else:
        logger.error(f"Unsupported TTS backend: {config.synthetic.tts_backend}")
        return

    # Generate synthetic entries in parallel
    logger.info(f"Generating {len(sentences)} synthetic entries...")
    valid_entries = []
    max_workers = config.synthetic.max_parallel_jobs

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                generate_synthetic_entry,
                text,
                i,
                tts_backend,
                wavs_dir,
                config,
            ): (i, text)
            for i, text in enumerate(sentences)
        }

        for future in as_completed(futures):
            i, text = futures[future]
            try:
                result = future.result()
                if result:
                    valid_entries.append(result)
                    if len(valid_entries) % 10 == 0:
                        logger.info(f"Generated {len(valid_entries)} valid entries...")
            except Exception as e:
                logger.error(f"Error generating entry {i}: {e}")

    # Write metadata
    logger.info(f"Writing metadata with {len(valid_entries)} entries")
    with open(metadata_csv, "w", encoding="utf-8") as f:
        for wav_path, text in valid_entries:
            # Escape pipe characters
            text_escaped = text.replace("|", " ")
            f.write(f"{wav_path}|{text_escaped}\n")

    logger.info(f"Synthetic dataset built: {len(valid_entries)} entries in {synth_dataset_dir}")

    # Now run phoneme verification on the synthetic dataset
    logger.info("Running phoneme verification on synthetic dataset...")
    # Note: Phoneme verification should be run separately via CLI
    # to avoid circular imports and allow more control

    # Temporarily update config paths to point to synthetic dataset
    original_real_dir = config.paths.real_dataset_dir
    config.paths.real_dataset_dir = synth_dataset_dir
    verify_dataset(config)
    config.paths.real_dataset_dir = original_real_dir


if __name__ == "__main__":
    import sys
    from pathlib import Path

    if len(sys.argv) < 2:
        print("Usage: python build_synthetic_dataset.py <config_path>")
        sys.exit(1)

    config_path = Path(sys.argv[1])
    config = VoiceConfig.from_yaml(config_path)
    config.ensure_directories()
    build_synthetic_dataset(config)

