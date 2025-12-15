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

from shallow_fake.config import MetaVoiceTeacherConfig, VoiceConfig, XTTSTeacherConfig
from shallow_fake.language_utils import convert_language_for_phoneme
from shallow_fake.utils import ensure_dir, setup_logging

logger = setup_logging()


class TTSBackend:
    """Abstract TTS backend interface."""

    def generate(self, text: str, output_path: Path) -> bool:
        """Generate audio from text. Returns True if successful."""
        raise NotImplementedError


class HTTPTTSBackend(TTSBackend):
    """HTTP-based TTS backend."""

    def __init__(self, base_url: str, voice_id: str, teacher_config=None):
        self.base_url = base_url.rstrip("/")
        self.voice_id = voice_id
        self.teacher_config = teacher_config
        # Create a session for connection pooling
        self.session = requests.Session()

    def generate(self, text: str, output_path: Path) -> bool:
        """Generate audio via HTTP API."""
        try:
            # Check if this is MetaVoice (uses different API format)
            if isinstance(self.teacher_config, MetaVoiceTeacherConfig):
                # MetaVoice API: POST /tts with X-Payload header
                url = f"{self.base_url}/tts"
                payload = {
                    "text": text,
                    "speaker_ref_path": self.teacher_config.speaker_ref_path,
                    "guidance": self.teacher_config.guidance,
                    "top_p": self.teacher_config.top_p,
                }
                if self.teacher_config.top_k is not None:
                    payload["top_k"] = self.teacher_config.top_k

                headers = {"X-Payload": json.dumps(payload)}
                # MetaVoice expects empty body when using speaker_ref_path
                response = self.session.post(url, headers=headers, data=b"", timeout=300)
            else:
                # XTTS API: POST /tts with JSON body
                url = f"{self.base_url}"
                params = {"text": text, "voice": self.voice_id}
                response = self.session.post(url, json=params, timeout=120)

            response.raise_for_status()

            # Save audio (assuming response is audio data)
            with open(output_path, "wb") as f:
                f.write(response.content)

            return output_path.exists()
        except requests.exceptions.HTTPError as e:
            # Log more details for 500 errors
            if e.response.status_code == 500:
                logger.error(f"HTTP TTS 500 error for text '{text[:50]}...': {e}")
                try:
                    error_detail = e.response.json().get("detail", "No details available")
                    logger.error(f"Server error details: {error_detail}")
                except Exception:
                    logger.error(f"Server response: {e.response.text[:200]}")
            else:
                logger.error(f"HTTP TTS error for text '{text[:50]}...': {e}")
            return False
        except requests.exceptions.ConnectionError as e:
            # Connection errors often indicate server crash or overload (e.g., GPU OOM)
            logger.error(f"HTTP TTS connection error for text '{text[:50]}...': {e}")
            if isinstance(self.teacher_config, XTTSTeacherConfig):
                logger.warning("This may indicate server overload or GPU memory exhaustion. Consider reducing workers.")
            return False
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

    # Phonemize both (convert language format for phoneme checking)
    phoneme_language = convert_language_for_phoneme(config.language)
    canonical_phonemes = phonemize_text(text, phoneme_language)
    whisper_phonemes = phonemize_text(whisper_text, phoneme_language)

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
    synth_dataset_dir = config.get_synth_dataset_dir()
    wavs_dir = synth_dataset_dir / "wavs"
    metadata_csv = synth_dataset_dir / "metadata.csv"

    ensure_dir(wavs_dir)

    # Load corpus
    sentences = load_corpus(corpus_path, config.synthetic.max_sentences)
    if not sentences:
        logger.error("No sentences loaded from corpus")
        return

    # Start teacher model service if configured
    teacher_started = False
    teacher_kind = None
    if config.synthetic.teacher:
        teacher_kind = config.synthetic.teacher.kind
        try:
            if teacher_kind == "xtts":
                from tools.xtts_teacher_orchestration import start_xtts_teacher, stop_xtts_teacher

                logger.info("Starting XTTS teacher model service...")
                teacher_config = config.synthetic.teacher
                if isinstance(teacher_config, XTTSTeacherConfig):
                    num_clips = teacher_config.num_reference_clips
                    if num_clips == 0:
                        logger.info(f"XTTS teacher configured to use ALL available reference audio files")
                    else:
                        logger.info(f"XTTS teacher configured to use {num_clips} reference audio files per request")
                start_xtts_teacher(config)
                teacher_started = True
                logger.info("XTTS teacher model service started successfully")
            elif teacher_kind == "metavoice":
                from tools.metavoice_teacher_orchestration import (
                    start_metavoice_teacher,
                    stop_metavoice_teacher,
                )

                logger.info("Starting MetaVoice teacher model service...")
                teacher_config = config.synthetic.teacher
                if isinstance(teacher_config, MetaVoiceTeacherConfig):
                    logger.info(
                        f"MetaVoice config: guidance={teacher_config.guidance}, "
                        f"top_p={teacher_config.top_p}, top_k={teacher_config.top_k}"
                    )
                start_metavoice_teacher(config)
                teacher_started = True
                logger.info("MetaVoice teacher model service started successfully")
            else:
                raise ValueError(f"Unsupported teacher kind: {teacher_kind}")
        except Exception as e:
            logger.error(f"Failed to start teacher model service: {e}")
            logger.error("Synthetic dataset generation aborted")
            return

    try:
        # Create TTS backend
        if config.synthetic.tts_backend == "http":
            tts_backend = HTTPTTSBackend(
                config.synthetic.tts_http.base_url,
                config.synthetic.tts_http.voice_id,
                teacher_config=config.synthetic.teacher,
            )
        else:
            logger.error(f"Unsupported TTS backend: {config.synthetic.tts_backend}")
            return

        # Generate synthetic entries in parallel
        logger.info(f"Generating {len(sentences)} synthetic entries...")
        valid_entries = []
        max_workers = config.synthetic.max_parallel_jobs
        if config.synthetic.teacher:
            if isinstance(config.synthetic.teacher, XTTSTeacherConfig):
                logger.info(
                    f"Using {max_workers} parallel jobs "
                    f"(auto-calculated as {config.synthetic.teacher.workers} workers * 2 to keep workers busy)"
                )
            else:
                logger.info(f"Using {max_workers} parallel jobs")

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

            completed = 0
            total = len(sentences)
            for future in as_completed(futures):
                completed += 1
                i, text = futures[future]
                try:
                    result = future.result()
                    if result:
                        valid_entries.append(result)
                    logger.info(f"Progress: {completed}/{total} entries processed ({len(valid_entries)} valid)")
                except Exception as e:
                    logger.error(f"Error generating entry {i}: {e}")
                    logger.info(f"Progress: {completed}/{total} entries processed ({len(valid_entries)} valid)")

        # Write metadata
        logger.info(f"Writing metadata with {len(valid_entries)} entries")
        with open(metadata_csv, "w", encoding="utf-8") as f:
            for wav_path, text in valid_entries:
                # Escape pipe characters
                text_escaped = text.replace("|", " ")
                f.write(f"{wav_path}|{text_escaped}\n")

        logger.info(f"Synthetic dataset built: {len(valid_entries)} entries in {synth_dataset_dir}")

    finally:
        # Stop teacher model service if we started it
        if teacher_started:
            try:
                import subprocess

                # Check container logs before stopping if there were errors
                if len(valid_entries) == 0 and sentences:
                    logger.warning("No valid entries generated. Checking container logs...")
                    container_name = (
                        f"{config.voice_id}_xtts_teacher"
                        if teacher_kind == "xtts"
                        else f"{config.voice_id}_metavoice_teacher"
                    )
                    try:
                        # Try to get logs from running or stopped container
                        result = subprocess.run(
                            ["docker", "logs", container_name],
                            capture_output=True,
                            text=True,
                            timeout=10,
                        )
                        if result.stdout:
                            logger.error("Container logs (stdout):")
                            for line in result.stdout.split("\n")[-100:]:  # Last 100 lines
                                if line.strip():
                                    logger.error(f"  {line}")
                        if result.stderr:
                            logger.error("Container logs (stderr):")
                            for line in result.stderr.split("\n")[-100:]:  # Last 100 lines
                                if line.strip():
                                    logger.error(f"  {line}")
                    except subprocess.TimeoutExpired:
                        logger.warning("Timeout retrieving container logs")
                    except Exception as log_error:
                        logger.warning(f"Could not retrieve container logs: {log_error}")
                        # Try alternative: check if container exists and get status
                        try:
                            status_result = subprocess.run(
                                [
                                    "docker",
                                    "ps",
                                    "-a",
                                    "--filter",
                                    f"name={container_name}",
                                    "--format",
                                    "{{.Status}}",
                                ],
                                capture_output=True,
                                text=True,
                                timeout=5,
                            )
                            if status_result.stdout.strip():
                                logger.info(f"Container status: {status_result.stdout.strip()}")
                        except Exception:
                            pass

                logger.info("Stopping teacher model service...")
                if teacher_kind == "xtts":
                    from tools.xtts_teacher_orchestration import stop_xtts_teacher

                    stop_xtts_teacher(config)
                elif teacher_kind == "metavoice":
                    from tools.metavoice_teacher_orchestration import stop_metavoice_teacher

                    stop_metavoice_teacher(config)
                logger.info("Teacher model service stopped")
            except Exception as e:
                logger.warning(f"Error stopping teacher model service: {e}")

    # Now run phoneme verification on the synthetic dataset
    logger.info("Running phoneme verification on synthetic dataset...")
    # Note: Phoneme verification should be run separately via CLI
    # to avoid circular imports and allow more control

    # Temporarily update config paths to point to synthetic dataset
    original_real_dir = config.paths.real_dataset_dir
    config.paths.real_dataset_dir = synth_dataset_dir
    from tools.verify_phonemes import verify_dataset
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

