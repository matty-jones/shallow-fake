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

from shallow_fake.config import (
    MetaVoiceTeacherConfig,
    OpenVoiceTeacherConfig,
    VoiceConfig,
    XTTSTeacherConfig,
)
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
        """Generate audio via HTTP API with retry logic for connection errors."""
        max_retries = 3
        retry_delay = 2.0  # seconds
        
        for attempt in range(max_retries):
            try:
                # Check if this is MetaVoice (uses different API format)
                if isinstance(self.teacher_config, MetaVoiceTeacherConfig):
                    # MetaVoice API: POST /tts with multipart form data
                    # base_url should be http://localhost:9010 (no /tts), we add /tts here
                    # Remove /tts from base_url if present to avoid double /tts/tts
                    base = self.base_url.rstrip("/")
                    if base.endswith("/tts"):
                        base = base[:-4]  # Remove /tts
                    url = f"{base}/tts"
                    
                    # MetaVoice expects multipart form data, not JSON headers
                    form_data = {
                        "text": text,
                        "speaker_ref_path": self.teacher_config.speaker_ref_path,
                        "guidance": str(self.teacher_config.guidance),
                        "top_p": str(self.teacher_config.top_p),
                    }
                    if self.teacher_config.top_k is not None:
                        form_data["top_k"] = str(self.teacher_config.top_k)

                    logger.debug(f"MetaVoice TTS request: URL={url}, form_data={form_data}")
                    response = self.session.post(url, data=form_data, timeout=300)
                    logger.debug(f"MetaVoice TTS response: status={response.status_code}, content_length={len(response.content)}")
                    
                    # Check if response is an error (JSON) instead of audio
                    content_type = response.headers.get("content-type", "").lower()
                    if "json" in content_type or (len(response.content) < 100 and response.content.startswith(b"{")):
                        try:
                            error_data = response.json()
                            logger.error(f"MetaVoice returned error: {error_data}")
                            return False
                        except Exception:
                            pass  # Not JSON, continue
                else:
                    # XTTS API: POST /tts with JSON body
                    # base_url might be http://localhost:9010 or http://localhost:9010/tts
                    # Ensure it ends with /tts
                    base = self.base_url.rstrip("/")
                    if not base.endswith("/tts"):
                        base = f"{base}/tts"
                    url = base
                    params = {"text": text, "voice": self.voice_id}
                    response = self.session.post(url, json=params, timeout=120)

                response.raise_for_status()

                # Check if response contains valid audio data
                content_length = len(response.content)
                if content_length == 0:
                    logger.error(f"TTS response is empty for text '{text[:50]}...'")
                    return False
                
                # Check content type to ensure it's audio
                content_type = response.headers.get("content-type", "").lower()
                if "audio" not in content_type and content_length < 100:
                    # Very small responses are likely error messages, not audio
                    logger.warning(
                        f"TTS response may not be audio (content-type: {content_type}, "
                        f"size: {content_length} bytes) for text '{text[:50]}...'"
                    )
                    # Try to log the response content if it's small enough
                    if content_length < 1000:
                        try:
                            response_text = response.content.decode('utf-8', errors='ignore')
                            logger.warning(f"Response content: {response_text[:200]}")
                        except Exception:
                            pass

                # Save audio (assuming response is audio data)
                with open(output_path, "wb") as f:
                    f.write(response.content)

                if not output_path.exists() or output_path.stat().st_size == 0:
                    logger.error(f"Failed to save TTS audio for text '{text[:50]}...'")
                    return False

                logger.debug(f"TTS generation successful: {content_length} bytes for text '{text[:50]}...'")
                return True
                
            except requests.exceptions.HTTPError as e:
                # HTTP errors (4xx, 5xx) - don't retry, these are server-side issues
                logger.error(f"HTTP TTS {e.response.status_code} error for text '{text[:50]}...': {e}")
                try:
                    error_detail = e.response.json().get("detail", "No details available")
                    logger.error(f"Server error details: {error_detail}")
                except Exception:
                    logger.error(f"Server response: {e.response.text[:500]}")
                return False
                
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
                # Connection/timeout errors - retry with backoff
                if attempt < max_retries - 1:
                    logger.warning(
                        f"HTTP TTS connection error (attempt {attempt + 1}/{max_retries}) "
                        f"for text '{text[:50]}...': {e}. Retrying in {retry_delay}s..."
                    )
                    import time
                    time.sleep(retry_delay)
                    retry_delay *= 1.5  # Exponential backoff
                    continue
                else:
                    # Final attempt failed
                    logger.error(f"HTTP TTS connection error after {max_retries} attempts for text '{text[:50]}...': {e}")
                    if isinstance(self.teacher_config, XTTSTeacherConfig):
                        logger.warning("This may indicate server overload or GPU memory exhaustion. Consider reducing workers.")
                    return False
                    
            except Exception as e:
                # Other errors - don't retry
                logger.error(f"HTTP TTS error for text '{text[:50]}...': {e}")
                return False
        
        # Should never reach here, but just in case
        return False


def normalize_audio_to_piper(input_path: Path, output_path: Path) -> bool:
    """Normalize audio to Piper format (22.05 kHz mono 16-bit PCM) with high-quality resampling."""
    try:
        # Try soxr resampler (best quality) with volume normalization
        (
            ffmpeg.input(str(input_path))
            .output(
                str(output_path),
                acodec="pcm_s16le",
                ac=1,
                ar=22050,
                af="aresample=resampler=soxr:precision=28:cheby=1,volume=-3dB",
                loglevel="error",
            )
            .overwrite_output()
            .run(quiet=True)
        )
        return True
    except ffmpeg.Error:
        # Fallback to swr resampler if soxr not available
        try:
            (
                ffmpeg.input(str(input_path))
                .output(
                    str(output_path),
                    acodec="pcm_s16le",
                    ac=1,
                    ar=22050,
                    af="aresample=resampler=swr:osr=22050,volume=-3dB",
                    loglevel="error",
                )
                .overwrite_output()
                .run(quiet=True)
            )
            return True
        except ffmpeg.Error as e2:
            error_msg = e2.stderr.decode() if e2.stderr else str(e2)
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
    try:
        if not tts_backend.generate(text, raw_audio_path):
            logger.warning(
                f"TTS generation failed for entry {index}: '{text[:50]}...' "
                f"(full text: '{text}')"
            )
            raw_audio_path.unlink(missing_ok=True)
            return None
    except Exception as e:
        logger.error(
            f"Exception during TTS generation for entry {index}: {e}",
            exc_info=True
        )
        logger.error(f"Failed text was: '{text}'")
        raw_audio_path.unlink(missing_ok=True)
        return None

    # Normalize to Piper format
    synth_id = f"synth_{index:04d}"
    normalized_path = synth_dir / f"{synth_id}.wav"
    if not normalize_audio_to_piper(raw_audio_path, normalized_path):
        logger.warning(f"Audio normalization failed for entry {index} ({synth_id})")
        raw_audio_path.unlink(missing_ok=True)
        return None

    raw_audio_path.unlink()

    # Verify quality
    is_valid, distance = verify_synthetic_entry(text, normalized_path, config)
    if not is_valid:
        logger.warning(
            f"Rejected synthetic entry {synth_id}: phoneme distance={distance:.4f} "
            f"(threshold={config.phoneme_check.max_phoneme_distance})"
        )
        logger.warning(f"  Original text: '{text}'")
        normalized_path.unlink()
        return None

    logger.info(f"✓ Generated valid entry {synth_id}: distance={distance:.4f}")
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
        error_msg = "No sentences loaded from corpus"
        logger.error(error_msg)
        raise RuntimeError(error_msg)

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
            elif teacher_kind == "openvoice":
                from tools.openvoice_teacher_orchestration import (
                    start_openvoice_teacher,
                    stop_openvoice_teacher,
                )

                logger.info("Starting OpenVoice teacher model service...")
                teacher_config = config.synthetic.teacher
                if isinstance(teacher_config, OpenVoiceTeacherConfig):
                    logger.info(
                        f"OpenVoice config: language={teacher_config.language}, "
                        f"base_speaker_key={teacher_config.base_speaker_key}, "
                        f"device={teacher_config.device}"
                    )
                start_openvoice_teacher(config)
                teacher_started = True
                logger.info("OpenVoice teacher model service started successfully")
            else:
                raise ValueError(f"Unsupported teacher kind: {teacher_kind}")
        except Exception as e:
            logger.error(f"Failed to start teacher model service: {e}")
            logger.error("Synthetic dataset generation aborted")
            raise RuntimeError(f"Failed to start teacher model service: {e}") from e

    try:
        # Create TTS backend
        if config.synthetic.tts_backend == "http":
            tts_backend = HTTPTTSBackend(
                config.synthetic.tts_http.base_url,
                config.synthetic.tts_http.voice_id,
                teacher_config=config.synthetic.teacher,
            )
        else:
            error_msg = f"Unsupported TTS backend: {config.synthetic.tts_backend}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Generate synthetic entries in parallel
        logger.info(f"Generating {len(sentences)} synthetic entries...")
        valid_entries = []
        failed_entries = []  # Track failed entries for summary
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
            # Log every N entries to show progress even with progress bars
            log_interval = max(1, total // 20)  # Log every 5% or at least every entry
            for future in as_completed(futures):
                completed += 1
                i, text = futures[future]
                try:
                    result = future.result()
                    if result:
                        valid_entries.append(result)
                        if completed % log_interval == 0 or completed == total:
                            logger.info(f"Progress: {completed}/{total} entries processed ({len(valid_entries)} valid, {len(failed_entries)} failed)")
                    else:
                        failed_entries.append((i, text, "Generation or verification failed"))
                        # Log failures immediately so they're visible
                        logger.warning(f"Entry {i} failed: '{text[:80]}...'")
                        if completed % log_interval == 0 or completed == total:
                            logger.info(f"Progress: {completed}/{total} entries processed ({len(valid_entries)} valid, {len(failed_entries)} failed)")
                except Exception as e:
                    failed_entries.append((i, text, f"Exception: {e}"))
                    logger.error(f"Error generating entry {i}: {e}")
                    logger.error(f"  Failed text: '{text[:80]}...'")
                    if completed % log_interval == 0 or completed == total:
                        logger.info(f"Progress: {completed}/{total} entries processed ({len(valid_entries)} valid, {len(failed_entries)} failed)")

        # Check if any files were generated before writing metadata
        wav_files = list(wavs_dir.glob("*.wav"))
        if wav_files and len(valid_entries) == 0:
            logger.warning(f"Found {len(wav_files)} WAV files in {wavs_dir} but 0 valid entries!")
            logger.warning("This suggests all entries failed verification. Check rejection logs above.")

        # Write metadata
        logger.info(f"Writing metadata with {len(valid_entries)} entries")
        with open(metadata_csv, "w", encoding="utf-8") as f:
            for wav_path, text in valid_entries:
                # Escape pipe characters
                text_escaped = text.replace("|", " ")
                f.write(f"{wav_path}|{text_escaped}\n")

        # Summary - use error level to ensure it's visible
        logger.error("=" * 60)
        logger.error("Synthetic Dataset Generation Summary")
        logger.error("=" * 60)
        logger.error(f"Total entries attempted: {len(sentences)}")
        logger.error(f"Valid entries: {len(valid_entries)}")
        logger.error(f"Failed entries: {len(failed_entries)}")
        if failed_entries:
            logger.error("")
            logger.error("Failed entries (first 20):")
            for i, text, reason in failed_entries[:20]:
                logger.error(f"  Entry {i}: {reason}")
                logger.error(f"    Text: '{text[:150]}'")
        logger.error("")
        logger.error(f"Synthetic dataset built: {len(valid_entries)} entries in {synth_dataset_dir}")
        logger.error("=" * 60)

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
                            logger.info("Container logs (stdout):")
                            for line in result.stdout.split("\n")[-100:]:  # Last 100 lines
                                if line.strip():
                                    logger.info(f"  {line}")
                        if result.stderr:
                            logger.info("Container logs (stderr):")
                            for line in result.stderr.split("\n")[-100:]:  # Last 100 lines
                                if line.strip():
                                    logger.info(f"  {line}")
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
                elif teacher_kind == "openvoice":
                    from tools.openvoice_teacher_orchestration import stop_openvoice_teacher

                    stop_openvoice_teacher(config)
                logger.info("Teacher model service stopped")
            except Exception as e:
                logger.warning(f"Error stopping teacher model service: {e}")

    # Now run phoneme verification on the synthetic dataset
    logger.info("Running phoneme verification on synthetic dataset...")
    from tools.verify_phonemes import verify_dataset
    # Pass synthetic dataset directory directly instead of modifying config
    verify_dataset(config, dataset_dir=synth_dataset_dir)


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

