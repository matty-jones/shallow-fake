"""Phoneme-based sanity checker for dataset quality."""

import json
import shutil
import subprocess
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Tuple

from faster_whisper import WhisperModel
from piper_phonemize import phonemize_espeak

from shallow_fake.config import VoiceConfig
from shallow_fake.utils import ensure_dir, setup_logging

logger = setup_logging()

# Thread-local storage for Whisper models (one per worker thread)
_thread_local = threading.local()
# Lock for model initialization to prevent CUDA kernel errors
# when multiple threads try to initialize models simultaneously
_model_init_lock = threading.Lock()


def phonemize_text(text: str, language: str) -> str:
    """Convert text to phoneme sequence using piper-phonemize."""
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
        logger.error(f"Error phonemizing text '{text}': {e}")
        return ""


def synthesize_with_piper(text: str, piper_model_path: str, output_path: Path) -> bool:
    """Synthesize text using Piper TTS."""
    try:
        # Use piper command-line tool to synthesize
        # Format: echo "text" | piper --model model.onnx --output_file output.wav
        cmd = [
            "piper",
            "--model", piper_model_path,
            "--output_file", str(output_path),
        ]
        process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        stdout, stderr = process.communicate(input=text)
        if process.returncode != 0:
            logger.error(f"Piper synthesis failed: {stderr}")
            return False
        return output_path.exists()
    except FileNotFoundError:
        logger.error("piper command not found. Please install piper-tts.")
        return False
    except Exception as e:
        logger.error(f"Error synthesizing with Piper: {e}")
        return False


def get_whisper_model(model_size: str = "base.en", device: str = "cpu") -> WhisperModel:
    """Get or create a Whisper model for the current thread (thread-local)."""
    # Use device as part of the cache key to support both CPU and CUDA models
    cache_key = f"{model_size}_{device}"
    
    if not hasattr(_thread_local, 'whisper_models'):
        _thread_local.whisper_models = {}
    
    if cache_key not in _thread_local.whisper_models:
        # Use lock to serialize model initialization and prevent CUDA kernel errors
        # when multiple threads try to initialize models simultaneously
        with _model_init_lock:
            # Double-check after acquiring lock (another thread might have initialized)
            if cache_key not in _thread_local.whisper_models:
                # Use appropriate compute type based on device
                compute_type = "float16" if device == "cuda" else "int8"
                _thread_local.whisper_models[cache_key] = WhisperModel(
                    model_size, 
                    device=device, 
                    compute_type=compute_type
                )
    return _thread_local.whisper_models[cache_key]


def transcribe_with_whisper(audio_path: Path, model_size: str = "base.en", device: str = "cpu") -> str:
    """Transcribe audio using Whisper (reuses thread-local model)."""
    try:
        model = get_whisper_model(model_size, device)
        segments, _ = model.transcribe(str(audio_path), language="en")
        # Concatenate all segments
        text = " ".join(segment.text for segment in segments)
        return text.strip()
    except Exception as e:
        logger.error(f"Error transcribing with Whisper: {e}")
        return ""


def normalized_edit_distance(s1: str, s2: str) -> float:
    """Compute normalized Levenshtein distance between two strings."""
    if not s1 and not s2:
        return 0.0
    if not s1 or not s2:
        return 1.0

    # Convert to lists of tokens (phonemes)
    tokens1 = s1.split()
    tokens2 = s2.split()

    # Levenshtein distance
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
                    dp[i - 1][j] + 1,  # deletion
                    dp[i][j - 1] + 1,  # insertion
                    dp[i - 1][j - 1] + 1,  # substitution
                )

    max_len = max(m, n)
    return dp[m][n] / max_len if max_len > 0 else 0.0


def verify_dataset_entry(
    canonical_text: str,
    audio_path: Path,
    config: VoiceConfig,
    baseline_piper_model: str = None,
    use_tts_roundtrip: bool = True,
) -> Tuple[bool, float, str]:
    """
    Verify a dataset entry using phoneme comparison.

    Returns: (is_valid, phoneme_distance, whisper_text)
    """
    if use_tts_roundtrip and baseline_piper_model:
        # Synthesize canonical text with baseline Piper
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            synth_path = Path(tmp.name)

        if not synthesize_with_piper(canonical_text, baseline_piper_model, synth_path):
            logger.warning(f"Failed to synthesize text, skipping verification")
            return False, 1.0, ""

        # Transcribe synthesized audio
        # Use config device (default to CPU for safety, but allow CUDA if configured)
        device = getattr(config.asr, 'device', 'cpu')
        whisper_text = transcribe_with_whisper(
            synth_path,
            model_size="base.en",
            device=device,
        )
        synth_path.unlink()  # Clean up
    else:
        # Use the actual audio file
        # Use config device (default to CPU for safety, but allow CUDA if configured)
        device = getattr(config.asr, 'device', 'cpu')
        whisper_text = transcribe_with_whisper(
            audio_path,
            model_size="base.en",
            device=device,
        )

    if not whisper_text:
        logger.warning(f"Failed to transcribe audio, skipping verification")
        return False, 1.0, ""

    # Phonemize both texts
    canonical_phonemes = phonemize_text(canonical_text, config.phoneme_check.language)
    whisper_phonemes = phonemize_text(whisper_text, config.phoneme_check.language)

    if not canonical_phonemes or not whisper_phonemes:
        logger.warning(f"Failed to phonemize text, skipping verification")
        return False, 1.0, whisper_text

    # Compute distance
    distance = normalized_edit_distance(canonical_phonemes, whisper_phonemes)
    is_valid = distance <= config.phoneme_check.max_phoneme_distance

    return is_valid, distance, whisper_text


def verify_entry_worker(
    entry_data: Tuple[int, str, Path, Path, VoiceConfig, str, bool]
) -> Tuple[int, bool, float, str, str, Path]:
    """
    Worker function for parallel verification.
    
    Returns: (index, is_valid, distance, whisper_text, wav_path, audio_path)
    """
    i, wav_path, text, audio_path, config, baseline_piper_model, use_tts_roundtrip = entry_data
    
    if not audio_path.exists():
        logger.warning(f"Audio file not found: {audio_path}")
        return (i, False, 1.0, "", wav_path, audio_path)
    
    is_valid, distance, whisper_text = verify_dataset_entry(
        text,
        audio_path,
        config,
        baseline_piper_model,
        use_tts_roundtrip,
    )
    
    return (i, is_valid, distance, whisper_text, wav_path, audio_path)


def verify_dataset(config: VoiceConfig, baseline_piper_model: str = None):
    """Verify dataset using phoneme-based quality checks (parallelized)."""
    real_dataset_dir = config.paths.real_dataset_dir
    metadata_csv = real_dataset_dir / "metadata.csv"
    wavs_dir = real_dataset_dir / "wavs"

    if not metadata_csv.exists():
        logger.error(f"Metadata file not found: {metadata_csv}")
        raise FileNotFoundError(f"Metadata file not found: {metadata_csv}")

    # Create cleaned dataset directory
    cleaned_dir = real_dataset_dir.parent / f"{real_dataset_dir.name}_clean"
    cleaned_wavs_dir = cleaned_dir / "wavs"
    cleaned_metadata_csv = cleaned_dir / "metadata.csv"
    ensure_dir(cleaned_wavs_dir)

    # Read metadata
    entries = []
    with open(metadata_csv, "r", encoding="utf-8") as f:
        for line in f:
            if "|" in line:
                parts = line.strip().split("|", 1)
                if len(parts) == 2:
                    wav_path, text = parts
                    entries.append((wav_path, text))

    logger.info(f"Verifying {len(entries)} dataset entries")
    
    # Get parallel workers from config
    num_workers = config.phoneme_check.parallel_workers
    logger.info(f"Using {num_workers} parallel workers for verification")
    
    # Pre-warm Whisper model initialization in main thread to avoid CUDA conflicts
    # This ensures the library is ready before parallel processing starts
    device = getattr(config.asr, 'device', 'cpu')
    compute_type = "float16" if device == "cuda" else "int8"
    logger.debug(f"Pre-initializing Whisper library on {device}...")
    try:
        # Initialize one model in main thread to ensure library is ready
        # This helps avoid CUDA initialization race conditions when multiple
        # threads start initializing models simultaneously
        test_model = WhisperModel("base.en", device=device, compute_type=compute_type)
        del test_model  # Clean up immediately
        logger.debug(f"Whisper library pre-initialized on {device}")
    except Exception as e:
        logger.warning(f"Could not pre-initialize Whisper library (non-critical): {e}")

    # Prepare entry data for workers (store original text for later use)
    entry_data_list = []
    original_texts = {}  # Store original texts by index
    for i, (wav_path, text) in enumerate(entries):
        audio_path = wavs_dir / Path(wav_path).name
        original_texts[i] = text
        entry_data_list.append((
            i,
            wav_path,
            text,
            audio_path,
            config,
            baseline_piper_model,
            config.phoneme_check.use_tts_roundtrip,
        ))

    # Process entries in parallel
    valid_entries = []
    invalid_entries = []
    distances = []
    results_dict = {}  # Store results by index to maintain order
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Submit all tasks
        futures = {
            executor.submit(verify_entry_worker, entry_data): entry_data[0]
            for entry_data in entry_data_list
        }
        
        # Process completed tasks
        completed = 0
        for future in as_completed(futures):
            completed += 1
            try:
                i, is_valid, distance, whisper_text, wav_path, audio_path = future.result()
                results_dict[i] = (is_valid, distance, whisper_text, wav_path, audio_path)
                
                # Log progress every 10 entries or at milestones
                if completed % 10 == 0 or completed == len(entries):
                    logger.info(f"Processing entry {completed}/{len(entries)}")
            except Exception as e:
                logger.error(f"Error processing entry: {e}")
                # Mark as invalid
                original_index = futures[future]
                results_dict[original_index] = (False, 1.0, "", "", None)

    # Process results in order and copy files
    for i in range(len(entries)):
        if i not in results_dict:
            continue
            
        is_valid, distance, whisper_text, wav_path, audio_path = results_dict[i]
        original_text = original_texts[i]
        
        distances.append(distance)
        
        if is_valid and audio_path:
            # Copy to cleaned dataset
            dest_wav = cleaned_wavs_dir / audio_path.name
            shutil.copy2(audio_path, dest_wav)
            valid_entries.append((wav_path, original_text))
        else:
            invalid_entries.append((wav_path, original_text, distance, whisper_text))

    # Write cleaned metadata
    logger.info(f"Writing cleaned metadata with {len(valid_entries)} entries")
    with open(cleaned_metadata_csv, "w", encoding="utf-8") as f:
        for wav_path, text in valid_entries:
            f.write(f"{wav_path}|{text}\n")

    # Generate report
    logger.info("=" * 60)
    logger.info("Phoneme Verification Report")
    logger.info("=" * 60)
    logger.info(f"Total entries: {len(entries)}")
    logger.info(f"Valid entries: {len(valid_entries)}")
    logger.info(f"Rejected entries: {len(invalid_entries)}")
    if distances:
        logger.info(f"Average phoneme distance: {sum(distances) / len(distances):.4f}")
        logger.info(f"Min distance: {min(distances):.4f}")
        logger.info(f"Max distance: {max(distances):.4f}")
    logger.info(f"Cleaned dataset: {cleaned_dir}")
    logger.info("=" * 60)

    if invalid_entries:
        logger.info(f"\nRejected entries (first 10):")
        for wav_path, text, distance, whisper_text in invalid_entries[:10]:
            logger.info(f"  {wav_path}: distance={distance:.4f}")
            logger.info(f"    Original: {text}")
            logger.info(f"    Whisper:  {whisper_text}")
        
        logger.info("")
        logger.info("=" * 60)
        logger.info("Manual Recovery Instructions")
        logger.info("=" * 60)
        logger.info("If you agree with the Original transcription (not the Whisper transcription),")
        logger.info("you can manually add these rejected entries to the cleaned dataset.")
        logger.info("")
        logger.info("Run the following commands to add all rejected entries:")
        logger.info("")
        
        # Generate commands for all rejected entries
        for wav_path, text, distance, whisper_text in invalid_entries:
            logger.info(f'echo "{wav_path}|{text}" >> {cleaned_metadata_csv}')
            logger.info(f"cp {wavs_dir / Path(wav_path).name} {cleaned_wavs_dir / Path(wav_path).name}")
        
        logger.info("")
        logger.info("=" * 60)


if __name__ == "__main__":
    import sys
    from pathlib import Path

    if len(sys.argv) < 2:
        print("Usage: python verify_phonemes.py <config_path> [--baseline-model <path>]")
        sys.exit(1)

    config_path = Path(sys.argv[1])
    config = VoiceConfig.from_yaml(config_path)
    config.ensure_directories()

    baseline_model = None
    args = sys.argv[2:]
    i = 0
    while i < len(args):
        if args[i] == "--baseline-model" and i + 1 < len(args):
            baseline_model = args[i + 1]
            i += 2
        else:
            i += 1

    verify_dataset(config, baseline_model)


