# Voice Piper Clone – Product Spec & Technical Design

## 1. Overview

### 1.1 Problem

You want a **repeatable, low‑touch pipeline** that can:

- Take a handful of existing recordings of your voice (talks, demos, etc.).
- Automatically:
  - segment and transcribe them,
  - clean and verify the resulting text–audio pairs,
  - expand them into a larger synthetic corpus in your voice,
- Then fine‑tune a **Piper** base model via **TextyMcSpeechy (TMS)** to produce a
  **Piper‑compatible ONNX model** that sounds as close as possible to you.

No manual phoneme entry, no hand‑aligning 2,000 clips. Heavy lifting is done
by ASR + phonemisation + synthetic data and a training container.

### 1.2 Goals

- **G1 – One‑command training runs.** Given a config file and a folder of raw audio,
  a single CLI command should run the complete pipeline:
  `raw audio → segments → transcripts → dataset → synthetic expansion → training → ONNX`.
- **G2 – Strong automation.** No manual transcription or phoneme work.
  All alignment/quality checks are automated.
- **G3 – Reusable architecture.** The pipeline should be reusable for other
  voices in future by changing config and input data.
- **G4 – HA/Piper ready.** Output must be a `.onnx` + `.onnx.json` pair that
  works out‑of‑the‑box with `piper-tts` and Home Assistant’s Piper integration.
- **G5 – Training feedback.** You should be able to inspect loss curves and
  listen to training samples while TextyMcSpeechy runs.

### 1.3 Non‑Goals

- No web UI or front‑end beyond whatever TextyMcSpeechy already ships.
- No deployment of the resulting voice into Home Assistant; repo stops
  at “produce a valid Piper voice model”.

---

## 2. High‑Level Architecture

### 2.1 Data Flow

1. **Raw audio ingestion**
   - Input: long recordings (talks, screen‑captures, etc.) in arbitrary formats.
   - Output: normalised WAV files (mono, 22.05 kHz, 16‑bit).

2. **Segmentation + ASR**
   - Use **Whisper/faster‑whisper** to:
     - detect speech segments,
     - produce English transcripts for each segment.
   - Output: `segments/` directory of short WAVs + a JSONL file
     (`segment_id`, `audio_path`, `text`, `duration`, `confidence`, etc.).

3. **Dataset builder**
   - Filter segments (duration, confidence, basic text rules).
   - Produce a Piper‑style `metadata.csv`:
     ```text
     wavs/seg_0001.wav|This is what I said in that segment.
     ```

4. **Phoneme‑based sanity checker**
   - Use **piper-phonemize** (espeak‑ng backend) to convert:
     - the canonical text,
     - a Whisper‑re‑transcription of TTS‑rendered canonical text,
     into phoneme sequences.
   - Reject pairs where phoneme distance exceeds a threshold.

5. **Synthetic data expansion**
   - Use a configurable TTS/VC backend to generate **synthetic voice‑ish audio**
     from a large English text corpus.
   - Use the same Whisper + phoneme verification to keep only high‑quality pairs.
   - Append accepted synthetic entries to `metadata.csv`.

6. **Training via TextyMcSpeechy (TMS)**
   - Copy dataset into TMS’s expected dataset layout.
   - Use a TMS config (TTS Dojo) to:
     - choose a base Piper checkpoint,
     - configure training schedule,
     - launch training inside TMS’s **Docker training container**.

7. **Monitoring**
   - Monitor training via:
     - Container logs (loss per epoch, etc.).
     - Optional TensorBoard port exposed from training container.
     - TMS’s own “listen to your model as it is training” audio samples.

8. **Export**
   - Use Piper’s `piper_train.export_onnx` (or TMS’s built‑in export)
     to produce a `voice.onnx` + `voice.onnx.json` pair.

9. **Evaluation**
   - Generate a standard set of sample phrases and write them to `samples/`
     for quick subjective evaluation.

---

## 3. Tech Stack & Dependencies

### 3.1 Languages & Runtimes

- **Python** 3.10 or 3.11 on the host.
- **Docker** + NVIDIA Container Toolkit for GPU access.
- Optional: **Poetry** or `uv` for Python dependency management.

### 3.2 Core Third‑Party Components

1. **Piper (TTS framework)**
   - Training docs and export guidance from the archived Rhasspy Piper repo
     and the successor `OHF-Voice/piper1-gpl`.  citeturn0search1turn0search13  
   - Training expectations:
     - Dataset in `metadata.csv` (LJSpeech format). citeturn0search1turn0search23  
     - Training via `piper_train` with PyTorch Lightning.
     - Export via `python3 -m piper_train.export_onnx checkpoint.ckpt model.onnx`. citeturn0search1turn0search15turn0search23  

2. **TextyMcSpeechy (TMS)**
   - GitHub: `domesticatedviking/TextyMcSpeechy`. citeturn0search0  
   - Provides:
     - A **Dockerised TTS Dojo** with Piper training built in.
     - Scripts to organise datasets and checkpoints, and **generate
       Piper‑compatible voices** ready for Home Assistant. citeturn0search10turn0search14  
     - Ability to listen to model output during training and manage
       multiple datasets/checkpoints.

3. **Whisper / faster‑whisper (ASR)**
   - Used to auto‑segment and transcribe source recordings.
   - Also used to transcribe synthetic TTS audio for phoneme‑based quality checks,
     mirroring Cal Bryant’s “4‑word voice” pipeline where Whisper is used
     to validate synthetic training data. citeturn0search2turn0search3turn0search11  

4. **piper-phonemize + espeak‑ng**
   - Python library for converting text to phoneme sequences with the same
     phonemisation logic used by Piper. citeturn0search1turn0search9  
   - Used to compare phoneme strings between:
     - canonical text,
     - Whisper‑decoded synthetic audio,
     to drop mismatched pairs as in Cal’s phoneme‑distance filtering. citeturn0search2turn0search11  

5. **Audio tools**
   - `ffmpeg` or `sox` for:
     - resampling to 22.05 kHz mono,
     - volume normalisation,
     - optional silence trimming.

6. **Synthetic TTS / VC backend (pluggable)**
   - Abstracted behind a simple Python interface, e.g.:
     - `generate_tts(text: str) -> wav_path`
   - The initial implementation can target:
     - A local HTTP TTS API (e.g. any self‑hosted cloning stack).
     - Or an RVC / VC pipeline output directory.
   - This mirrors Cal’s use of Chatterbox TTS as the “teacher”
     to generate synthetic training data. citeturn0search2turn0search3turn0search11  

7. **Monitoring**
   - TextyMcSpeechy’s TTS Dojo logs and audio preview tools. citeturn0search10turn0search17  
   - TensorBoard (if exposed) to inspect `lightning_logs` from training. citeturn0search1turn0search15  

### 3.3 Container Stack

- **Training container**
  - Base: TMS Piper training image (from TMS docs) or a custom image derived from:
    - `pytorch/pytorch:2.2.2-cuda11.8-cudnn8-devel` (or similar),
    - Python 3.10,
    - `pytorch-lightning==1.9.x` (known good with Piper). citeturn0search1turn0search9turn0search18  
  - Contains:
    - Piper training code (`piper_train`),
    - `piper-phonemize`,
    - monotonic alignment op,
    - TMS TTS Dojo utilities.

---

## 4. Repository Structure

Proposed repo layout:

```text
ShallowFaker/
  README.md
  pyproject.toml / requirements.txt
  docker/
    Dockerfile.training          # (optional) if not using TMS image directly
    docker-compose.training.yml  # wires up TMS training container + volumes
  config/
    voice1.yaml                   # main pipeline config for your voice
    tms_training.yaml            # template for TMS / TTS Dojo
  data_raw/
    input_audio/                 # original long-form recordings
    external_corpus/             # text corpora for synthetic expansion
  data_processed/
    normalized/                  # resampled long-form WAVs
    segments/                    # short segments from ASR
    asr_segments.jsonl           # Whisper output metadata
  datasets/
    voice1/
      real/                      # real-voice dataset (wavs + metadata.csv)
      synth/                     # synthetic dataset (wavs + metadata.csv)
      combined/                  # merged dataset used for training
  tms_workspace/
    datasets/                    # mounted into TMS container
    checkpoints/
    logs/
    audio_samples/               # training audio previews
  training/
    scripts/                     # wrappers to call TMS / piper_train
  models/
    voice1/                       # final ONNX + JSON exported voice
  samples/
    voice1/                       # evaluation WAVs from final model
  tools/
    asr_segment.py
    build_metadata.py
    verify_phonemes.py
    build_synthetic_dataset.py
    launch_training.py
    monitor_training.py
    export_onnx.py
    eval_model.py
```

---

## 5. Detailed Features & Requirements

### 5.1 Config System

Single YAML config per voice, e.g. `config/voice1.yaml`:

```yaml
voice_id: <person name>
language: en_GB

paths:
  raw_audio_dir: data_raw/input_audio
  normalized_dir: data_processed/normalized
  segments_dir: data_processed/segments
  asr_metadata: data_processed/asr_segments.jsonl
  real_dataset_dir: datasets/voice1/real
  synth_dataset_dir: datasets/voice1/synth
  combined_dataset_dir: datasets/voice1/combined
  tms_workspace_dir: tms_workspace
  output_models_dir: models/voice1

asr:
  model_size: medium.en
  device: cuda
  beam_size: 5
  max_segment_seconds: 15
  min_segment_seconds: 1.0
  min_confidence: 0.7

phoneme_check:
  language: en-gb
  max_phoneme_distance: 0.1  # e.g. normalised Levenshtein distance
  use_tts_roundtrip: true    # render TTS from canonical text before phonemisation

synthetic:
  enabled: true
  corpus_text_path: data_raw/external_corpus/corpus.txt
  max_sentences: 2000
  tts_backend: http
  tts_http:
    base_url: http://localhost:9000/tts
    voice_id: voice1_clone
  max_parallel_jobs: 4

training:
  base_checkpoint: en_GB-base-medium.ckpt
  batch_size: 32
  max_epochs: 1000
  quality: medium
  accelerator: gpu
  devices: 1

tms:
  enable_tts_dojo: true
  docker_compose_file: docker/docker-compose.training.yml
  project_name: voice1-voice
  expose_tensorboard: true
  tensorboard_port: 6006
```

### 5.2 Data Ingestion & Normalisation

**Tool:** `tools/asr_segment.py`

Responsibilities:

- Walk `raw_audio_dir` and:
  - Convert everything to `normalized_dir` using `ffmpeg`:
    - PCM 16‑bit, mono, 22,050 Hz.
- Run Whisper (or faster‑whisper) with:
  - language forced to English,
  - model size chosen via config,
  - segmenting on silence / ASR timestamps.
- Emit:
  - `segments/seg_XXXX.wav` cropped using Whisper timestamps.
  - `asr_segments.jsonl` with entries:
    ```json
    {
      "id": "seg_0001",
      "audio_path": "data_processed/segments/seg_0001.wav",
      "text": "This is an example",
      "start": 12.3,
      "end": 15.6,
      "duration": 3.3,
      "confidence": 0.92
    }
    ```

Filtering logic:

- Drop segments shorter than `min_segment_seconds` or longer than `max_segment_seconds`.
- Drop segments where `confidence < min_confidence` (if Whisper backend provides it).
- Optionally drop segments with:
  - too many non‑ASCII symbols,
  - abnormal number of numbers/punctuation.

### 5.3 Dataset Builder

**Tool:** `tools/build_metadata.py`

Responsibilities:

- Read `asr_segments.jsonl`.
- Apply further filtering (e.g. minimum word count).
- Copy/canonicalise selected segment WAVs into:
  - `datasets/voice1/real/wavs/`.
- Create `datasets/voice1/real/metadata.csv` with LJSpeech format:
  ```text
  wavs/seg_0001.wav|This is what I said in that segment.
  wavs/seg_0002.wav|Here is another thrilling sentence.
  ```

Configurable knobs:

- `min_words`, `max_words`.
- Max number of segments to include (just in case you have a huge corpus).

### 5.4 Phoneme‑Based Sanity Checker

**Tool:** `tools/verify_phonemes.py`

Pipeline (for each dataset entry):

1. Take canonical text `T`.
2. Use a **baseline Piper voice** (e.g. `en_GB-alan`) to synthesise `T` → audio `A_TTS`.
3. Run Whisper on `A_TTS` → `T_whisper`.
4. Use `piper_phonemize` to compute phoneme sequences:
   - `P(T)` and `P(T_whisper)`.
5. Compute normalised edit distance between `P(T)` and `P(T_whisper)`.

If distance > threshold (e.g. > 0.1):

- Mark this line as **bad** and drop it from `metadata.csv`.

This mirrors Cal Bryant’s approach of using phoneme comparison to reject
misgenerated synthetic lines, but applied generically to both real and synthetic
datasets. citeturn0search2turn0search11  

Outputs:

- Cleaned dataset directory (e.g. `datasets/voice1/real_clean/`).
- A report summarising:
  - total lines,
  - kept vs rejected,
  - phoneme distance distribution.

### 5.5 Synthetic Data Expansion

**Tool:** `tools/build_synthetic_dataset.py`

Inputs:

- Configured **text corpus** (`synthetic.corpus_text_path`).
- Pluggable TTS/VC backend.

Steps:

1. **Load corpus** (one sentence per line, or split long paragraphs).
2. For each sentence:
   - Call TTS backend to produce synthetic voice‑ish audio:
     - either via HTTP,
     - or via a CLI wrapper around an RVC / TTS engine.
   - Normalise audio to 22.05 kHz mono.
   - Save as `datasets/voice1/synth/wavs/synth_XXXX.wav`.
3. Build `datasets/voice1/synth/metadata.csv` in LJSpeech format.
4. Run **phoneme‑based sanity check** just like for real data:
   - Whisper decode the synthetic audio.
   - Compare phoneme strings of target text vs Whisper transcript.
   - Drop mismatches.

Outputs:

- `datasets/voice1/synth_clean/` with `wavs/` + cleaned `metadata.csv`.

### 5.6 Combined Dataset

**Tool:** `tools/combine_datasets.py`

Responsibilities:

- Merge:
  - `voice1/real_clean/metadata.csv`
  - `voice1/synth_clean/metadata.csv`
  into `datasets/voice1/combined/metadata.csv`.
- Optionally:
  - Down‑weight synthetic examples (e.g. by subsampling if they vastly outnumber real data).

---

## 6. Training Integration (TextyMcSpeechy)

### 6.1 TMS Workspace Layout

This repo will treat `tms_workspace/` as a bind‑mount into the TMS container,
matching its TTS Dojo expectations: citeturn0search0turn0search10turn0search14  

```text
tms_workspace/
  datasets/
    voice1/
      combined/
        wavs/
        metadata.csv
  checkpoints/
    base_checkpoints/
      en_GB-base-medium.ckpt
  logs/
  audio_samples/
```

### 6.2 Training Launch Script

**Tool:** `tools/launch_training.py`

Responsibilities:

- Read `config/voice1.yaml`.
- Ensure `tms_workspace/datasets/voice1/combined` exists and is populated.
- Ensure base checkpoint is available in `tms_workspace/checkpoints/base_checkpoints/`.
- Call `docker compose` using `docker/docker-compose.training.yml` with:
  - GPU access (`runtime: nvidia` / `--gpus all`),
  - volume mounts for `tms_workspace`,
  - environment variables for training config (dataset name, base checkpoint, etc.).

Example `docker-compose.training.yml` (conceptual):

```yaml
services:
  tms-trainer:
    image: domesticatedviking/textymcspeechy:piper-latest
    container_name: tms-voice1-trainer
    restart: unless-stopped
    environment:
      - TMS_DATASETS=/workspace/datasets
      - TMS_CHECKPOINTS=/workspace/checkpoints
      - TMS_LOGS=/workspace/logs
      - TMS_AUDIO_SAMPLES=/workspace/audio_samples
    volumes:
      - ../tms_workspace:/workspace
    deploy:
      resources:
        reservations:
          devices:
            - capabilities: [gpu]
    ports:
      - "6006:6006"   # TensorBoard (optional)
      - "8765:8765"   # any TMS web UI, if present
```

The actual command inside the container can be something like:

```bash
python3 tts_dojo/train_piper.py --dataset voice1/combined --base-checkpoint en_GB-base-medium.ckpt
```

(Exact script name/args to be aligned with TMS’ latest TTS Dojo docs.)

### 6.3 Training Monitoring

**Tool:** `tools/monitor_training.py`

Functionality:

- Tail logs from `tms_workspace/logs/` (or from `docker logs`).
- Optionally:
  - Parse training loss/val loss per epoch from logs and display as a simple CLI
    progress report.
  - Launch TensorBoard pointing at `tms_workspace/logs/lightning_logs/` if
    TensorBoard is enabled in the container. citeturn0search1turn0search15  
- Informally: give you “is this converging or a dumpster fire?” feedback
  without touching the container manually.

TMS itself already supports listening to model audio as it trains; this repo
just ensures `audio_samples/` is bind‑mounted so you can play them from the host. citeturn0search10turn0search14  

---

## 7. Export & Evaluation

### 7.1 ONNX Export

**Tool:** `tools/export_onnx.py`

Responsibilities:

- Find the “best” checkpoint in `tms_workspace/checkpoints/`:
  - default heuristic: latest epoch, or lowest val loss if logs provide it.
- Invoke `piper_train.export_onnx` **inside the training container** or a dedicated
  Piper export container, e.g.:

```bash
python3 -m piper_train.export_onnx \
  /workspace/checkpoints/version_3/checkpoints/epoch=0900-step=XXXX.ckpt \
  /workspace/models/voice1/voice-medium.onnx
```

- Copy the corresponding `config.json` to `voice-medium.onnx.json`,
  as recommended in Piper tutorials. citeturn0search23turn0search1turn0search15  

Output:

```text
models/voice1/
  voice-medium.onnx
  voice-medium.onnx.json
```

These should be directly usable with `piper-tts` and Home Assistant.

### 7.2 Evaluation Script

**Tool:** `tools/eval_model.py`

Responsibilities:

- Take a list of standard evaluation phrases (configurable) and:
  - call `piper` CLI with the exported ONNX model,
  - generate WAVs in `samples/voice1/`.
- Optionally:
  - compare to baseline Piper model outputs for A/B listening.

Example:

```bash
piper \
  --model models/voice1/voice-medium.onnx \
  --output_file samples/voice1/line_01.wav \
  <<< "This is Matty, your unreasonably customised house AI."
```

---

## 8. CLI UX

Optionally, wrap all these tools in a Typer‑based CLI entrypoint, `cli.py`:

```bash
shallow-fake asr-segment   # run ASR + segmentation
shallow-fake build-dataset # build real metadata.csv
shallow-fake verify        # run phoneme sanity check
shallow-fake build-synth   # generate synthetic dataset
shallow-fake train         # launch TMS training container
shallow-fake monitor       # tail logs / open TensorBoard
shallow-fake export        # export ONNX
shallow-fake eval          # generate sample phrases
```

Each subcommand reads `config/voice1.yaml` and orchestrates the steps above.

---

## 9. Milestones

1. **M1 – Baseline real‑only pipeline**
   - ASR segmentation + `metadata.csv`.
   - Basic training via TMS using `real` only.
   - Manual ONNX export and sample generation.

2. **M2 – Phoneme sanity check**
   - Integrate `piper-phonemize` and Whisper round‑trip.
   - Filter bad pairs and demonstrate improved training set stats.

3. **M3 – Synthetic expansion**
   - Pluggable TTS backend + corpus ingestion.
   - Synthetic dataset verified via phoneme distance.
   - Combined dataset training.

4. **M4 – Monitoring & polish**
   - `monitor_training.py` with log parsing.
   - TensorBoard support, sample audio previews wired up.
   - Single‑command `shallow-fake train` tying it all together.

5. **M5 – Quality iteration**
   - Hyperparameter tweaks (epochs, batch size, learning rate) guided by
     subjective listening.
