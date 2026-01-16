# ShallowFaker - Voice Piper Clone Pipeline

Automated pipeline for creating Piper-compatible voice clones from raw audio recordings.

## Overview

ShallowFaker provides a repeatable, low-touch pipeline that:

- Takes raw audio recordings (talks, demos, etc.)
- Automatically segments and transcribes them using Whisper
- Cleans and verifies text-audio pairs using phoneme-based validation
- Expands the dataset with synthetic voice data using teacher models (XTTS, OpenVoice, or MetaVoice)
- Fine-tunes a Piper base model via TextyMcSpeechy (TMS)
- Produces a Piper-compatible ONNX model ready for Home Assistant

## Requirements

- Python 3.10 or 3.11
- Docker and NVIDIA Container Toolkit (for GPU training and teacher model services)
- ffmpeg (for audio processing)
- CUDA and cuDNN (for GPU-accelerated ASR and teacher models)
- Baseline Piper checkpoints (for training)

## Installation

1. Run the setup script to create and configure the virtual environment:
```bash
./setup.sh
```

2. Activate the virtual environment:
```bash
source .venv/bin/activate
```

## Quick Start

1. **Initialize a new project:**
```bash
shallow-fake init <project_name>
```

This creates:
- Required directory structure organized by project
- Configuration file at `config/<project_name>.yaml`
- Project-specific directories (`input/<project_name>/audio/`, `workspace/<project_name>/`, `models/<project_name>/`)
- Shared resources (`models/shared/`, `input/shared/`)
- Placeholder corpus file at `input/shared/corpus.txt` (shared across all projects)

2. **Place your raw audio files:**
```bash
# Copy your audio files to:
input/<project_name>/audio/
```

3. **Configure your project:**
Edit `config/<project_name>.yaml` to set:
- Language code (e.g., `en_GB`, `en_US`)
- Training parameters (batch size, epochs, quality)
- Teacher model selection (XTTS, OpenVoice, or MetaVoice) for synthetic data generation

4. **Run the pipeline:**
```bash
shallow-fake asr-segment --config <project_name>
shallow-fake build-dataset --config <project_name>
shallow-fake verify --config <project_name>
shallow-fake build-synth --config <project_name>
shallow-fake combine --config <project_name>
shallow-fake train --config <project_name>
shallow-fake monitor --config <project_name>
shallow-fake export --config <project_name>
shallow-fake eval --config <project_name>
```

## Usage

The pipeline is controlled via a Typer-based CLI:

```bash
shallow-fake init <project_name>              # Initialize a new project
shallow-fake status [--config <project>]       # Show pipeline progress and status
shallow-fake asr-segment [--config <project>] # Run ASR + segmentation
shallow-fake build-dataset [--config <project>] # Build real dataset
shallow-fake verify [--config <project>]      # Run phoneme sanity check
shallow-fake build-synth [--config <project>] # Generate synthetic dataset
shallow-fake combine [--config <project>]    # Merge datasets
shallow-fake train [--config <project>]       # Launch TMS training container
shallow-fake save-checkpoint <version> [--config <project>] # Save checkpoint with version
shallow-fake monitor [--config <project>]    # Monitor training progress
shallow-fake export [--config <project>]     # Export ONNX model
shallow-fake eval [--config <project>]        # Generate evaluation samples
shallow-fake cleanup-docker [--all]          # Clean up Docker resources
```

**Status Command**: Use `shallow-fake status` to see which pipeline stages have been completed:
```bash
shallow-fake status                    # Show status for all projects
shallow-fake status --config claudia   # Show status for specific project
```

All commands accept a `--config` option to specify the config file (defaults to `voice1.yaml`). You can provide:
- Just the project name: `--config claudia` (automatically looks for `config/claudia.yaml`)
- The filename: `--config claudia.yaml` (automatically looks in `config/` directory)
- Full path: `--config config/claudia.yaml` (if needed)

**CPU Mode**: Use the `--cpu` flag with `asr-segment` to force CPU mode even if CUDA is configured:
```bash
shallow-fake asr-segment --config claudia --cpu
```

**Training Resume**: Resume training from a versioned checkpoint:
```bash
shallow-fake train --config claudia --resume-from-version 1
```

**Checkpoint Versioning**: Save checkpoints with version numbers for easy resumption:
```bash
shallow-fake save-checkpoint 1 --config claudia
shallow-fake save-checkpoint 2 --config claudia --overwrite
```

## Teacher Models

The `build-synth` command automatically starts a teacher model service (if configured) to generate high-quality synthetic voice data. Three teacher model options are supported:

### XTTS (Coqui TTS)
- **Zero-shot voice cloning** using reference audio
- Multilingual support
- Configurable number of reference clips (or use all available)
- Multiple worker processes for parallel generation
- Docker service: `docker-compose.xtts.yml`

### OpenVoice
- **Zero-shot voice cloning** with tone color conversion
- Uses MeloTTS for base speech synthesis
- Supports multiple language variants (EN_BR, EN_US, etc.)
- Docker service: `docker-compose.openvoice.yml`

### MetaVoice
- **Zero-shot voice cloning** with advanced guidance controls
- Configurable guidance scale, top-p, and top-k sampling
- HuggingFace model integration
- Docker service: `docker-compose.metavoice.yml`

The teacher model service:
- Uses reference audio from your real dataset (`workspace/<project_name>/datasets/real/wavs`)
- Starts automatically when `build-synth` runs
- Stops automatically after synthetic data generation completes
- Requires Docker to be running
- Uses zero-shot voice cloning (no fine-tuning needed)

Configure the teacher model in your project's YAML config file under `synthetic.teacher`:

```yaml
synthetic:
  teacher:
    kind: xtts  # or 'openvoice' or 'metavoice'
    port: 9010
    device: cuda
    reference_audio_dir: workspace/<project_name>/datasets/real/wavs
    # XTTS-specific options
    num_reference_clips: 3  # or 0 to use all available
    workers: 3
    # OpenVoice-specific options
    openvoice_language: EN
    openvoice_base_speaker_key: EN-BR
    # MetaVoice-specific options
    guidance: 3.0
    top_p: 0.95
    top_k: 200
```

## Project Structure

The repository uses a unified directory structure to minimize duplication and simplify organization:

- `input/<project_name>/audio/` - User input audio files (one clear location)
- `workspace/<project_name>/` - All working data for a project (unified workspace)
  - `segments/` - Extracted audio segments from ASR
  - `datasets/` - Dataset directories
    - `real/` - Real voice dataset
    - `synth-xtts/` - Synthetic dataset (XTTS teacher)
    - `synth-openvoice/` - Synthetic dataset (OpenVoice teacher)
    - `synth-metavoice/` - Synthetic dataset (MetaVoice teacher)
    - `combined/` - Combined dataset (for training)
    - `prepared/` - Preprocessed dataset (training cache, logs, checkpoints)
  - `training/` - Training outputs
    - `checkpoints/` - Exported model checkpoints
    - `samples/` - Sample audio outputs
- `models/<project_name>/` - Final exported ONNX models
- `models/shared/` - Shared resources
  - `base_checkpoints/` - Base model checkpoints (shared across projects)
  - `xtts_baseline/` - Teacher model cache (shared across all projects)
- `input/shared/` - Shared input files
  - `corpus.txt` - Text corpus for synthetic data generation (shared across all projects)
- `config/` - YAML configuration files (one per project)
- `tools/` - Pipeline scripts
- `services/` - Teacher model service implementations
- `docker/` - Docker compose files and Dockerfiles

## Training System

Training uses TextyMcSpeechy (TMS), a Docker-based training system for Piper models. The training container:

- Runs from `/app` (compose `working_dir`)
- Uses `piper_train.preprocess` to prepare datasets at 22.05 kHz in LJSpeech format
- Trains with configurable batch size, epochs, and quality settings
- Supports checkpoint resumption with automatic epoch calculation
- Exposes TensorBoard on port 6006 (if enabled)
- Requires shared memory size of 8GB (`shm_size: 8g`) to avoid DataLoader errors

Training checkpoints are saved in `workspace/<project_name>/training/checkpoints/` and can be versioned using the `save-checkpoint` command for easy resumption.

## Export and Evaluation

Exported models are saved to `models/<project_name>/` in ONNX format with accompanying JSON configuration files. The export process:

- Automatically injects espeak voice mapping based on language code
- Supports versioned exports (v1, v2, etc.) with v1 protected from overwrite
- Generates model files compatible with Piper and Home Assistant

Evaluation generates sample audio files from test phrases to verify model quality.

## Docker Management

The pipeline uses Docker for:
- Teacher model services (XTTS, OpenVoice, MetaVoice)
- Training container (TextyMcSpeechy)

Use `shallow-fake cleanup-docker` to free disk space by cleaning Docker build cache. Add `--all` to also remove unused images, containers, and volumes.

### Troubleshooting Docker Services

If a teacher model service fails to start or exits unexpectedly:

1. **Check container logs:**
   ```bash
   docker logs <container-name>  # View logs
   docker logs -f <container-name>  # Follow logs in real-time
   ```

2. **List containers:**
   ```bash
   docker ps -a  # Show all containers (including stopped)
   docker ps -a --filter "name=<service>"  # Filter by name
   ```

3. **Check container status:**
   ```bash
   docker inspect <container-name> --format='{{.State.ExitCode}}'  # Exit code
   ```

4. **Common issues:**
   - **Container exits immediately**: Check logs for errors (missing dependencies, permission issues, GPU not available)
   - **GPU not available**: Ensure NVIDIA Container Toolkit is installed and `--gpus all` or CDI devices are configured
   - **Permission errors**: Verify volume mount paths exist and have correct permissions
   - **ModuleNotFoundError**: Missing dependency in Dockerfile - check service implementation

## Configuration

Each project has its own YAML configuration file in `config/`. Key configuration sections:

- `voice_id`: Project identifier
- `language`: Language code (e.g., `en_GB`, `en_US`)
- `paths`: Directory paths (auto-generated, rarely need modification)
- `asr`: Whisper ASR settings (model size, device, beam size, segment length)
- `phoneme_check`: Phoneme verification thresholds
- `synthetic`: Synthetic data generation settings and teacher model configuration
- `training`: Training parameters (checkpoint, batch size, epochs, quality)
- `tms`: Training container settings (Docker compose file, TensorBoard)

See `config/voice1.yaml` or `config/claudia.yaml` for example configurations.

## Notes

- Python 3.10+ virtual environment is created in `.venv` by `setup.sh`
- All tools are in `tools/` directory and can be run standalone or via CLI
- CLI entrypoint: `shallow-fake` (or `python -m shallow_fake.cli`)
- Docker and NVIDIA Container Toolkit required for GPU training and teacher models
- TextyMcSpeechy Docker image and baseline Piper checkpoints needed for training
- Training compose image: `textymcspeechy-piper:5080-cu128` (locally built on top of `domesticatedviking/textymcspeechy-piper:latest` with PyTorch nightly cu128; see `docker/Dockerfile.piper5080`)
