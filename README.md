# ShallowFaker - Voice Piper Clone Pipeline

Automated pipeline for creating Piper-compatible voice clones from raw audio recordings.

## Overview

ShallowFaker provides a repeatable, low-touch pipeline that:

- Takes raw audio recordings (talks, demos, etc.)
- Automatically segments and transcribes them using Whisper
- Cleans and verifies text-audio pairs using phoneme-based validation
- Expands the dataset with synthetic voice data
- Fine-tunes a Piper base model via TextyMcSpeechy (TMS)
- Produces a Piper-compatible ONNX model ready for Home Assistant

## Requirements

- Python 3.10 or 3.11
- Docker and NVIDIA Container Toolkit (for GPU training and teacher model service)
- ffmpeg (for audio processing)
- TextyMcSpeechy Docker image
- Baseline Piper checkpoints
- CUDA and cuDNN (for GPU-accelerated ASR) - see [CUDA_SETUP.md](CUDA_SETUP.md) for installation help
- Coqui TTS (installed automatically in teacher model Docker container)

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

3. **Run the pipeline:**
```bash
shallow-fake asr-segment --config <project_name>.yaml
shallow-fake build-dataset --config <project_name>.yaml
# ... continue with other commands
```

## Usage

The pipeline is controlled via a Typer-based CLI:

```bash
shallow-fake init <project_name>    # Initialize a new project
shallow-fake status                 # Show pipeline progress and status
shallow-fake asr-segment            # Run ASR + segmentation
shallow-fake build-dataset          # Build real dataset
shallow-fake verify                 # Run phoneme sanity check
shallow-fake build-synth            # Generate synthetic dataset (starts teacher model service automatically if configured)
shallow-fake combine                # Merge datasets
shallow-fake train                  # Launch TMS training container
shallow-fake monitor                # Monitor training progress
shallow-fake export                 # Export ONNX model
shallow-fake eval                   # Generate evaluation samples
```

**Status Command**: Use `shallow-fake status` to see which pipeline stages have been completed and what the next step should be:
```bash
shallow-fake status --config claudia.yaml
```

All commands accept a `--config` option to specify the config file (defaults to `voice1.yaml`). You can provide:
- Just the project name: `--config claudia` (automatically looks for `config/claudia.yaml`)
- The filename: `--config claudia.yaml` (automatically looks in `config/` directory)
- Full path: `--config config/claudia.yaml` (if needed)

**CPU Mode**: Use the `--cpu` flag with `asr-segment` to force CPU mode even if CUDA is configured:
```bash
shallow-fake asr-segment --config claudia.yaml --cpu
```
This is useful if you encounter CUDA/cuDNN issues and want to use CPU without editing the config file.

**Teacher Model Integration**: The `build-synth` command automatically starts a teacher model service (if configured) to generate high-quality synthetic voice data. The service:
- Uses reference audio from your real dataset (`workspace/<project_name>/datasets/real/wavs`)
- Starts automatically when `build-synth` runs
- Stops automatically after synthetic data generation completes
- Requires Docker to be running
- Uses zero-shot voice cloning (no fine-tuning needed)

## Project Structure

The repository uses a unified directory structure to minimize duplication and simplify organization:

- `input/<project_name>/audio/` - User input audio files (one clear location)
- `workspace/<project_name>/` - All working data for a project (unified workspace)
  - `segments/` - Extracted audio segments from ASR
  - `datasets/` - Dataset directories
    - `real/` - Real voice dataset
    - `synth/` - Synthetic dataset
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
- `services/` - Teacher model service implementation
- `docker/` - Docker compose files and Dockerfiles

## Documentation

- `PRODUCT_SPEC.md` - Complete product specification and technical design
- `CUDA_SETUP.md` - Guide for setting up CUDA/cuDNN for GPU acceleration

