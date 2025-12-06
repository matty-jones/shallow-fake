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
- Docker and NVIDIA Container Toolkit (for GPU training)
- ffmpeg (for audio processing)
- TextyMcSpeechy Docker image
- Baseline Piper checkpoints
- CUDA and cuDNN (for GPU-accelerated ASR) - see [CUDA_SETUP.md](CUDA_SETUP.md) for installation help

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
- Required directory structure organized by project (`datasets/<project_name>/`, `models/<project_name>/`, etc.)
- Configuration file at `config/<project_name>.yaml`
- Project-specific data directories (`data_raw/<project_name>/`, `data_processed/<project_name>/`)
- Placeholder corpus file at `data_raw/<project_name>/external_corpus/corpus.txt`

2. **Place your raw audio files:**
```bash
# Copy your audio files to:
data_raw/<project_name>/input_audio/
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
shallow-fake asr-segment            # Run ASR + segmentation
shallow-fake build-dataset          # Build real dataset
shallow-fake verify                 # Run phoneme sanity check
shallow-fake build-synth            # Generate synthetic dataset
shallow-fake combine                # Merge datasets
shallow-fake train                  # Launch TMS training container
shallow-fake monitor                # Monitor training progress
shallow-fake export                 # Export ONNX model
shallow-fake eval                   # Generate evaluation samples
```

All commands accept a `--config` option to specify the config file name (defaults to `voice1.yaml`). The config file should be in the `config/` directory - you only need to provide the filename.

**CPU Mode**: Use the `--cpu` flag with `asr-segment` to force CPU mode even if CUDA is configured:
```bash
shallow-fake asr-segment --config claudia.yaml --cpu
```
This is useful if you encounter CUDA/cuDNN issues and want to use CPU without editing the config file.

## Project Structure

- `tools/` - Pipeline scripts
- `config/` - YAML configuration files (one per project)
- `data_raw/<project_name>/` - Input audio and text corpora (organized by project)
- `data_processed/<project_name>/` - Normalized audio and segments (organized by project)
- `datasets/<project_name>/` - Real, synthetic, and combined datasets (organized by project)
- `tms_workspace/` - TMS training workspace (shared)
- `models/<project_name>/` - Final exported ONNX models (organized by project)
- `samples/<project_name>/` - Evaluation audio samples (organized by project)

## Documentation

- `PRODUCT_SPEC.md` - Complete product specification and technical design
- `CUDA_SETUP.md` - Guide for setting up CUDA/cuDNN for GPU acceleration

