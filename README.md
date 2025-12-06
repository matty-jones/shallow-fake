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

## Installation

1. Activate the virtual environment:
```bash
source .venv/bin/activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

1. **Initialize a new project:**
```bash
shallow-fake init <project_name>
```

This creates:
- Required directory structure (`datasets/<project_name>/`, `models/<project_name>/`, etc.)
- Configuration file at `config/<project_name>.yaml`
- Placeholder corpus file at `data_raw/external_corpus/corpus.txt`

2. **Place your raw audio files:**
```bash
# Copy your audio files to:
data_raw/input_audio/
```

3. **Run the pipeline:**
```bash
shallow-fake asr-segment --config config/<project_name>.yaml
shallow-fake build-dataset --config config/<project_name>.yaml
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

All commands accept a `--config` option to specify the config file (defaults to `config/voice1.yaml`).

## Project Structure

- `tools/` - Pipeline scripts
- `config/` - YAML configuration files
- `data_raw/` - Input audio and text corpora
- `data_processed/` - Normalized audio and segments
- `datasets/` - Real, synthetic, and combined datasets
- `tms_workspace/` - TMS training workspace
- `models/` - Final exported ONNX models
- `samples/` - Evaluation audio samples

## Documentation

See `PRODUCT_SPEC.md` for the complete product specification and technical design.

