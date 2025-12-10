# Progress Tracking

## Implementation Status

### Phase 1: Project Setup & Dependencies
- [x] Project structure created
- [x] pyproject.toml and requirements.txt created
- [x] .gitignore configured
- [x] Config template (voice1.yaml) created
- [x] Config parser with Pydantic validation

### Phase 2: ASR Segmentation
- [x] Audio normalization with ffmpeg
- [x] Whisper integration (faster-whisper)
- [x] Segment extraction and JSONL output

### Phase 3: Dataset Builder
- [x] Metadata processing
- [x] LJSpeech format generation

### Phase 4: Phoneme Verification
- [x] Phoneme comparison pipeline
- [x] Filtering and reporting

### Phase 5: Synthetic Data Expansion
- [x] TTS backend abstraction (HTTP backend implemented)
- [x] Corpus processing
- [x] Quality verification

### Phase 6: Training Integration
- [x] Docker setup (docker-compose.training.yml)
- [x] Training orchestration (launch_training.py)
- [x] Monitoring tools (monitor_training.py)

### Phase 7: Export & Evaluation
- [x] ONNX export (export_onnx.py)
- [x] Evaluation script (eval_model.py)

### Phase 8: CLI Integration
- [x] Typer CLI with all subcommands
- [x] Error handling and logging
- [x] Config loading

## Implementation Complete

All components from the product specification have been implemented:

1. **Project Structure**: Complete directory layout as specified
2. **Configuration System**: YAML-based config with Pydantic validation
3. **ASR Segmentation**: Audio normalization and Whisper-based segmentation
4. **Dataset Building**: LJSpeech format metadata generation
5. **Phoneme Verification**: Quality checking using phoneme distance
6. **Synthetic Data**: TTS backend abstraction with HTTP implementation
7. **Dataset Combination**: Merging real and synthetic datasets
8. **Training Integration**: Docker-based TMS training setup
9. **Monitoring**: Log tailing and TensorBoard support
10. **Export**: ONNX model export with config JSON
11. **Evaluation**: Sample phrase generation
12. **CLI**: Complete Typer-based command-line interface

## Next Steps

1. Install dependencies: `pip install -r requirements.txt`
2. Configure `config/voice1.yaml` for your voice
3. Place raw audio files in `data_raw/input_audio/`
4. Run the pipeline: `shallow-fake asr-segment` (or use individual commands)

## Notes

- Python 3.10 virtual environment exists in .venv
- All tools are in `tools/` directory and can be run standalone or via CLI
- CLI entrypoint: `shallow-fake` (or `python -m shallow_fake.cli`)
- Docker and NVIDIA Container Toolkit required for training
- TextyMcSpeechy Docker image and baseline Piper checkpoints needed for training
- Training compose image name: `textymcspeechy-piper:5080-cu128` (locally built on top of `domesticatedviking/textymcspeechy-piper:latest` with PyTorch nightly cu128; see `docker/Dockerfile.piper5080`).
- Training container runs from `/app` (compose `working_dir`).
- Training command now uses `piper_train.preprocess` -> `piper_train` with dataset at `/workspace/datasets/<voice>/combined_prepared` (preprocess input `/workspace/datasets/<voice>/combined`, 22.05 kHz, ljspeech single-speaker, language `en-us`); training arg uses `--max_epochs` (underscore).
- `launch_training.py` now reads checkpoint `epoch` and sets `MAX_EPOCHS` to `checkpoint_epoch + config.training.max_epochs` so resumed training runs for the requested additional epochs.
- Training compose sets `shm_size: 8g` to avoid DataLoader shared-memory bus errors.
- **Export Fix**: Added espeak voice mapping to export process. The `espeak.voice` field is now automatically injected into `onnx.json` based on the language code in config.yaml. For `en_GB`, uses `en-gb-x-rp` (Received Pronunciation) as the espeak voice. Mapping function `get_espeak_voice()` in `language_utils.py` handles language-to-espeak-voice conversion with fallbacks.
