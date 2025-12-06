"""Configuration management with Pydantic validation."""

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings


class PathsConfig(BaseModel):
    """Path configuration for data directories."""

    raw_audio_dir: Path
    normalized_dir: Path
    segments_dir: Path
    asr_metadata: Path
    real_dataset_dir: Path
    synth_dataset_dir: Path
    combined_dataset_dir: Path
    tms_workspace_dir: Path
    output_models_dir: Path


class ASRConfig(BaseModel):
    """ASR (Whisper) configuration."""

    model_size: str = Field(default="medium.en")
    device: Literal["cuda", "cpu"] = Field(default="cuda")
    beam_size: int = Field(default=5, ge=1)
    max_segment_seconds: float = Field(default=15.0, gt=0)
    min_segment_seconds: float = Field(default=1.0, gt=0)
    min_confidence: float = Field(default=0.7, ge=0.0, le=1.0)


class PhonemeCheckConfig(BaseModel):
    """Phoneme verification configuration."""

    language: str = Field(default="en-gb")
    max_phoneme_distance: float = Field(default=0.1, ge=0.0, le=1.0)
    use_tts_roundtrip: bool = Field(default=True)


class TTSHTTPConfig(BaseModel):
    """HTTP TTS backend configuration."""

    base_url: str
    voice_id: str


class SyntheticConfig(BaseModel):
    """Synthetic data expansion configuration."""

    enabled: bool = Field(default=True)
    corpus_text_path: Path
    max_sentences: int = Field(default=2000, ge=0)
    tts_backend: Literal["http"] = Field(default="http")
    tts_http: TTSHTTPConfig
    max_parallel_jobs: int = Field(default=4, ge=1)


class TrainingConfig(BaseModel):
    """Training configuration."""

    base_checkpoint: str
    batch_size: int = Field(default=32, ge=1)
    max_epochs: int = Field(default=1000, ge=1)
    quality: Literal["low", "medium", "high"] = Field(default="medium")
    accelerator: Literal["gpu", "cpu"] = Field(default="gpu")
    devices: int = Field(default=1, ge=1)


class TMSConfig(BaseModel):
    """TextyMcSpeechy configuration."""

    enable_tts_dojo: bool = Field(default=True)
    docker_compose_file: Path
    project_name: str
    expose_tensorboard: bool = Field(default=True)
    tensorboard_port: int = Field(default=6006, ge=1024, le=65535)


class VoiceConfig(BaseModel):
    """Complete voice pipeline configuration."""

    voice_id: str = Field(default="")  # Will be auto-detected from path if not set
    language: str = Field(default="en_GB")
    paths: PathsConfig
    asr: ASRConfig
    phoneme_check: PhonemeCheckConfig
    synthetic: SyntheticConfig
    training: TrainingConfig
    tms: TMSConfig

    @field_validator("paths", mode="before")
    @classmethod
    def convert_paths(cls, v):
        """Convert path strings to Path objects."""
        if isinstance(v, dict):
            return {k: Path(v) if isinstance(v, str) else v for k, v in v.items()}
        return v

    @field_validator("synthetic", mode="before")
    @classmethod
    def convert_synthetic_paths(cls, v):
        """Convert corpus_text_path to Path object."""
        if isinstance(v, dict) and "corpus_text_path" in v:
            v["corpus_text_path"] = Path(v["corpus_text_path"])
        return v

    @field_validator("tms", mode="before")
    @classmethod
    def convert_tms_paths(cls, v):
        """Convert docker_compose_file to Path object."""
        if isinstance(v, dict) and "docker_compose_file" in v:
            v["docker_compose_file"] = Path(v["docker_compose_file"])
        return v

    @classmethod
    def from_yaml(cls, config_path: Path) -> "VoiceConfig":
        """Load configuration from YAML file."""
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        # Resolve relative paths relative to current working directory (project root)
        # This allows config files to be in config/ while paths are relative to project root
        project_root = Path.cwd()
        if "paths" in data:
            for key, value in data["paths"].items():
                if isinstance(value, str):
                    data["paths"][key] = str(project_root / value)

        if "synthetic" in data and "corpus_text_path" in data["synthetic"]:
            corpus_path = data["synthetic"]["corpus_text_path"]
            if isinstance(corpus_path, str):
                data["synthetic"]["corpus_text_path"] = str(project_root / corpus_path)

        if "tms" in data and "docker_compose_file" in data["tms"]:
            compose_path = data["tms"]["docker_compose_file"]
            if isinstance(compose_path, str):
                data["tms"]["docker_compose_file"] = str(project_root / compose_path)

        # Auto-detect project name from real_dataset_dir
        # Project name is the directory name in datasets/<project_name>/real
        if "paths" in data and "real_dataset_dir" in data["paths"]:
            real_dataset_path = Path(data["paths"]["real_dataset_dir"])
            # Extract project name from path: datasets/<project_name>/real
            if "datasets" in real_dataset_path.parts:
                datasets_idx = real_dataset_path.parts.index("datasets")
                if datasets_idx + 1 < len(real_dataset_path.parts):
                    detected_project_name = real_dataset_path.parts[datasets_idx + 1]
                    # Always use detected project name from directory structure
                    # This ensures voice_id matches the actual project directory
                    if "voice_id" not in data or not data.get("voice_id"):
                        data["voice_id"] = detected_project_name
                    elif data["voice_id"] != detected_project_name:
                        # Warn if mismatch and override with detected name
                        import warnings
                        warnings.warn(
                            f"voice_id '{data['voice_id']}' doesn't match project name "
                            f"'{detected_project_name}' from directory structure. "
                            f"Using '{detected_project_name}' as voice_id."
                        )
                        data["voice_id"] = detected_project_name

        # Ensure voice_id is set
        if not data.get("voice_id"):
            raise ValueError(
                "Could not determine project name from real_dataset_dir path. "
                "Path must contain 'datasets/<project_name>/real' structure."
            )

        return cls(**data)

    def ensure_directories(self):
        """Create all required directories if they don't exist."""
        # Safeguard: prevent creating directories in config/ directory
        config_dir = Path("config").resolve()
        
        for path in [
            self.paths.raw_audio_dir,
            self.paths.normalized_dir,
            self.paths.segments_dir,
            self.paths.real_dataset_dir,
            self.paths.synth_dataset_dir,
            self.paths.combined_dataset_dir,
            self.paths.tms_workspace_dir,
            self.paths.output_models_dir,
        ]:
            resolved_path = path.resolve()
            # Check if path would be created inside config/ directory
            if str(resolved_path).startswith(str(config_dir)):
                raise ValueError(
                    f"Invalid path: {path} would be created inside config/ directory. "
                    f"This indicates a path resolution error. Resolved path: {resolved_path}"
                )
            path.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        (self.paths.real_dataset_dir / "wavs").mkdir(parents=True, exist_ok=True)
        (self.paths.synth_dataset_dir / "wavs").mkdir(parents=True, exist_ok=True)
        (self.paths.combined_dataset_dir / "wavs").mkdir(parents=True, exist_ok=True)

