"""Configuration management with Pydantic validation."""

from pathlib import Path
from typing import Annotated, Literal, Optional, Union

import yaml
from pydantic import BaseModel, Discriminator, Field, field_validator, model_validator
from pydantic_settings import BaseSettings


class PathsConfig(BaseModel):
    """Simplified path configuration for unified directory structure."""

    input_audio_dir: Path  # input/{voice_id}/audio
    workspace_dir: Path  # workspace/{voice_id}
    models_dir: Path  # models/{voice_id}
    shared_models_dir: Path  # models/shared
    corpus_path: Path  # input/shared/corpus.txt

    # Computed properties for backward compatibility and convenience
    @property
    def segments_dir(self) -> Path:
        """Segments directory within workspace."""
        return self.workspace_dir / "segments"

    @property
    def asr_metadata(self) -> Path:
        """ASR metadata file."""
        return self.workspace_dir / "segments" / "asr_metadata.jsonl"

    @property
    def real_dataset_dir(self) -> Path:
        """Real dataset directory."""
        return self.workspace_dir / "datasets" / "real"

    def synth_dataset_dir(self, teacher_kind: Optional[str] = None) -> Path:
        """
        Synthetic dataset directory.
        
        Args:
            teacher_kind: Teacher model kind ('xtts' or 'metavoice'). 
                         If None, defaults to 'synth' for backward compatibility.
        
        Returns:
            Path to synthetic dataset directory (synth-<teacher_kind> or synth if teacher_kind is None)
        """
        if teacher_kind:
            return self.workspace_dir / "datasets" / f"synth-{teacher_kind}"
        else:
            # Backward compatibility: return 'synth' if teacher_kind not specified
            return self.workspace_dir / "datasets" / "synth"

    @property
    def combined_dataset_dir(self) -> Path:
        """Combined dataset directory."""
        return self.workspace_dir / "datasets" / "combined"

    @property
    def prepared_dataset_dir(self) -> Path:
        """Prepared dataset directory for training."""
        return self.workspace_dir / "datasets" / "prepared"

    @property
    def training_dir(self) -> Path:
        """Training outputs directory."""
        return self.workspace_dir / "training"

    @property
    def training_checkpoints_dir(self) -> Path:
        """Training checkpoints directory."""
        return self.workspace_dir / "training" / "checkpoints"

    @property
    def training_samples_dir(self) -> Path:
        """Training samples directory."""
        return self.workspace_dir / "training" / "samples"

    @property
    def output_models_dir(self) -> Path:
        """Output models directory (alias for models_dir)."""
        return self.models_dir

    @property
    def base_checkpoints_dir(self) -> Path:
        """Base checkpoints directory."""
        return self.shared_models_dir / "base_checkpoints"


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

    max_phoneme_distance: float = Field(default=0.1, ge=0.0, le=1.0)
    use_tts_roundtrip: bool = Field(default=True)
    parallel_workers: int = Field(default=4, ge=1, le=16, description="Number of parallel workers for verification. Each worker loads its own Whisper model (~500MB RAM per worker).")


class TTSHTTPConfig(BaseModel):
    """HTTP TTS backend configuration."""

    base_url: str
    voice_id: str


class XTTSTeacherConfig(BaseModel):
    """XTTS teacher model service configuration."""

    kind: Literal["xtts"] = Field(default="xtts")
    port: int = Field(default=9010, ge=1024, le=65535)
    model_name: str = Field(default="tts_models/multilingual/multi-dataset/xtts_v2")
    language: str = Field(default="en")
    device: Literal["cuda", "cpu"] = Field(default="cuda")
    reference_audio_dir: Path
    num_reference_clips: int = Field(default=3, ge=0, description="Number of reference clips to use. Set to 0 to use all available clips.")
    workers: int = Field(default=3, ge=1, le=20, description="Number of uvicorn workers. Each worker loads its own model copy (~2GB GPU memory per worker). max_parallel_jobs will be auto-set to workers * 2.")

    @field_validator("reference_audio_dir", mode="before")
    @classmethod
    def convert_reference_audio_dir(cls, v):
        """Convert reference_audio_dir to Path object."""
        if isinstance(v, str):
            return Path(v)
        return v


class MetaVoiceTeacherConfig(BaseModel):
    """MetaVoice teacher model service configuration."""

    kind: Literal["metavoice"] = Field(default="metavoice")
    base_url: str = Field(default="http://localhost:58003", description="Base URL for MetaVoice HTTP server")
    huggingface_repo_id: str = Field(default="metavoiceio/metavoice-1B-v0.1", description="HuggingFace repository ID for MetaVoice model")
    speaker_ref_path: str = Field(default="/speakers/voice_ref.wav", description="Path inside container to reference audio file")
    guidance: float = Field(default=3.0, ge=0.0, description="Guidance scale for MetaVoice generation")
    top_p: float = Field(default=0.95, ge=0.0, le=1.0, description="Top-p sampling parameter")
    top_k: Optional[int] = Field(default=200, ge=1, description="Top-k sampling parameter")
    port: int = Field(default=58003, ge=1024, le=65535, description="Port for MetaVoice HTTP server")
    reference_audio_dir: Path = Field(description="Directory containing reference audio files (used to select best clip)")

    @field_validator("reference_audio_dir", mode="before")
    @classmethod
    def convert_reference_audio_dir(cls, v):
        """Convert reference_audio_dir to Path object."""
        if isinstance(v, str):
            return Path(v)
        return v


# Union type for teacher configs with discriminated union
# Pydantic will use the "kind" field to determine which config type to use
TeacherConfig = Annotated[
    Union[XTTSTeacherConfig, MetaVoiceTeacherConfig],
    Discriminator("kind"),
]


class SyntheticConfig(BaseModel):
    """Synthetic data expansion configuration."""

    enabled: bool = Field(default=True)
    corpus_text_path: Path
    max_sentences: int = Field(default=2000, ge=0)
    tts_backend: Literal["http"] = Field(default="http")
    tts_http: TTSHTTPConfig
    teacher: Optional[TeacherConfig] = Field(default=None)
    max_parallel_jobs: Optional[int] = Field(default=None, ge=1, description="Number of parallel HTTP requests. If None, auto-calculated as teacher.workers * 2 to keep workers busy with a ready queue.")

    @model_validator(mode="after")
    def auto_calculate_parallel_jobs(self):
        """Auto-calculate max_parallel_jobs if not specified."""
        if self.max_parallel_jobs is not None:
            return self
        # Auto-calculate based on teacher type
        if self.teacher:
            if isinstance(self.teacher, XTTSTeacherConfig):
                # XTTS: use workers * 2 to keep workers busy
                self.max_parallel_jobs = self.teacher.workers * 2
            elif isinstance(self.teacher, MetaVoiceTeacherConfig):
                # MetaVoice: default to 4 parallel jobs (MetaVoice doesn't have workers config)
                self.max_parallel_jobs = 4
            else:
                # Fallback for unknown teacher types
                self.max_parallel_jobs = 4
        else:
            # Default fallback if no teacher configured
            self.max_parallel_jobs = 4
        return self


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
    language: str = Field(description="Language code in format 'en_GB' or 'en-US'. Required for model naming and phoneme checking.")
    language_code: Optional[str] = Field(default=None, description="Language code (e.g., 'en'). Auto-parsed from 'language' if not set.")
    region: Optional[str] = Field(default=None, description="Region code (e.g., 'GB'). Auto-parsed from 'language' if not set.")
    paths: PathsConfig
    asr: ASRConfig
    phoneme_check: PhonemeCheckConfig
    synthetic: SyntheticConfig
    training: TrainingConfig
    tms: TMSConfig

    def get_synth_dataset_dir(self) -> Path:
        """
        Get synthetic dataset directory based on configured teacher kind.
        
        Returns:
            Path to teacher-kind-specific synthetic dataset directory (synth-xtts or synth-metavoice)
            Falls back to 'synth' if no teacher is configured.
        """
        if self.synthetic.teacher:
            teacher_kind = self.synthetic.teacher.kind
            return self.paths.synth_dataset_dir(teacher_kind=teacher_kind)
        else:
            # No teacher configured, use default 'synth' directory
            return self.paths.synth_dataset_dir(teacher_kind=None)

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

    @model_validator(mode="after")
    def parse_language_fields(self):
        """Parse language_code and region from language field if not explicitly set."""
        from shallow_fake.language_utils import parse_language_code
        
        if self.language_code is None or self.region is None:
            lang_code, region = parse_language_code(self.language)
            if self.language_code is None:
                self.language_code = lang_code
            if self.region is None:
                self.region = region
        return self

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

        # Resolve reference_audio_dir path if teacher is configured
        if "synthetic" in data and "teacher" in data["synthetic"] and data["synthetic"]["teacher"]:
            teacher = data["synthetic"]["teacher"]
            if "reference_audio_dir" in teacher and isinstance(teacher["reference_audio_dir"], str):
                teacher["reference_audio_dir"] = str(project_root / teacher["reference_audio_dir"])
            # Handle MetaVoice base_url default if not set
            if teacher.get("kind") == "metavoice" and "base_url" not in teacher:
                port = teacher.get("port", 58003)
                teacher["base_url"] = f"http://localhost:{port}"

        if "tms" in data and "docker_compose_file" in data["tms"]:
            compose_path = data["tms"]["docker_compose_file"]
            if isinstance(compose_path, str):
                data["tms"]["docker_compose_file"] = str(project_root / compose_path)

        # Auto-detect project name from workspace_dir or input_audio_dir
        # New structure: workspace/<project_name> or input/<project_name>/audio
        detected_project_name = None
        if "paths" in data:
            # Try workspace_dir first (new structure)
            if "workspace_dir" in data["paths"]:
                workspace_path = Path(data["paths"]["workspace_dir"])
                if "workspace" in workspace_path.parts:
                    workspace_idx = workspace_path.parts.index("workspace")
                    if workspace_idx + 1 < len(workspace_path.parts):
                        detected_project_name = workspace_path.parts[workspace_idx + 1]
            # Try input_audio_dir (new structure)
            elif "input_audio_dir" in data["paths"]:
                input_path = Path(data["paths"]["input_audio_dir"])
                if "input" in input_path.parts:
                    input_idx = input_path.parts.index("input")
                    if input_idx + 1 < len(input_path.parts):
                        detected_project_name = input_path.parts[input_idx + 1]
            # Fallback to old structure detection
            elif "real_dataset_dir" in data["paths"]:
                real_dataset_path = Path(data["paths"]["real_dataset_dir"])
                if "datasets" in real_dataset_path.parts:
                    datasets_idx = real_dataset_path.parts.index("datasets")
                    if datasets_idx + 1 < len(real_dataset_path.parts):
                        detected_project_name = real_dataset_path.parts[datasets_idx + 1]

        # Set voice_id from detected name
        if detected_project_name:
            if "voice_id" not in data or not data.get("voice_id"):
                data["voice_id"] = detected_project_name
            elif data["voice_id"] != detected_project_name:
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
                "Could not determine project name from paths. "
                "Please set voice_id explicitly or ensure paths contain project name."
            )

        # Ensure language is set (required field)
        if "language" not in data or not data.get("language"):
            raise ValueError(
                "Language is required. Please set 'language' at the top level of the config "
                "(e.g., 'language: en_GB' or 'language: en-US')."
            )

        return cls(**data)

    def ensure_directories(self):
        """Create all required directories if they don't exist."""
        # Safeguard: prevent creating directories in config/ directory
        config_dir = Path("config").resolve()
        
        # Core directories
        directories = [
            self.paths.input_audio_dir,
            self.paths.workspace_dir,
            self.paths.models_dir,
            self.paths.shared_models_dir,
            self.paths.segments_dir,
            self.paths.real_dataset_dir,
            # synth_dataset_dir is teacher-kind-specific and created on-demand
            # when build-synth runs, so we don't create it here
            self.paths.combined_dataset_dir,
            self.paths.prepared_dataset_dir,
            self.paths.training_dir,
            self.paths.training_checkpoints_dir,
            self.paths.training_samples_dir,
            self.paths.base_checkpoints_dir,
        ]
        
        for path in directories:
            resolved_path = path.resolve()
            # Check if path would be created inside config/ directory
            if str(resolved_path).startswith(str(config_dir)):
                raise ValueError(
                    f"Invalid path: {path} would be created inside config/ directory. "
                    f"This indicates a path resolution error. Resolved path: {resolved_path}"
                )
            path.mkdir(parents=True, exist_ok=True)

        # Create dataset subdirectories
        (self.paths.real_dataset_dir / "wavs").mkdir(parents=True, exist_ok=True)
        # Synth dataset directory is created on-demand based on teacher kind
        # when build-synth runs, so we don't create it here
        (self.paths.combined_dataset_dir / "wavs").mkdir(parents=True, exist_ok=True)
        
        # Ensure corpus directory exists
        self.paths.corpus_path.parent.mkdir(parents=True, exist_ok=True)

