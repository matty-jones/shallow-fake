# Configuration Files

## Project Name Detection

The project name (voice_id) is **automatically detected** from the directory structure. The system looks for the pattern:

```
datasets/<project_name>/real
```

The project name is extracted from the `real_dataset_dir` path. For example:
- If `real_dataset_dir: datasets/myvoice/real`, the project name will be `myvoice`
- If `real_dataset_dir: datasets/alice/real`, the project name will be `alice`

You can optionally specify `voice_id` in the config file, but it must match the directory name. If it doesn't match, a warning will be issued and the directory name will be used.

## Example Config

```yaml
# voice_id is optional - will be auto-detected from real_dataset_dir
# voice_id: myproject
language: en_GB

paths:
  real_dataset_dir: datasets/myproject/real
  synth_dataset_dir: datasets/myproject/synth
  combined_dataset_dir: datasets/myproject/combined
  output_models_dir: models/myproject
  # ... other paths
```

## Multiple Projects

To work with multiple projects, create separate config files:
- `config/project1.yaml` with paths pointing to `datasets/project1/...`
- `config/project2.yaml` with paths pointing to `datasets/project2/...`

Each config file will automatically detect the correct project name from its paths.

