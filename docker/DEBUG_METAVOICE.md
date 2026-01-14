# Debugging MetaVoice Container

## Quick Debug Commands

### 1. Check if container exists (even if exited)
```bash
docker ps -a --filter "name=metavoice"
```

### 2. View logs from exited container
```bash
# View last 50 lines
docker logs docker-metavoice-teacher-1 2>&1 | tail -50

# View all logs
docker logs docker-metavoice-teacher-1

# Follow logs in real-time (if container is running)
docker logs -f docker-metavoice-teacher-1
```

### 3. Run container manually with docker compose

From the project root:
```bash
cd /mnt/m/Creativity/VoiceModels/ShallowFaker
docker compose -f docker/docker-compose.metavoice.yml --env-file .env.claudia.metavoice up --build
```

Or from the docker directory:
```bash
cd /mnt/m/Creativity/VoiceModels/ShallowFaker/docker
docker compose -f docker-compose.metavoice.yml --env-file ../.env.claudia.metavoice up --build
```

### 4. Run container interactively to debug

First, build the image:
```bash
cd /mnt/m/Creativity/VoiceModels/ShallowFaker/docker
docker build -f Dockerfile.metavoice.inference -t metavoice-5080 .
```

Then run interactively:
```bash
docker run -it --rm \
  --gpus all \
  -p 58003:58003 \
  -v ~/.cache/huggingface:/root/.cache/huggingface:rw \
  -v /mnt/m/Creativity/VoiceModels/ShallowFaker/workspace/claudia/reference:/speakers:ro \
  -e CUDA_VISIBLE_DEVICES=all \
  -e HUGGINGFACE_REPO_ID=metavoiceio/metavoice-1B-v0.1 \
  metavoice-5080 \
  /bin/bash
```

Inside the container, you can then:
```bash
# Check Python environment
python -c "import fastapi; import uvicorn; print('OK')"

# Try running serving.py manually
python serving.py --port=58003

# Or check what's missing
python -c "import serving"
```

### 5. Check container exit code
```bash
docker inspect docker-metavoice-teacher-1 --format='{{.State.ExitCode}}'
```

### 6. Inspect container configuration
```bash
docker inspect docker-metavoice-teacher-1
```

### 7. Run with different entrypoint to keep it alive
```bash
docker run -it --rm \
  --gpus all \
  --entrypoint /bin/bash \
  metavoice-5080
```

## Common Issues

1. **Container exits immediately**: Check logs with `docker logs <container-name>`
2. **ModuleNotFoundError**: Missing dependency in Dockerfile - add to pip install
3. **Permission errors**: Check volume mount paths and permissions
4. **GPU not available**: Ensure `--gpus all` or CDI devices are configured

## Finding Container Name

If you're not sure of the container name:
```bash
# List all containers
docker ps -a

# Filter by image
docker ps -a --filter "ancestor=metavoice-5080"

# Filter by name pattern
docker ps -a --filter "name=metavoice"
```
























