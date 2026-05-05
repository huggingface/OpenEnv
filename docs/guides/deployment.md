# Deployment

!!! note "Coming Soon"
    This page is under construction.

Deploy your OpenEnv environments to Docker, Hugging Face Spaces, and more.

## Docker Deployment

### Building the Image

```bash
cd my_env
docker build -t my-env:latest .
```

### Running Locally

```bash
docker run -p 8000:8000 my-env:latest
```

### Using with OpenEnv CLI

```bash
openenv run my-env:latest
```

## Hugging Face Spaces

### Create a Space

1. Go to [huggingface.co/new-space](https://huggingface.co/new-space)
2. Select "Docker" as the SDK
3. Upload your environment code

### Dockerfile for HF Spaces

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY . .

RUN pip install -e .

EXPOSE 7860
CMD ["uvicorn", "my_env.server:app", "--host", "0.0.0.0", "--port", "7860"]
```

### Connect to Your Space

```python
from openenv import AutoEnv

env = AutoEnv.from_env("your-username/my-env")
```

## Next Steps

- [Environment Anatomy](environment-anatomy.md) - Understand the structure
- [CLI Reference](../reference/cli.md) - CLI commands for deployment
