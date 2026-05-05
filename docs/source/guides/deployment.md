# Deployment

Deploy your OpenEnv environments to Hugging Face Spaces or a custom Docker registry using the `openenv push` command.

## Quick deploy to Hugging Face Spaces

From inside an environment directory (must contain `openenv.yaml`):

```bash
openenv push
```

This builds the Docker image, packages the environment, and pushes it to a Hugging Face Space. The Space name defaults to `<your-hf-username>/<env-name>` from `openenv.yaml`; override it with `--repo-id`:

```bash
openenv push --repo-id my-org/my-env
```

## Local development workflow

Test your environment locally before pushing:

```bash
# Build the image
openenv build

# Run it locally
docker run -p 8000:8000 my-env:latest
```

```python
# Connect a client
from openenv import AutoEnv
env = AutoEnv.from_env("http://localhost:8000")
```

Once satisfied, `openenv push` deploys the same image to HF Spaces.

## Configuring the Space

### Public variables and private secrets

Pass runtime configuration at push time:

```bash
# Public Space variable (visible in Space settings)
openenv push -e MODEL_NAME=Qwen3-1.7B

# Private secret (never logged, stored encrypted)
openenv push --secret OPENAI_API_KEY=sk-...

# Both at once
openenv push -e DATASET=chain_sum --secret HF_TOKEN=hf_...
```

Both flags are repeatable. For defaults that belong with the environment, declare them in `openenv.yaml` under `variables:` — `openenv push` applies them automatically and CLI `-e` overrides matching keys. Secrets should only ever be passed via `--secret`, never committed to `openenv.yaml`.

See the [environment builder guide](../getting_started/environment-builder.md) for the full `variables:` reference.

### Hardware

Request a specific accelerator for the Space:

```bash
openenv push --hardware t4-medium   # NVIDIA T4
openenv push --hardware a10g-small  # NVIDIA A10G
openenv push --hardware cpu-basic   # CPU-only (default)
```

See the [HF Spaces hardware docs](https://huggingface.co/docs/hub/spaces-gpus) for available tiers.

### Visibility

```bash
openenv push --private   # deploy as a private Space
```

### Multiple instances

Deploy `N` copies of the same environment, each with a numeric suffix
(`my-env-1`, `my-env-2`, …):

```bash
openenv push --count 3   # deploys my-env-1, my-env-2, my-env-3
```

Useful for load distribution across parallel training runs. Cannot be combined with `--registry` or `--create-pr`.

## Staging changes with a pull request

To review changes before they go live, push to a new branch and open a PR on the Space repo:

```bash
openenv push --create-pr
```

See the [contributing environments guide](../getting_started/contributing-envs.md) for the full PR-based update workflow.

## Pushing to a custom Docker registry

```bash
openenv push --registry ghcr.io/my-org
openenv push --registry docker.io/myuser
```

The web interface is disabled by default for custom registry pushes. `-e`/`--env-var` and `--secret` are not available with `--registry` (HF Space settings only).

## Next Steps

- [Environment builder](../getting_started/environment-builder.md) — full `openenv push` flag reference and `openenv.yaml` `variables:` docs
- [Contributing environments](../getting_started/contributing-envs.md) — PR-based update workflow and forking existing environments
- [CLI reference](../cli.md) — all CLI commands
