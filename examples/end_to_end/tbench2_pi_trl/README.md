# Terminus + TRL Async GRPO

Start a Terminus server and a vLLM server with weight transfer enabled, then run:

```bash
TERMINUS_ENV_URL=http://localhost:8000 \
TERMINUS_VLLM_SERVER_URL=http://localhost:8001 \
uv run train_terminus_grpo.py
```
