# vLLM 0.21.0 — supports Qwen3_5ForCausalLM (text-only), transformers >=5.0,
# and the four-phase weight-transfer protocol (init/start/update/finish).
# TRL AsyncGRPO uses vLLM as an external HTTP server; its optional-dep pin
# (<=0.18.0) does not apply when vLLM is installed independently.
FROM vllm/vllm-openai:v0.21.0

# System deps + Docker CLI (for Docker sandbox backend)
RUN apt-get update -qq && \
    apt-get install -y -qq --no-install-recommends curl ca-certificates docker.io git && \
    update-ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# HF Spaces requires uid 1000
RUN useradd -m -u 1000 user && usermod -aG docker user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Pre-create HF cache dir (writable by user)
RUN mkdir -p /home/user/.cache/huggingface && chown -R user:user /home/user/.cache

# Copy repo
WORKDIR $HOME/app
COPY --chown=user . .

# Install deps:
# 1. Force-upgrade huggingface-hub (resolves vLLM vs hf-sandbox conflict).
# 2. Force-install transformers >=5.2.0 (required by TRL v1.6.0 AsyncGRPO,
#    and for native Qwen3.5-4B model type support).  --no-deps avoids
#    pulling vLLM's bundled transformers which may conflict.
# 3. TRL emits a warning about vLLM >0.18 but functions correctly —
#    the version check is advisory, not blocking.
RUN uv pip install --system --no-cache "huggingface-hub>=1.5" && \
    uv pip install --system --no-cache --no-deps \
        -e "." \
        -e "envs/mini_swe_env" && \
    uv pip install --system --no-cache \
        "trl>=1.5.1" \
        "accelerate>=1.13.0" \
        "hf-sandbox[named-tunnels] @ git+https://github.com/rycerzes/hf-sandbox@feat/named-tunnels" \
        "datasets>=4.8.0" \
        "aiohttp>=3.13.5" \
        "fastmcp>=3.3.0" \
        "trackio>=0.25.0" \
        "bitsandbytes>=0.43.0" && \
    uv pip install --system --no-cache --no-deps "transformers>=5.2.0"

# Runtime defaults
ENV PYTHONPATH=$HOME/app/src:$HOME/app/envs \
    PYTHONUNBUFFERED=1 \
    INTERCEPTION_HOST=0.0.0.0 \
    INTERCEPTION_PORT=7860

USER user
EXPOSE 7860

ENTRYPOINT []
CMD ["bash", "start.sh"]
