# Pin vLLM to 0.18.1 — TRL AsyncGRPO supports 0.12-0.18.x for NCCL weight sync.
# vLLM 0.19+ has breaking changes in the weight transfer API.
FROM vllm/vllm-openai:v0.18.1

# System deps + Docker CLI (for Docker sandbox backend)
RUN apt-get update -qq && \
    apt-get install -y -qq --no-install-recommends curl ca-certificates docker.io && \
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

# Install deps: force-upgrade huggingface-hub first (resolves vLLM <1.0 vs hf-sandbox >=1.5 conflict)
# vLLM 0.18.1 works fine with newer huggingface-hub at runtime.
RUN uv pip install --system --no-cache "huggingface-hub>=1.5" && \
    uv pip install --system --no-cache --no-deps \
        -e "." \
        -e "envs/mini_swe_env" && \
    uv pip install --system --no-cache \
        "trl>=1.4.0" \
        "accelerate>=1.13.0" \
        "hf-sandbox>=0.1.1" \
        "datasets>=4.8.0" \
        "aiohttp>=3.13.5" \
        "fastmcp>=3.3.0" \
        "trackio>=0.25.0"

# Runtime defaults
ENV PYTHONPATH=$HOME/app/src:$HOME/app/envs \
    PYTHONUNBUFFERED=1 \
    INTERCEPTION_HOST=0.0.0.0 \
    INTERCEPTION_PORT=7860

USER user
EXPOSE 7860

ENTRYPOINT []
CMD ["bash", "start.sh"]
