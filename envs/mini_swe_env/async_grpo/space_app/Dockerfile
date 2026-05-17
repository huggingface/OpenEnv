FROM vllm/vllm-openai:latest

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

# Install: openenv-core + mini_swe_env[train] into system Python
RUN uv pip install --system --no-cache \
    -e "." \
    -e "envs/mini_swe_env[train]"

# Runtime defaults
ENV PYTHONPATH=$HOME/app/src:$HOME/app/envs \
    PYTHONUNBUFFERED=1 \
    INTERCEPTION_HOST=0.0.0.0 \
    INTERCEPTION_PORT=7860

USER user
EXPOSE 7860

ENTRYPOINT []
CMD ["bash", "envs/mini_swe_env/async_grpo/space_app/start.sh"]
