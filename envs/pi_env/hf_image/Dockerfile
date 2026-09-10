# Pre-baked image for HFSandboxBackend: Node + Pi + proxy deps installed ahead
# of time so rollouts skip the cold install. Use with PiConfig(sandbox_home="/root").
#
# The interception proxy source lives in opencode_env (outside this build
# context), so it is uploaded by the factory at runtime rather than baked here.
#
# Build context = envs/pi_env:
#   docker build -f hf_image/Dockerfile -t <user>/pi-rl:latest .
#   docker push <user>/pi-rl:latest

FROM python:3.12

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl ca-certificates xz-utils \
    && rm -rf /var/lib/apt/lists/*

# Proxy deps that interception.py imports.
RUN pip install --no-cache-dir "fastapi>=0.104" "uvicorn[standard]>=0.24" "httpx>=0.27"

# Pi's launcher uses `#!/usr/bin/env node`; keep both pre-baked tools available
# while building the image and when the sandbox executes Pi at runtime.
ENV PATH="/root/.node/bin:/root/.pi-npm/bin:${PATH}"

# Node 22 (pi requires >=22.19) + the pi CLI, at the paths pi_runtime expects.
RUN A=$(uname -m); case $A in x86_64) A=x64;; aarch64|arm64) A=arm64;; esac \
    && curl -fsSL "https://nodejs.org/dist/v22.19.0/node-v22.19.0-linux-$A.tar.xz" | tar -xJ -C /root \
    && ln -sfn "/root/node-v22.19.0-linux-$A" /root/.node \
    && /root/.node/bin/npm install -g --prefix /root/.pi-npm @mariozechner/pi-coding-agent \
    && /root/.pi-npm/bin/pi --version

# Directory layout the harness expects.
RUN mkdir -p /root/.pi/agent \
             /root/logs/agent \
             /root/logs/verifier \
             /root/task \
             /root/workdir \
             /root/proxy

WORKDIR /root/workdir
