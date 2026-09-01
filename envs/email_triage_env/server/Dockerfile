ARG BASE_IMAGE=ghcr.io/meta-pytorch/openenv-base:latest
FROM ${BASE_IMAGE}

WORKDIR /app

COPY . /app

ENV PYTHONPATH=/app/src:/app
ENV HOST=0.0.0.0
ENV PORT=8000

CMD ["sh", "-c", "if [ -d /app/server ]; then uvicorn server.app:app --host 0.0.0.0 --port 8000; else uvicorn envs.email_triage_env.server.app:app --host 0.0.0.0 --port 8000; fi"]
