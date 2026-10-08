FROM python:3.13-slim
WORKDIR /opt/openenv
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY envs/echo_env ./envs/echo_env
RUN test -x /bin/tar \
    && pip install --no-cache-dir -e . -e ./envs/echo_env \
    && groupadd --gid 1000 sandbox \
    && useradd --uid 1000 --gid 1000 --no-create-home --home-dir /sandbox sandbox \
    && mkdir -p /sandbox \
    && chown 1000:1000 /sandbox
USER 1000:1000
WORKDIR /sandbox
