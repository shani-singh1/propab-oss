# syntax=docker/dockerfile:1.7
FROM python:3.11-bookworm

RUN --mount=type=cache,target=/var/cache/apt,id=apt-worker,sharing=locked \
    apt-get update \
    && apt-get install -y --no-install-recommends docker.io ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md ARCHITECTURE.md ./
COPY packages ./packages
COPY services ./services
COPY migrations ./migrations

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -e ".[dl]"

CMD ["celery", "-A", "services.worker.celery_app", "worker", "--loglevel=info", "--concurrency=2"]
