FROM python:3.11-bookworm

RUN apt-get update \
    && apt-get install -y --no-install-recommends docker.io ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md ARCHITECTURE.md ./
COPY packages ./packages
COPY services ./services
COPY migrations ./migrations

RUN pip install --no-cache-dir -e ".[dl]"

CMD ["celery", "-A", "services.worker.celery_app", "worker", "--loglevel=info", "--concurrency=2"]
