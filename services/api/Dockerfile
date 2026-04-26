# syntax=docker/dockerfile:1.7
FROM python:3.11-bookworm

RUN --mount=type=cache,target=/var/cache/apt,id=apt-api,sharing=locked \
    apt-get update \
    && apt-get install -y --no-install-recommends \
        texlive-latex-base \
        texlive-latex-recommended \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md ARCHITECTURE.md ./
COPY packages ./packages
COPY services ./services
COPY migrations ./migrations

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -e .

EXPOSE 8000
CMD ["uvicorn", "services.api.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
