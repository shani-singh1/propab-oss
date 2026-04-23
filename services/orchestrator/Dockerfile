FROM python:3.11-bookworm

RUN apt-get update \
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

RUN pip install --no-cache-dir -e .

EXPOSE 8010
CMD ["uvicorn", "services.orchestrator.main:app", "--host", "0.0.0.0", "--port", "8010"]
