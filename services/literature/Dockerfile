# syntax=docker/dockerfile:1.7
FROM python:3.11-bookworm

RUN --mount=type=cache,target=/var/cache/apt,id=apt-literature,sharing=locked \
    apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY services/literature/requirements.txt ./requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

COPY services/literature/app ./services/literature/app
COPY services/__init__.py ./services/__init__.py

ENV PYTHONPATH=/app
EXPOSE 8020
CMD ["uvicorn", "services.literature.app.main:app", "--host", "0.0.0.0", "--port", "8020"]
