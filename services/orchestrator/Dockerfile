# Phase 1: orchestrator service stub (health only). See stub_app.py.
FROM python:3.12-slim-bookworm
RUN pip install --no-cache-dir "fastapi>=0.115.0" "uvicorn[standard]>=0.32.0"
WORKDIR /app
COPY services/orchestrator/stub_app.py ./stub_app.py
EXPOSE 8010
CMD ["uvicorn", "stub_app:app", "--host", "0.0.0.0", "--port", "8010"]
