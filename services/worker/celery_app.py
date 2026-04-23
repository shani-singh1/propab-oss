from __future__ import annotations

import os

from celery import Celery

_broker = os.environ.get("CELERY_BROKER_URL") or os.environ.get("REDIS_URL", "redis://localhost:6379/0")

app = Celery(
    "propab",
    broker=_broker,
    result_backend=_broker,
    include=["services.worker.tasks"],
)

app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    task_track_started=True,
)
