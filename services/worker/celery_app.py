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

# Long MNIST / think-act traces can exceed Celery defaults; override via env in prod.
_soft = int(os.environ.get("CELERY_TASK_SOFT_TIME_LIMIT_SEC", "3600"))
_hard = int(os.environ.get("CELERY_TASK_TIME_LIMIT_SEC", str(_soft + 300)))

app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    task_track_started=True,
    task_soft_time_limit=_soft,
    task_time_limit=_hard,
    # Worker-loss resilience: a hypothesis task is only acked after it completes,
    # so if a worker is killed mid-hypothesis the broker redelivers it to another
    # worker instead of silently dropping it. Requires broker visibility_timeout
    # >= task_time_limit (Redis) so an in-flight task is not double-dispatched.
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    broker_transport_options={"visibility_timeout": _hard + 60},
)
