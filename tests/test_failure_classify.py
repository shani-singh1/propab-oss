from __future__ import annotations

import asyncio

from services.worker.failure_classify import classify_exception, compact_failure_summary


class SoftTimeLimitExceeded(Exception):
    """Name-matched like Celery's soft limit (no celery import in unit test)."""


def test_classify_soft_time_limit_by_exception_name():
    e = SoftTimeLimitExceeded("worker soft time limit")
    assert classify_exception(e)["failure_kind"] == "celery_soft_time_limit"


def test_timeout_error_is_async_timeout():
    e = TimeoutError("read timed out")
    assert classify_exception(e)["failure_kind"] == "async_timeout"


def test_asyncio_timeout_error_alias():
    e = asyncio.TimeoutError()
    assert classify_exception(e)["failure_kind"] == "async_timeout"


def test_compact_failure_summary_reads_payload():
    s = compact_failure_summary(
        {"failure_kind": "http_timeout", "exc_types": "ReadTimeout", "message": "x"},
    )
    assert "http_timeout" in s
