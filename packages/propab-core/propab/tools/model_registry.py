"""
Model registry: in-process store with optional Redis persistence.

Tools that build models (build_mlp, build_transformer) write here.
Tools that consume models (train_model, compare_optimizers, compute_flops)
read from here. Because Celery uses process isolation, the in-process dict
may not have entries created in a different task. Redis persistence bridges
this gap so model configs survive across task boundaries within the same
hypothesis run (or across hypotheses in the same session).

Only the architecture config (dims, kind, activation, param_count) is stored
in Redis — not model weights — which is sufficient for compute_flops and
for rebuilding fresh model instances in train_model.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

# ─── In-process fast path ─────────────────────────────────────────────────────
_STORE: dict[str, dict[str, Any]] = {}

# ─── Redis slow path (optional) ───────────────────────────────────────────────
_REDIS_TTL = 1800  # 30 minutes
_redis_client = None


def _get_redis():
    global _redis_client
    if _redis_client is not None:
        return _redis_client
    try:
        import redis as _redis_lib  # sync redis-py
        url = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
        _redis_client = _redis_lib.from_url(url, socket_timeout=2, socket_connect_timeout=2, decode_responses=True)
        _redis_client.ping()
        return _redis_client
    except Exception as exc:
        logger.debug("Redis model registry unavailable: %s", exc)
        _redis_client = None
        return None


def _redis_key(model_id: str) -> str:
    return f"propab:model_registry:{model_id}"


# ─── Public API ───────────────────────────────────────────────────────────────

def put_model(model_id: str, info: dict[str, Any]) -> None:
    """Store model config in-process and optionally in Redis."""
    _STORE[model_id] = info
    try:
        r = _get_redis()
        if r is not None:
            r.setex(_redis_key(model_id), _REDIS_TTL, json.dumps(info, default=str))
    except Exception as exc:
        logger.debug("Redis put_model failed (non-fatal): %s", exc)


def get_model(model_id: str) -> dict[str, Any] | None:
    """Return model config, checking in-process cache first then Redis."""
    # Fast path
    info = _STORE.get(model_id)
    if info is not None:
        return info
    # Slow path: Redis
    try:
        r = _get_redis()
        if r is not None:
            raw = r.get(_redis_key(model_id))
            if raw:
                info = json.loads(raw)
                _STORE[model_id] = info  # warm local cache
                return info
    except Exception as exc:
        logger.debug("Redis get_model failed (non-fatal): %s", exc)
    return None


def _param_count_from_dims(dims: list[int] | None) -> int:
    """Dense MLP parameter count from layer dims (weights + biases)."""
    if not dims or len(dims) < 2:
        return 0
    total = 0
    for i in range(len(dims) - 1):
        total += dims[i] * dims[i + 1] + dims[i + 1]
    return int(total)


def resolve_model(model_id: str) -> dict[str, Any] | None:
    """Resolve a model_id robustly across the build → train chain.

    Agents frequently pass the wrong handle to downstream tools: a base build id to an
    eval tool, or a ``<id>:trained`` handle to a profiling tool. Both should work. This
    looks up the exact id, then tries the trained variant, then the base variant, and
    backfills ``param_count`` from ``dims`` (carried from the base build) so profiling
    and counting tools never silently report 0.
    """
    mid = str(model_id or "").strip()
    if not mid:
        return None

    candidates = [mid]
    if mid.endswith(":trained"):
        candidates.append(mid[: -len(":trained")])
    else:
        candidates.append(f"{mid}:trained")

    info: dict[str, Any] | None = None
    for cand in candidates:
        got = get_model(cand)
        if got is not None:
            info = dict(got)
            break
    if info is None:
        return None

    # Backfill param_count from the base entry / dims so trained handles profile correctly.
    if not info.get("param_count"):
        base_id = info.get("base")
        base_info = get_model(str(base_id)) if base_id else None
        if base_info and base_info.get("param_count"):
            info["param_count"] = int(base_info["param_count"])
            info.setdefault("activation", base_info.get("activation"))
        else:
            pc = _param_count_from_dims(info.get("dims"))
            if pc:
                info["param_count"] = pc
    return info


def clear_model(model_id: str) -> None:
    _STORE.pop(model_id, None)
    try:
        r = _get_redis()
        if r is not None:
            r.delete(_redis_key(model_id))
    except Exception:
        pass
