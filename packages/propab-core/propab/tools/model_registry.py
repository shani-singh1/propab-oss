"""In-process registry for ephemeral ``model_id`` handles between DL tools."""

from __future__ import annotations

from typing import Any

_STORE: dict[str, dict[str, Any]] = {}


def put_model(model_id: str, info: dict[str, Any]) -> None:
    _STORE[model_id] = info


def get_model(model_id: str) -> dict[str, Any] | None:
    return _STORE.get(model_id)


def clear_model(model_id: str) -> None:
    _STORE.pop(model_id, None)
