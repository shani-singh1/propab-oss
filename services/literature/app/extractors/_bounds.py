"""Shared numeric-bound parsing used by both the contradiction detector and
the gap mapper — "X > 0.5" style assertions reduced to (subject, low, high)
intervals so two claims about the same quantity can be compared."""
from __future__ import annotations

import re

_BOUND_RE = re.compile(
    r"([A-Za-z_][\w\(\),./]{0,24})\s*(>=|<=|≥|≤|>|<|=|≈|\\approx)\s*"
    r"(-?\d+(?:\.\d+)?(?:\s*/\s*\d+(?:\.\d+)?)?)"
)
_EPS = 1e-9


def to_float(raw: str) -> float | None:
    raw = raw.strip()
    if "/" in raw:
        num, _, den = raw.partition("/")
        try:
            return float(num) / float(den)
        except (ValueError, ZeroDivisionError):
            return None
    try:
        return float(raw)
    except ValueError:
        return None


def parse_bounds(text: str) -> list[tuple[str, float, float]]:
    """Every comparison found in ``text`` as (normalized subject, low, high)."""
    out = []
    for subject, op, value_raw in _BOUND_RE.findall(text):
        value = to_float(value_raw)
        if value is None:
            continue
        subject_key = re.sub(r"\s+", "", subject.lower())
        if op == ">":
            out.append((subject_key, value + _EPS, float("inf")))
        elif op in (">=", "≥"):
            out.append((subject_key, value, float("inf")))
        elif op == "<":
            out.append((subject_key, -float("inf"), value - _EPS))
        elif op in ("<=", "≤"):
            out.append((subject_key, -float("inf"), value))
        elif op in ("=", "≈", "\\approx"):
            out.append((subject_key, value - 1e-6, value + 1e-6))
    return out


def intervals_disjoint(a: tuple[float, float], b: tuple[float, float]) -> bool:
    return a[1] < b[0] or b[1] < a[0]
