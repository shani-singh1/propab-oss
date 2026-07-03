"""Domain-specific belief promotion (fixes.md Track A1)."""
from __future__ import annotations

import re
from typing import Any

SIDON_METRICS = frozenset({
    "sidon_ratio_to_sqrt_n",
    "sidon_density",
    "sidon_set_density",
})
CAP_METRICS = frozenset({"cap_set_clp_ratio", "cap_set_density"})
BC_METRICS = frozenset({"bose_chowla_vs_greedy_ratio", "bc_matched_win_rate"})


def _belief_subject(text: str) -> str:
    t = text.lower()
    if any(k in t for k in ("cap set", "cap-set", "f_3", "clp")):
        return "cap_set"
    if any(k in t for k in ("bose-chowla", "bose chowla", "bc ")):
        return "bc"
    if any(k in t for k in ("sidon", "f(n)", "sqrt")):
        return "sidon"
    return "other"


def _node_evidence(node: dict[str, Any]) -> dict[str, Any]:
    finding = node.get("finding") or {}
    evs = node.get("evidence_summary") or {}
    if isinstance(finding, dict) and finding:
        return finding
    if isinstance(evs, dict):
        return evs
    return {}


def _extract_n_from_node(node: dict[str, Any]) -> int | None:
    text = str(node.get("text") or "")
    for m in re.finditer(r"\bn\s*=\s*(\d+)\b", text, re.I):
        return int(m.group(1))
    m = re.search(r"n\s+in\s+\[(\d+)", text, re.I)
    if m:
        return int(m.group(1))
    ev = _node_evidence(node)
    if ev.get("n") is not None:
        return int(ev["n"])
    sweep = ev.get("sweep") or ev.get("comparison_sweep")
    if isinstance(sweep, list) and sweep:
        first = sweep[0]
        if isinstance(first, dict) and first.get("n") is not None:
            return int(first["n"])
    return None


def _metric_matches_subject(metric_name: str | None, subject: str) -> bool:
    m = str(metric_name or "")
    if subject == "sidon":
        return m in SIDON_METRICS
    if subject == "cap_set":
        return m in CAP_METRICS
    if subject == "bc":
        return m in BC_METRICS
    return True


def collect_confirmed_metric_nodes(
    tree_nodes: dict[str, Any],
    *,
    subject: str,
    min_count: int = 3,
) -> list[tuple[str, float, int | None]]:
    """Return (node_id, metric_value, n) for confirmed nodes matching subject."""
    rows: list[tuple[str, float, int | None]] = []
    for nid, node in tree_nodes.items():
        if not isinstance(node, dict):
            continue
        if node.get("verdict") != "confirmed":
            continue
        ev = _node_evidence(node)
        metric = ev.get("metric_name")
        if not _metric_matches_subject(metric, subject):
            continue
        val = ev.get("metric_value")
        if val is None:
            val = ev.get("mean_ratio_to_sqrt_n")
        if val is None:
            continue
        try:
            fval = float(val)
        except (TypeError, ValueError):
            continue
        rows.append((str(nid), fval, _extract_n_from_node(node)))
    if len(rows) < min_count:
        return []
    rows.sort(key=lambda r: (r[2] if r[2] is not None else 10**9, r[0]))
    return rows


def is_consistent_trend(values: list[float], *, decreasing: bool | None = None) -> bool:
    if len(values) < 3:
        return False
    if decreasing is None:
        decreasing = values[0] > values[-1]
    if decreasing:
        return all(values[i] >= values[i + 1] for i in range(len(values) - 1))
    return all(values[i] <= values[i + 1] for i in range(len(values) - 1))


def try_trend_promotion(
    belief_statement: str,
    tree_nodes: dict[str, Any],
    threshold: dict[str, Any],
) -> list[str] | None:
    """
    If trend promotion applies, return supporting node IDs; else None.
    """
    if not threshold.get("allow_trend_promotion"):
        return None
    min_nodes = int(threshold.get("requires_supporting_nodes") or 3)
    subject = _belief_subject(belief_statement)
    rows = collect_confirmed_metric_nodes(tree_nodes, subject=subject, min_count=min_nodes)
    if len(rows) < min_nodes:
        return None
    values = [r[1] for r in rows]
    stmt = belief_statement.lower()
    want_decrease = any(k in stmt for k in ("decreas", "monoton", "descend", "fall", "drop", "below"))
    want_increase = any(k in stmt for k in ("increas", "grow", "rise"))
    if want_decrease and not want_increase:
        if not is_consistent_trend(values, decreasing=True):
            return None
    elif want_increase and not want_decrease:
        if not is_consistent_trend(values, decreasing=False):
            return None
    elif not is_consistent_trend(values, decreasing=values[0] > values[-1]):
        return None
    return [r[0] for r in rows[: max(min_nodes, 5)]]


def apply_trend_promotion_to_beliefs(
    belief_state: Any,
    tree_nodes: dict[str, Any],
    threshold: dict[str, Any],
) -> int:
    """Promote ungrounded beliefs when trend threshold met. Returns count promoted."""
    if not threshold.get("allow_trend_promotion"):
        return 0
    promoted = 0
    still_ungrounded: list[Any] = []
    for belief in belief_state.proposed_ungrounded_beliefs:
        node_ids = try_trend_promotion(belief.statement, tree_nodes, threshold)
        if node_ids:
            belief.supporting_nodes = node_ids
            belief.confidence = "weak"  # type: ignore[assignment]
            belief.status = "active"
            belief_state.active_beliefs.append(belief)
            promoted += 1
        else:
            still_ungrounded.append(belief)
    belief_state.proposed_ungrounded_beliefs = still_ungrounded
    cap = 3
    if len(belief_state.active_beliefs) > cap:
        belief_state.active_beliefs = belief_state.active_beliefs[:cap]
    return promoted
