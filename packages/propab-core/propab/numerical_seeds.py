"""Cross-campaign numerical seed extraction (fixes.md Track A2, D2)."""
from __future__ import annotations

import re
from typing import Any

THRESHOLDS = (0.95, 0.90, 0.80, 0.70, 0.60)

# Scoped-claim template lines appended to every hypothesis — must not drive diversity buckets.
_SCOPE_BOILERPLATE_MARKERS = (
    "vector space f_3",
    "admissible combinatorial structures",
    "claimed generalization: structural/density",
    "expected failure modes: greedy",
)


def claim_core_for_bucket(text: str) -> str:
    """First falsifiable claim lines only — strip enrich_entry_with_scope boilerplate."""
    lines: list[str] = []
    for raw in (text or "").splitlines():
        ln = raw.strip()
        if not ln:
            continue
        low = ln.lower()
        if ln.startswith("Population:") and any(m in low for m in _SCOPE_BOILERPLATE_MARKERS):
            continue
        if ln.startswith("Distribution:") and "admissible combinatorial" in low:
            continue
        if ln.startswith("Claimed generalization:") or ln.startswith("Expected failure modes:"):
            continue
        lines.append(ln)
    core = "\n".join(lines).strip()
    return core or (text or "")


def claim_has_numeric_falsifier(text: str, methodology: str = "") -> bool:
    """Reject structural/monotonic claims with no numeric band, threshold, or n-scale (B2)."""
    core = claim_core_for_bucket(text)
    t = f"{core}\n{methodology}".lower()
    if re.search(r"\d\.\d{2,4}", t):
        return True
    if re.search(r"\bn\s*[=:]\s*\d{3,6}\b", t):
        return True
    if re.search(r"n\s+in\s+\[\s*\d", t):
        return True
    if re.search(r"below\s+0\.\d+|above\s+0\.\d+|band\s*\[|threshold|first\s+n\s+where", t):
        return True
    if re.search(r"f_\d+\^?\d+|dim(?:ension)?\s*[=:]?\s*\d", t):
        return True
    structural_only = any(
        k in t for k in ("monoton", "structural", "variance", "distribution", "geometric")
    )
    return not structural_only


def _node_evidence(node: dict[str, Any]) -> dict[str, Any]:
    finding = node.get("finding") or {}
    evs = node.get("evidence_summary")
    if isinstance(finding, dict) and finding:
        return finding
    if isinstance(evs, dict) and evs:
        return evs
    if isinstance(evs, str) and evs.strip():
        from propab.hypothesis_tree import HypothesisTree

        parsed = HypothesisTree._parse_evidence_blob(evs)
        if parsed:
            return parsed
    return {}


def _parse_sweep_points(node: dict[str, Any]) -> list[tuple[int, float]]:
    ev = _node_evidence(node)
    points: list[tuple[int, float]] = []
    sweep = ev.get("sweep") or []
    for pt in sweep:
        if not isinstance(pt, dict):
            continue
        n = pt.get("n")
        ratio = pt.get("ratio_to_sqrt_n")
        if n is not None and ratio is not None:
            points.append((int(n), float(ratio)))
    if not points:
        notes = str(ev.get("notes") or "")
        m = re.search(r"ratios=\[([^\]]+)\]", notes)
        ns_m = re.search(r"n=(\[[^\]]+\]|\{[^\}]+\})", notes)
        if m:
            ratios = [float(x.strip()) for x in m.group(1).split(",") if x.strip()]
            ns: list[int] = []
            if ns_m:
                ns = [int(x) for x in re.findall(r"\d+", ns_m.group(1))]
            for i, r in enumerate(ratios):
                if i < len(ns):
                    points.append((ns[i], r))
    return sorted(set(points))


def _parse_text_sweep_points(node: dict[str, Any]) -> list[tuple[int, float]]:
    """Fallback when verifier metrics were not persisted on the node dict."""
    text = str(node.get("text") or "")
    if "sidon" not in text.lower():
        return []
    points: list[tuple[int, float]] = []
    for m in re.finditer(
        r"F\((\d+)\)/sqrt\(\1\)[^\d]{0,48}(\d\.\d{2,4})",
        text,
        re.I,
    ):
        points.append((int(m.group(1)), float(m.group(2))))
    for m in re.finditer(
        r"For n=(\d{3,6})[^.]{0,120}?fall below (\d\.\d{2,4})",
        text,
        re.I,
    ):
        points.append((int(m.group(1)), float(m.group(2))))
    for m in re.finditer(
        r"(?:for|at)\s+n=(\d{4,6})[^\d]{0,40}(\d\.\d{2,4})",
        text,
        re.I,
    ):
        points.append((int(m.group(1)), float(m.group(2))))
    return sorted(set(points))


def crossing_from_node(node: dict[str, Any]) -> dict[str, Any] | None:
    """Structured threshold_search from verifier output, if present."""
    ev = _node_evidence(node)
    ts = ev.get("threshold_search")
    if not isinstance(ts, dict):
        return None
    crossing_n = ts.get("crossing_n")
    if crossing_n is None:
        return None
    return {
        "crossing_n": int(crossing_n),
        "crossing_ratio": ts.get("crossing_ratio"),
        "target_ratio": ts.get("target_ratio"),
        "prev_n": ts.get("prev_n"),
        "prev_ratio": ts.get("prev_ratio"),
    }


def _threshold_search_seed(node: dict[str, Any], crossing: dict[str, Any]) -> dict[str, Any]:
    target = crossing.get("target_ratio") or 0.60
    cn = int(crossing["crossing_n"])
    ratio = crossing.get("crossing_ratio")
    ratio_s = f"{float(ratio):.4f}" if ratio is not None else "?"
    return {
        "finding_type": "threshold_crossing",
        "claim": f"F(n)/sqrt(n) first drops below {target} at n={cn} (ratio={ratio_s})",
        "parameters": {
            "threshold": float(target),
            "crossing_n": cn,
            "ratio": ratio,
            "prev_n": crossing.get("prev_n"),
        },
        "source_node_ids": [str(node.get("id") or "")],
        "next_hypotheses": [
            f"Pin the exact crossing n below {target} between n={crossing.get('prev_n')} and n={cn}",
        ],
    }


def best_q1_crossing_from_nodes(
    nodes: dict[str, Any] | list[dict[str, Any]],
    *,
    target: float = 0.60,
) -> dict[str, Any] | None:
    """Best (smallest) confirmed crossing_n at or below target from tree nodes."""
    node_list = nodes.values() if isinstance(nodes, dict) else nodes
    best: dict[str, Any] | None = None
    for node in node_list:
        if not isinstance(node, dict):
            continue
        if str(node.get("verdict") or "") != "confirmed":
            continue
        crossing = crossing_from_node(node)
        if not crossing:
            continue
        ts_target = crossing.get("target_ratio")
        if ts_target is not None and abs(float(ts_target) - target) > 0.02:
            continue
        cn = int(crossing["crossing_n"])
        if best is None or cn < int(best["crossing_n"]):
            best = {**crossing, "node_id": str(node.get("id") or "")}
    return best


def in_campaign_crossing_context_block(
    crossing: dict[str, Any],
    *,
    target: float = 0.60,
) -> str:
    """Factual in-campaign crossing summary for lifetime context (no experiment directives)."""
    cn = int(crossing["crossing_n"])
    ratio = crossing.get("crossing_ratio")
    ratio_s = f"{float(ratio):.4f}" if ratio is not None else "?"
    return (
        f"In-campaign finding: F(n)/sqrt(n) crossed below {target:.2f} at n={cn} (ratio={ratio_s})."
    )


def refresh_lifetime_context_with_crossings(
    lifetime_context: str,
    tree: Any,
    question: str,
) -> str:
    """Append confirmed threshold-crossing facts from the current tree (evidence only)."""
    from propab.synthesis_diversity import q1_threshold_target

    target = q1_threshold_target(question)
    node_dicts = {
        nid: (n.to_dict() if hasattr(n, "to_dict") else n)
        for nid, n in tree.nodes.items()
    }
    crossing = best_q1_crossing_from_nodes(node_dicts, target=target)
    if not crossing:
        return lifetime_context
    block = in_campaign_crossing_context_block(crossing, target=target)
    marker = "In-campaign finding:"
    if marker in lifetime_context:
        before = lifetime_context.split(marker)[0].rstrip()
        return f"{before}\n\n{block}".strip()
    base = (lifetime_context or "").strip()
    return f"{base}\n\n{block}".strip() if base else block


def extract_math_combinatorics_seeds(confirmed_nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Extract structured numerical seeds from confirmed hypothesis nodes."""
    seeds: list[dict[str, Any]] = []
    seen_crossings: set[tuple[float, int]] = set()

    for node in confirmed_nodes:
        crossing = crossing_from_node(node)
        if not crossing:
            continue
        target = float(crossing.get("target_ratio") or 0.60)
        cn = int(crossing["crossing_n"])
        key = (target, cn)
        if key in seen_crossings:
            continue
        seen_crossings.add(key)
        seeds.append(_threshold_search_seed(node, crossing))

    all_points: list[tuple[int, float, str]] = []
    for node in confirmed_nodes:
        nid = str(node.get("id") or "")
        for n, ratio in _parse_sweep_points(node):
            all_points.append((n, ratio, nid))
        for n, ratio in _parse_text_sweep_points(node):
            all_points.append((n, ratio, nid))

    if not all_points:
        for node in confirmed_nodes:
            ev = _node_evidence(node)
            if ev.get("metric_name") != "sidon_ratio_to_sqrt_n":
                continue
            n = ev.get("n") or ev.get("max_n")
            ratio = ev.get("metric_value") or ev.get("mean_ratio_to_sqrt_n")
            if n and ratio is not None:
                all_points.append((int(n), float(ratio), str(node.get("id") or "")))

    by_n: dict[int, tuple[float, str]] = {}
    for n, ratio, nid in all_points:
        if n not in by_n or ratio < by_n[n][0]:
            by_n[n] = (ratio, nid)

    if len(by_n) < 2:
        return seeds

    sorted_ns = sorted(by_n)
    ratios = [by_n[n][0] for n in sorted_ns]

    if all(ratios[i] >= ratios[i + 1] for i in range(len(ratios) - 1)):
        seeds.append({
            "finding_type": "monotonic_trend",
            "claim": (
                f"Greedy Sidon F(n)/sqrt(n) monotonically decreasing at "
                f"n={sorted_ns} with ratios {[round(r, 4) for r in ratios]}"
            ),
            "parameters": {"n_values": sorted_ns, "ratios": ratios},
            "source_node_ids": list({by_n[n][1] for n in sorted_ns}),
            "next_hypotheses": [
                "Is the descent rate constant between consecutive doublings of n?",
                "Where does F(n)/sqrt(n) first drop below 0.60?",
            ],
        })

    for target in THRESHOLDS:
        crossing_n = None
        for n in sorted_ns:
            if by_n[n][0] < target:
                crossing_n = n
                break
        if crossing_n is None:
            continue
        key = (target, crossing_n)
        if key in seen_crossings:
            continue
        seen_crossings.add(key)
        prev_n = max(n for n in sorted_ns if n < crossing_n) if any(n < crossing_n for n in sorted_ns) else None
        seeds.append({
            "finding_type": "threshold_crossing",
            "claim": f"F(n)/sqrt(n) first drops below {target} at n={crossing_n} (ratio={by_n[crossing_n][0]:.4f})",
            "parameters": {"threshold": target, "crossing_n": crossing_n, "ratio": by_n[crossing_n][0]},
            "source_node_ids": [by_n[crossing_n][1]],
            "next_hypotheses": [
                f"Does F(n)/sqrt(n) drop below {target - 0.10:.2f} before n={crossing_n * 2}?",
            ],
        })

    return seeds


def format_seeds_for_question(seeds: list[dict[str, Any]], *, max_items: int = 12) -> str:
    if not seeds:
        return ""
    lines = ["Established numerical baseline from prior campaigns:"]
    for seed in seeds[:max_items]:
        lines.append(f"- [{seed.get('finding_type')}] {seed.get('claim')}")
        for nh in (seed.get("next_hypotheses") or [])[:2]:
            lines.append(f"  Open follow-up: {nh}")
    return "\n".join(lines)


def classify_hypothesis_bucket(text: str, methodology: str = "") -> dict[str, str]:
    """Diversity buckets for synthesis tracking (fixes.md D2)."""
    core = claim_core_for_bucket(text)
    t = f"{core}\n{methodology}".lower()
    if any(k in t for k in ("sidon", "f(n)/sqrt", "f(n)/√n", "greedy sidon")):
        problem = "sidon"
    elif "ap-free" in t or "arithmetic progression" in t or "ap free" in t:
        problem = "ap_free"
    elif any(k in t for k in ("cap set", "cap-set", "clp")) or (
        "f_3" in t and "sidon" not in t and "ap-free" not in t
    ):
        problem = "cap_set"
    elif "sumset" in t or "|a+a|" in t:
        problem = "sumset"
    elif "bose" in t or "chowla" in t:
        problem = "bc_comparison"
    else:
        problem = "sidon"

    n_vals = [int(x) for x in re.findall(r"\bn\s*[=:]\s*(\d{3,6})\b", t)]
    n_vals += [int(x) for x in re.findall(r"\{(\d{3,6})", t)]
    max_n = max(n_vals) if n_vals else 0
    if max_n >= 10000:
        scale = "large"
    elif max_n >= 1000:
        scale = "medium"
    else:
        scale = "small"

    if any(k in t for k in ("first", "cross", "threshold", "smallest n")):
        claim = "threshold"
    elif any(k in t for k in ("compare", "versus", "vs ", "exceed")):
        claim = "comparison"
    elif any(k in t for k in ("monoton", "decreas", "structural", "variance")):
        claim = "structural"
    else:
        claim = "density"

    return {"problem_type": problem, "parameter_scale": scale, "claim_type": claim}
