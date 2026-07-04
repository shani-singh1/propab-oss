"""Cross-campaign numerical seed extraction (fixes.md Track A2, D2)."""
from __future__ import annotations

import re
from typing import Any

THRESHOLDS = (0.95, 0.90, 0.80, 0.70, 0.60)


def _node_evidence(node: dict[str, Any]) -> dict[str, Any]:
    finding = node.get("finding") or {}
    evs = node.get("evidence_summary") or {}
    if isinstance(finding, dict) and finding:
        return finding
    return evs if isinstance(evs, dict) else {}


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


def extract_math_combinatorics_seeds(confirmed_nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Extract structured numerical seeds from confirmed hypothesis nodes."""
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
        return []

    sorted_ns = sorted(by_n)
    ratios = [by_n[n][0] for n in sorted_ns]
    seeds: list[dict[str, Any]] = []

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
    t = f"{text}\n{methodology}".lower()
    if any(k in t for k in ("cap set", "cap-set", "f_3", "clp")):
        problem = "cap_set"
    elif "ap-free" in t or "arithmetic progression" in t:
        problem = "ap_free"
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
