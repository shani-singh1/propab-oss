"""Compare root hypotheses between two campaigns (with vs without literature prior)."""
from __future__ import annotations

import json
import re
import urllib.request
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").lower()).strip()


def _similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, _normalize(a), _normalize(b)).ratio()


def fetch_campaign_hypotheses(campaign_id: str, *, api: str = "http://localhost:8000") -> list[dict[str, Any]]:
    url = f"{api.rstrip('/')}/campaigns/{campaign_id}"
    with urllib.request.urlopen(url, timeout=60) as resp:
        data = json.loads(resp.read())
    campaign = data.get("campaign") or {}
    tree = campaign.get("hypothesis_tree") or {}
    nodes_map = tree.get("nodes") or {}
    if isinstance(nodes_map, dict):
        nodes = list(nodes_map.values())
    else:
        nodes = nodes_map if isinstance(nodes_map, list) else []
    out: list[dict[str, Any]] = []
    for node in nodes:
        if not isinstance(node, dict):
            continue
        depth = node.get("depth", node.get("level", 0))
        parent_id = node.get("parent_id")
        if parent_id:
            continue
        text = node.get("text") or node.get("hypothesis") or ""
        if text:
            out.append({
                "id": node.get("id"),
                "text": text,
                "status": node.get("status") or node.get("verdict"),
                "depth": depth,
            })
    return out


def fetch_hypotheses_from_events(campaign_id: str, *, api: str = "http://localhost:8000") -> list[str]:
    """Fallback: pull dispatched hypothesis texts from session events."""
    url = f"{api.rstrip('/')}/sessions/{campaign_id}/events?limit=500"
    with urllib.request.urlopen(url, timeout=60) as resp:
        data = json.loads(resp.read())
    texts: list[str] = []
    seen: set[str] = set()
    for ev in data.get("events") or []:
        if ev.get("event_type") != "hypothesis.dispatched":
            continue
        p = ev.get("payload_json") or {}
        text = p.get("text") or p.get("hypothesis_text") or ""
        norm = _normalize(text)
        if norm and norm not in seen:
            seen.add(norm)
            texts.append(text)
    return texts


def compare_hypothesis_sets(
    baseline: list[str],
    with_prior: list[str],
    *,
    similarity_threshold: float = 0.72,
) -> dict[str, Any]:
    matched: list[dict[str, Any]] = []
    novel_with_prior: list[str] = []
    used_baseline: set[int] = set()

    for hyp in with_prior:
        best_i, best_score = -1, 0.0
        for i, base in enumerate(baseline):
            if i in used_baseline:
                continue
            score = _similarity(hyp, base)
            if score > best_score:
                best_score, best_i = score, i
        if best_i >= 0 and best_score >= similarity_threshold:
            used_baseline.add(best_i)
            matched.append({
                "with_prior": hyp,
                "baseline": baseline[best_i],
                "similarity": round(best_score, 3),
            })
        else:
            novel_with_prior.append(hyp)

    only_baseline = [h for i, h in enumerate(baseline) if i not in used_baseline]

    def _token_set(texts: list[str]) -> Counter[str]:
        words: list[str] = []
        for t in texts:
            words.extend(re.findall(r"[a-z]{4,}", _normalize(t)))
        return Counter(words)

    base_tokens = _token_set(baseline)
    prior_tokens = _token_set(with_prior)
    new_terms = [w for w, c in prior_tokens.items() if c >= 2 and base_tokens.get(w, 0) == 0]

    return {
        "baseline_count": len(baseline),
        "with_prior_count": len(with_prior),
        "matched_pairs": matched,
        "novel_with_prior": novel_with_prior,
        "only_in_baseline": only_baseline,
        "novel_term_hints": sorted(new_terms)[:25],
        "overlap_ratio": round(len(matched) / max(1, len(with_prior)), 3),
    }


def load_hypothesis_snapshot(path: Path) -> list[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return [str(x) for x in data]
    return list(data.get("hypotheses") or data.get("texts") or [])


def snapshot_campaign(
    campaign_id: str,
    out_path: Path,
    *,
    api: str = "http://localhost:8000",
) -> dict[str, Any]:
    hyps = fetch_campaign_hypotheses(campaign_id, api=api)
    texts = [h["text"] for h in hyps]
    if not texts:
        texts = fetch_hypotheses_from_events(campaign_id, api=api)
    payload = {
        "campaign_id": campaign_id,
        "hypothesis_count": len(texts),
        "hypotheses": texts,
        "nodes": hyps,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload
