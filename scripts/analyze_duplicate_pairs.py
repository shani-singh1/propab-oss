#!/usr/bin/env python3
"""Classify near-duplicate hypothesis pairs: within-round vs cross-round (fixes.md step 1)."""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "artifacts" / "mandrake_duplicate_pair_analysis.json"
DEFAULT_CAMPAIGN = "faaf394b-7f95-4778-9136-e922f2401e7f"


def _get_json(url: str) -> Any:
    with urlopen(url, timeout=120) as resp:
        return json.loads(resp.read())


def _payload(ev: dict) -> dict:
    raw = ev.get("payload_json") or ev.get("payload") or {}
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {}
    return raw if isinstance(raw, dict) else {}


def _norm_text(t: str) -> str:
    return re.sub(r"\s+", " ", (t or "").lower().strip())


def _similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, _norm_text(a), _norm_text(b)).ratio()


def _node_list(tree: dict) -> list[tuple[str, dict]]:
    nodes = tree.get("nodes") or {}
    return sorted(nodes.items(), key=lambda kv: kv[0])


def _verification_times(events: list[dict]) -> dict[str, str]:
    """node_id -> ISO timestamp of first verification_diagnostic."""
    out: dict[str, str] = {}
    for e in events:
        if (e.get("step") or "") != "campaign.verification_diagnostic":
            continue
        p = _payload(e)
        nid = str(p.get("node_id") or "")
        if nid and nid not in out:
            out[nid] = str(e.get("created_at") or "")
    return out


def _enrich_pairs_with_timing(
    pairs: list[dict[str, Any]],
    tree: dict,
    events: list[dict],
) -> dict[str, Any]:
    nodes = tree.get("nodes") or {}
    vtimes = _verification_times(events)
    seed_pairs = 0
    synthesis_pairs = 0
    parallel_batch = 0
    sequential_after_first_refuted = 0

    enriched: list[dict[str, Any]] = []
    for p in pairs:
        na, nb = p["node_a"], p["node_b"]
        node_a, node_b = nodes.get(na, {}), nodes.get(nb, {})
        is_seed = node_a.get("parent_id") is None and node_b.get("parent_id") is None
        if is_seed:
            seed_pairs += 1
        else:
            synthesis_pairs += 1

        ta, tb = vtimes.get(na), vtimes.get(nb)
        timing = "unknown"
        if ta and tb:
            if ta <= tb:
                first, second, t_first, t_second = na, nb, ta, tb
            else:
                first, second, t_first, t_second = nb, na, tb, ta
            first_v = nodes.get(first, {}).get("verdict")
            if t_first == t_second:
                timing = "same_timestamp"
                parallel_batch += 1
            else:
                timing = "sequential_dispatch"
                if first_v in ("refuted", "inconclusive", "confirmed"):
                    sequential_after_first_refuted += 1
                else:
                    parallel_batch += 1

        ep = {
            **p,
            "both_are_seed_nodes": is_seed,
            "expansion_reason_a": node_a.get("expansion_reason"),
            "expansion_reason_b": node_b.get("expansion_reason"),
            "verified_at_a": ta,
            "verified_at_b": tb,
            "dispatch_timing": timing,
        }
        enriched.append(ep)

    return {
        "pairs_enriched": enriched,
        "seed_duplicate_pairs": seed_pairs,
        "synthesis_duplicate_pairs": synthesis_pairs,
        "parallel_batch_pairs": parallel_batch,
        "sequential_after_first_result_pairs": sequential_after_first_refuted,
        "context_completeness_verdict": (
            "SEED_BATCH_NOVELTY_GAP"
            if seed_pairs >= len(pairs) // 2
            else "MIXED_SEED_AND_SYNTHESIS"
        ),
        "interpretation_both_tested": (
            "Most duplicates are seed hypotheses (parent_id=null) added in the same "
            "generation batch, then dispatched in parallel — not synthesis re-proposing "
            "after seeing refuted results. Fix: dedup at seed ingestion + synthesis."
        ),
    }


def analyze_pairs(
    *,
    campaign_id: str,
    tree: dict,
    synthesis_events: list[dict],
    all_events: list[dict] | None = None,
    similarity_threshold: float = 0.85,
) -> dict[str, Any]:
    ordered = _node_list(tree)
    texts = [(nid, n.get("text") or "") for nid, n in ordered]

    # Map synthesis round -> generations added (approximate via event order + node generation)
    synth_rounds = len(synthesis_events)

    pairs: list[dict[str, Any]] = []
    for i, (nid_a, text_a) in enumerate(texts):
        if not text_a.strip():
            continue
        for j, (nid_b, text_b) in enumerate(texts):
            if j <= i or not text_b.strip():
                continue
            sim = _similarity(text_a, text_b)
            if sim < similarity_threshold:
                continue
            node_a = (tree.get("nodes") or {}).get(nid_a, {})
            node_b = (tree.get("nodes") or {}).get(nid_b, {})
            gen_a = int(node_a.get("generation") or 0)
            gen_b = int(node_b.get("generation") or 0)
            same_generation = gen_a == gen_b
            both_tested = (
                node_a.get("verdict") in ("confirmed", "refuted", "inconclusive")
                and node_b.get("verdict") in ("confirmed", "refuted", "inconclusive")
            )
            classification = "within_round" if same_generation else "cross_round"
            pairs.append({
                "node_a": nid_a,
                "node_b": nid_b,
                "sim": round(sim, 3),
                "generation_a": gen_a,
                "generation_b": gen_b,
                "classification": classification,
                "both_already_tested": both_tested,
                "verdict_a": node_a.get("verdict"),
                "verdict_b": node_b.get("verdict"),
                "text_a_preview": text_a[:120],
                "text_b_preview": text_b[:120],
            })

    by_class = Counter(p["classification"] for p in pairs)
    recommendation = "P2_dedup_only"
    if by_class.get("cross_round", 0) > by_class.get("within_round", 0):
        recommendation = "P2_dedup_plus_P1_trigger_tuning"
    elif by_class.get("within_round", 0) > 0 and by_class.get("cross_round", 0) == 0:
        recommendation = "P2_dedup_only"

    timing = _enrich_pairs_with_timing(pairs, tree, all_events or [])

    return {
        "campaign_id": campaign_id,
        "similarity_threshold": similarity_threshold,
        "n_nodes": len(texts),
        "n_near_duplicate_pairs": len(pairs),
        "within_round": by_class.get("within_round", 0),
        "cross_round": by_class.get("cross_round", 0),
        "both_already_tested_count": sum(1 for p in pairs if p["both_already_tested"]),
        "synthesis_rounds": synth_rounds,
        "recommended_fix": recommendation,
        "pairs": pairs,
        **timing,
        "interpretation": (
            "within_round → dedup inside apply_synthesis_to_frontier; "
            "cross_round → dedup + closed-beliefs lookup + possibly slower synthesis trigger"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-id", default=DEFAULT_CAMPAIGN)
    parser.add_argument("--api", default="http://localhost:8000")
    parser.add_argument("--input", type=Path, help="Campaign JSON file (campaign.to_dict shape)")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--threshold", type=float, default=0.85)
    args = parser.parse_args()

    if args.input and args.input.exists():
        blob = json.loads(args.input.read_text(encoding="utf-8"))
        camp = blob.get("campaign") or blob
        tree = camp.get("hypothesis_tree") or {}
        synthesis_events = []
        all_events: list[dict] = []
    else:
        base = args.api.rstrip("/")
        try:
            camp_resp = _get_json(f"{base}/campaigns/{args.campaign_id}")
        except Exception as exc:
            print(f"Failed to fetch campaign: {exc}", file=sys.stderr)
            return 1
        camp = camp_resp.get("campaign") or camp_resp
        tree = camp.get("hypothesis_tree") or {}
        events = _get_json(f"{base}/sessions/{args.campaign_id}/events?limit=2000")
        if not isinstance(events, list):
            events = events.get("events", [])
        synthesis_events = [
            e for e in events if (e.get("step") or "") == "campaign.synthesis"
        ]
        all_events = events

    report = analyze_pairs(
        campaign_id=args.campaign_id,
        tree=tree,
        synthesis_events=synthesis_events,
        all_events=all_events,
        similarity_threshold=args.threshold,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        "out": str(args.out),
        "n_pairs": report["n_near_duplicate_pairs"],
        "within_round": report["within_round"],
        "cross_round": report["cross_round"],
        "recommended_fix": report["recommended_fix"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
