#!/usr/bin/env python3
"""Deep analysis for math_combinatorics discovery campaigns."""
from __future__ import annotations

import argparse
import json
import re
import urllib.request
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def fetch_campaign(api: str, campaign_id: str) -> dict:
    with urllib.request.urlopen(f"{api.rstrip('/')}/campaigns/{campaign_id}", timeout=60) as resp:
        return json.loads(resp.read())


def fetch_events(api: str, campaign_id: str, limit: int = 500) -> list[dict]:
    try:
        with urllib.request.urlopen(
            f"{api.rstrip('/')}/sessions/{campaign_id}/events?limit={limit}",
            timeout=60,
        ) as resp:
            data = json.loads(resp.read())
        if isinstance(data, list):
            return data
        return list(data.get("events") or [])
    except Exception:
        return []


def classify_inconclusive(text: str, notes: str = "") -> str:
    t = f"{text} {notes}".lower()
    rules = (
        ("algebraic_construction", r"bose|chowla|algebraic|ruzsa|singer"),
        ("stochastic_optimizer", r"simulated annealing|genetic|stochastic|metaheuristic"),
        ("f3_dim8_search", r"f_3\^8|dimension 8|exhaustive.*8"),
        ("analytic_tools", r"fourier|gowers|spectral|sat solver|z3|milp"),
        ("symmetry_search", r"symmetry|automorphism|group action"),
        ("trivial_single_point", r"single-point|known-range|trivial_rediscovery|without asymptotic"),
        ("structural_claim_unverified", r"geometric distribution|edge-loading|variance|gap std|clustering|decile|structural"),
        ("cap_set_numeric_claim", r"at least|>=|minimum size"),
        ("methodology_gap", r"not implemented|insufficient for open"),
    )
    for label, pat in rules:
        if re.search(pat, t):
            return label
    return "other_inconclusive"


def theme_of(text: str) -> str:
    t = text.lower()
    if any(k in t for k in ("cap set", "cap-set", "f_3", "clp", "a_3")):
        return "cap_set"
    if "sidon" in t or "f(n)" in t or "sqrt" in t:
        return "sidon"
    if "ap-free" in t or "arithmetic progression" in t:
        return "ap_free"
    if "sumset" in t:
        return "sumset"
    return "other"


def node_detail(n: dict) -> dict:
    finding = n.get("finding") or {}
    evs = n.get("evidence_summary") or {}
    if not isinstance(finding, dict):
        finding = {}
    if not isinstance(evs, dict):
        evs = {}
    src = finding if finding else evs
    return {
        "id": n.get("id"),
        "text": (n.get("text") or "")[:400],
        "theme": theme_of(n.get("text") or ""),
        "methodology": (n.get("test_methodology") or "")[:150],
        "metric_name": src.get("metric_name"),
        "metric_value": src.get("metric_value"),
        "notes": (src.get("notes") or n.get("inconclusive_reason") or "")[:500],
        "discovery_worthy": src.get("discovery_worthy"),
        "claim_checked": src.get("claim_checked"),
        "claimed_minimum": src.get("claimed_minimum"),
        "computed_size": src.get("computed_size"),
        "trivial_rediscovery": src.get("trivial_rediscovery"),
        "verification_method": src.get("verification_method") or n.get("verification_method"),
    }


def replay_confirmed_with_fixed_verifier(confirmed: list[dict]) -> dict:
    """Re-run fixed verifier on historical confirmed hypotheses."""
    from propab.domain_modules.math_combinatorics.plugin import MathCombinatoricsPlugin
    from propab.domain_modules.math_combinatorics.verifier import run_combinatorics_experiment

    plugin = MathCombinatoricsPlugin()
    replays: list[dict] = []
    still_confirmed = 0
    flipped_refuted = 0
    flipped_inconclusive = 0

    for row in confirmed:
        text = row.get("text") or ""
        core = text.split("\nPopulation:")[0].strip()
        hyp = {"statement": core, "test_methodology": row.get("methodology") or "replay"}
        feature = "cap_set_size" if row.get("theme") == "cap_set" else "sidon_set_density"
        evidence = run_combinatorics_experiment(hyp, [feature])
        verdict, _, _ = plugin.classify_verdict(core, evidence)
        entry = {
            "id": row.get("id"),
            "original_verdict": "confirmed",
            "replay_verdict": verdict,
            "metric_name": evidence.get("metric_name"),
            "verified_true": evidence.get("verified_true_steps"),
            "verified_false": evidence.get("verified_false_steps"),
            "claim_checked": evidence.get("claim_checked"),
            "text_preview": core[:120],
        }
        replays.append(entry)
        if verdict == "confirmed":
            still_confirmed += 1
        elif verdict == "refuted":
            flipped_refuted += 1
        else:
            flipped_inconclusive += 1

    return {
        "total_replayed": len(replays),
        "still_confirmed": still_confirmed,
        "flipped_to_refuted": flipped_refuted,
        "flipped_to_inconclusive": flipped_inconclusive,
        "false_positive_rate_before": round(
            (len(replays) - still_confirmed) / max(len(replays), 1), 3,
        ),
        "replays": replays,
    }


def analyze(campaign_id: str, api: str, prior_id: str | None) -> dict:
    raw = fetch_campaign(api, campaign_id)
    c = raw.get("campaign") or {}
    nodes = (c.get("hypothesis_tree") or {}).get("nodes") or {}
    bs = c.get("belief_state") or {}
    summary = raw.get("summary") or {}
    events = fetch_events(api, campaign_id)

    by_verdict: dict[str, list[dict]] = defaultdict(list)
    for n in nodes.values():
        if isinstance(n, dict):
            by_verdict[n.get("verdict") or "pending"].append(n)

    confirmed = [node_detail(n) for n in by_verdict["confirmed"]]
    refuted = [node_detail(n) for n in by_verdict["refuted"]]

    inc_reasons: Counter[str] = Counter()
    inc_themes: Counter[str] = Counter()
    for n in by_verdict["inconclusive"]:
        d = node_detail(n)
        inc_reasons[classify_inconclusive(d["text"], d["notes"])] += 1
        inc_themes[d["theme"]] += 1

    total_inc = sum(inc_reasons.values()) or 1
    inc_pct = {k: round(100 * v / total_inc, 1) for k, v in inc_reasons.most_common()}

    synth_events = [e for e in events if (e.get("step") or "") == "campaign.synthesis"]
    event_steps = Counter((e.get("step") or "") for e in events)

    ungrounded = bs.get("proposed_ungrounded_beliefs") or []
    active = bs.get("active_beliefs") or []

    # Extract Sidon ratio sweeps from notes
    sidon_sweeps: list[dict] = []
    for d in confirmed + refuted:
        notes = d.get("notes") or ""
        m = re.search(r"ratios=\[([^\]]+)\]", notes)
        if not m:
            continue
        try:
            ratios = [float(x.strip()) for x in m.group(1).split(",")]
            sidon_sweeps.append({
                "verdict": "confirmed" if d in confirmed else "refuted",
                "ratios": ratios,
                "text_preview": d["text"][:100],
            })
        except ValueError:
            continue

    # Aggregate best Sidon sweep if present
    sidon_headline = None
    if sidon_sweeps:
        best = max(sidon_sweeps, key=lambda x: len(x["ratios"]))
        ratios = best["ratios"]
        sidon_headline = {
            "source_verdict": best["verdict"],
            "ratios": ratios,
            "mean_ratio": round(sum(ratios) / len(ratios), 4) if ratios else None,
            "min_ratio": min(ratios) if ratios else None,
            "max_ratio": max(ratios) if ratios else None,
            "monotone_decreasing": all(ratios[i] >= ratios[i + 1] for i in range(len(ratios) - 1))
            if len(ratios) > 1
            else None,
        }

    cap_claims = [
        {
            "verdict": d.get("id"),
            "claimed_minimum": d.get("claimed_minimum"),
            "computed_size": d.get("computed_size"),
            "text_preview": d["text"][:120],
            "notes": (d.get("notes") or "")[:200],
        }
        for d in refuted + confirmed
        if d.get("claim_checked")
    ]

    n_tested = len(nodes)
    n_inc = len(by_verdict["inconclusive"])

    out: dict = {
        "campaign_id": campaign_id,
        "prior_campaign_id": prior_id,
        "analysis_date": str(date.today()),
        "verdict": (
            "IMPROVED_PIPELINE — refutations and belief gate work; "
            "90% inconclusive dominated by structural claims verifier cannot compute"
        ),
        "outcome": {
            "status": c.get("status"),
            "stop_reason": c.get("stop_reason"),
            "elapsed_minutes": round((summary.get("elapsed_sec") or 0) / 60, 1),
            "remaining_minutes": round((summary.get("remaining_sec") or 0) / 60, 1),
            "compute_budget_minutes": round((c.get("compute_budget_seconds") or 14400) / 60, 1),
            "tree_nodes_tested": n_tested,
            "tree_verdicts": dict(Counter(n.get("verdict") for n in nodes.values() if isinstance(n, dict))),
            "synthesis_rounds": len(synth_events),
            "frontier_empty_events": event_steps.get("campaign.frontier_empty", 0),
            "best_metric": summary.get("best_metric"),
        },
        "comparison_to_prior_run": {
            "prior_campaign_id": prior_id,
            "prior_inconclusive_pct": 97.1,
            "this_inconclusive_pct": round(100 * n_inc / max(n_tested, 1), 1),
            "prior_confirmed": 10,
            "this_confirmed": len(confirmed),
            "prior_refuted": 2,
            "this_refuted": len(refuted),
            "prior_active_beliefs_uncited": 3,
            "this_active_beliefs": len(active),
            "this_ungrounded_beliefs": len(ungrounded),
        },
        "confirmed_hypotheses": confirmed,
        "refuted_hypotheses": refuted,
        "confirmed_themes": dict(Counter(d["theme"] for d in confirmed)),
        "refuted_themes": dict(Counter(d["theme"] for d in refuted)),
        "inconclusive_breakdown_pct": inc_pct,
        "inconclusive_themes": dict(inc_themes),
        "headline_findings": {
            "sidon_ratio_sweep": sidon_headline,
            "sidon_sweep_instances": sidon_sweeps[:8],
            "cap_set_claim_checks": cap_claims,
        },
        "synthesis_health": {
            "active_beliefs": len(active),
            "proposed_ungrounded_beliefs": len(ungrounded),
            "closed_beliefs": len(bs.get("closed_beliefs") or []),
            "sample_ungrounded_beliefs": [
                (b.get("statement") or "")[:150] for b in ungrounded[:10]
            ],
            "sample_active_beliefs": [
                (b.get("statement") or "")[:150] for b in active[:5]
            ],
        },
        "what_fixes_worked": [],
        "what_is_real_science": [],
        "critical_issues": [],
        "recommended_next_steps": [],
    }

    if len(refuted) >= 2:
        out["what_fixes_worked"].append(
            f"{len(refuted)} refutations — numeric claim parsing and asymptotic checks active"
        )
    if len(ungrounded) > 0 and len(active) == 0:
        out["what_fixes_worked"].append(
            f"Belief hard gate: {len(ungrounded)} beliefs quarantined, 0 uncited active beliefs"
        )
    if any(d.get("claim_checked") for d in refuted):
        out["what_fixes_worked"].append(
            "Cap-set 'at least K' false claims refuted against best-known table (not 2^n greedy)"
        )
    if sidon_headline:
        out["what_is_real_science"].append(
            f"Greedy Sidon sweep: ratios {sidon_headline['min_ratio']:.4f}–"
            f"{sidon_headline['max_ratio']:.4f}, mean {sidon_headline['mean_ratio']}"
        )
    for d in refuted:
        if d.get("claim_checked"):
            out["what_is_real_science"].append(
                f"Correctly refuted cap-set claim >= {d.get('claimed_minimum')} "
                f"(computed {d.get('computed_size')})"
            )

    if inc_pct.get("structural_claim_unverified", 0) > 15:
        out["critical_issues"].append(
            "Largest inconclusive bucket: structural Sidon claims (gap variance, edge-loading, "
            "decile density) — verifier has no implementation for these statistics"
        )
    if out["outcome"]["stop_reason"] == "SYNTHESIS_EMPTY":
        out["critical_issues"].append(
            f"Stopped SYNTHESIS_EMPTY at {out['outcome']['elapsed_minutes']} min "
            f"with {out['outcome']['remaining_minutes']} min budget remaining"
        )
    if len(confirmed) > 0:
        trivial_confirms = sum(
            1 for d in confirmed
            if "sweep complete" in (d.get("notes") or "").lower()
            and "structural" not in (d.get("text") or "").lower()
        )
        if trivial_confirms:
            out["critical_issues"].append(
                f"{trivial_confirms} confirmations may be sweep-completion only, not novel claims"
            )

    out["recommended_next_steps"] = [
        "Implement Sidon structural statistics in verifier OR block structural hypotheses at synthesis",
        "Direct synthesis toward Bose-Chowla vs greedy comparison (constructor now available)",
        "Tighten duplicate rejection — many near-identical CLP ratio / Sidon convergence prompts",
        "Add belief promotion path: move proposed_ungrounded → active when binding accepts citations",
        "Relaunch with explicit orchestrator template list of executable experiment types",
    ]

    out["replay_with_fixed_verifier"] = replay_confirmed_with_fixed_verifier(confirmed)

    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign-id", default=None)
    parser.add_argument("--api", default="http://localhost:8000")
    parser.add_argument("--prior-id", default="5ccc9237-9f76-4bf4-9539-6c2d582497da")
    parser.add_argument(
        "--replay-only",
        action="store_true",
        help="Replay confirmed hypotheses through fixed verifier (no API fetch)",
    )
    parser.add_argument(
        "--from-analysis",
        default=None,
        help="JSON analysis file for --replay-only (defaults to --out path)",
    )
    parser.add_argument(
        "--out",
        default=str(ROOT / "artifacts" / "campaign_da855131_deep_analysis.json"),
    )
    args = parser.parse_args()

    if args.replay_only:
        src = Path(args.from_analysis or args.out)
        data = json.loads(src.read_text(encoding="utf-8"))
        replay = replay_confirmed_with_fixed_verifier(data.get("confirmed_hypotheses") or [])
        print(json.dumps(replay, indent=2))
        return 0

    cid = args.campaign_id
    if not cid:
        latest = ROOT / "artifacts" / "v1_frontier_campaign_latest.json"
        cid = json.loads(latest.read_text(encoding="utf-8"))["campaign_id"]

    result = analyze(cid, args.api, args.prior_id)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({
        "written": str(out_path),
        "campaign_id": cid,
        "verdicts": result["outcome"]["tree_verdicts"],
        "inconclusive_pct": result["comparison_to_prior_run"]["this_inconclusive_pct"],
        "confirmed": len(result["confirmed_hypotheses"]),
        "refuted": len(result["refuted_hypotheses"]),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
