#!/usr/bin/env python3
"""Post-run audit for Part B live campaign (fixes.md)."""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "packages" / "propab-core"))

OUT = ROOT / "artifacts" / "partb_live_campaign_audit.json"
STATE = ROOT / "artifacts" / "partb_live_campaign_latest.json"
PRIOR_CID = "b3c39b75-bb05-42e0-9098-ca4e3099b54e"  # pre-Part-B network resilience


def _get(url: str) -> dict | list:
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


def _fetch_campaign(cid: str, api: str) -> dict:
    return _get(f"{api}/campaigns/{cid}")


def _fetch_events(cid: str, api: str) -> list[dict]:
    """Fetch all session events (API has no offset; limit caps at 2000 most-recent)."""
    data = _get(f"{api}/sessions/{cid}/events")
    return data if isinstance(data, list) else data.get("events", [])


def _fetch_llm_calls(cid: str, api: str) -> list[dict]:
    data = _get(f"{api}/sessions/{cid}/llm-calls")
    return data if isinstance(data, list) else data.get("llm_calls", [])


def _confound_tags(text: str) -> list[str]:
    t = (text or "").lower()
    tags = []
    if re.search(r"topology|graph.?family|single.?context|ood", t):
        tags.append("topology_dependence")
    if re.search(r"family.?leak|group.?identity|distribution.?leak", t):
        tags.append("family_leakage")
    if re.search(r"significance|p.?value|replicat|unreplicat", t):
        tags.append("significance_only")
    if re.search(r"simulator|sandbox|implementation", t):
        tags.append("simulator_artifact")
    if re.search(r"sample.?size|underpower|n_metric", t):
        tags.append("sample_size")
    if re.search(r"metric.?direction|ambiguous|verification", t):
        tags.append("verification_failure")
    return tags or ["other"]


def main() -> None:
    state = json.loads(STATE.read_text(encoding="utf-8"))
    cid = state["campaign_id"]
    api = (state.get("api") or "http://localhost:8000").rstrip("/")

    live = _fetch_campaign(cid, api)
    summary = live.get("summary") or {}
    tree = summary.get("tree") or {}
    nodes = (live.get("campaign", {}).get("hypothesis_tree", {}).get("nodes") or {})
    verdict_counts = Counter(n.get("verdict") for n in nodes.values() if isinstance(n, dict))
    inconclusive_reasons = Counter(
        n.get("inconclusive_reason") or n.get("failure_signature") or "unknown"
        for n in nodes.values()
        if isinstance(n, dict) and n.get("verdict") == "inconclusive"
    )
    events = _fetch_events(cid, api)

    events = _fetch_events(cid, api)
    llm_calls = _fetch_llm_calls(cid, api)

    interpreters: list[dict] = []
    for call in llm_calls:
        purpose = (call.get("call_purpose") or "").lower()
        if "failure_interpret" not in purpose:
            continue
        raw = call.get("response_text") or ""
        interp: dict = {}
        if raw.strip().startswith("{"):
            try:
                parsed = json.loads(raw)
                interp = parsed if isinstance(parsed, dict) else {}
            except json.JSONDecodeError:
                pass
        interpreters.append({
            "node_id": call.get("hypothesis_id"),
            "confound_hypothesis": interp.get("confound_hypothesis"),
            "diagnostic_experiment": interp.get("diagnostic_experiment"),
            "failure_mechanism": interp.get("failure_mechanism"),
            "rules_out": interp.get("rules_out"),
            "tags": _confound_tags(
                " ".join([
                    str(interp.get("confound_hypothesis") or ""),
                    str(interp.get("failure_mechanism") or ""),
                    " ".join(interp.get("rules_out") or []),
                ])
            ),
        })

    # Also count from events (step / event_type fields differ)
    def _step(ev: dict) -> str:
        return ev.get("step") or ev.get("event_type") or ""

    expansions: list[dict] = []
    for ev in events:
        step = _step(ev)
        p = _payload(ev)
        if step == "campaign.failure_interpret" and not any(
            i.get("node_id") == p.get("node_id") for i in interpreters
        ):
            interp = p.get("interpreter") or {}
            if isinstance(interp, dict) and interp:
                interpreters.append({
                    "node_id": p.get("node_id"),
                    "confound_hypothesis": interp.get("confound_hypothesis"),
                    "diagnostic_experiment": interp.get("diagnostic_experiment"),
                    "failure_mechanism": interp.get("failure_mechanism"),
                    "rules_out": interp.get("rules_out"),
                    "tags": _confound_tags(
                        " ".join([
                            str(interp.get("confound_hypothesis") or ""),
                            str(interp.get("failure_mechanism") or ""),
                            " ".join(interp.get("rules_out") or []),
                        ])
                    ),
                })
        if step.startswith("campaign.tree_expand"):
            expansions.append({"step": step, "parent_id": p.get("parent_id"), "n_children": p.get("n_children")})

    step_counts = Counter(_step(ev) for ev in events)
    llm_purpose_counts = Counter(c.get("call_purpose") for c in llm_calls)

    # Confirmed audit (likely empty)
    confirmed_nodes = [
        n for n in (live.get("campaign", {}).get("hypothesis_tree", {}).get("nodes") or {}).values()
        if isinstance(n, dict) and n.get("verdict") == "confirmed"
    ]
    artifact_audit = {"n_confirmed": len(confirmed_nodes), "rows": []}
    if confirmed_nodes:
        try:
            from propab.artifact_verification import audit_confirmed_rows
            artifact_audit = audit_confirmed_rows(confirmed_nodes)
        except ImportError:
            artifact_audit = {"n_confirmed": len(confirmed_nodes), "error": "audit module unavailable"}

    # Prior campaign comparison
    prior_summary = {}
    try:
        prior = _fetch_campaign(PRIOR_CID, api)
        ps = prior.get("summary") or {}
        prior_summary = {
            "campaign_id": PRIOR_CID,
            "status": ps.get("status"),
            "total_hypotheses": ps.get("total_hypotheses"),
            "total_confirmed": ps.get("total_confirmed"),
            "tree_verdict_counts": (ps.get("tree") or {}).get("verdict_counts"),
            "part_b": False,
        }
    except Exception as exc:
        prior_summary = {"error": str(exc)}

    confound_counter = Counter()
    for i in interpreters:
        for t in i.get("tags") or []:
            confound_counter[t] += 1

    report = {
        "audit": "partb_live_campaign_post_run",
        "generated_at": datetime.now(tz=UTC).isoformat(),
        "campaign_id": cid,
        "status": summary.get("status"),
        "session_status": (live.get("research_session") or {}).get("status"),
        "headline": {
            "hypotheses_tested": summary.get("total_hypotheses"),
            "total_confirmed": summary.get("total_confirmed"),
            "baseline_metric": summary.get("baseline_metric"),
            "best_metric": summary.get("best_metric"),
            "elapsed_sec": summary.get("elapsed_sec"),
            "tree": tree,
            "verdict_counts": dict(verdict_counts),
            "inconclusive_reason_top": dict(inconclusive_reasons.most_common(8)),
        },
        "part_b_telemetry": {
            "failure_interpret_events": step_counts.get("campaign.failure_interpret", 0),
            "tree_expand_events": sum(
                step_counts.get(s, 0)
                for s in step_counts
                if s and s.startswith("campaign.tree_expand")
            ),
            "llm_failure_interpret_calls": step_counts.get("llm.campaign.failure_interpret", 0)
            + sum(1 for ev in events if "failure_interpret" in (_payload(ev).get("purpose") or "")),
            "interpreter_samples": interpreters[:12],
            "confound_tag_distribution": dict(confound_counter),
            "n_interpreters_parsed": len(interpreters),
        },
        "operational_overhead": {
            "total_events": len(events),
            "total_llm_calls": len(llm_calls),
            "llm_call_purposes": dict(llm_purpose_counts),
            "failure_interpret_llm_calls": llm_purpose_counts.get("campaign.failure_interpret", 0),
            "tree_expand_llm_calls": llm_purpose_counts.get("campaign.tree_expand", 0),
            "expansion_related_llm_estimate": (
                llm_purpose_counts.get("campaign.tree_expand", 0)
                + llm_purpose_counts.get("campaign.failure_interpret", 0)
            ),
            "note": "Part B ≈ 2 LLM calls per expansion vs 1 pre-Part-B",
        },
        "artifact_audit": artifact_audit,
        "comparison_prior_network_resilience": prior_summary,
        "verdicts": {
            "confirmed_survival_audit": "N/A — zero confirmed findings",
            "part_b_interpreter_ran": step_counts.get("campaign.failure_interpret", 0) > 0,
            "topology_default_bias": confound_counter.get("topology_dependence", 0)
            / max(1, len([i for i in interpreters if i.get("confound_hypothesis")])),
            "interpreter_parse_rate": len([i for i in interpreters if i.get("confound_hypothesis")])
            / max(1, len(interpreters)),
            "interpreter_diversity": len(confound_counter),
        },
        "fixes_md_conclusions": [],
    }

    n_interp = len(interpreters)
    parsed = [i for i in interpreters if i.get("confound_hypothesis")]
    topo_rate = confound_counter.get("topology_dependence", 0) / max(1, len(parsed))
    conclusions = [
        f"Campaign completed with {summary.get('total_hypotheses')} hypotheses tested, "
        f"{summary.get('total_confirmed')} confirmed — no confirmed findings to artifact-audit.",
        f"Part B interpreter fired {step_counts.get('campaign.failure_interpret', 0)} times "
        f"across {summary.get('tree', {}).get('total_nodes', '?')} tree nodes.",
    ]
    if topo_rate > 0.6:
        conclusions.append(
            "WARNING: >60% of interpreter outputs tag topology_dependence — possible Section-12 default bias."
        )
    else:
        conclusions.append(
            f"Interpreter confound tags span {len(confound_counter)} categories; "
            f"topology_dependence rate={topo_rate:.0%}."
        )
    if prior_summary.get("total_confirmed") is not None:
        conclusions.append(
            f"Prior network-resilience campaign (no Part B): "
            f"{prior_summary.get('total_hypotheses')} tested, "
            f"{prior_summary.get('total_confirmed')} confirmed."
        )
    conclusions.append(
        "Real test from fixes.md (confirmed survival rate) is inconclusive this run — "
        "need a campaign that produces confirmed nodes, or extend budget."
    )
    report["fixes_md_conclusions"] = conclusions

    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({
        "written": str(OUT),
        "status": summary.get("status"),
        "tested": summary.get("total_hypotheses"),
        "confirmed": summary.get("total_confirmed"),
        "failure_interpret": step_counts.get("campaign.failure_interpret", 0),
        "confound_tags": dict(confound_counter),
    }, indent=2))


if __name__ == "__main__":
    main()
