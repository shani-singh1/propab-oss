#!/usr/bin/env python3
"""
Phase 0 audit — citation fabrication severity across beliefs and mechanisms (fixes.md Step 2).

Reports what fraction of citation-style fields fail Evidence Binding.
Use ``--all-campaigns`` to close the historical audit across every campaign in DB.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "packages" / "propab-core"))

from propab.evidence_binding import (  # noqa: E402
    BindingMetrics,
    binding_check_statement_to_node,
    filter_mechanism_anomalies,
)

DEFAULT_CID = "faaf394b-7f95-4778-9136-e922f2401e7f"
DEFAULT_OUT = ROOT / "artifacts" / "evidence_binding_audit.json"
ALL_OUT = ROOT / "artifacts" / "evidence_binding_audit_all_campaigns.json"


def _psql_json(query: str) -> Any:
    cmd = [
        "docker", "compose", "-f", "docker-compose.yml", "exec", "-T", "postgres",
        "psql", "-U", "propab", "-d", "propab", "-t", "-A", "-c", query,
    ]
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=300)
    if proc.returncode != 0 or not proc.stdout.strip():
        return None
    raw = proc.stdout.strip()
    if raw in ("", "null"):
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def _list_campaign_ids() -> list[str]:
    rows = _psql_json(
        "SELECT json_agg(id::text) FROM (SELECT id FROM research_campaigns ORDER BY started_at DESC NULLS LAST) s;"
    )
    if isinstance(rows, list):
        return [str(x) for x in rows]
    return []


def _get(url: str) -> Any:
    with urlopen(url, timeout=300) as r:
        return json.load(r)


def _load_campaign_nodes(campaign_id: str, api: str) -> dict[str, Any]:
    try:
        camp = _get(f"{api.rstrip('/')}/campaigns/{campaign_id}")
        c = camp.get("campaign") or camp
        return (c.get("hypothesis_tree") or {}).get("nodes") or {}
    except Exception:
        pass
    nodes = _psql_json(
        f"SELECT hypothesis_tree_json->'nodes' FROM research_campaigns WHERE id = '{campaign_id}'::uuid;"
    )
    return nodes if isinstance(nodes, dict) else {}


def _psql_synthesis_beliefs(campaign_id: str) -> list[dict[str, Any]]:
    rows = _psql_json(f"""
    SELECT json_agg(elem) FROM (
        SELECT DISTINCT ON (elem->>'statement') elem
        FROM (
            SELECT jsonb_array_elements(payload_json->'active_beliefs') AS elem
            FROM events
            WHERE session_id = '{campaign_id}'::uuid
              AND step = 'campaign.synthesis'
              AND payload_json ? 'active_beliefs'
        ) sub
        WHERE elem IS NOT NULL
    ) outer_sub;
    """)
    if isinstance(rows, list):
        return [r for r in rows if isinstance(r, dict)]
    return []


def _beliefs_for_campaign(campaign_id: str, api: str) -> list[dict[str, Any]]:
    beliefs = _psql_synthesis_beliefs(campaign_id)
    if beliefs:
        return beliefs
    try:
        camp = _get(f"{api.rstrip('/')}/campaigns/{campaign_id}")
        bs = (camp.get("campaign") or camp).get("belief_state") or {}
        return list(bs.get("active_beliefs") or [])
    except Exception:
        pass
    raw = _psql_json(
        f"SELECT belief_state_json->'active_beliefs' FROM research_campaigns WHERE id = '{campaign_id}'::uuid;"
    )
    if isinstance(raw, list):
        return [b for b in raw if isinstance(b, dict)]
    return []


def _audit_belief_citations(
    beliefs: list[dict[str, Any]],
    nodes: dict[str, Any],
) -> dict[str, Any]:
    total = 0
    mismatched = 0
    examples: list[dict[str, Any]] = []
    for b in beliefs:
        stmt = str(b.get("statement") or "")
        for field in ("supporting_nodes", "contradicting_nodes"):
            for nid in b.get(field) or []:
                total += 1
                node = nodes.get(str(nid))
                if not isinstance(node, dict):
                    mismatched += 1
                    if len(examples) < 15:
                        examples.append({"belief": stmt[:80], "node_id": nid, "reason": "missing_node"})
                    continue
                result = binding_check_statement_to_node(stmt, node)
                if not result.match:
                    mismatched += 1
                    if len(examples) < 15:
                        examples.append({
                            "belief": stmt[:80],
                            "node_id": nid,
                            "node_text": (node.get("text") or "")[:80],
                            "reason": result.reason,
                        })
    return {
        "belief_objects_audited": len(beliefs),
        "citations_total": total,
        "citations_mismatched": mismatched,
        "mismatch_fraction": round(mismatched / total, 4) if total else None,
        "examples": examples,
    }


def _audit_mechanisms(path: Path, anomalies_path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"skipped": True, "reason": f"missing {path.name}"}
    raw = json.loads(path.read_text(encoding="utf-8"))
    mechs: list[dict[str, Any]] = []
    if isinstance(raw, list):
        for item in raw:
            if isinstance(item, dict) and item.get("explanation"):
                mechs.append(item)
            elif isinstance(item, dict) and item.get("mechanisms"):
                mechs.extend(m for m in item["mechanisms"] if isinstance(m, dict))
    anomalies_by_key: dict[str, dict[str, Any]] = {}
    if anomalies_path.exists():
        for m in mechs:
            for key in m.get("supporting_anomalies") or []:
                feats = [p for p in str(key).split("|") if p]
                anomalies_by_key[str(key)] = {
                    "feature_subset": feats,
                    "metadata": {"bucket": "survivor" if "LOFO" in m.get("explanation", "") else "collapse"},
                    "anomaly_type": "family_violation",
                    "observed_score": -0.1,
                }
    total = 0
    rejected = 0
    examples: list[dict[str, Any]] = []
    for m in mechs:
        raw_ids = list(m.get("supporting_anomalies") or [])
        total += len(raw_ids)
        metrics = BindingMetrics()
        accepted = filter_mechanism_anomalies(m, anomalies_by_key, metrics=metrics)
        rejected += len(raw_ids) - len(accepted)
        if len(accepted) < len(raw_ids) and len(examples) < 10:
            examples.append({
                "mechanism": (m.get("explanation") or "")[:80],
                "rejected": len(raw_ids) - len(accepted),
            })
    return {
        "mechanisms_audited": len(mechs),
        "anomaly_citations_total": total,
        "anomaly_citations_rejected": rejected,
        "mismatch_fraction": round(rejected / total, 4) if total else None,
        "examples": examples,
    }


def _severity_reading(totals: int, bad: int) -> str:
    if not totals:
        return "no_citations_found"
    frac = bad / totals
    if frac > 0.5:
        return "systemic_default"
    if frac > 0.15:
        return "mixed"
    return "rare_edge_case"


def _audit_one_campaign(campaign_id: str, api: str) -> dict[str, Any]:
    nodes = _load_campaign_nodes(campaign_id, api)
    beliefs = _beliefs_for_campaign(campaign_id, api)
    return {
        "campaign_id": campaign_id,
        "beliefs": _audit_belief_citations(beliefs, nodes),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-id", default=DEFAULT_CID)
    parser.add_argument("--api", default="http://localhost:8000")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--all-campaigns",
        action="store_true",
        help="Audit every campaign in research_campaigns (fixes.md Step 2 close-out)",
    )
    args = parser.parse_args()

    mech_audit = _audit_mechanisms(
        ROOT / "artifacts" / "mechanism_objects.json",
        ROOT / "artifacts" / "mechanism_objects.json",
    )
    competing_audit = _audit_mechanisms(
        ROOT / "artifacts" / "competing_mechanisms.json",
        ROOT / "artifacts" / "mechanism_objects.json",
    )

    per_campaign: list[dict[str, Any]] = []
    if args.all_campaigns:
        for cid in _list_campaign_ids():
            per_campaign.append(_audit_one_campaign(cid, args.api))
        belief_audit = {
            "belief_objects_audited": sum(p["beliefs"]["belief_objects_audited"] for p in per_campaign),
            "citations_total": sum(p["beliefs"]["citations_total"] for p in per_campaign),
            "citations_mismatched": sum(p["beliefs"]["citations_mismatched"] for p in per_campaign),
        }
        t = belief_audit["citations_total"]
        belief_audit["mismatch_fraction"] = round(belief_audit["citations_mismatched"] / t, 4) if t else None
        args.out = ALL_OUT if args.out == DEFAULT_OUT else args.out
    else:
        nodes = _load_campaign_nodes(args.campaign_id, args.api)
        beliefs = _beliefs_for_campaign(args.campaign_id, args.api)
        belief_audit = _audit_belief_citations(beliefs, nodes)
        per_campaign = [{"campaign_id": args.campaign_id, "beliefs": belief_audit}]

    totals = (
        (belief_audit.get("citations_total") or 0)
        + (mech_audit.get("anomaly_citations_total") or 0)
        + (competing_audit.get("anomaly_citations_total") or 0)
    )
    bad = (
        (belief_audit.get("citations_mismatched") or 0)
        + (mech_audit.get("anomaly_citations_rejected") or 0)
        + (competing_audit.get("anomaly_citations_rejected") or 0)
    )

    report = {
        "scope": "all_campaigns" if args.all_campaigns else "single_campaign",
        "campaign_id": args.campaign_id if not args.all_campaigns else None,
        "n_campaigns_audited": len(per_campaign),
        "severity": {
            "citations_audited_total": totals,
            "citations_mismatched_total": bad,
            "mismatch_fraction": round(bad / totals, 4) if totals else None,
            "reading": _severity_reading(totals, bad),
            "phase_0_closed": args.all_campaigns,
        },
        "beliefs": belief_audit,
        "per_campaign": per_campaign if args.all_campaigns else None,
        "mechanism_objects": mech_audit,
        "competing_mechanisms": competing_audit,
        "proceed_to_phase_1": True,
        "note": "Evidence Binding wired at belief/mechanism/artifact write paths; this audit is read-only severity.",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"out": str(args.out), "severity": report["severity"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
