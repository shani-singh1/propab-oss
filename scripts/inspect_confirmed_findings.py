#!/usr/bin/env python3
"""Inspect confirmed campaign findings — fixes.md priority #1."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "packages" / "propab-core"))

from propab.finding_audit import audit_confirmed_findings


def _fetch_confirmed(api: str, campaign_id: str) -> list[dict]:
    with urlopen(f"{api.rstrip('/')}/campaigns/{campaign_id}", timeout=120) as resp:
        camp = json.load(resp)
    tree = (camp.get("campaign") or camp).get("hypothesis_tree") or {}
    nodes = tree.get("nodes") or {}
    if isinstance(nodes, list):
        nodes = {n.get("id", str(i)): n for i, n in enumerate(nodes)}
    rows = [n for n in nodes.values() if n.get("verdict") == "confirmed"]
    if rows:
        return rows
    # Fallback: session hypotheses via events not available — use summary only
    return []


def _fetch_confirmed_db(campaign_id: str) -> list[dict]:
    import subprocess

    sql = (
        "SELECT DISTINCT ON (LEFT(text, 100)) id::text, text, evidence_summary, key_finding, confidence "
        f"FROM hypotheses WHERE session_id = '{campaign_id}' AND verdict = 'confirmed' "
        "ORDER BY LEFT(text, 100), confidence DESC;"
    )
    try:
        proc = subprocess.run(
            ["docker", "compose", "-f", "docker-compose.yml", "exec", "-T", "postgres",
             "psql", "-U", "propab", "-d", "propab", "-t", "-A", "-F", "\t", "-c", sql],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return []
    rows: list[dict] = []
    for line in proc.stdout.splitlines():
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) < 5:
            continue
        rows.append({
            "id": parts[0],
            "text": parts[1],
            "evidence_summary": parts[2],
            "key_finding": parts[3],
            "confidence": float(parts[4]) if parts[4] else 0.0,
        })
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit confirmed findings (fixes.md)")
    parser.add_argument("--campaign-id", default="")
    parser.add_argument("--api", default="http://localhost:8000")
    parser.add_argument("--out", default=str(ROOT / "artifacts" / "confirmed_findings_audit.json"))
    parser.add_argument("--use-db", action="store_true", help="Query Postgres directly")
    args = parser.parse_args()

    cid = args.campaign_id.strip()
    if not cid:
        latest = ROOT / "artifacts" / "mandrake_campaign_latest.json"
        if latest.is_file():
            cid = str(json.loads(latest.read_text(encoding="utf-8")).get("campaign_id") or "")
    if not cid:
        print("Provide --campaign-id", file=sys.stderr)
        return 1

    rows = _fetch_confirmed_db(cid) if args.use_db else _fetch_confirmed(args.api, cid)
    if not rows:
        rows = _fetch_confirmed_db(cid)

    if not rows:
        print(f"No confirmed findings for {cid}", file=sys.stderr)
        return 1

    report = audit_confirmed_findings(rows)
    report["campaign_id"] = cid

    try:
        from propab.artifact_verification import audit_confirmed_rows
        artifact_report = audit_confirmed_rows(rows)
        report["artifact_audit"] = {
            "artifact_survival_rate": artifact_report["artifact_survival_rate"],
            "n_confirmed_under_artifact_gate": artifact_report["n_confirmed_under_artifact_gate"],
            "n_refuted_by_artifact": artifact_report["n_refuted_by_artifact"],
            "n_inconclusive_artifact": artifact_report["n_inconclusive_artifact"],
            "artifact_failure_distribution": artifact_report["artifact_failure_distribution"],
            "recommendation": artifact_report["recommendation"],
        }
    except Exception as exc:
        report["artifact_audit"] = {"error": str(exc)}

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps({
        "written": str(out),
        "n_confirmed": report["n_confirmed"],
        "by_disposition": report["by_disposition"],
        "by_primary_family": report["by_primary_family"],
        "fake_diversity": report["fake_diversity"],
        "n_gold": len(report.get("gold_findings") or []),
        "recommendation": report.get("recommendation"),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
