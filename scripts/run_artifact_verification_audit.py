#!/usr/bin/env python3
"""
Artifact-adversarial audit for confirmed campaign findings (fixes.md P7).

Replaces raw confirmed_count with artifact_survival_rate and failure distribution.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "packages" / "propab-core"))

from propab.artifact_verification import (
    audit_confirmed_row,
    audit_confirmed_rows,
    format_paper_artifact_section,
)


def _load_confirmed(*, campaign_id: str, tree_path: Path) -> list[dict]:
    if tree_path.is_file():
        tree = json.loads(tree_path.read_text(encoding="utf-8"))
        nodes = tree.get("nodes") or {}
        rows = [n for n in nodes.values() if n.get("verdict") == "confirmed"]
        if rows:
            return rows

    import subprocess

    sql = (
        "SELECT id::text, text, evidence_summary, key_finding, confidence, primary_theme "
        f"FROM hypotheses WHERE session_id = '{campaign_id}' AND verdict = 'confirmed' "
        "ORDER BY confidence DESC;"
    )
    proc = subprocess.run(
        [
            "docker", "compose", "exec", "-T", "postgres",
            "psql", "-U", "propab", "-d", "propab", "-t", "-A", "-F", "\t", "-c", sql,
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    rows: list[dict] = []
    for line in proc.stdout.splitlines():
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) < 4:
            continue
        rows.append({
            "id": parts[0],
            "text": parts[1],
            "evidence_summary": parts[2],
            "key_finding": parts[3],
            "confidence": float(parts[4]) if len(parts) > 4 and parts[4] else 0.0,
            "primary_theme": parts[5] if len(parts) > 5 else "general",
        })
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Artifact verification audit (fixes.md)")
    parser.add_argument("--campaign-id", default="19063c76-e039-4f96-bc2e-de989eb4afc7")
    parser.add_argument(
        "--tree",
        default=str(ROOT / "artifacts" / "demo" / "main" / "hypothesis_tree.json"),
    )
    parser.add_argument("--out", default=str(ROOT / "artifacts" / "artifact_verification_audit.json"))
    parser.add_argument("--paper-md", default=str(ROOT / "artifacts" / "artifact_verification_paper.md"))
    args = parser.parse_args()

    rows = _load_confirmed(campaign_id=args.campaign_id, tree_path=Path(args.tree))
    if not rows:
        print(f"No confirmed findings for {args.campaign_id}", file=sys.stderr)
        return 1

    report = audit_confirmed_rows(rows)
    report["campaign_id"] = args.campaign_id
    report["legacy_confirmed_count"] = len(rows)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    paper_pairs = [(str(r.get("text") or "")[:300], audit_confirmed_row(r)) for r in rows]
    paper_md = format_paper_artifact_section(paper_pairs)
    Path(args.paper_md).write_text(paper_md, encoding="utf-8")

    summary = {
        "written": str(out),
        "paper_md": args.paper_md,
        "legacy_confirmed": len(rows),
        "artifact_survival_rate": report["artifact_survival_rate"],
        "confirmed_under_artifact_gate": report["n_confirmed_under_artifact_gate"],
        "refuted_by_artifact": report["n_refuted_by_artifact"],
        "inconclusive_artifact": report["n_inconclusive_artifact"],
        "failure_distribution": report["artifact_failure_distribution"],
        "recommendation": report["recommendation"],
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
