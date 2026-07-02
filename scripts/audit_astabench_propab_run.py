#!/usr/bin/env python3
"""Fabrication audit for a completed Propab AstaBench eval log directory."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "packages" / "propab-core"))

from integrations.astabench.audit import audit_campaign_answer
from integrations.astabench.campaign_client import fetch_campaign


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", help="propab_run_manifest.json or directory containing it")
    parser.add_argument("--api", default="http://localhost:8000")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    path = Path(args.manifest)
    if path.is_dir():
        path = path / "propab_run_manifest.json"
    if not path.is_file():
        print(f"Not found: {path}", file=sys.stderr)
        return 1

    manifest = json.loads(path.read_text(encoding="utf-8"))
    # Campaign IDs must be recorded per-sample in solver store; re-fetch from sidecar if present
    sidecar = path.parent / "propab_campaign_ids.json"
    if not sidecar.is_file():
        print("No propab_campaign_ids.json — run eval with propab solver first.", file=sys.stderr)
        return 1

    records = json.loads(sidecar.read_text(encoding="utf-8"))
    audits = []
    for rec in records:
        payload = fetch_campaign(args.api.rstrip("/"), rec["campaign_id"])
        answer = rec.get("answer") or {}
        audits.append(
            {
                "sample_id": rec.get("sample_id"),
                "campaign_id": rec["campaign_id"],
                **audit_campaign_answer(payload, answer),
            }
        )

    out = Path(args.out) if args.out else path.parent / "propab_fabrication_audit.json"
    out.write_text(json.dumps({"audits": audits}, indent=2), encoding="utf-8")
    print(json.dumps({"n": len(audits), "clean": sum(1 for a in audits if a.get("clean"))}, indent=2))
    print(f"written: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
