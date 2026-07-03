#!/usr/bin/env python3
"""Inspect hypothesis routing without running a campaign (fixes.md)."""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "packages" / "propab-core"))

from propab.domain_modules.math_combinatorics.routing_inspector import (  # noqa: E402
    inspect_corpus,
    inspect_routing,
)

DEFAULT_EXAMPLE = {
    "statement": "F(n)/sqrt(n) is below 0.90 for all n >= 1000",
    "test_methodology": "greedy Sidon search with band validation",
    "scope": "Population: integers {1,...,n}...",
}

CORPUS_ARTIFACTS = (
    ROOT / "artifacts" / "campaign_ac60fcda_deep_analysis.json",
    ROOT / "artifacts" / "campaign_da855131_deep_analysis.json",
)


def _load_corpus_from_artifacts() -> list[dict]:
    out: list[dict] = []
    for path in CORPUS_ARTIFACTS:
        if not path.is_file():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        for key in ("confirmed_hypotheses", "refuted_hypotheses"):
            for row in data.get(key) or []:
                text = row.get("text") or ""
                out.append({
                    "id": row.get("id"),
                    "text": text,
                    "statement": text.split("\nPopulation:")[0].strip(),
                    "test_methodology": row.get("methodology") or "",
                    "source": path.name,
                    "prior_verdict": key.replace("_hypotheses", ""),
                })
    return out


def _load_corpus_from_api(campaign_id: str, api: str) -> list[dict]:
    with urllib.request.urlopen(f"{api.rstrip('/')}/campaigns/{campaign_id}", timeout=60) as resp:
        raw = json.loads(resp.read())
    nodes = raw["campaign"]["hypothesis_tree"]["nodes"]
    out: list[dict] = []
    for nid, n in nodes.items():
        if n.get("verdict") not in ("confirmed", "refuted", "inconclusive"):
            continue
        out.append({
            "id": nid,
            "text": n.get("text") or "",
            "statement": (n.get("text") or "").split("\nPopulation:")[0].strip(),
            "test_methodology": n.get("test_methodology") or "",
            "prior_verdict": n.get("verdict"),
        })
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--statement", default=None, help="Single hypothesis statement")
    parser.add_argument("--methodology", default="", help="test_methodology for single inspect")
    parser.add_argument("--json", default=None, help="Hypothesis JSON file (statement/text keys)")
    parser.add_argument(
        "--corpus",
        action="store_true",
        help="Run all known-bad texts from prior campaign analysis artifacts (no API)",
    )
    parser.add_argument("--campaign-id", default=None, help="Load corpus from live API campaign")
    parser.add_argument("--api", default="http://localhost:8000")
    parser.add_argument("--out", default=None, help="Write corpus report JSON here")
    parser.add_argument(
        "--require-zero-mismatches",
        action="store_true",
        help="Exit 1 if any routing mismatch (for CI preflight)",
    )
    args = parser.parse_args()

    if args.corpus or args.campaign_id:
        corpus = (
            _load_corpus_from_api(args.campaign_id, args.api)
            if args.campaign_id
            else _load_corpus_from_artifacts()
        )
        if not corpus:
            print("No corpus hypotheses found.", file=sys.stderr)
            return 1
        report = inspect_corpus(corpus)
        text = json.dumps(
            {
                "total": report["total"],
                "routing_ok": report["routing_ok"],
                "routing_mismatches": report["routing_mismatches"],
                "mismatch_rate": report["mismatch_rate"],
                "mismatches": [
                    {
                        "id": m.get("id"),
                        "text_preview": m.get("text_preview"),
                        "resolved_verifier": m.get("resolved_verifier"),
                        "expected_metric_name": m.get("expected_metric_name"),
                        "actual_metric_name": m.get("actual_metric_name"),
                    }
                    for m in report["mismatches"][:30]
                ],
            },
            indent=2,
        )
        print(text)
        if args.out:
            Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
            print(f"Full report: {args.out}", file=sys.stderr)
        if args.require_zero_mismatches and report["routing_mismatches"] > 0:
            return 1
        return 0

    if args.json:
        hypothesis = json.loads(Path(args.json).read_text(encoding="utf-8"))
    else:
        hypothesis = {
            "statement": args.statement or DEFAULT_EXAMPLE["statement"],
            "test_methodology": args.methodology or DEFAULT_EXAMPLE.get("test_methodology", ""),
        }

    result = inspect_routing(hypothesis)
    print(json.dumps(result, indent=2))
    return 0 if result.get("routing_ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
