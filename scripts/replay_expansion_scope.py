#!/usr/bin/env python3
"""
P4/P5 — Offline expansion scope replay (fixes.md).

Replays scope validation on existing campaign trees (no LLM calls).
Audits child nodes (depth > 0) for missing scope vs parent inheritance.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "packages" / "propab-core"))

from propab.hypothesis_tree import HypothesisTree
from propab.scoped_claim import (
    EXPANSION_BOILERPLATE,
    EXPANSION_MISSING_SCOPE,
    EXPANSION_VALID_INHERITANCE,
    EXPANSION_VALID_MUTATION,
    classify_expansion_scope,
    parse_scope_from_methodology,
    validate_expansion_child,
)


def _audit_tree(tree: HypothesisTree, question: str) -> dict[str, Any]:
    children = [n for n in tree.nodes.values() if n.depth > 0]
    n_missing = 0
    class_counts: dict[str, int] = {}
    rows: list[dict[str, Any]] = []

    for child in children:
        parent = tree.nodes.get(child.parent_id) if child.parent_id else None
        parent_scope = (
            parse_scope_from_methodology(parent.text, parent.test_methodology)
            if parent else None
        )
        entry = {
            "text": child.text,
            "test_methodology": child.test_methodology or "",
            "population": (child.claim_scope or {}).get("population", ""),
            "distribution": (child.claim_scope or {}).get("distribution", ""),
            "claimed_generalization": (child.claim_scope or {}).get("claimed_generalization", ""),
            "expected_failure_modes": (child.claim_scope or {}).get("expected_failure_modes", ""),
            "ood_test": (child.claim_scope or {}).get("ood_test", ""),
        }
        scope = parse_scope_from_methodology(child.text, child.test_methodology)
        if scope is None and child.claim_scope:
            scope = parse_scope_from_methodology(child.text, json.dumps(child.claim_scope))
        audit_class = classify_expansion_scope(scope, parent_scope, question=question)
        if audit_class == EXPANSION_MISSING_SCOPE:
            n_missing += 1
        class_counts[audit_class] = class_counts.get(audit_class, 0) + 1
        rows.append({
            "node_id": child.id,
            "depth": child.depth,
            "audit_class": audit_class,
            "scope_delta": child.scope_delta,
            "text_snippet": child.text[:240],
            "parent_id": child.parent_id,
        })

    n_children = len(children)
    return {
        "n_children_generated": n_children,
        "n_children_rejected": n_missing,
        "n_children_passed": n_children - n_missing,
        "scope_rejection_rate": round(n_missing / max(1, n_children), 4),
        "missing_scope_rate": round(n_missing / max(1, n_children), 4),
        "audit_class_counts": class_counts,
        "samples": rows,
    }


def _simulate_gate_on_raw_items(
    tree: HypothesisTree,
    question: str,
) -> dict[str, Any]:
    """Simulate P3 gate on child text as if freshly expanded (no template fill)."""
    total_gen = 0
    total_rej = 0
    reasons: dict[str, int] = {}
    for child in tree.nodes.values():
        if child.depth == 0:
            continue
        parent = tree.nodes.get(child.parent_id) if child.parent_id else None
        parent_scope = (
            parse_scope_from_methodology(parent.text, parent.test_methodology)
            if parent else None
        )
        entry = {"text": child.text, "test_methodology": child.test_methodology or ""}
        if child.claim_scope:
            entry.update(child.claim_scope)
        total_gen += 1
        ok, _, reason = validate_expansion_child(entry, parent=parent_scope, question=question)
        if not ok:
            total_rej += 1
            r = reason or "missing_scope"
            reasons[r] = reasons.get(r, 0) + 1
    return {
        "n_children_generated": total_gen,
        "n_children_rejected": total_rej,
        "scope_rejection_rate": round(total_rej / max(1, total_gen), 4),
        "missing_scope_rate": round(
            reasons.get("missing_scope", 0) / max(1, total_gen), 4,
        ),
        "rejection_reasons": reasons,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Offline expansion scope replay")
    parser.add_argument("--campaign-id", action="append", default=[])
    parser.add_argument("--all-recent", type=int, default=3, help="Replay N most recent campaigns")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample-size", type=int, default=10)
    parser.add_argument("--out", default=str(ROOT / "artifacts" / "expansion_scope_replay.json"))
    args = parser.parse_args()

    try:
        import psycopg
    except ImportError:
        print("psycopg required", file=sys.stderr)
        return 1

    campaign_ids = list(args.campaign_id)
    with psycopg.connect("postgresql://propab:propab@localhost:5432/propab") as conn:
        with conn.cursor() as cur:
            if not campaign_ids:
                cur.execute(
                    """
                    SELECT id::text FROM research_campaigns
                    ORDER BY started_at DESC NULLS LAST LIMIT %s
                    """,
                    (args.all_recent,),
                )
                campaign_ids = [r[0] for r in cur.fetchall()]

            reports: list[dict[str, Any]] = []
            all_samples: list[dict[str, Any]] = []
            for cid in campaign_ids:
                cur.execute(
                    "SELECT question, hypothesis_tree_json FROM research_campaigns WHERE id = %s",
                    (cid,),
                )
                row = cur.fetchone()
                if not row:
                    continue
                question, tree_json = row[0], row[1]
                if isinstance(tree_json, str):
                    tree_json = json.loads(tree_json)
                tree = HypothesisTree.from_dict(tree_json or {})
                audit = _audit_tree(tree, str(question or ""))
                simulated = _simulate_gate_on_raw_items(tree, str(question or ""))
                reports.append({
                    "campaign_id": cid,
                    "question": question,
                    "audit": audit,
                    "simulated_gate": simulated,
                })
                for s in audit["samples"]:
                    s["campaign_id"] = cid
                    all_samples.append(s)

    rng = random.Random(args.seed)
    manual = rng.sample(all_samples, min(args.sample_size, len(all_samples))) if all_samples else []
    manual_inspection = {
        "n_sampled": len(manual),
        "target_missing_scope": 0,
        "samples": manual,
        "audit_class_counts": {
            c: sum(1 for s in manual if s["audit_class"] == c)
            for c in {s["audit_class"] for s in manual}
        },
    }

    out_doc = {
        "campaigns": reports,
        "aggregate": {
            "n_campaigns": len(reports),
            "total_children": sum(r["audit"]["n_children_generated"] for r in reports),
            "total_missing_scope": sum(r["audit"]["n_children_rejected"] for r in reports),
            "missing_scope_rate": round(
                sum(r["audit"]["n_children_rejected"] for r in reports)
                / max(1, sum(r["audit"]["n_children_generated"] for r in reports)),
                4,
            ),
        },
        "manual_inspection": manual_inspection,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out_doc, indent=2), encoding="utf-8")
    print(json.dumps({
        "out": str(out_path),
        "missing_scope_rate": out_doc["aggregate"]["missing_scope_rate"],
        "manual_missing": manual_inspection["audit_class_counts"].get(EXPANSION_MISSING_SCOPE, 0),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
