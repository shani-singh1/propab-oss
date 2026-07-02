#!/usr/bin/env python3
"""
Family-label permutation audit for graph contagion confirmed findings.

Mirrors scripts/lofo_family_permutation_test.py: shuffle topology_family labels,
recompute leave-one-family-out R², count survivors.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "packages" / "propab-core"))

from propab.benchmarks.graph_contagion_benchmark import (
    benchmark_arrays,
    generate_benchmark,
    infer_finding_proxy,
)
from propab.domain_adapters.mandrake_adapter import _leave_one_family_out_r2, _make_model


def _parse_evidence(evidence_summary: str) -> dict[str, Any]:
    if not evidence_summary:
        return {}
    m = re.search(r"evidence=(\{.*?\});", evidence_summary)
    if not m:
        return {}
    try:
        obj = json.loads(m.group(1))
        return obj if isinstance(obj, dict) else {}
    except json.JSONDecodeError:
        return {}


def _load_confirmed(*, campaign_id: str, tree_path: Path | None) -> list[dict[str, Any]]:
    if tree_path and tree_path.is_file():
        tree = json.loads(tree_path.read_text(encoding="utf-8"))
        nodes = tree.get("nodes") or {}
        rows = [n for n in nodes.values() if n.get("verdict") == "confirmed"]
        if rows:
            return rows

    import subprocess

    sql = (
        "SELECT id::text, text, evidence_summary, key_finding, confidence "
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
    rows: list[dict[str, Any]] = []
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
        })
    return rows


def family_label_permutation_null(
    X: np.ndarray,
    y: np.ndarray,
    families: np.ndarray,
    model,
    *,
    n_perm: int,
    seed: int,
) -> tuple[float, list[float]]:
    observed = _leave_one_family_out_r2(X, y, families, model)
    rng = np.random.default_rng(seed)
    null: list[float] = []
    for _ in range(n_perm):
        perm_families = families.copy()
        rng.shuffle(perm_families)
        null.append(_leave_one_family_out_r2(X, y, perm_families, model))
    return observed, null


def summarize_null(observed: float, null: list[float]) -> dict[str, Any]:
    arr = np.asarray(null, dtype=float)
    p_ge_adj = float((np.sum(arr >= observed) + 1) / (len(arr) + 1))
    rank = int(np.sum(arr < observed)) + 1
    percentile = 100.0 * rank / len(arr)
    p95 = float(np.percentile(arr, 95))
    survives = observed > p95 and p_ge_adj < 0.05
    return {
        "observed_lofo_r2": round(observed, 6),
        "n_permutations": len(null),
        "null_mean": round(float(np.mean(arr)), 6),
        "null_std": round(float(np.std(arr)), 6),
        "null_p95": round(p95, 6),
        "empirical_p_value_ge": round(p_ge_adj, 4),
        "percentile_rank": round(percentile, 2),
        "outside_noise_band_p95": observed > p95,
        "survives_permutation_audit": survives,
        "verdict": (
            "survives label-shuffle null"
            if survives
            else "not outside noise — cross-topology signal not supported at this n"
        ),
    }


def audit_finding(
    finding: dict[str, Any],
    rows: list,
    *,
    n_perm: int,
    seed: int,
    model_name: str,
) -> dict[str, Any]:
    text = str(finding.get("text") or finding.get("key_finding") or "")
    theme = str(finding.get("primary_theme") or finding.get("theme") or "general")
    proxy = infer_finding_proxy(text, theme)
    feature_cols: list[str] = proxy["feature_cols"]
    target: str = proxy["target"]
    X, y, families = benchmark_arrays(rows, feature_cols=feature_cols, target=target)
    model = _make_model(model_name)

    observed, null = family_label_permutation_null(
        X, y, families, model, n_perm=n_perm, seed=seed,
    )
    summary = summarize_null(observed, null)
    ev = _parse_evidence(str(finding.get("evidence_summary") or ""))

    return {
        "hypothesis_id": finding.get("id"),
        "theme": theme,
        "text_snippet": text[:200],
        "campaign_p_value": ev.get("p_value"),
        "campaign_metric_value": ev.get("metric_value"),
        "campaign_effect_size": ev.get("effect_size"),
        "proxy_features": feature_cols,
        "proxy_target": target,
        **summary,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Contagion confirmed findings permutation audit")
    parser.add_argument("--campaign-id", default="19063c76-e039-4f96-bc2e-de989eb4afc7")
    parser.add_argument("--tree", default=str(ROOT / "artifacts" / "demo" / "main" / "hypothesis_tree.json"))
    parser.add_argument("--n-perm", type=int, default=1000)
    parser.add_argument("--per-family", type=int, default=40)
    parser.add_argument("--model", default="ridge")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default=str(ROOT / "artifacts" / "contagion_confirmed_permutation_audit.json"))
    args = parser.parse_args()

    confirmed = _load_confirmed(campaign_id=args.campaign_id, tree_path=Path(args.tree))
    if not confirmed:
        print(f"No confirmed findings for {args.campaign_id}", file=sys.stderr)
        return 1

    print(f"Generating benchmark ({args.per_family}/family)...", flush=True)
    bench_rows = generate_benchmark(per_family=args.per_family, seed=args.seed)

    audits: list[dict[str, Any]] = []
    for i, finding in enumerate(confirmed):
        audit = audit_finding(
            finding,
            bench_rows,
            n_perm=args.n_perm,
            seed=args.seed + i + 1,
            model_name=args.model,
        )
        audits.append(audit)
        print(
            f"  [{i + 1}/{len(confirmed)}] lofo={audit['observed_lofo_r2']:.3f} "
            f"p95={audit['null_p95']:.3f} survive={audit['survives_permutation_audit']}",
            flush=True,
        )

    survivors = [a for a in audits if a["survives_permutation_audit"]]
    report = {
        "campaign_id": args.campaign_id,
        "domain": "graphs_sis_contagion",
        "n_confirmed": len(confirmed),
        "n_survives": len(survivors),
        "n_fails": len(confirmed) - len(survivors),
        "survival_rate": round(len(survivors) / len(confirmed), 4),
        "methodology": "topology_family_label_shuffle_lofo",
        "n_permutations": args.n_perm,
        "benchmark": {
            "per_family": args.per_family,
            "n_samples": len(bench_rows),
            "families": sorted({r.topology_family for r in bench_rows}),
            "model": args.model,
        },
        "headline_verdict": (
            f"{len(survivors)}/{len(confirmed)} confirmed findings survive family-label "
            f"permutation null (observed LOFO R² above 95th percentile, p<0.05)"
        ),
        "findings": audits,
        "survivors": [a["hypothesis_id"] for a in survivors],
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        "written": str(out),
        "n_confirmed": len(confirmed),
        "n_survives": len(survivors),
        "headline": report["headline_verdict"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
