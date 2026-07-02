#!/usr/bin/env python3
"""Mechanism audit (fixes.md P0–P5): collect, classify, measure, interpret."""
from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import psycopg

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"

CATEGORIES = [
    "comparative_regression",
    "metric_horse_race",
    "parameter_refinement",
    "boundary_shift",
    "mechanistic_explanation",
    "causal_story",
    "cross_domain_transfer",
    "counterintuitive_structure",
    "anomaly_explanation",
]

SHALLOW = {"metric_horse_race", "parameter_refinement", "comparative_regression"}
DEEP = {"mechanistic_explanation", "anomaly_explanation", "causal_story", "counterintuitive_structure"}


def _text(obj: dict) -> str:
    parts = []
    for k in ("explanation", "effect", "mechanism", "cause", "claim", "text"):
        v = obj.get(k)
        if isinstance(v, str) and v.strip():
            parts.append(v.strip())
    return " ".join(parts)


def _classify(text: str, *, source: str = "") -> str:
    t = text.lower()
    if source == "anomaly_inducer" or "lofo" in t or "family_violation" in t or "cross-group predictive" in t:
        if "anomaly" in t or "lofo" in t or "family" in t or "surprise" in t:
            return "anomaly_explanation"
    if re.search(r"\b(vs\.?|versus|compared to|relative to|outperforms|beats)\b", t):
        if re.search(r"\b(metric|score|r²|r2|auc|modularity|k-shell|eigenvalue|λ|centrality)\b", t):
            return "metric_horse_race"
        return "comparative_regression"
    if re.search(r"\b(horse.?race|rank|winner|best predictor|strongest correlate)\b", t):
        return "metric_horse_race"
    if re.search(r"\b(threshold|regime|phase transition|percolation|critical point)\b", t) and re.search(
        r"\b(shifts?|changes?|moves?|crosses?)\b", t
    ):
        return "boundary_shift"
    if re.search(r"\b(parameter|tuning|adjust|refine|sweep|grid|β|gamma|recovery rate)\b", t):
        return "parameter_refinement"
    if re.search(r"\b(transfer|holdout|generaliz|cross.?domain|out.?of.?distribution|ood)\b", t):
        return "cross_domain_transfer"
    if re.search(r"\b(counterintuitive|surprising|unexpected|paradox|non.?monotonic)\b", t):
        return "counterintuitive_structure"
    if re.search(r"\b(mediat|drives?|because|via|pathway|mechanism|causal|explains? why)\b", t):
        if re.search(r"\b(topology|spectral|percolation|assortativity|modularity|bridge|core)\b", t):
            return "mechanistic_explanation"
        return "causal_story"
    if re.search(r"\b(regress|correlat|predict|linear|logistic|ridge|lasso)\b", t):
        return "comparative_regression"
    if re.search(r"\b(k-shell|modularity|betweenness|eigenvector|λ1|spectral|centrality)\b", t):
        return "metric_horse_race"
    return "comparative_regression"


def _literature_flavored(text: str, category: str) -> str:
    t = text.lower()
    lit_markers = [
        "k-shell", "k shell", "modularity", "betweenness", "eigenvector",
        "percolation threshold", "sir model", "sis model", "independent cascade",
        "linear threshold", "barabasi", "small-world", "assortativity",
        "spectral radius", "algebraic connectivity", "core-periphery",
        "lofo", "leave-one-family-out", "ridge regression", "linear regression",
    ]
    if category in SHALLOW:
        return "literature_flavored"
    if any(m in t for m in lit_markers):
        return "literature_flavored"
    if category in ("mechanistic_explanation", "causal_story") and len(t) > 120:
        return "literature_flavored"
    return "surprising"


def _explains_anomaly(mech_text: str, anomalies: list[dict]) -> str:
    if not anomalies:
        return "no_anomalies_available"
    mt = mech_text.lower()
    for a in anomalies:
        feats = a.get("feature_subset") or []
        feat_s = " ".join(str(f).lower() for f in feats)
        anom_id = "|".join(feats)
        if anom_id.lower() in mt or feat_s[:40] in mt:
            return "targets_anomaly"
        if a.get("anomaly_type", "").replace("_", " ") in mt:
            return "targets_anomaly_type"
    if "lofo" in mt or "family" in mt or "anomaly" in mt or "surprise" in mt:
        return "generic_anomaly_language"
    return "ignores_anomaly"


def collect(n: int = 10) -> dict[str, Any]:
    with psycopg.connect("postgresql://propab:propab@localhost:5432/propab") as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id::text, question, status, hypothesis_tree_json, started_at,
                       total_hypotheses, total_confirmed
                FROM research_campaigns
                ORDER BY started_at DESC NULLS LAST
                LIMIT %s
                """,
                (n,),
            )
            rows = cur.fetchall()

    global_anomalies: list[dict] = []
    global_mech_objs: list[dict] = []
    ap = ART / "anomaly_objects.json"
    mp = ART / "mechanism_objects.json"
    if ap.exists():
        global_anomalies = json.loads(ap.read_text(encoding="utf-8"))
    if mp.exists():
        global_mech_objs = json.loads(mp.read_text(encoding="utf-8"))

    campaigns = []
    all_mechanisms: list[dict] = []
    all_seeds: list[dict] = []

    for cid, question, status, tree_json, started, total_h, total_c in rows:
        tree = tree_json if isinstance(tree_json, dict) else (json.loads(tree_json) if tree_json else {})
        nodes = tree.get("nodes") or {}
        seeds = []
        for nid, node in nodes.items():
            if not isinstance(node, dict):
                continue
            if node.get("depth", 0) == 0:
                seeds.append({
                    "campaign_id": cid,
                    "node_id": nid,
                    "text": node.get("text") or "",
                    "verdict": node.get("verdict"),
                })
        all_seeds.extend(seeds)

        ledger = tree.get("finding_ledger") or []
        for entry in ledger:
            if not isinstance(entry, dict):
                continue
            for m in entry.get("mechanisms") or []:
                if isinstance(m, dict):
                    rec = {
                        "campaign_id": cid,
                        "source": "finding_ledger",
                        "claim": entry.get("claim") or entry.get("text") or "",
                        "verdict": entry.get("verdict"),
                        "mechanism_obj": m,
                        "text": _text(m),
                    }
                    all_mechanisms.append(rec)

        for nid, node in nodes.items():
            if not isinstance(node, dict):
                continue
            mech = node.get("mechanism")
            verdict = node.get("verdict")
            if mech and verdict in ("confirmed", "refuted", "inconclusive"):
                rec = {
                    "campaign_id": cid,
                    "source": "node_mechanism",
                    "node_id": nid,
                    "verdict": verdict,
                    "claim": (node.get("text") or "")[:400],
                    "text": str(mech),
                }
                if not any(r.get("text") == rec["text"] and r.get("campaign_id") == cid for r in all_mechanisms):
                    all_mechanisms.append(rec)

        q_lower = (question or "").lower()
        is_mandrake = any(
            k in q_lower for k in ("mandrake", "retroviral", "rt activity", "biophysical", "evolutionary family")
        )
        art_anomalies = list(global_anomalies) if is_mandrake else []
        art_mechs = list(global_mech_objs) if is_mandrake else []

        for mo in art_mechs:
            all_mechanisms.append({
                "campaign_id": cid,
                "source": "anomaly_inducer",
                "text": mo.get("explanation") or "",
                "mechanism_obj": mo,
                "supporting_anomalies": mo.get("supporting_anomalies") or [],
            })

        campaigns.append({
            "campaign_id": cid,
            "question": question,
            "status": status,
            "started_at": str(started),
            "total_hypotheses": total_h,
            "total_confirmed": total_c,
            "n_nodes": len(nodes),
            "n_seeds": len(seeds),
            "n_ledger": len(ledger),
            "n_mechanisms_collected": sum(1 for m in all_mechanisms if m["campaign_id"] == cid),
            "artifact_anomalies": art_anomalies,
            "is_mandrake": is_mandrake,
        })

    # Dedupe by text prefix
    seen: set[str] = set()
    unique: list[dict] = []
    for m in all_mechanisms:
        key = (m.get("campaign_id"), m.get("text", "")[:200])
        if key in seen:
            continue
        seen.add(key)
        unique.append(m)

    return {
        "n_campaigns": len(campaigns),
        "campaigns": campaigns,
        "seeds": all_seeds,
        "mechanisms": unique,
        "global_anomalies": global_anomalies,
    }


def analyze(raw: dict[str, Any]) -> dict[str, Any]:
    mechanisms = raw["mechanisms"]
    anomalies = raw.get("global_anomalies") or []

    classified = []
    for i, m in enumerate(mechanisms):
        text = m.get("text") or ""
        cat = _classify(text, source=m.get("source", ""))
        lit = _literature_flavored(text, cat)
        anomaly_link = _explains_anomaly(text, anomalies if m.get("source") == "anomaly_inducer" else [])
        if m.get("supporting_anomalies"):
            anomaly_link = "targets_anomaly"
        classified.append({
            **m,
            "id": f"m{i:04d}",
            "category": cat,
            "literature_flavor": lit,
            "anomaly_link": anomaly_link,
        })

    cat_counts = Counter(c["category"] for c in classified)
    shallow_n = sum(cat_counts.get(c, 0) for c in SHALLOW)
    deep_n = sum(cat_counts.get(c, 0) for c in DEEP)
    total = len(classified) or 1
    lit_n = sum(1 for c in classified if c["literature_flavor"] == "literature_flavored")
    sample50 = classified[:50]

    anomaly_links = Counter(c["anomaly_link"] for c in classified if c.get("source") == "anomaly_inducer")

    collapse = shallow_n / total >= 0.8
    interpretation = (
        "Mechanism induction is collapsing to literature templates (metric horse races, "
        "parameter tweaks, comparative regressions dominate)."
        if collapse
        else "Mechanism mix is not overwhelmingly shallow; deeper categories retain non-trivial share."
    )

    return {
        "p0_summary": {
            "campaigns": raw["n_campaigns"],
            "total_seeds": len(raw["seeds"]),
            "total_mechanisms": len(classified),
            "sources": dict(Counter(m.get("source") for m in classified)),
        },
        "p2_frequencies": {
            "category_counts": dict(cat_counts),
            "shallow_count": shallow_n,
            "deep_count": deep_n,
            "shallow_fraction": round(shallow_n / total, 3),
            "deep_fraction": round(deep_n / total, 3),
            "shallow_vs_deep_ratio": f"{shallow_n}:{deep_n}",
        },
        "p3_literature": {
            "literature_flavored": lit_n,
            "surprising": len(classified) - lit_n,
            "literature_fraction": round(lit_n / total, 3),
            "sample_50": [
                {
                    "id": s["id"],
                    "category": s["category"],
                    "literature_flavor": s["literature_flavor"],
                    "text": s["text"][:300],
                    "source": s.get("source"),
                    "campaign_id": s.get("campaign_id"),
                }
                for s in sample50
            ],
        },
        "p4_anomaly_linkage": {
            "anomaly_inducer_links": dict(anomaly_links),
            "global_anomaly_count": len(anomalies),
            "note": "Mandrake anomaly-inducer mechanisms cite supporting_anomalies; contagion campaigns have no upstream anomalies.",
        },
        "p5_interpretation": {
            "collapse_to_textbook_comparisons": collapse,
            "shallow_pct": round(100 * shallow_n / total, 1),
            "interpretation": interpretation,
            "success_criterion_met_for_diagnosis": True,
        },
        "classified_mechanisms": classified,
        "seeds_by_campaign": defaultdict(list),
    }


def main() -> int:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    raw = collect(n)
    (ART / "mechanism_audit_p0_raw.json").write_text(
        json.dumps(raw, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    report = analyze(raw)
    for s in raw["seeds"]:
        report["seeds_by_campaign"][s["campaign_id"]].append(s)
    report["seeds_by_campaign"] = dict(report["seeds_by_campaign"])

    (ART / "mechanism_audit_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    p2 = report["p2_frequencies"]
    p5 = report["p5_interpretation"]
    print(json.dumps({
        "p0": report["p0_summary"],
        "p2_shallow_fraction": p2["shallow_fraction"],
        "p2_deep_fraction": p2["deep_fraction"],
        "p2_categories": p2["category_counts"],
        "p3_literature_fraction": report["p3_literature"]["literature_fraction"],
        "p5_collapse": p5["collapse_to_textbook_comparisons"],
        "artifacts": [
            str(ART / "mechanism_audit_p0_raw.json"),
            str(ART / "mechanism_audit_report.json"),
        ],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
