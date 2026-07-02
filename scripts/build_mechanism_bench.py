#!/usr/bin/env python3
"""Build MechanismBench (~50 cases) from artifacts (fixes.md)."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"
OUT = ART / "mechanism_bench.json"

MANDRAKE_Q = (
    "Which biophysical properties predict RT activity independently of "
    "evolutionary family membership?"
)
CONTAGION_Q = (
    "Investigate which structural properties of complex networks most strongly "
    "determine the speed and extent of contagion spreading under competing diffusion models."
)
MANDRAKE_CTX = (
    "Mandrake Retroviral Wall dataset. Target: pe_efficiency_pct (RT activity). "
    "Groups: evolutionary families. LOFO = leave-one-family-out R². "
    "Features span thermal, geometry, electrostatics, surface; ESM excluded."
)


def _load(name: str):
    p = ART / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def _anomaly_prompt(anomaly: dict, *, question: str, domain_context: str = "") -> str:
    ctx = f"\nDomain context:\n{domain_context.strip()}\n" if domain_context else ""
    return f"""You are a scientific mechanism analyst. Do NOT generate research hypotheses.

Research question:
{question}
{ctx}
Observed anomaly from systematic feature-subset sweep:
{json.dumps(anomaly, indent=2)}

Task: Propose 1–3 causal mechanism explanations for this anomaly.
Each must include:
- explanation (why this pattern occurs, not just restating LOFO numbers)
- counterfactual_prediction (what would change if the mechanism were wrong)
- candidate_features or candidate_structures (list)
- assumptions_challenged (list)
- confidence (0–1)

Return JSON ONLY: {{"mechanisms": [{{"explanation": "...", "counterfactual_prediction": "...", "candidate_features": [], "assumptions_challenged": [], "confidence": 0.0}}]}}
Do NOT output hypotheses or experiment plans.
"""


def _foil_prompt(anomaly: dict, shallow: str, *, question: str) -> str:
    return f"""You are a scientific mechanism analyst. Do NOT generate research hypotheses.

Research question:
{question}

Domain context:
{MANDRAKE_CTX}

Observed anomaly:
{json.dumps(anomaly, indent=2)}

A shallow baseline explanation was proposed:
"{shallow}"

Task: Propose 1–2 DEEPER competing mechanisms that:
1. Explain WHY the anomaly occurs (causal chain, not metric restatement)
2. Include a falsifiable counterfactual prediction
3. Differ substantively from the baseline (not rephrased LOFO-gap language)

Return JSON ONLY: {{"mechanisms": [{{"explanation": "...", "counterfactual_prediction": "...", "candidate_features": [], "assumptions_challenged": [], "confidence": 0.0}}]}}
"""


def _dead_finding_prompt(text: str, failure_types: list[str], note: str, *, domain: str) -> str:
    q = MANDRAKE_Q if domain == "mandrake" else CONTAGION_Q
    return f"""You are a scientific mechanism analyst reviewing a FAILED finding.

Research question:
{q}

Failed finding claim:
{text}

Failure modes detected: {", ".join(failure_types)}
Auditor note: {note}

Task: Either (a) propose a NON-shallow mechanistic explanation that would legitimately explain the underlying pattern if true, with counterfactual predictions and scope limits, OR (b) explain concretely why no valid mechanism can be induced from this claim (what is missing).

Return JSON ONLY: {{"mechanisms": [{{"explanation": "...", "counterfactual_prediction": "...", "scope_limits": "...", "assumptions_challenged": [], "confidence": 0.0}}]}}
Do NOT restate the claim as a metric horse race.
"""


def _topology_prompt(text: str, gate: dict) -> str:
    return f"""You are a scientific mechanism analyst. A contagion hypothesis was REFUTED by artifact verification.

Research question:
{CONTAGION_Q}

Hypothesis:
{text}

Artifact gate result:
{json.dumps(gate, indent=2)}

The top artifact was topology_dependence: the effect likely does not generalize across network topology families (ER/BA/WS/SBM).

Task: Propose 1–2 mechanistic explanations that would SURVIVE cross-topology OOD transfer, OR explain why this hypothesis class cannot produce transferable mechanisms.

Each mechanism must include counterfactual_prediction and ood_testable_claim.

Return JSON ONLY: {{"mechanisms": [{{"explanation": "...", "counterfactual_prediction": "...", "ood_testable_claim": "...", "confidence": 0.0}}]}}
"""


def _lineage_texts() -> dict[str, str]:
    texts: dict[str, str] = {}
    lineage = _load("demo/main/lineage.json")
    if not lineage:
        return texts
    for lin in lineage.get("confirmed_lineages") or []:
        nid = lin.get("node_id")
        if nid and lin.get("text"):
            texts[nid] = lin["text"]
        for node in lin.get("path") or []:
            if node.get("id") and node.get("text"):
                texts.setdefault(node["id"], node["text"])
    return texts


def build() -> dict:
    anomalies = _load("anomaly_objects.json") or []
    competing = _load("competing_mechanisms.json") or []
    dead = (_load("dead_findings_classification.json") or {}).get("findings") or []
    artifact_audit = _load("artifact_verification_audit.json") or {}
    lineage_texts = _lineage_texts()

    cases: list[dict] = []
    idx = 0

    # 9 anomaly objects
    for a in anomalies[:9]:
        idx += 1
        cases.append({
            "id": f"bench-{idx:03d}",
            "type": "anomaly_object",
            "domain": "mandrake",
            "source_id": a.get("id"),
            "prompt": _anomaly_prompt(a, question=MANDRAKE_Q, domain_context=MANDRAKE_CTX),
            "reference": None,
        })

    # 15 Mandrake competing-mechanism foil cases
    foil_count = 0
    for cset in competing:
        anom_id = cset.get("anomaly_id")
        anomaly = next((a for a in anomalies if a.get("id") == anom_id), None)
        if not anomaly:
            anomaly = {
                "feature_subset": cset.get("feature_subset"),
                "anomaly_type": "family_violation",
                "metadata": {"bucket": cset.get("bucket")},
            }
        for mech in cset.get("mechanisms") or []:
            if foil_count >= 15:
                break
            foil_count += 1
            idx += 1
            cases.append({
                "id": f"bench-{idx:03d}",
                "type": "mandrake_foil",
                "domain": "mandrake",
                "source_id": mech.get("id"),
                "prompt": _foil_prompt(anomaly, mech.get("explanation") or "", question=MANDRAKE_Q),
                "reference": mech,
            })
        if foil_count >= 15:
            break

    # 20 dead findings
    dead_count = 0
    for d in dead:
        if dead_count >= 20:
            break
        hid = d.get("id")
        text = lineage_texts.get(hid) or d.get("note") or ""
        if not text:
            continue
        dead_count += 1
        idx += 1
        cases.append({
            "id": f"bench-{idx:03d}",
            "type": "dead_finding",
            "domain": d.get("domain", "contagion"),
            "source_id": hid,
            "prompt": _dead_finding_prompt(
                text, d.get("failure_types") or [], d.get("note") or "", domain=d.get("domain", "contagion")
            ),
            "reference": d,
        })

    # Pad dead findings to 20 from artifact audit if needed
    seen_dead = {c["source_id"] for c in cases if c["type"] == "dead_finding"}
    for f in artifact_audit.get("findings") or []:
        if dead_count >= 20:
            break
        hid = f.get("hypothesis_id")
        if hid in seen_dead:
            continue
        text = f.get("text_snippet") or ""
        if not text:
            continue
        dead_count += 1
        idx += 1
        cases.append({
            "id": f"bench-{idx:03d}",
            "type": "dead_finding",
            "domain": "contagion",
            "source_id": hid,
            "prompt": _dead_finding_prompt(
                text,
                ["topology_dependence", "scope_inflation"],
                "Refuted by artifact gate — insufficient OOD transfer",
                domain="contagion",
            ),
            "reference": f,
        })
    # Second pass: contagion deep analysis confirmed nodes if still short
    if dead_count < 20:
        deep = _load("contagion_campaign_deep_analysis.json") or {}
        for node in (deep.get("tree_structure") or {}).get("confirmed_nodes") or []:
            if dead_count >= 20:
                break
            hid = node.get("id")
            if not hid or hid in seen_dead:
                continue
            text = node.get("text") or ""
            if len(text) < 40:
                continue
            seen_dead.add(hid)
            dead_count += 1
            idx += 1
            cases.append({
                "id": f"bench-{idx:03d}",
                "type": "dead_finding",
                "domain": "contagion",
                "source_id": hid,
                "prompt": _dead_finding_prompt(
                    text,
                    ["scope_inflation", "single_context"],
                    "Previously confirmed but audit-classified as shallow",
                    domain="contagion",
                ),
                "reference": node,
            })

    # topology_dependence failures (fill to ~50 total)
    topo_count = 0
    target_total = 50
    for f in artifact_audit.get("findings") or []:
        if len(cases) >= target_total:
            break
        gate = f.get("gate") or {}
        reason = gate.get("verdict_reason") or ""
        if "topology_dependence" not in reason:
            continue
        hid = f.get("hypothesis_id")
        if any(c.get("source_id") == hid and c["type"] == "topology_dependence" for c in cases):
            continue
        text = f.get("text_snippet") or lineage_texts.get(hid) or ""
        if not text:
            continue
        topo_count += 1
        idx += 1
        cases.append({
            "id": f"bench-{idx:03d}",
            "type": "topology_dependence",
            "domain": "contagion",
            "source_id": hid,
            "prompt": _topology_prompt(text, gate),
            "reference": gate,
        })

    bench = {
        "version": 1,
        "n_cases": len(cases),
        "composition": {
            "anomaly_object": sum(1 for c in cases if c["type"] == "anomaly_object"),
            "mandrake_foil": sum(1 for c in cases if c["type"] == "mandrake_foil"),
            "dead_finding": sum(1 for c in cases if c["type"] == "dead_finding"),
            "topology_dependence": sum(1 for c in cases if c["type"] == "topology_dependence"),
        },
        "models_to_compare": [
            "gemini-3-flash-preview",
            "gemini-3.1-pro-preview",
        ],
        "cases": cases[:target_total],
    }
    OUT.write_text(json.dumps(bench, indent=2, ensure_ascii=False), encoding="utf-8")
    return bench


if __name__ == "__main__":
    b = build()
    print(json.dumps({"out": str(OUT), "n_cases": b["n_cases"], "composition": b["composition"]}, indent=2))
