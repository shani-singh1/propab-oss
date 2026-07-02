#!/usr/bin/env python3
"""Audit: is sequential conditioning happening in tree expansion? (fixes.md)"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "packages" / "propab-core"))

from propab.hypothesis_tree import HypothesisTree  # noqa: E402
from services.orchestrator.campaign_diagnostics import parse_evidence_obj  # noqa: E402


def _load_tree(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _analyze_nodes(tree_data: dict) -> dict:
    nodes = tree_data.get("nodes") or {}
    by_verdict: dict[str, list[dict]] = {"refuted": [], "inconclusive": [], "confirmed": []}
    for nid, n in nodes.items():
        v = n.get("verdict")
        if v not in by_verdict:
            continue
        ev = n.get("evidence_summary") or ""
        ev_obj = parse_evidence_obj(ev)
        by_verdict[v].append({
            "id": nid,
            "text_snippet": (n.get("text") or "")[:120],
            "verdict": v,
            "confidence": n.get("confidence"),
            "mechanism": n.get("mechanism"),
            "inconclusive_reason": n.get("inconclusive_reason"),
            "failure_signature": n.get("failure_signature"),
            "has_verdict_reason_in_evidence": bool(ev_obj.get("verdict_reason")),
            "verdict_reason": ev_obj.get("verdict_reason"),
            "has_artifact_gate": "artifact_gate" in ev_obj,
            "artifact_gate_verdict": (ev_obj.get("artifact_gate") or {}).get("verdict")
            if isinstance(ev_obj.get("artifact_gate"), dict) else None,
            "top_artifact": _top_artifact(ev_obj),
            "evidence_summary_chars": len(ev),
            "children_count": len(n.get("children") or []),
            "expansion_reason": n.get("expansion_reason"),
        })

    return {
        "counts": {k: len(v) for k, v in by_verdict.items()},
        "samples": {k: v[:5] for k, v in by_verdict.items()},
        "inconclusive_reason_distribution": dict(
            Counter(x["inconclusive_reason"] for x in by_verdict["inconclusive"])
        ),
        "refuted_with_mechanism": sum(1 for x in by_verdict["refuted"] if x["mechanism"]),
        "refuted_with_artifact_gate": sum(1 for x in by_verdict["refuted"] if x["has_artifact_gate"]),
    }


def _top_artifact(ev_obj: dict) -> str | None:
    ag = ev_obj.get("artifact_gate")
    if not isinstance(ag, dict):
        return None
    ranked = ag.get("ranked_artifacts") or []
    if ranked and isinstance(ranked[0], dict):
        return ranked[0].get("artifact_type") or ranked[0].get("name")
    return ag.get("top_artifact")


def _expand_prompt_fields(node: dict, question: str = "") -> dict:
    tree = HypothesisTree.from_dict({"nodes": {node["id"]: node}, "frontier": [], "confirmed": [], "exhausted": []})
    # HypothesisNode.from_dict expects full node dict keyed by id
    nid = node["id"]
    prompt = tree.build_expand_prompt(nid, question=question)
    if not prompt:
        return {"prompt_built": False}
    return {
        "prompt_built": True,
        "prompt_chars": len(prompt),
        "includes_verdict_label": node.get("verdict", "") in prompt,
        "includes_mechanism_block": "Extracted mechanism:" in prompt,
        "includes_inconclusive_reason": bool(node.get("inconclusive_reason") and node["inconclusive_reason"] in prompt),
        "includes_failure_signature": bool(node.get("failure_signature") and node["failure_signature"] in prompt),
        "includes_verdict_reason_explicit": bool(
            node.get("verdict_reason") and str(node["verdict_reason"]) in prompt
        ),
        "includes_artifact_type_explicit": bool(
            node.get("top_artifact") and str(node["top_artifact"]) in prompt
        ),
        "prompt_excerpt": prompt[:800],
    }


def main() -> None:
    tree_path = ROOT / "artifacts" / "demo" / "main" / "hypothesis_tree.json"
    deep_path = ROOT / "artifacts" / "contagion_campaign_deep_analysis.json"

    tree_analysis = _analyze_nodes(_load_tree(tree_path)) if tree_path.exists() else {}

    # Build expand-prompt field checks on real refuted/inconclusive nodes
    tree_raw = _load_tree(tree_path) if tree_path.exists() else {"nodes": {}}
    expand_checks = []
    for nid, n in tree_raw.get("nodes", {}).items():
        if n.get("verdict") not in ("refuted", "inconclusive"):
            continue
        if not n.get("evidence_summary"):
            continue
        ev_obj = parse_evidence_obj(n["evidence_summary"])
        enriched = dict(n)
        enriched["id"] = nid
        enriched["verdict_reason"] = ev_obj.get("verdict_reason")
        enriched["top_artifact"] = _top_artifact(ev_obj)
        check = _expand_prompt_fields(enriched)
        check["node_id"] = nid
        check["verdict"] = n.get("verdict")
        expand_checks.append(check)
        if len(expand_checks) >= 8:
            break

    report = {
        "audit_title": "Sequential conditioning audit (fixes.md)",
        "verdict_summary": (
            "Sequential conditioning is partially present in the expansion path but weak: "
            "failure semantics are embedded inside evidence_summary JSON (verdict_reason) "
            "rather than surfaced as dedicated prompt fields; refuted learnings flow to "
            "later seeds mainly as dead-end exclusion, not positive narrowing; "
            "expansion is a single LLM call with no explicit interpret-failure step."
        ),
        "questions": {
            "q1_failure_reason_not_just_verdict": {
                "question": "Does expansion receive failure reason, not just verdict?",
                "answer": "PARTIAL — verdict_reason is inside evidence_summary JSON blob; "
                "inconclusive_reason and failure_signature are computed on nodes but NOT passed to build_expand_prompt; "
                "optional mechanism_block carries generic gate text for some refuted nodes.",
                "expand_prompt_template_fields": [
                    "verdict",
                    "confidence",
                    "parent_text",
                    "evidence_summary",
                    "mechanism_block (optional)",
                    "depth",
                    "scope fields",
                    "expansion_instructions (verdict-keyed generic bullets)",
                ],
                "missing_from_expand_prompt": [
                    "inconclusive_reason",
                    "failure_signature",
                    "verdict_reason (as dedicated field)",
                    "artifact_gate / top_artifact",
                    "explicit 'what this rules out' statement",
                ],
                "code_refs": {
                    "expand_template": "packages/propab-core/propab/hypothesis_tree.py:_EXPAND_PROMPT_TEMPLATE",
                    "build_expand_prompt": "packages/propab-core/propab/hypothesis_tree.py:build_expand_prompt",
                    "diagnostics_not_wired": "services/orchestrator/campaign_loop.py:_apply_result_diagnostics",
                },
                "sample_expand_checks": expand_checks,
            },
            "q2_refuted_as_conditioning_vs_exclusion": {
                "question": "Is refuted information used as positive conditioning or only dead-end exclusion?",
                "answer": "MOSTLY EXCLUSION — dead_ends list in seed prompts says 'do not repeat'; "
                "lifetime graph stores failure text as dead_end_texts; no prompt states "
                "'X is false therefore Y must hold'. Expansion from refuted parent uses generic "
                "'generate alternatives' instructions, not structured implication.",
                "seed_path": {
                    "confirmed_findings_in_prior": True,
                    "refuted_in_prior": "dead_ends only (exclusion)",
                    "lifetime_positive_conditioning": "theme_boost/penalty, blocked_failure_signatures (avoid patterns)",
                },
                "expansion_path": {
                    "prior_snippets_source": "established_facts only (_prior_snippets)",
                    "refuted_siblings_in_expand_prompt": False,
                },
                "code_refs": {
                    "seed_dead_ends": "services/orchestrator/hypotheses.py:_build_hypothesis_prompt",
                    "prior_snippets": "services/orchestrator/campaign_loop.py:_prior_snippets",
                    "lifetime_dead_ends": "services/orchestrator/lifetime_knowledge.py:enrich_prior_from_lifetime",
                    "failure_records": "packages/propab-core/propab/negative_knowledge.py:extract_failures_from_campaign",
                },
            },
            "q3_explicit_reasoning_step": {
                "question": "Is there an explicit reasoning step between failure and child generation?",
                "answer": "NO — expand_tree_node is one LLM call: build_expand_prompt → llm.call → parse_expanded_nodes. "
                "No intermediate step that states what the failure rules out or implies before generating children.",
                "flow": [
                    "sub_agent_loop builds evidence_summary string with embedded verdict_reason",
                    "_apply_result_diagnostics sets inconclusive_reason/failure_signature/mechanism on node",
                    "campaign_loop triggers _expand_node_async on refuted/confirmed/inconclusive",
                    "expand_tree_node: single llm.call(purpose=campaign.tree_expand)",
                ],
                "contrast": "think_act agent loop has multi-step reasoning during testing, but NOT at expansion time",
                "code_refs": {
                    "expand_tree_node": "services/orchestrator/campaign_loop.py:expand_tree_node",
                    "no_interpret_step": "grep 'rules out|failure implies' → no matches in codebase",
                },
            },
        },
        "path_comparison": {
            "seed_path": {
                "sequential_conditioning": "weak",
                "notes": "Prior round confirmed findings + lifetime theories; refuted → dead_ends exclusion only",
            },
            "expansion_path": {
                "sequential_conditioning": "weak-to-partial",
                "notes": "Raw evidence blob + generic verdict-keyed instructions; mechanism_block when set",
            },
            "sub_agent_test_path": {
                "sequential_conditioning": "moderate",
                "notes": "Multi-step think_act with peer_findings and learned_from; not reused for expansion",
            },
        },
        "empirical_samples": {
            "demo_tree": tree_analysis,
        },
        "design_vs_practice": {
            "design_doc_intent": "Refuted → alternatives; inconclusive → retest; confirmed → mechanistic/boundary children",
            "practice_gap": [
                "Diagnostics fields (inconclusive_reason, failure_signature) are written but not read back into expansion prompts",
                "artifact_gate results never appear in sampled campaign artifacts; verdict_reason is often generic gate text",
                "Positive conditioning ('given X false, pursue Y') is nowhere encoded explicitly",
            ],
        },
        "recommendations_if_fixing": [
            "Add dedicated failure_interpretation block to build_expand_prompt: verdict_reason, inconclusive_reason, top_artifact, failure_signature",
            "Optional LLM interpret step: evidence → structured 'rules_out' / 'implies' before child generation",
            "Feed refutation mechanisms into seed prior as structured negative facts, not just dead-end text",
            "Include sibling refuted/confirmed summaries in expansion prompt, not only parent evidence",
        ],
    }

    if deep_path.exists():
        deep = json.loads(deep_path.read_text(encoding="utf-8"))
        records = deep.get("node_records") or deep.get("nodes") or []
        if isinstance(records, list):
            refuted = [r for r in records if r.get("verdict") == "refuted"][:3]
            inc = [r for r in records if r.get("verdict") == "inconclusive"][:3]
            report["empirical_samples"]["contagion_deep"] = {
                "refuted_samples": refuted,
                "inconclusive_samples": inc,
            }

    out = ROOT / "artifacts" / "sequential_conditioning_audit.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
