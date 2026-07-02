#!/usr/bin/env python3
"""
Campaign failure synthesis bench (fixes.md).

Test: given FULL visibility of all failed nodes in one real campaign (hypothesis,
evidence, failure reasons), can the model synthesize the common failure pattern
and state what that implies for where to search next?

No new orchestrator plumbing — one-shot LLM with complete failure dossier.
"""
from __future__ import annotations

import json
import os
import re
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "packages" / "propab-core"))

from propab.research_quality import NODE_ROLE_CONTROL  # noqa: E402
from services.orchestrator.campaign_diagnostics import parse_evidence_obj  # noqa: E402

ART = ROOT / "artifacts"
OUT = ART / "campaign_failure_synthesis_bench.json"

SYNTHESIS_PROMPT = """You are reviewing a completed research campaign that produced NO breakthrough.

Research question:
{question}

Below are ALL non-confirmed DISCOVERY hypotheses from this campaign ({n_failures} nodes).
Each entry includes the hypothesis text and whatever evidence / failure diagnostics exist.

{failure_dossier}

Your task — with EVERYTHING above visible at once:

1. **Common failure pattern**: What pattern is shared across these failures? Be specific
   (mechanism-level), not generic ("hypotheses were wrong"). Cite concrete recurring mistakes.

2. **What this rules out**: What classes of claims or approaches are now ruled out?

3. **Where to look next**: Given the pattern, what 3-5 *specific* next hypotheses would
   actually address the gap — not more variants of what already failed?

Return JSON ONLY:
{{
  "common_failure_pattern": "...",
  "rules_out": ["...", "..."],
  "where_to_look_next": [
    {{"hypothesis_sketch": "...", "why_different_from_failures": "..."}}
  ],
  "confidence": 0.0-1.0
}}
"""

# Hand-curated ground truth from dead_findings_classification + Section 10 counterfactual
GROUND_TRUTH: dict[str, dict[str, Any]] = {
    "contagion_demo_tree": {
        "question_contains": "contagion",
        "required_themes": [
            # at least 2 of these must appear (case-insensitive) in synthesis
            ("narrow_regime", r"single.{0,20}(context|regime|topology|graph)|narrow|one graph|without.{0,30}(transfer|holdout|ood)|modular.{0,20}only|gamma\s*[=>]"),
            ("significance_or_replication", r"replicat|unreplicat|p[\-\s]?value|significance|metric.{0,15}(missing|ambiguous|direction)|insufficient sample|verification"),
            ("scope_or_generalization", r"scope|generaliz|transfer|holdout|ood|cross.{0,10}(family|topology|graph)|confound"),
        ],
        "forbidden_defaults": [
            # shallow pattern-completion defaults (Section 12 failure mode)
            (r"^the dominant (theme|pattern) is topology", "topology_only_without_evidence"),
            (r"topology.?dependence.{0,40}$", "topology_only_without_evidence"),
        ],
        "good_next_steps": [
            r"holdout|leave.{0,5}out|transfer|cross.{0,10}(family|topology|graph|regime)|replicat|null.{0,10}(control|baseline)",
        ],
    },
    "mandrake": {
        "question_contains": "biophysical",
        "required_themes": [
            ("family_leakage", r"family|lineage|group|logo|lofo|leave.{0,5}one.{0,5}(group|family)|confound|proxy"),
            ("thermal_features", r"t55|t70|t75|thermal|unfold|denatur|temperature|mid.{0,10}range"),
        ],
        "forbidden_defaults": [
            (r"^the dominant (theme|pattern) is topology", "topology_only_without_evidence"),
        ],
        "good_next_steps": [
            r"shuffle|permut|null|family.{0,20}split|residual|after regress|finer.{0,20}(group|family)|counterfactual",
        ],
    },
}


def _load_api_key() -> str:
    key = (os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY") or "").strip()
    if key:
        return key
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            if line.startswith("GOOGLE_API_KEY="):
                return line.split("=", 1)[1].strip()
            if line.startswith("GEMINI_API_KEY="):
                return line.split("=", 1)[1].strip()
    return ""


def _load_model() -> str:
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            if line.startswith("LLM_MODEL="):
                return line.split("=", 1)[1].strip()
    return os.environ.get("LLM_MODEL", "gemini-3-flash-preview")


def _gemini(prompt: str, model: str, api_key: str) -> str:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    payload = {"contents": [{"parts": [{"text": prompt}]}], "generationConfig": {"temperature": 0.2}}
    with httpx.Client(timeout=180.0) as client:
        r = client.post(url, params={"key": api_key}, json=payload)
    r.raise_for_status()
    parts = (((r.json().get("candidates") or [{}])[0].get("content") or {}).get("parts")) or []
    return "".join(p.get("text", "") for p in parts if isinstance(p, dict))


def _parse_json(raw: str) -> dict[str, Any]:
    text = (raw or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        m = re.search(r"\{[\s\S]*\}", text)
        if not m:
            return {"_parse_error": True, "_raw": raw[:2000]}
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            return {"_parse_error": True, "_raw": raw[:2000]}


def _node_from_tree_dict(nid: str, n: dict[str, Any]) -> dict[str, Any] | None:
    role = n.get("node_role") or "DISCOVERY"
    verdict = n.get("verdict")
    if role == NODE_ROLE_CONTROL or verdict in ("confirmed", "pending", None):
        return None
    ev = n.get("evidence_summary") or ""
    ev_obj = parse_evidence_obj(ev)
    return {
        "id": nid,
        "verdict": verdict,
        "text": (n.get("text") or "")[:600],
        "depth": n.get("depth"),
        "generation": n.get("generation"),
        "parent_id": n.get("parent_id"),
        "inconclusive_reason": n.get("inconclusive_reason"),
        "failure_signature": n.get("failure_signature"),
        "mechanism": n.get("mechanism"),
        "verdict_reason": ev_obj.get("verdict_reason"),
        "claim_type": ev_obj.get("claim_type") or n.get("claim_type"),
        "metric_value": ev_obj.get("metric_value"),
        "p_value": ev_obj.get("p_value"),
        "n_metric_steps": ev_obj.get("n_metric_steps"),
        "evidence_excerpt": ev[:500] if ev else None,
    }


def _load_contagion_demo_failures() -> tuple[str, list[dict[str, Any]]]:
    path = ART / "demo" / "main" / "hypothesis_tree.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    question = (
        "Investigate which structural properties of complex networks most strongly determine "
        "the speed and extent of contagion spreading under competing diffusion models."
    )
    failures: list[dict[str, Any]] = []
    for nid, n in (data.get("nodes") or {}).items():
        row = _node_from_tree_dict(nid, n)
        if row:
            failures.append(row)
    return question, failures


def _load_contagion_deep_failures() -> tuple[str, list[dict[str, Any]]]:
    path = ART / "contagion_campaign_deep_analysis.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    question = data.get("headline", {}).get("question") or data.get("session", {}).get("question") or ""
    failures = []
    for r in data.get("node_records") or []:
        if r.get("node_role") == NODE_ROLE_CONTROL:
            continue
        if r.get("verdict") in ("confirmed", "pending"):
            continue
        pe = r.get("parsed_evidence") or {}
        ev = pe.get("evidence") if isinstance(pe.get("evidence"), dict) else {}
        failures.append({
            "id": r.get("id"),
            "verdict": r.get("verdict"),
            "text": (r.get("text_snippet") or "")[:600],
            "depth": r.get("depth"),
            "generation": r.get("generation"),
            "parent_id": r.get("parent_id"),
            "inconclusive_reason": r.get("inconclusive_reason"),
            "failure_signature": r.get("failure_signature"),
            "mechanism": r.get("mechanism"),
            "verdict_reason": ev.get("verdict_reason"),
            "claim_type": ev.get("claim_type") or r.get("claim_type"),
            "metric_value": ev.get("metric_value"),
            "p_value": ev.get("p_value"),
            "n_metric_steps": ev.get("n_metric_steps"),
            "evidence_excerpt": json.dumps(ev, ensure_ascii=False)[:500] if ev else None,
        })
    return question, failures


def _load_mandrake_failures() -> tuple[str, list[dict[str, Any]]]:
    path = ART / "mandrake_run_analysis.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    question = data.get("summary", {}).get("question") or data.get("campaign_fields", {}).get("question") or ""
    failures = []
    for n in data.get("nodes") or []:
        if n.get("verdict") not in ("refuted", "inconclusive"):
            continue
        text = n.get("text") or ""
        if text.lower().startswith("null hypothesis"):
            continue
        failures.append({
            "id": n.get("id"),
            "verdict": n.get("verdict"),
            "text": text[:600],
            "depth": None,
            "generation": n.get("generation"),
            "parent_id": None,
            "inconclusive_reason": None,
            "failure_signature": None,
            "mechanism": None,
            "verdict_reason": None,
            "claim_type": n.get("claim_type"),
            "metric_value": None,
            "p_value": None,
            "n_metric_steps": None,
            "evidence_excerpt": None,
            "note": "Mandrake export lacks per-node evidence_summary — text only",
        })
    return question, failures


def _format_dossier(failures: list[dict[str, Any]], *, max_nodes: int = 45) -> str:
    lines: list[str] = []
    for i, f in enumerate(failures[:max_nodes], 1):
        lines.append(f"--- Failure #{i} [{f.get('verdict')}] gen={f.get('generation')} depth={f.get('depth')} ---")
        lines.append(f"Hypothesis: {f.get('text')}")
        if f.get("inconclusive_reason"):
            lines.append(f"inconclusive_reason: {f['inconclusive_reason']}")
        if f.get("failure_signature"):
            lines.append(f"failure_signature: {f['failure_signature']}")
        if f.get("verdict_reason"):
            lines.append(f"verdict_reason: {f['verdict_reason']}")
        if f.get("mechanism"):
            lines.append(f"mechanism: {f['mechanism']}")
        if f.get("claim_type"):
            lines.append(f"claim_type: {f['claim_type']}")
        if f.get("n_metric_steps") is not None:
            lines.append(f"n_metric_steps: {f['n_metric_steps']}")
        if f.get("evidence_excerpt"):
            lines.append(f"evidence: {f['evidence_excerpt']}")
        if f.get("note"):
            lines.append(f"note: {f['note']}")
        lines.append("")
    if len(failures) > max_nodes:
        lines.append(f"... ({len(failures) - max_nodes} additional failures omitted for length)")
    return "\n".join(lines)


def _score_synthesis(case_id: str, synthesis: dict[str, Any]) -> dict[str, Any]:
    gt_key = case_id if case_id in GROUND_TRUTH else (
        "contagion_demo_tree" if case_id.startswith("contagion") else case_id
    )
    gt = GROUND_TRUTH.get(gt_key, {})
    blob = json.dumps(synthesis, ensure_ascii=False).lower()
    pattern = (synthesis.get("common_failure_pattern") or "").lower()
    rules = " ".join(str(x) for x in (synthesis.get("rules_out") or [])).lower()
    next_h = json.dumps(synthesis.get("where_to_look_next") or [], ensure_ascii=False).lower()
    full = f"{pattern} {rules} {next_h} {blob}"

    theme_hits: dict[str, bool] = {}
    for name, pat in gt.get("required_themes", []):
        theme_hits[name] = bool(re.search(pat, full, re.I | re.DOTALL))

    forbidden_hit: list[str] = []
    for pat, label in gt.get("forbidden_defaults", []):
        if re.search(pat, pattern, re.I):
            forbidden_hit.append(label)

    good_next = any(re.search(p, next_h, re.I) for p in gt.get("good_next_steps", []))

    n_required = len(gt.get("required_themes", []))
    n_hit = sum(1 for v in theme_hits.values() if v)
    theme_score = n_hit / max(1, n_required)

    # Pass bar: >=60% themes AND at least one good next-step AND no forbidden default
    passed = theme_score >= 0.6 and good_next and not forbidden_hit

    return {
        "theme_hits": theme_hits,
        "theme_score": round(theme_score, 3),
        "good_next_step": good_next,
        "forbidden_defaults_hit": forbidden_hit,
        "passed": passed,
    }


def _run_case(
    case_id: str,
    question: str,
    failures: list[dict[str, Any]],
    *,
    model: str,
    api_key: str,
) -> dict[str, Any]:
    dossier = _format_dossier(failures)
    prompt = SYNTHESIS_PROMPT.format(
        question=question,
        n_failures=len(failures),
        failure_dossier=dossier,
    )
    raw = _gemini(prompt, model, api_key)
    synthesis = _parse_json(raw)
    score = _score_synthesis(case_id, synthesis)
    return {
        "case_id": case_id,
        "question": question,
        "n_failures": len(failures),
        "verdict_counts": _verdict_counts(failures),
        "prompt_chars": len(prompt),
        "model": model,
        "synthesis": synthesis,
        "score": score,
        "raw_llm_response_chars": len(raw),
    }


def _verdict_counts(failures: list[dict]) -> dict[str, int]:
    c: dict[str, int] = {}
    for f in failures:
        v = str(f.get("verdict") or "?")
        c[v] = c.get(v, 0) + 1
    return c


def main() -> None:
    api_key = _load_api_key()
    if not api_key:
        print("ERROR: GOOGLE_API_KEY not set", file=sys.stderr)
        sys.exit(1)
    model = _load_model()

    cases = [
        ("contagion_demo_tree", *_load_contagion_demo_failures()),
        ("contagion_deep_records", *_load_contagion_deep_failures()),
        ("mandrake", *_load_mandrake_failures()),
    ]

    results = []
    for case_id, question, failures in cases:
        print(f"Running {case_id}: {len(failures)} failures, prompt building...")
        if not failures:
            results.append({"case_id": case_id, "error": "no failures"})
            continue
        try:
            r = _run_case(case_id, question, failures, model=model, api_key=api_key)
            results.append(r)
            print(f"  theme_score={r['score']['theme_score']} passed={r['score']['passed']}")
        except Exception as exc:
            results.append({"case_id": case_id, "error": str(exc)})

    n_pass = sum(1 for r in results if r.get("score", {}).get("passed"))
    report = {
        "bench": "campaign_failure_synthesis (fixes.md cheap test)",
        "generated_at": datetime.now(tz=UTC).isoformat(),
        "model": model,
        "interpretation": (
            "If passed on contagion with full evidence: information access may be the bottleneck — "
            "campaign memory/summarization worth building. "
            "If failed despite full dossier: synthesis capability is the bottleneck — "
            "bigger context will inherit the same shallow defaults."
        ),
        "summary": {
            "cases_run": len(results),
            "cases_passed": n_pass,
            "verdict": "SYNTHESIS_CAPABLE" if n_pass >= 2 else ("MIXED" if n_pass == 1 else "SYNTHESIS_WEAK"),
        },
        "ground_truth_notes": {
            "contagion": "Expect narrow simulator regimes, missing OOD/transfer, significance-without-replication — NOT topology_dependence-only",
            "mandrake": "Expect family/LOFO leakage via t55/t70/t75 thermal features — confirmed by counterfactual test",
        },
        "results": results,
    }
    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote {OUT}")
    print(f"Verdict: {report['summary']['verdict']} ({n_pass}/{len(results)} passed)")


if __name__ == "__main__":
    main()
