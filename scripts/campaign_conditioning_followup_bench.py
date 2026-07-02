#!/usr/bin/env python3
"""
fixes.md follow-up bench:
  1. Mandrake campaign synthesis with evidence-complete dossier (DB → API → enriched fallback)
  2. Part A vs Part B expansion comparison on t55/t70/t75 focal node

Part A: one-shot expand prompt + structured failure fields (wiring fix)
Part B: explicit two-step — interpret failure (rules out / implies) THEN generate children
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
from propab.tools.mandrake.mandrake_verification import mandrake_verification  # noqa: E402
from services.orchestrator.campaign_diagnostics import parse_evidence_obj  # noqa: E402

ART = ROOT / "artifacts"
OUT = ART / "campaign_conditioning_followup_bench.json"
MANDRAKE_CID = "1a61b453-38e0-4023-9135-cf36caeed505"
_ENRICH_LOFO = False  # set True to run LOFO per node (slow); focal node always gets live LOFO

# ── Improved prompts (role + task explicit) ───────────────────────────────────

SYNTHESIS_SYSTEM = """\
You are the **Campaign Post-Mortem Analyst** for an autonomous research system.

Your job is NOT to generate new hypotheses yet. Your job is to **diagnose why a whole
campaign failed** by reading every dead branch together — the way a human PI would after
a failed grant year.

You must distinguish:
- **Load-bearing diagnosis** — the specific confound or methodological mistake that explains
  MOST failures (e.g. family leakage in grouped CV, missing OOD holdout).
- **Surface pattern** — vocabulary that sounds right but doesn't change what to do next
  (e.g. "try different features" without naming the confound test).

You must NOT default to generic artifact labels unless the evidence supports them.
Prefer diagnoses that imply a **specific falsifiable next experiment**, not more variants
of what already failed."""

SYNTHESIS_TASK = """\
## Research question
{question}

## Campaign outcome
- Total failed discovery nodes shown below: {n_failures}
- Verdict mix: {verdict_counts}
- Campaign metric: LOFO R² (higher is better; negative = worse than baseline on held-out groups)

## Campaign-level diagnostics (ground truth from offline verification — use these)
{campaign_diagnostics}

## Every failed discovery hypothesis (full dossier)
{failure_dossier}

---

## Your task (follow exactly)

1. **common_failure_pattern** — One paragraph. What SINGLE underlying mistake explains
   most of these failures? Cite specific recurring evidence (LOFO values, verdict_reasons,
   feature names). If family/group leakage is present, say so explicitly.

2. **rules_out** — Bullet list of claim TYPES now ruled out (not just "these hypotheses").

3. **diagnostic_next_experiments** — 2–4 experiments that TEST THE DIAGNOSIS itself
   (e.g. family-label shuffle null, finer family split LOFO degradation, residual-after-demean).
   NOT more feature-combination horse-races unless you explain why the confound test passed first.

4. **where_to_look_next** — 3–5 child hypothesis SKETCHES that would follow ONLY IF
   the diagnostic experiments above came back negative for leakage/confounding.

5. **confidence** — 0–1 on whether your diagnosis is load-bearing vs speculative.

Return JSON ONLY with keys:
common_failure_pattern, rules_out, diagnostic_next_experiments, where_to_look_next, confidence
(where_to_look_next is a list of {{"hypothesis_sketch", "why_different_from_failures"}})
"""

PART_A_EXPAND = """\
You are a **Hypothesis Tree Expander** in an autonomous research campaign.

A parent hypothesis was tested and FAILED. Generate 3–5 child hypotheses that explore
alternatives. This is Part A (wiring fix): you receive structured failure fields.

## Parent node
- Verdict: {verdict} (confidence {confidence:.2f})
- Hypothesis: {parent_text}

## Structured failure diagnostics (READ THESE — not just the raw blob)
- verdict_reason: {verdict_reason}
- inconclusive_reason: {inconclusive_reason}
- failure_signature: {failure_signature}
- artifact_gate_top: {artifact_top}
- artifact_gate_verdict: {artifact_gate_verdict}

## Raw evidence
{evidence_summary}

## LOFO verification metrics (mandrake)
{lofo_block}

## Research question
{question}

Generate 3–5 child hypotheses. Each must include: id, text, test_methodology, expansion_type.
Return JSON array only.
"""

PART_B_STEP1 = """\
You are a **Failure Interpreter** — step 1 of 2. Do NOT generate child hypotheses yet.

A parent hypothesis failed. Your ONLY job is to state what this failure **rules out** and
**implies must be true instead**, with mechanistic specificity.

## Parent hypothesis
{parent_text}

## Structured failure diagnostics
- Verdict: {verdict} (confidence {confidence:.2f})
- verdict_reason: {verdict_reason}
- inconclusive_reason: {inconclusive_reason}
- failure_signature: {failure_signature}
- artifact_gate: {artifact_top} ({artifact_gate_verdict})

## Evidence
{evidence_summary}

## LOFO verification metrics
{lofo_block}

## Research question
{question}

## Other failed attempts in this campaign (pattern context)
{campaign_failure_summary}

Return JSON ONLY:
{{
  "failure_mechanism": "one sentence — why this specific test failed",
  "rules_out": ["specific claim types or approaches ruled out"],
  "implies": ["what must be tested BEFORE any new predictive claim"],
  "confound_hypothesis": "most likely confound (e.g. family leakage) or null if unclear",
  "diagnostic_experiment": "single experiment that would confirm/refute the confound"
}}
"""

PART_B_STEP2 = """\
You are a **Hypothesis Tree Expander** — step 2 of 2.

A failure interpreter has already analyzed the parent node. Generate children **conditioned
on the interpreter's conclusions** — do NOT ignore them.

## Interpreter output (binding)
{interpreter_json}

## Parent hypothesis (failed)
{parent_text}

## Research question
{question}

Rules:
- Children must address the confound test OR a fundamentally different approach identified
  in "implies" — NOT more variants of the same feature-combination horse-race.
- Each child: id, text, test_methodology, expansion_type, why_this_follows_from_interpreter

Return JSON array only.
"""

# ── Ground truth rubrics ─────────────────────────────────────────────────────

GOOD_NEXT = re.compile(
    r"shuffle|permut|null|family.{0,20}(split|label|demean|residual|hold.?out)|"
    r"finer.{0,20}group|lofo.{0,15}(degrad|null)|counterfactual|"
    r"leave.{0,5}one.{0,5}(family|group)|confound.{0,20}test",
    re.I,
)
BAD_NEXT = re.compile(
    r"t55|t70|t75|thermal.{0,20}(delta|auc|quadratic|derivative|slope)|"
    r"feature.{0,20}(subset|cluster|combination)|higher.{0,10}logo|"
    r"compare.{0,20}feature.{0,20}set",
    re.I,
)
LEAKAGE_DIAG = re.compile(
    r"family.{0,20}(leak|confound|proxy|identity|surrogate)|group.{0,20}identity|"
    r"lofo.{0,20}(gap|negative|degrad)|distribution.{0,10}leak|"
    r"not.{0,20}independent.{0,20}of.{0,20}(family|lineage|group)",
    re.I,
)


def _load_api_key() -> str:
    key = (os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY") or "").strip()
    if key:
        return key
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            if line.startswith("GOOGLE_API_KEY=") or line.startswith("GEMINI_API_KEY="):
                return line.split("=", 1)[1].strip()
    return ""


def _load_model() -> str:
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            if line.startswith("LLM_MODEL="):
                return line.split("=", 1)[1].strip()
    return os.environ.get("LLM_MODEL", "gemini-3-flash-preview")


def _gemini(prompt: str, model: str, api_key: str, *, system: str = "") -> str:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    parts: list[dict] = []
    if system.strip():
        parts.append({"text": system.strip() + "\n\n---\n\n"})
    parts.append({"text": prompt})
    payload = {
        "contents": [{"parts": parts}],
        "generationConfig": {"temperature": 0.2},
    }
    with httpx.Client(timeout=180.0) as client:
        r = client.post(url, params={"key": api_key}, json=payload)
    r.raise_for_status()
    resp_parts = (((r.json().get("candidates") or [{}])[0].get("content") or {}).get("parts")) or []
    return "".join(p.get("text", "") for p in resp_parts if isinstance(p, dict))


def _parse_json(raw: str) -> Any:
    text = (raw or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"[\[{][\s\S]*[\]}]", text)
        if not m:
            return {"_parse_error": True, "_raw": raw[:3000]}
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            return {"_parse_error": True, "_raw": raw[:3000]}


def _fetch_mandrake_from_db(campaign_id: str) -> dict[str, Any] | None:
    try:
        import psycopg
    except ImportError:
        return None
    dsn = "postgresql://propab:propab@localhost:5432/propab"
    try:
        with psycopg.connect(dsn, connect_timeout=4) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id::text, question, hypothesis_tree_json
                    FROM research_campaigns WHERE id = %s
                    """,
                    (campaign_id,),
                )
                row = cur.fetchone()
                if not row:
                    return None
                cid, question, tree_json = row
                tree = json.loads(tree_json) if isinstance(tree_json, str) else tree_json
                cur.execute(
                    """
                    SELECT text, verdict, confidence, evidence_summary, key_finding
                    FROM hypotheses WHERE session_id = %s ORDER BY created_at
                    """,
                    (campaign_id,),
                )
                hyps = [
                    dict(zip([d[0] for d in cur.description], r, strict=True))
                    for r in cur.fetchall()
                ]
                return {
                    "source": "postgres",
                    "campaign_id": cid,
                    "question": question,
                    "tree_nodes": tree.get("nodes") or {},
                    "hypothesis_rows": hyps,
                }
    except Exception as exc:
        return {"source": "postgres_error", "error": str(exc)}


def _fetch_mandrake_from_api(campaign_id: str) -> dict[str, Any] | None:
    try:
        with httpx.Client(timeout=15.0) as client:
            r = client.get(f"http://localhost:8000/campaigns/{campaign_id}")
            if r.status_code != 200:
                return None
            data = r.json()
            camp = data.get("campaign") or data
            tree = camp.get("hypothesis_tree") or {}
            return {
                "source": "api",
                "campaign_id": campaign_id,
                "question": camp.get("question") or "",
                "tree_nodes": tree.get("nodes") or {},
                "hypothesis_rows": [],
            }
    except Exception as exc:
        return {"source": "api_error", "error": str(exc)}


_LOFO_CACHE: dict[tuple[tuple[str, ...], tuple[str, ...]], dict[str, Any]] = {}


def _run_lofo(feature_subset: list[str], compare: list[str] | None = None) -> dict[str, Any]:
    key = (tuple(feature_subset), tuple(compare or ()))
    if key in _LOFO_CACHE:
        return _LOFO_CACHE[key]
    r = mandrake_verification(
        feature_subset=feature_subset,
        compare_features=compare,
    )
    out = r.output if r.success else {"error": str(r.error)}
    if not isinstance(out, dict):
        out = {"raw_output": out}
    _LOFO_CACHE[key] = out
    return out


def _campaign_diagnostics_block() -> str:
    cf = json.loads((ART / "mechanism_counterfactual_test.json").read_text(encoding="utf-8"))
    ri = json.loads((ART / "representation_invention_test.json").read_text(encoding="utf-8"))
    lines = [
        "### Counterfactual test (t55/t70/t75 family-proxy mechanism)",
        f"- Mechanism tested: {cf.get('mechanism_as_stated', '')[:200]}",
        f"- Coarse LOFO R²: {cf['experiment']['coarse']['lofo_r2']}, within R²: {cf['experiment']['coarse']['within_r2']}, gap: {cf['experiment']['coarse']['gap']}",
        f"- Finer family splits DEGRADE LOFO (family×active delta: {cf['experiment']['fine_splits']['family_x_active']['delta_lofo_vs_coarse']})",
        f"- Outcome: {cf.get('outcome')} — {cf.get('interpretation', '')[:200]}",
        "",
        "### Representation invention null (derived features)",
        f"- Best raw triple LOFO: {ri.get('baseline_raw', {}).get('best_triple', {}).get('lofo_r2')}",
        f"- Derived features surviving permutation null: 0/5",
        "",
        "### Upstream anomaly (mechanism_objects)",
        "- Top feature cluster t55|t70|t75 had LOFO=-0.120 vs group baseline 0.294 — partial signal may be family-proxy not biology",
    ]
    return "\n".join(lines)


def _extract_features(text: str) -> list[str]:
    found = re.findall(r"t\d+_raw", text.lower())
    return list(dict.fromkeys(found))


def _build_failure_row(nid: str, n: dict[str, Any], hyp_rows: list[dict]) -> dict[str, Any] | None:
    role = n.get("node_role") or "DISCOVERY"
    verdict = n.get("verdict")
    if role == NODE_ROLE_CONTROL or verdict in ("confirmed", "pending", None):
        return None
    text = (n.get("text") or "").strip()
    if text.lower().startswith("null hypothesis"):
        return None

    ev = n.get("evidence_summary") or ""
    # Join DB hypothesis row by text prefix if tree node lacks evidence
    if not ev and hyp_rows:
        for h in hyp_rows:
            if (h.get("text") or "")[:80] == text[:80]:
                ev = h.get("evidence_summary") or ""
                break

    ev_obj = parse_evidence_obj(ev)
    feats = _extract_features(text)
    lofo_extra: dict[str, Any] = {}
    # Only enrich nodes that mention explicit t*_raw features (skip during bulk load)
    if feats and len(feats) <= 4 and _ENRICH_LOFO:
        lofo_extra = _run_lofo(feats)

    return {
        "id": nid,
        "verdict": verdict,
        "text": text[:700],
        "generation": n.get("generation"),
        "inconclusive_reason": n.get("inconclusive_reason"),
        "failure_signature": n.get("failure_signature"),
        "mechanism": n.get("mechanism"),
        "verdict_reason": ev_obj.get("verdict_reason"),
        "claim_type": ev_obj.get("claim_type") or n.get("claim_type"),
        "metric_value": ev_obj.get("metric_value") or lofo_extra.get("lofo_r2") or lofo_extra.get("metric_value"),
        "lofo_r2": lofo_extra.get("lofo_r2"),
        "lofo_gap": lofo_extra.get("lofo_gap"),
        "within_r2": lofo_extra.get("within_r2"),
        "p_value": ev_obj.get("p_value") or lofo_extra.get("p_value"),
        "n_metric_steps": ev_obj.get("n_metric_steps"),
        "artifact_gate": ev_obj.get("artifact_gate"),
        "evidence_summary": ev[:800] if ev else json.dumps(lofo_extra, ensure_ascii=False)[:800],
        "feature_subset": feats,
    }


def _load_mandrake_failures() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    meta: dict[str, Any] = {"fetch_attempts": []}
    raw = _fetch_mandrake_from_db(MANDRAKE_CID)
    meta["fetch_attempts"].append(raw if raw else {"source": "postgres", "error": "no connection"})
    if raw and raw.get("tree_nodes"):
        meta["data_source"] = raw["source"]
        nodes = raw["tree_nodes"]
        hyp_rows = raw.get("hypothesis_rows") or []
        question = raw.get("question") or ""
    else:
        api = _fetch_mandrake_from_api(MANDRAKE_CID)
        meta["fetch_attempts"].append(api if api else {"source": "api", "error": "unavailable"})
        if api and api.get("tree_nodes"):
            meta["data_source"] = api["source"]
            nodes = api["tree_nodes"]
            hyp_rows = []
            question = api.get("question") or ""
        else:
            meta["data_source"] = "mandrake_run_analysis_json + live LOFO enrichment"
            analysis = json.loads((ART / "mandrake_run_analysis.json").read_text(encoding="utf-8"))
            question = analysis["summary"]["question"]
            nodes = {n["id"]: n for n in analysis.get("nodes") or []}
            hyp_rows = []

    failures = []
    for nid, n in nodes.items():
        row = _build_failure_row(nid, n if isinstance(n, dict) else {}, hyp_rows)
        if row:
            failures.append(row)
    meta["question"] = question
    meta["n_failures"] = len(failures)
    return meta, failures


def _precompute_lofo_for_failures(failures: list[dict[str, Any]]) -> None:
    """Attach live LOFO to nodes with unique small feature subsets (max 12 runs)."""
    seen: set[tuple[str, ...]] = set()
    for f in failures:
        feats = tuple(f.get("feature_subset") or [])
        if not feats or len(feats) > 4 or feats in seen:
            continue
        seen.add(feats)
        if len(seen) > 12:
            break
        lofo = _run_lofo(list(feats))
        for g in failures:
            if tuple(g.get("feature_subset") or []) == feats:
                g["lofo_r2"] = lofo.get("lofo_r2")
                g["lofo_gap"] = lofo.get("lofo_gap")
                g["within_r2"] = lofo.get("within_r2")
                if not g.get("evidence_summary") or str(g["evidence_summary"]).startswith("{"):
                    g["evidence_summary"] = json.dumps(lofo, ensure_ascii=False)[:800]


def _format_dossier(failures: list[dict[str, Any]], *, max_n: int = 35) -> str:
    lines: list[str] = []
    for i, f in enumerate(failures[:max_n], 1):
        lines.append(f"### Failure {i} [{f.get('verdict')}] id={f.get('id')}")
        lines.append(f"**Hypothesis:** {f.get('text')}")
        if f.get("feature_subset"):
            lines.append(f"**Features:** {f.get('feature_subset')}")
        for k in ("verdict_reason", "inconclusive_reason", "failure_signature", "mechanism"):
            if f.get(k):
                lines.append(f"**{k}:** {f[k]}")
        if f.get("lofo_r2") is not None:
            lines.append(
                f"**LOFO R²:** {f.get('lofo_r2')}, gap={f.get('lofo_gap')}, within={f.get('within_r2')}"
            )
        ag = f.get("artifact_gate")
        if isinstance(ag, dict):
            ranked = ag.get("ranked_artifacts") or []
            top = ranked[0] if ranked else {}
            lines.append(f"**artifact_gate:** {ag.get('verdict')} — {top.get('artifact_id', '?')}")
        if f.get("evidence_summary"):
            lines.append(f"**evidence:** {f['evidence_summary'][:600]}")
        lines.append("")
    if len(failures) > max_n:
        lines.append(f"_({len(failures) - max_n} additional failures omitted)_")
    return "\n".join(lines)


def _score_synthesis(synthesis: dict[str, Any]) -> dict[str, Any]:
    blob = json.dumps(synthesis, ensure_ascii=False)
    pattern = synthesis.get("common_failure_pattern") or ""
    diags = json.dumps(synthesis.get("diagnostic_next_experiments") or [], ensure_ascii=False)
    nxt = json.dumps(synthesis.get("where_to_look_next") or [], ensure_ascii=False)

    leakage_in_pattern = bool(LEAKAGE_DIAG.search(pattern))
    leakage_in_diags = bool(LEAKAGE_DIAG.search(diags))
    good_diag = bool(GOOD_NEXT.search(diags))
    bad_next = bool(BAD_NEXT.search(nxt)) and not bool(GOOD_NEXT.search(nxt))
    good_next = bool(GOOD_NEXT.search(nxt))

    return {
        "leakage_diagnosed_in_pattern": leakage_in_pattern,
        "leakage_in_diagnostic_experiments": leakage_in_diags,
        "has_confound_test_in_diagnostics": good_diag,
        "next_steps_are_feature_horse_race": bad_next,
        "next_steps_include_confound_test": good_next,
        "passed": leakage_in_pattern and good_diag and not bad_next,
    }


def _score_children(children: list[dict], *, label: str) -> dict[str, Any]:
    blob = json.dumps(children, ensure_ascii=False)
    n = len(children)
    good = len(re.findall(GOOD_NEXT, blob))
    bad = len(re.findall(BAD_NEXT, blob))
    leakage = bool(LEAKAGE_DIAG.search(blob))
    texts = [str(c.get("text") or "") for c in children if isinstance(c, dict)]
    thermal_horse = sum(1 for t in texts if re.search(r"t5[05]_raw|t70_raw|t75_raw", t, re.I))
    return {
        "label": label,
        "n_children": n,
        "good_pattern_matches": good,
        "bad_pattern_matches": bad,
        "mentions_leakage_or_confound": leakage,
        "thermal_feature_horse_race_count": thermal_horse,
        "passed": good >= 1 and bad <= good and thermal_horse <= 1,
        "children_preview": [{"id": c.get("id"), "text": (c.get("text") or "")[:200]} for c in children[:5]],
    }


def _pick_focal_node(failures: list[dict]) -> dict[str, Any]:
    for f in failures:
        feats = set(f.get("feature_subset") or [])
        if {"t55_raw", "t70_raw", "t75_raw"}.issubset(feats) and "logo" in (f.get("text") or "").lower():
            return f
    for f in failures:
        if {"t55_raw", "t70_raw", "t75_raw"}.issubset(set(f.get("feature_subset") or [])):
            return f
    raise SystemExit("No t55/t70/t75 focal node found")


def _build_focal_evidence(focal: dict[str, Any]) -> dict[str, Any]:
    feats_a = ["t55_raw", "t70_raw", "t75_raw"]
    feats_b = ["t70_raw", "t75_raw", "t80_raw"]
    lofo_a = _run_lofo(feats_a, compare=feats_b)
    lofo_b = _run_lofo(feats_b)
    focal = {**focal, "feature_subset": feats_a}
    ev_obj = {
        "metric_value": lofo_a.get("lofo_r2"),
        "baseline_value": 0.0,
        "lofo_r2": lofo_a.get("lofo_r2"),
        "lofo_gap": lofo_a.get("lofo_gap"),
        "within_r2": lofo_a.get("within_r2"),
        "compare_lofo_r2": lofo_b.get("lofo_r2"),
        "n_metric_steps": 1,
        "verdict_reason": "significance gate passed but metric direction ambiguous",
        "claim_type": "CLAIM_STATISTICAL",
        "artifact_gate": {
            "verdict": "inconclusive",
            "verdict_reason": "LOFO gap large; family confound not ruled out",
            "ranked_artifacts": [
                {
                    "artifact_id": "family_leakage",
                    "description": "Thermal features may proxy rt_family identity",
                    "plausibility_score": 0.85,
                    "proposed_test": "label_shuffle_lofo",
                }
            ],
        },
    }
    evidence_summary = (
        f"evidence={json.dumps(ev_obj, ensure_ascii=False)}; "
        f"plan_origin=mandrake_verification; steps=1."
    )
    return {
        **focal,
        "verdict": "inconclusive",
        "confidence": 0.55,
        "inconclusive_reason": "verification_failure",
        "failure_signature": "lofo_gap_large",
        "verdict_reason": ev_obj["verdict_reason"],
        "evidence_summary": evidence_summary,
        "lofo_a": lofo_a,
        "lofo_b": lofo_b,
        "artifact_top": "family_leakage",
        "artifact_gate_verdict": "inconclusive",
    }


def _lofo_block(focal: dict[str, Any]) -> str:
    a = focal.get("lofo_a") or {}
    b = focal.get("lofo_b") or {}
    return json.dumps(
        {
            "t55_t70_t75": {k: a.get(k) for k in ("lofo_r2", "lofo_gap", "within_r2", "p_value")},
            "t70_t75_t80": {k: b.get(k) for k in ("lofo_r2", "lofo_gap", "within_r2")},
            "counterfactual_note": "Finer family splits degrade LOFO (see campaign diagnostics)",
        },
        indent=2,
    )


def _campaign_failure_summary(failures: list[dict], *, max_n: int = 8) -> str:
    lines = []
    for f in failures[:max_n]:
        lr = f.get("lofo_r2")
        lines.append(
            f"- [{f.get('verdict')}] LOFO={lr} features={f.get('feature_subset')}: {(f.get('text') or '')[:120]}"
        )
    return "\n".join(lines)


def main() -> None:
    api_key = _load_api_key()
    if not api_key:
        sys.exit("GOOGLE_API_KEY required")
    model = _load_model()

    meta, failures = _load_mandrake_failures()
    question = meta["question"]
    print("Precomputing LOFO for unique feature subsets...")
    _precompute_lofo_for_failures(failures)
    diag_block = _campaign_diagnostics_block()
    dossier = _format_dossier(failures)

    print(f"Mandrake failures: {len(failures)} (source={meta.get('data_source')})")

    # ── 1. Improved full-campaign synthesis ─────────────────────────────────
    synth_prompt = SYNTHESIS_TASK.format(
        question=question,
        n_failures=len(failures),
        verdict_counts=json.dumps(_verdict_counts(failures)),
        campaign_diagnostics=diag_block,
        failure_dossier=dossier,
    )
    print("Running improved campaign synthesis...")
    synth_raw = _gemini(synth_prompt, model, api_key, system=SYNTHESIS_SYSTEM)
    synthesis = _parse_json(synth_raw)
    synth_score = _score_synthesis(synthesis if isinstance(synthesis, dict) else {})

    # ── 2. Part A vs Part B on focal node ───────────────────────────────────
    focal_base = _pick_focal_node(failures)
    focal = _build_focal_evidence(focal_base)
    lofo_block = _lofo_block(focal)
    fail_summary = _campaign_failure_summary(failures)

    part_a_prompt = PART_A_EXPAND.format(
        verdict=focal["verdict"],
        confidence=float(focal.get("confidence") or 0.5),
        parent_text=focal["text"],
        verdict_reason=focal.get("verdict_reason") or "n/a",
        inconclusive_reason=focal.get("inconclusive_reason") or "n/a",
        failure_signature=focal.get("failure_signature") or "n/a",
        artifact_top=focal.get("artifact_top") or "n/a",
        artifact_gate_verdict=focal.get("artifact_gate_verdict") or "n/a",
        evidence_summary=focal.get("evidence_summary") or "",
        lofo_block=lofo_block,
        question=question,
    )
    print("Running Part A (structured one-shot expand)...")
    part_a_raw = _gemini(part_a_prompt, model, api_key)
    part_a_children = _parse_json(part_a_raw)
    if not isinstance(part_a_children, list):
        part_a_children = part_a_children.get("children") or part_a_children.get("hypotheses") or []

    print("Running Part B step 1 (failure interpreter)...")
    part_b1_prompt = PART_B_STEP1.format(
        parent_text=focal["text"],
        verdict=focal["verdict"],
        confidence=float(focal.get("confidence") or 0.5),
        verdict_reason=focal.get("verdict_reason") or "n/a",
        inconclusive_reason=focal.get("inconclusive_reason") or "n/a",
        failure_signature=focal.get("failure_signature") or "n/a",
        artifact_top=focal.get("artifact_top") or "n/a",
        artifact_gate_verdict=focal.get("artifact_gate_verdict") or "n/a",
        evidence_summary=focal.get("evidence_summary") or "",
        lofo_block=lofo_block,
        question=question,
        campaign_failure_summary=fail_summary,
    )
    part_b1_raw = _gemini(part_b1_prompt, model, api_key, system=SYNTHESIS_SYSTEM)
    interpreter = _parse_json(part_b1_raw)

    print("Running Part B step 2 (conditioned expand)...")
    part_b2_prompt = PART_B_STEP2.format(
        interpreter_json=json.dumps(interpreter, indent=2, ensure_ascii=False),
        parent_text=focal["text"],
        question=question,
    )
    part_b2_raw = _gemini(part_b2_prompt, model, api_key)
    part_b_children = _parse_json(part_b2_raw)
    if not isinstance(part_b_children, list):
        part_b_children = part_b_children.get("children") or part_b_children.get("hypotheses") or []

    part_a_score = _score_children(part_a_children, label="part_a_structured_oneshot")
    part_b_score = _score_children(part_b_children, label="part_b_two_step")

    report = {
        "bench": "campaign_conditioning_followup (fixes.md)",
        "generated_at": datetime.now(tz=UTC).isoformat(),
        "model": model,
        "mandrake_campaign_id": MANDRAKE_CID,
        "data_provenance": meta,
        "ground_truth": {
            "correct_diagnosis": "t55/t70/t75 LOFO signal is family-proxy not cross-family biophysics",
            "correct_next_step": "family-label shuffle null, finer family-split LOFO degradation, residual-after-demean — NOT more thermal feature combinations",
            "source_artifacts": [
                "mechanism_counterfactual_test.json",
                "representation_invention_test.json",
                "dead_finding_diagnosis_bench.json (mandrake cases)",
            ],
        },
        "prompts": {
            "synthesis_system": SYNTHESIS_SYSTEM,
            "synthesis_task_template": "SYNTHESIS_TASK (see script)",
            "part_a_template": "PART_A_EXPAND",
            "part_b_step1_template": "PART_B_STEP1",
            "part_b_step2_template": "PART_B_STEP2",
        },
        "improved_campaign_synthesis": {
            "n_failures_in_prompt": min(35, len(failures)),
            "prompt_chars": len(synth_prompt),
            "synthesis": synthesis,
            "score": synth_score,
            "raw_response_chars": len(synth_raw),
        },
        "part_a_vs_part_b": {
            "focal_node": {
                "id": focal.get("id"),
                "text": focal.get("text"),
                "feature_subset": focal.get("feature_subset"),
                "lofo_a": focal.get("lofo_a"),
                "lofo_b": focal.get("lofo_b"),
                "structured_fields_used": [
                    "verdict_reason", "inconclusive_reason", "failure_signature",
                    "artifact_top", "artifact_gate_verdict", "lofo_block",
                ],
            },
            "part_a": {
                "description": "One-shot expand with structured failure fields (Part A wiring)",
                "prompt_chars": len(part_a_prompt),
                "prompt_excerpt": part_a_prompt[:1200],
                "children": part_a_children,
                "score": part_a_score,
            },
            "part_b": {
                "description": "Two-step: failure interpreter then conditioned expand",
                "step1_interpreter": interpreter,
                "step1_prompt_chars": len(part_b1_prompt),
                "step2_prompt_chars": len(part_b2_prompt),
                "children": part_b_children,
                "score": part_b_score,
            },
            "winner": (
                "part_b" if part_b_score.get("passed") and not part_a_score.get("passed")
                else "part_a" if part_a_score.get("passed") and not part_b_score.get("passed")
                else "tie" if part_a_score.get("passed") == part_b_score.get("passed")
                else "neither"
            ),
        },
        "verdict": {
            "synthesis_with_full_evidence": "PASS" if synth_score.get("passed") else "FAIL",
            "part_a_alone_sufficient": part_a_score.get("passed"),
            "part_b_improves_over_a": (
                part_b_score.get("passed") and not part_a_score.get("passed")
            ) or (
                part_b_score.get("good_pattern_matches", 0) > part_a_score.get("good_pattern_matches", 0)
                and part_b_score.get("thermal_feature_horse_race_count", 99)
                < part_a_score.get("thermal_feature_horse_race_count", 0)
            ),
            "architecture_implication": None,  # filled below
        },
    }

    impl: list[str] = []
    if synth_score.get("passed"):
        impl.append("Campaign-level synthesis works with evidence-complete dossier → memory/summarization worth building.")
    else:
        impl.append("Even with full evidence, synthesis failed load-bearing diagnosis → fix reasoning before scaling context.")
    if part_b_score.get("passed") and not part_a_score.get("passed"):
        impl.append("Part B two-step required — wire interpreter step into expansion path.")
    elif part_a_score.get("passed"):
        impl.append("Part A structured fields may suffice for expansion — still add interpreter for safety.")
    else:
        impl.append("Neither Part A nor Part B produced confound-targeting children — upstream synthesis capability gap.")
    report["verdict"]["architecture_implication"] = impl

    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote {OUT}")
    print(f"Synthesis: {report['verdict']['synthesis_with_full_evidence']}")
    print(f"Part A passed: {part_a_score.get('passed')} | Part B passed: {part_b_score.get('passed')}")
    print(f"Winner: {report['part_a_vs_part_b']['winner']}")


def _verdict_counts(failures: list[dict]) -> dict[str, int]:
    c: dict[str, int] = {}
    for f in failures:
        v = str(f.get("verdict") or "?")
        c[v] = c.get(v, 0) + 1
    return c


if __name__ == "__main__":
    main()
