#!/usr/bin/env python3
"""LIVE end-to-end biology validation (v1 checklist item 3).

Runs a REAL biology question through:
  1. a REAL LLM (Gemini, GOOGLE_API_KEY) — skills-grounded hypothesis generation
     using each domain's real hypothesis skill as the generation context;
  2. the REAL domain verifier — ``DomainPlugin.run_verification`` runs the real
     leave-one-group-out (LOFO) + label-shuffle null on the REAL cached data
     (GTEx v8 median-TPM for genomics; DLKcat BRENDA+SABIO-RK for enzyme_kinetics),
     then ``classify_verdict`` returns the honest verdict.

It records EXACTLY what happens — including honest refute/inconclusive verdicts,
which are the CORRECT outcome on subtle, confounded real biological signals. It
NEVER fabricates a confirmation. The transcript is written to
``artifacts/biology_live_validation_<ts>.{md,json}``.

Requires: real data cached (``scripts/build_real_domain_datasets.py``) and a real
GOOGLE_API_KEY (read from the repo ``.env`` if not already in the environment).

Usage:
    PYTHONPATH="packages/propab-core;." python artifacts/run_biology_live_validation.py
"""
from __future__ import annotations

import json
import os
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "packages" / "propab-core"))
sys.path.insert(0, str(ROOT))

from propab.domain_modules.enzyme_kinetics.adapter import real_data_cached as enzyme_real
from propab.domain_modules.enzyme_kinetics.plugin import EnzymeKineticsPlugin
from propab.domain_modules.genomics.adapter import real_data_cached as genomics_real
from propab.domain_modules.genomics.plugin import GenomicsPlugin


# --------------------------------------------------------------------------- #
# Config: load the real LLM key from the environment or the repo .env.
# --------------------------------------------------------------------------- #
def _load_env_key() -> tuple[str, str]:
    key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY") or ""
    model = os.getenv("LLM_MODEL") or ""
    if not key:
        for env_path in (ROOT / ".env", ROOT.parent.parent.parent / ".env"):
            if not env_path.is_file():
                continue
            for line in env_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line.startswith("GOOGLE_API_KEY=") and not key:
                    key = line.split("=", 1)[1].strip()
                elif line.startswith("GEMINI_API_KEY=") and not key:
                    key = line.split("=", 1)[1].strip()
                elif line.startswith("LLM_MODEL=") and not model:
                    model = line.split("=", 1)[1].strip()
            if key:
                break
    return key, (model or "gemini-2.0-flash")


def _gemini_generate(prompt: str, *, api_key: str, model: str, timeout: float = 120.0) -> str:
    mid = model[len("models/") :] if model.startswith("models/") else model
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/{mid}:generateContent"
        f"?key={api_key}"
    )
    body = json.dumps(
        {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.4, "maxOutputTokens": 8192},
        }
    ).encode("utf-8")
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 (trusted Google host)
        payload = json.loads(resp.read())
    cands = payload.get("candidates") or []
    if not cands:
        raise RuntimeError(f"no candidates in Gemini response: {json.dumps(payload)[:500]}")
    parts = cands[0].get("content", {}).get("parts", [])
    return "".join(p.get("text", "") for p in parts)


def _extract_json_array(text: str) -> list[dict]:
    """Pull the first JSON array out of an LLM response (tolerates markdown fences)."""
    t = text.strip()
    # Strip a leading ```json / ``` fence and the trailing ``` if present.
    if t.startswith("```"):
        t = t[3:]
        if t[:4].lower() == "json":
            t = t[4:]
        if "```" in t:
            t = t[: t.index("```")]
        t = t.strip()
    start = t.find("[")
    if start == -1:
        raise ValueError(f"no JSON array found in LLM output:\n{text[:800]}")
    body = t[start:]
    end = body.rfind("]")
    if end != -1:
        try:
            return json.loads(body[: end + 1])
        except json.JSONDecodeError:
            pass
    # Truncation-tolerant repair: keep only whole ``{...}`` objects and re-close.
    objs = []
    depth = 0
    obj_start = None
    for i, ch in enumerate(body):
        if ch == "{":
            if depth == 0:
                obj_start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and obj_start is not None:
                try:
                    objs.append(json.loads(body[obj_start : i + 1]))
                except json.JSONDecodeError:
                    pass
                obj_start = None
    if not objs:
        raise ValueError(f"no JSON objects recoverable from LLM output:\n{text[:800]}")
    return objs


# --------------------------------------------------------------------------- #
# Per-domain live run.
# --------------------------------------------------------------------------- #
def _skill_text(rel: str) -> str:
    p = ROOT / "packages" / "propab-core" / "propab" / "skills" / "domains" / rel
    return p.read_text(encoding="utf-8") if p.is_file() else ""


def _build_prompt(question: str, skill: str, features: list[str], n: int) -> str:
    return (
        "You are Propab's hypothesis-generation layer for a computational-biology "
        "research campaign. Use ONLY the domain skill below to frame falsifiable, "
        "LOFO-testable hypotheses.\n\n"
        f"RESEARCH QUESTION:\n{question}\n\n"
        f"DOMAIN SKILL (your generation guardrails):\n{skill}\n\n"
        f"AVAILABLE FEATURES you may reference: {features}\n\n"
        f"Propose {n} DISTINCT, non-obvious, falsifiable hypotheses that the "
        "leave-one-group-out + label-shuffle null could actually decide. Avoid "
        "rediscoveries of definitions/known values.\n"
        "Return ONLY a JSON array; each element: "
        '{"text": "<hypothesis>", "feature_subset": ["<feat>", ...]}. '
        "feature_subset must be 2-4 names drawn from AVAILABLE FEATURES."
    )


def _run_domain(
    *, name: str, plugin, question: str, skill_rel: str, api_key: str, model: str, n: int
) -> dict:
    skill = _skill_text(skill_rel)
    features = plugin.available_features()
    prompt = _build_prompt(question, skill, features, n)
    print(f"\n=== {name}: generating hypotheses with {model} ===")
    raw = _gemini_generate(prompt, api_key=api_key, model=model)
    hyps = _extract_json_array(raw)
    print(f"[{name}] LLM proposed {len(hyps)} hypotheses")

    records = []
    for i, hyp in enumerate(hyps):
        text = str(hyp.get("text") or "").strip()
        feats = [str(f) for f in (hyp.get("feature_subset") or []) if f in features]
        if not text:
            continue
        print(f"[{name}] verifying #{i + 1}: {text[:90]}...")
        try:
            result = plugin.run_verification({"text": text, "feature_subset": feats}, features=feats or None)
            verdict, reason, conf = plugin.classify_verdict(text, result)
            # Also run the domain-owned known-value / rediscovery check.
            redisc = None
            if hasattr(plugin, "known_value_check"):
                redisc = plugin.known_value_check(text, result)
        except Exception as exc:  # noqa: BLE001
            records.append({"hypothesis": text, "feature_subset": feats, "error": str(exc)})
            print(f"[{name}]   ERROR: {exc}")
            continue
        rec = {
            "hypothesis": text,
            "feature_subset": feats,
            "verdict": verdict,
            "reason": reason,
            "confidence": conf,
            "lofo_r2": result.get("lofo_r2"),
            "label_shuffle_null_p": result.get("label_shuffle_null_p") or result.get("label_shuffle_permutation_p"),
            "label_shuffle_null_p95": result.get("label_shuffle_null_p95"),
            "verified_true_steps": result.get("verified_true_steps"),
            "n_samples": result.get("n_genes") or result.get("n_samples") or result.get("n_enzymes"),
            "rediscovery": redisc,
        }
        records.append(rec)
        p = rec["label_shuffle_null_p"]
        print(
            f"[{name}]   verdict={verdict} lofo_r2={rec['lofo_r2']} "
            f"shuffle_p={p} conf={conf}"
        )
    return {
        "domain": name,
        "question": question,
        "model": model,
        "n_generated": len(hyps),
        "llm_raw_response": raw,
        "results": records,
    }


def main() -> int:
    if not (genomics_real() and enzyme_real()):
        print(
            "REAL data not cached. Run scripts/build_real_domain_datasets.py first "
            "(genomics_real=%s enzyme_real=%s)." % (genomics_real(), enzyme_real())
        )
        return 2
    api_key, model = _load_env_key()
    if not api_key:
        print("No GOOGLE_API_KEY / GEMINI_API_KEY available; cannot run the live LLM path.")
        return 2

    runs = []
    runs.append(
        _run_domain(
            name="genomics",
            plugin=GenomicsPlugin(),
            question=(
                "In the GTEx v8 cross-tissue expression atlas, which non-obvious "
                "gene-level expression feature predicts tissue-specificity (tau) or "
                "mean expression in a HELD-OUT tissue, surviving the tissue-label "
                "shuffle null?"
            ),
            skill_rel="genomics/tissue-specificity-and-cross-tissue-transfer.skill.md",
            api_key=api_key,
            model=model,
            n=4,
        )
    )
    runs.append(
        _run_domain(
            name="enzyme_kinetics",
            plugin=EnzymeKineticsPlugin(),
            question=(
                "In the DLKcat (BRENDA + SABIO-RK) kcat compilation, which enzyme "
                "sequence/physicochemical feature predicts log_kcat across a HELD-OUT "
                "EC class, surviving the EC-label shuffle null?"
            ),
            skill_rel="enzyme_kinetics/sequence-to-function-kcat-reasoning.skill.md",
            api_key=api_key,
            model=model,
            n=4,
        )
    )

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_json = ROOT / "artifacts" / "biology_live_validation.json"
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "llm_model": model,
        "note": (
            "Live end-to-end: real LLM generation + real LOFO/label-shuffle null on "
            "real cached data. Honest refute/inconclusive on subtle real signals is "
            "the CORRECT outcome; no confirmation is fabricated."
        ),
        "runs": runs,
    }
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # Human-readable transcript.
    lines = [
        f"# Biology live end-to-end validation ({ts})",
        "",
        f"LLM model: `{model}`  |  data: real GTEx v8 + real DLKcat (BRENDA+SABIO-RK)",
        "",
        payload["note"],
        "",
    ]
    for run in runs:
        lines += [f"## {run['domain']}", "", f"**Question:** {run['question']}", ""]
        lines.append(f"LLM proposed {run['n_generated']} hypotheses; each verified by the real LOFO + shuffle null.")
        lines.append("")
        for i, r in enumerate(run["results"], 1):
            if "error" in r:
                lines.append(f"{i}. `[ERROR]` {r['hypothesis']} — {r['error']}")
                continue
            rd = r.get("rediscovery")
            rd_note = f"  _rediscovery: {rd['rediscovery_identifier']}_" if rd else ""
            lines.append(
                f"{i}. **{r['verdict']}** (conf {r['confidence']}) — {r['hypothesis']}\n"
                f"   - features: `{r['feature_subset']}`; "
                f"LOFO R²={r['lofo_r2']}, shuffle p={r['label_shuffle_null_p']}, "
                f"n={r['n_samples']}\n"
                f"   - reason: {r['reason']}{rd_note}"
            )
        # Honest summary of verdict mix.
        verdicts = [r.get("verdict") for r in run["results"] if "verdict" in r]
        conf = verdicts.count("confirmed")
        lines += [
            "",
            f"**{run['domain']} verdict mix:** "
            + ", ".join(f"{v}={verdicts.count(v)}" for v in ("confirmed", "refuted", "inconclusive"))
            + (
                "  — an honest 'no novel confirmed signal' result on real, subtle data."
                if conf == 0
                else "  — a confirmation cleared the real LOFO + shuffle-null gate (inspect it critically)."
            ),
            "",
        ]
    out_md = ROOT / "artifacts" / "biology_live_validation.md"
    out_md.write_text("\n".join(lines), encoding="utf-8")

    print(f"\nwrote {out_json}")
    print(f"wrote {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
