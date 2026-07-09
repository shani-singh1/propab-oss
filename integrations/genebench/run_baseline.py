"""GeneBench-Pro baseline runner for Propab.

Runs a ReAct-style code agent (Propab's real gemini-3.5-flash client) on each open
GeneBench-Pro problem: the agent iteratively writes Python that analyses the staged data
files in a network-isolated sandbox, then emits a final `{answer, reasoning}` JSON, which
is graded DETERMINISTICALLY by the benchmark's own `reference_grader.py` (authoritative
`passed`). This establishes the FLOOR (base model + code interpreter) before we wire in
Propab's tools / skills / verification. The ground_truth + grader stay runner-side and are
never shown to the agent (only `task` + `data_files/`).

Usage (repo root, .env with GOOGLE_API_KEY, worker image built, Docker running):
  python integrations/genebench/run_baseline.py                # all 10 problems
  python integrations/genebench/run_baseline.py --only wf_selection --max-steps 8
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (_REPO_ROOT, _REPO_ROOT / "packages" / "propab-core"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from integrations.genebench.sandbox_data import run_code_with_data  # noqa: E402

_DATA = _REPO_ROOT / "integrations" / "genebench" / "data" / "public"
_GRADER = _DATA / "reference_grader.py"
_IMAGE = os.environ.get("SANDBOX_IMAGE", "propab-oss-worker:latest")


# ── LLM (reuse Propab's provider client, no DB/emitter) ───────────────────────
def _make_llm():
    from propab.config import settings
    from propab.llm import LLMClient

    return LLMClient(
        provider=str(settings.llm_provider),
        model=str(settings.llm_model),
        api_key=str(settings.llm_api_secret),
        emitter=None,          # unused: we call _call_provider_once directly
        session_factory=None,  # unused
    )


async def _ask(llm, prompt: str) -> str:
    # _call_provider runs the raw provider call WITH bounded retry on transient errors,
    # without touching the DB/emitter (those live in the public .call()).
    return await llm._call_provider(prompt)


def _bounded(history: str, max_chars: int = 14000) -> str:
    """Keep prompts under the model's context limit on heavy, many-step problems
    (unbounded history caused an LLM error -> null answer on carrier_cnv)."""
    if len(history) <= max_chars:
        return history
    return "\n...(earlier steps truncated)...\n" + history[-max_chars:]


def _extract_json(text: str) -> dict | None:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{[\s\S]*\}", text)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            return None
    return None


def _expected_keys(cfg: dict) -> set[str]:
    """The answer field names the grader will check (so we can recognise a printed answer)."""
    grader = cfg.get("grader") or {}
    g = grader.get("config") or {}
    gtype = grader.get("type")
    if gtype == "numeric_tolerance":
        return {g.get("key", "h2")}
    if gtype == "multi_numeric_tolerance":
        return set((g.get("keys") or {}).keys())
    if gtype == "composite":
        keys: set[str] = set()
        for kk in ("numeric_keys", "integer_keys", "exact_match_keys"):
            keys |= set((g.get(kk) or {}).keys())
        return keys
    return set()


def _candidate_from_stdout(parsed, expected: set[str]) -> tuple[dict | None, str]:
    """If code stdout's last JSON line carries all expected answer keys, accept it."""
    if not isinstance(parsed, dict):
        return None, ""
    ans = parsed.get("answer") if isinstance(parsed.get("answer"), dict) else parsed
    reasoning = str(parsed.get("reasoning", "")) if isinstance(parsed, dict) else ""
    if isinstance(ans, dict) and expected and expected.issubset(set(ans.keys())):
        return ans, reasoning
    return None, ""


# General computational-genomics rigor (methodology scaffold — NOT per-problem answers).
# Encodes the standard corrections a domain expert applies for each analysis TYPE, which the
# bare base model reaches for but executes imprecisely. Applies to any genomics analysis.
_RIGOR = """Work as a rigorous computational geneticist. Before coding, STATE THE ESTIMAND
precisely (what quantity, in which population/subset, on what scale). Then use the estimator
the field standardises on for that estimand — not a convenient shortcut:
- Single-cell / expression / eQTL: subtract AMBIENT background (empty-droplet profile) from
  counts, RESTRICT to the exact cell state/subpopulation named, model counts with a
  library-size OFFSET (Poisson / negative-binomial GLM), and report the effect on the stated
  (usually natural-log) rate scale — a per-allele log rate ratio is a GLM coefficient, not a
  ratio of raw means.
- Mapping in related/structured samples (QTL, multi-founder): reconstruct latent
  haplotypes/ancestry (HMM) FIRST, then test association with a mixed model (LMM) that
  controls relatedness/structure; report the estimate at the likelihood peak.
- Instrumental-variable / Mendelian-randomisation and selected effects: correct WINNER'S CURSE
  on discovery estimates and be LD-aware when combining instruments.
- Demographic / admixture inference: fit the explicit generative model; sex-biased processes
  need autosome-vs-X (or parent-specific) contrasts, not a single pooled estimate.
Handle messy data honestly: drop or mask zero-coverage / QC-fail rows, check units and allele
polarity (ancestral vs derived), and never silently ignore a provided covariate or file.
Before finalising, SANITY-CHECK the estimate: is its sign and magnitude plausible? If a naive
and a rigorous estimate disagree, trust the rigorous one and say why. Do not take shortcuts.
"""

# ── Prompt ────────────────────────────────────────────────────────────────────
_SYS = """You are a computational-biology research agent solving a GeneBench-Pro problem.
You work in a Python sandbox (NO network). The problem's data files are staged at
/work/data_files/. Available libraries: numpy, scipy, pandas, statsmodels, scikit-learn.

{rigor}
Work iteratively: first INSPECT the data (shapes, columns, head, dtypes), then run the
appropriate rigorous analysis, checking your assumptions. When (and only when) your estimate
is trustworthy, give the final answer.

TASK:
{task}

Data files available at /work/data_files/:
{files}

You have a BUDGET of {max_steps} analysis steps. Pace yourself: inspect the data in AT MOST 1-2 steps
(it is shown above once seen — do NOT re-print it), spend the middle steps on the real computation,
and RESERVE the last 1-2 steps to finalize. Do not run out of budget before answering.

Respond with a SINGLE JSON object, one of:
  {{"action": "code", "code": "<complete Python program; print results to stdout>"}}
  {{"action": "final", "answer": {{<the exact answer fields the task asked for>}}, "reasoning": "<method + QC>"}}

Rules:
- In a "code" action, write a COMPLETE self-contained program (re-import + re-load data each
  time; the sandbox is fresh every run). Read files with absolute paths under /work/data_files/.
- Print what you need to see (shapes, estimates) to stdout so you can reason about it next turn.
- When your analysis code has computed the result, PRINT the final JSON as its last stdout line:
  {{"answer": {{<exact fields>}}, "reasoning": "<method + QC>"}} — that is accepted as your answer.
- Or emit an "action": "final" object directly once your numeric answer is trustworthy.
- Match the answer field names EXACTLY: {expected}
- Keep each program's runtime modest (well under the sandbox timeout); prefer efficient methods.
- Output ONLY the JSON object, no markdown, no prose around it.
"""

_HISTORY_STEP = """
--- Your step {i} ({action}) ---
{body}
--- Sandbox output (exit={exit_code}{err}) ---
{stdout}
"""

_FINALIZE = """You have used your analysis budget. You MUST produce an answer now. Choose ONE:
- If your best numeric estimate is already computed above, output it directly:
  {"action": "final", "answer": {<exact fields>}, "reasoning": "<method + QC>"}
- Otherwise write ONE complete Python program that computes the answer from the data and
  PRINTS as its final stdout line exactly: {"answer": {<exact fields>}, "reasoning": "..."}
  as {"action": "code", "code": "..."}.
Do NOT ask for more inspection. Use your best estimate even if imperfect. Output ONLY the JSON.
"""

_LAST_RESORT = """FINAL CHANCE. Output your best numeric estimate NOW as a single JSON object.
Code is NOT allowed. Use the analysis above; if a value is uncertain, give your best guess
rather than nothing:
{"action": "final", "answer": {<exact fields>}, "reasoning": "<brief basis>"}
Output ONLY the JSON object.
"""


def _clip(s: str, n: int = 2500) -> str:
    s = s or ""
    return s if len(s) <= n else s[: n // 2] + "\n...(clipped)...\n" + s[-n // 2 :]


# ── One problem ───────────────────────────────────────────────────────────────
async def solve_problem(llm, prob_dir: Path, max_steps: int, code_timeout: int) -> dict:
    cfg = json.loads((prob_dir / "eval_config.json").read_text(encoding="utf-8"))
    task = cfg["task"]
    data_dir = prob_dir / "data_files"
    expected = _expected_keys(cfg)
    files = "\n".join(f"  - /work/data_files/{f}" for f in
                       [str(p.relative_to(data_dir)).replace(chr(92), "/")
                        for p in sorted(data_dir.rglob("*")) if p.is_file()])
    base = _SYS.format(task=task, files=files, max_steps=max_steps, rigor=_RIGOR,
                       expected=", ".join(sorted(expected)) or "(see task)")

    history = ""
    final_answer = None
    reasoning = ""
    printed_answer = None       # a valid answer PRINTED by code stdout (fallback)
    printed_reasoning = ""
    steps_used = 0
    code_runs = 0
    force_raw = ""
    t0 = time.time()

    async def _run_code_action(action: dict, i: int) -> None:
        nonlocal history, code_runs, printed_answer, printed_reasoning
        code = action.get("code") or ""
        if not code.strip():
            history += f"\n--- step {i}: empty code ---\n"
            return
        code_runs += 1
        res = run_code_with_data(code, data_dir, image=_IMAGE,
                                 timeout_sec=code_timeout, memory_mb=4096)
        cand, creason = _candidate_from_stdout(res.get("parsed"), expected)
        if cand is not None:
            printed_answer, printed_reasoning = cand, creason
        err = f", {res['error_type']}" if not res.get("ok") else ""
        history += _HISTORY_STEP.format(
            i=i, action="code", body=_clip(code, 1500),
            exit_code=res.get("exit_code", "-"), err=err, stdout=_clip(res.get("stdout", "")),
        )

    for i in range(1, max_steps + 1):
        steps_used = i
        left = max_steps - i
        finalize_hint = ("\n\nBUDGET NEARLY SPENT — finalize this step: run your final computation "
                         "and PRINT the answer JSON, or emit action=final.") if left <= 1 else ""
        prompt = base + _bounded(history) + finalize_hint + f"\n\n[step {i} of {max_steps}, {left} left] Your next action (JSON only):"
        try:
            raw = await _ask(llm, prompt)
        except Exception as exc:  # noqa: BLE001
            history += f"\n--- LLM error step {i}: {exc} ---\n"
            continue
        action = _extract_json(raw)
        if not action:
            history += _HISTORY_STEP.format(i=i, action="unparseable", body=_clip(raw, 800),
                                            exit_code="-", err="", stdout="(no valid JSON returned)")
            continue
        if action.get("action") == "final":
            final_answer = action.get("answer")
            reasoning = str(action.get("reasoning", ""))
            break
        await _run_code_action(action, i)

    # Finalize: up to 3 robust rounds. Accept a direct final answer, or RUN the model's
    # final program and capture the answer it PRINTS. History is bounded so a long run
    # can't blow the context window (which previously errored -> null).
    if final_answer is None:
        for j in range(3):
            try:
                force_raw = await _ask(llm, base + _bounded(history) + "\n\n" + _FINALIZE)
            except Exception as exc:  # noqa: BLE001
                history += f"\n--- finalize LLM error round {j}: {exc} ---\n"
                continue
            action = _extract_json(force_raw) or {}
            if action.get("action") == "final" or ("answer" in action and not action.get("code")):
                final_answer = action.get("answer")
                reasoning = str(action.get("reasoning", ""))
                break
            if action.get("code"):
                await _run_code_action(action, max_steps + 1 + j)  # its final computation
                if printed_answer is not None:
                    break

    # Fall back to an answer the code printed if the model never emitted a clean final.
    if final_answer is None and printed_answer is not None:
        final_answer, reasoning = printed_answer, printed_reasoning

    # Guarantee a non-null answer: force a direct best-guess (no code allowed) as the last
    # resort, so a hard problem yields a real attempt instead of null harness noise.
    if final_answer is None:
        try:
            raw = await _ask(llm, base + _bounded(history) + "\n\n" + _LAST_RESORT)
            act = _extract_json(raw) or {}
            final_answer = act.get("answer")
            reasoning = str(act.get("reasoning", ""))
        except Exception:  # noqa: BLE001
            pass

    passed, grade = _grade(prob_dir, final_answer, reasoning)
    if os.environ.get("GENEBENCH_DEBUG"):
        dbg = _DATA.parent / f"_debug_{cfg['id']}.txt"
        dbg.write_text(base + history + "\n\n=== FORCE-FINAL RAW ===\n" + force_raw, encoding="utf-8")
    return {
        "id": cfg["id"],
        "passed": passed,
        "answer": final_answer,
        "ground_truth": cfg.get("ground_truth"),
        "grade": grade,
        "steps": steps_used,
        "code_runs": code_runs,
        "elapsed_sec": round(time.time() - t0, 1),
    }


def _grade(prob_dir: Path, answer, reasoning: str) -> tuple[bool, dict]:
    if answer is None:
        return False, {"error": "no answer produced"}
    ans_path = prob_dir / "_eval_answer.json"
    ans_path.write_text(json.dumps({"answer": answer, "reasoning": reasoning}), encoding="utf-8")
    try:
        out = subprocess.run(
            [sys.executable, str(_GRADER), str(prob_dir / "eval_config.json"), str(ans_path)],
            capture_output=True, text=True, timeout=60,
        )
        if out.returncode != 0:
            return False, {"error": "grader_error", "stderr": out.stderr[-500:]}
        graded = json.loads(out.stdout)
        return bool(graded.get("passed")), graded
    except Exception as exc:  # noqa: BLE001
        return False, {"error": f"grade_exception: {exc}"}
    finally:
        try:
            ans_path.unlink()
        except Exception:
            pass


# ── Driver ────────────────────────────────────────────────────────────────────
async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, help="run a single problem id")
    ap.add_argument("--max-steps", type=int, default=8)
    ap.add_argument("--code-timeout", type=int, default=180)
    ap.add_argument("--out", default=None, help="write full results JSON here")
    args = ap.parse_args()

    prob_dirs = sorted((_DATA / "problems").iterdir())
    prob_dirs = [d for d in prob_dirs if d.is_dir() and (d / "eval_config.json").exists()]
    if args.only:
        prob_dirs = [d for d in prob_dirs if d.name == args.only]
        if not prob_dirs:
            print(f"no such problem: {args.only}")
            return 1

    llm = _make_llm()
    results = []
    for d in prob_dirs:
        print(f"\n=== {d.name} ===", flush=True)
        r = await solve_problem(llm, d, args.max_steps, args.code_timeout)
        results.append(r)
        mark = "PASS" if r["passed"] else "FAIL"
        print(f"[{mark}] {r['id']}  steps={r['steps']} code_runs={r['code_runs']} "
              f"{r['elapsed_sec']}s  answer={json.dumps(r['answer'])[:160]}", flush=True)

    n = len(results)
    npass = sum(1 for r in results if r["passed"])
    print("\n" + "=" * 60)
    print(f"GeneBench-Pro (public 10) baseline: {npass}/{n} passed = {npass / n:.1%}")
    for r in results:
        print(f"  {'PASS' if r['passed'] else 'FAIL'}  {r['id']}")
    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
        print(f"\nfull results -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
