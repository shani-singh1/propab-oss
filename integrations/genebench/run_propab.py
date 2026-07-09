"""GeneBench-Pro through the REAL Propab scaffold.

Same deterministic harness as run_baseline.py, but the agent is given Propab's ACTUAL
scaffold rather than a bare code loop:
  - REAL SKILLS: the curated analysis-phase methodology (skills_prompt_block, experiment
    + evidence phase, genomics) injected into context — not hand-written prose.
  - REAL TOOLS: the worker tool registry (registry.get_for("worker")); the agent may call
    any trusted tool via registry.call in addition to writing sandbox code.
This is the faithful A/B against the 10% bare-code-agent baseline: does Propab's real
tools+skills scaffold lift the same base model (gemini-3.5-flash) on the same problems?

Usage:
  python integrations/genebench/run_propab.py [--only <id>] [--max-steps 12] [--out FILE]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (_REPO_ROOT, _REPO_ROOT / "packages" / "propab-core"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# Reuse the proven harness pieces from the baseline runner.
from integrations.genebench.run_baseline import (  # noqa: E402
    _DATA, _IMAGE, _ask, _bounded, _candidate_from_stdout, _clip, _expected_keys,
    _extract_json, _grade, _make_llm, _HISTORY_STEP,
)
from integrations.genebench.sandbox_data import run_code_with_data  # noqa: E402

from propab.skills import skills_prompt_block  # noqa: E402
from propab.tools.registry import ToolRegistry  # noqa: E402

_REGISTRY = ToolRegistry()


def _skills_block() -> str:
    """Propab's REAL analysis methodology (experiment + evidence phase, genomics)."""
    parts = []
    for phase in ("experiment", "evidence"):
        b = skills_prompt_block(domain_id="genomics", phase=phase)
        if b:
            parts.append(b)
    block = "\n\n".join(parts)
    # Bound so it never dominates the context window.
    return block if len(block) <= 16000 else block[:16000] + "\n...(skills truncated)..."


def _tool_catalog(specs: list[dict]) -> str:
    lines = []
    for s in specs:
        flags = ""
        if s.get("significance_capable"):
            flags += " [SIGNIFICANCE]"
        if s.get("verification_capable"):
            flags += " [VERIFY]"
        lines.append(f"  {s['name']}{flags}: {s.get('description', '')[:180]}")
    return "\n".join(lines)


_SYS = """You are a computational-biology research agent solving a GeneBench-Pro problem, using
the Propab research scaffold. You work in a Python sandbox (NO network). The problem's data
files are staged at /work/data_files/. Sandbox libraries: numpy, scipy, pandas, statsmodels,
scikit-learn.

{skills}

TASK:
{task}

Data files available at /work/data_files/:
{files}

You may also call any of these trusted Propab TOOLS (they run outside the sandbox on values you
provide as params — use them for rigorous statistics on numbers your code has already extracted):
{tools}

Budget: {max_steps} steps. Inspect the data in 1-2 steps, run the rigorous analysis, RESERVE the
last 1-2 steps to finalize. Respond with a SINGLE JSON object, one of:
  {{"action": "code", "code": "<complete Python program; print results to stdout>"}}
  {{"action": "tool", "tool_name": "<name>", "params": {{<params>}}}}
  {{"action": "final", "answer": {{<the exact answer fields>}}, "reasoning": "<method + QC>"}}

Rules:
- Sandbox code is fresh each run (re-import + re-load). Read files under /work/data_files/.
- When your analysis code has the result, PRINT the final line {{"answer": {{<exact fields>}}, "reasoning": "..."}}.
- Match the answer field names EXACTLY: {expected}
- Output ONLY the JSON object, no markdown.
"""

_FINALIZE = """You have used your analysis budget. Produce an answer NOW. Either output
{"action": "final", "answer": {<exact fields>}, "reasoning": "..."} directly, or ONE program
that PRINTS {"answer": {<exact fields>}, "reasoning": "..."} as {"action": "code", "code": "..."}.
Use your best estimate even if imperfect. Output ONLY the JSON.
"""

_LAST_RESORT = """FINAL CHANCE. Output your best numeric estimate NOW as a single JSON object.
Code is NOT allowed. Give your best guess rather than nothing:
{"action": "final", "answer": {<exact fields>}, "reasoning": "<brief basis>"}
Output ONLY the JSON.
"""


def _run_tool(tool_name: str, params: dict) -> dict:
    try:
        res = _REGISTRY.call(tool_name, params if isinstance(params, dict) else {})
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": f"tool_exception: {exc}"}
    if getattr(res, "success", False):
        return {"ok": True, "output": getattr(res, "output", None)}
    err = getattr(res, "error", None)
    return {"ok": False, "error": str(getattr(err, "message", err))}


async def solve_problem(llm, prob_dir: Path, max_steps: int, code_timeout: int, specs: list[dict]) -> dict:
    cfg = json.loads((prob_dir / "eval_config.json").read_text(encoding="utf-8"))
    task = cfg["task"]
    data_dir = prob_dir / "data_files"
    expected = _expected_keys(cfg)
    files = "\n".join(f"  - /work/data_files/{p.relative_to(data_dir)}".replace("\\", "/")
                      for p in sorted(data_dir.rglob("*")) if p.is_file())
    base = _SYS.format(skills=_skills_block(), task=task, files=files, tools=_tool_catalog(specs),
                       max_steps=max_steps, expected=", ".join(sorted(expected)) or "(see task)")

    history = ""
    final_answer, reasoning = None, ""
    printed_answer, printed_reasoning = None, ""
    steps_used = code_runs = tool_runs = 0
    t0 = time.time()

    async def _do_code(action: dict, i: int) -> None:
        nonlocal history, code_runs, printed_answer, printed_reasoning
        code = action.get("code") or ""
        if not code.strip():
            return
        code_runs += 1
        res = run_code_with_data(code, data_dir, image=_IMAGE, timeout_sec=code_timeout, memory_mb=4096)
        cand, creason = _candidate_from_stdout(res.get("parsed"), expected)
        if cand is not None:
            printed_answer, printed_reasoning = cand, creason
        err = f", {res['error_type']}" if not res.get("ok") else ""
        history += _HISTORY_STEP.format(i=i, action="code", body=_clip(code, 1400),
                                        exit_code=res.get("exit_code", "-"), err=err,
                                        stdout=_clip(res.get("stdout", "")))

    def _do_tool(action: dict, i: int) -> None:
        nonlocal history, tool_runs
        tool_runs += 1
        out = _run_tool(str(action.get("tool_name", "")), action.get("params") or {})
        body = f"tool={action.get('tool_name')} params={json.dumps(action.get('params'))[:400]}"
        history += _HISTORY_STEP.format(i=i, action="tool", body=body,
                                        exit_code="ok" if out.get("ok") else "err", err="",
                                        stdout=_clip(json.dumps(out.get("output", out.get("error")), default=str)))

    for i in range(1, max_steps + 1):
        steps_used = i
        left = max_steps - i
        hint = ("\n\nBUDGET NEARLY SPENT — finalize this step." if left <= 1 else "")
        try:
            raw = await _ask(llm, base + _bounded(history) + hint + f"\n\n[step {i}/{max_steps}, {left} left] Next action (JSON only):")
        except Exception as exc:  # noqa: BLE001
            history += f"\n--- LLM error step {i}: {exc} ---\n"
            continue
        action = _extract_json(raw)
        if not action:
            history += f"\n--- step {i}: unparseable: {_clip(raw, 400)} ---\n"
            continue
        kind = action.get("action")
        if kind == "final":
            final_answer, reasoning = action.get("answer"), str(action.get("reasoning", ""))
            break
        if kind == "tool":
            _do_tool(action, i)
        else:
            await _do_code(action, i)

    if final_answer is None:
        for j in range(3):
            try:
                raw = await _ask(llm, base + _bounded(history) + "\n\n" + _FINALIZE)
            except Exception as exc:  # noqa: BLE001
                history += f"\n--- finalize err {j}: {exc} ---\n"
                continue
            action = _extract_json(raw) or {}
            if action.get("action") == "final" or ("answer" in action and not action.get("code")):
                final_answer, reasoning = action.get("answer"), str(action.get("reasoning", ""))
                break
            if action.get("code"):
                await _do_code(action, max_steps + 1 + j)
                if printed_answer is not None:
                    break

    if final_answer is None and printed_answer is not None:
        final_answer, reasoning = printed_answer, printed_reasoning
    if final_answer is None:
        try:
            raw = await _ask(llm, base + _bounded(history) + "\n\n" + _LAST_RESORT)
            action = _extract_json(raw) or {}
            final_answer, reasoning = action.get("answer"), str(action.get("reasoning", ""))
        except Exception:  # noqa: BLE001
            pass

    passed, grade = _grade(prob_dir, final_answer, reasoning)
    if os.environ.get("GENEBENCH_DEBUG"):
        (_DATA.parent / f"_debugP_{cfg['id']}.txt").write_text(base + history, encoding="utf-8")
    return {"id": cfg["id"], "passed": passed, "answer": final_answer, "ground_truth": cfg.get("ground_truth"),
            "grade": grade, "steps": steps_used, "code_runs": code_runs, "tool_runs": tool_runs,
            "elapsed_sec": round(time.time() - t0, 1)}


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None)
    ap.add_argument("--max-steps", type=int, default=12)
    ap.add_argument("--code-timeout", type=int, default=240)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    prob_dirs = sorted(d for d in (_DATA / "problems").iterdir() if d.is_dir() and (d / "eval_config.json").exists())
    if args.only:
        prob_dirs = [d for d in prob_dirs if d.name == args.only] or prob_dirs
        prob_dirs = [d for d in prob_dirs if d.name == args.only]
        if not prob_dirs:
            print(f"no such problem: {args.only}")
            return 1

    llm = _make_llm()
    specs = _REGISTRY.get_for("worker")
    print(f"[propab scaffold] {len(specs)} tools, skills block {len(_skills_block())} chars", flush=True)
    results = []
    for d in prob_dirs:
        print(f"\n=== {d.name} ===", flush=True)
        r = await solve_problem(llm, d, args.max_steps, args.code_timeout, specs)
        results.append(r)
        print(f"[{'PASS' if r['passed'] else 'FAIL'}] {r['id']} steps={r['steps']} "
              f"code={r['code_runs']} tools={r['tool_runs']} {r['elapsed_sec']}s "
              f"answer={json.dumps(r['answer'])[:150]}", flush=True)

    n = len(results); npass = sum(r["passed"] for r in results)
    print("\n" + "=" * 60)
    print(f"GeneBench-Pro (public 10) PROPAB SCAFFOLD: {npass}/{n} = {npass / n:.1%}")
    for r in results:
        print(f"  {'PASS' if r['passed'] else 'FAIL'}  {r['id']}")
    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
