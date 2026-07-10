"""
Real LitQA2 evaluation run — downloads actual questions from the public
futurehouse/lab-bench dataset, retrieves evidence live through this
service's own pipeline, and uses Gemini to answer, scored with LAB-bench's
exact accuracy/coverage/precision metrics.

This is the honest comparison point against the real AstaBench leaderboard
(see CHANGELOG.md for the real baseline numbers pulled from
allenai/asta-bench-results): ReAct + Claude Opus 4.7 scores 0.84 accuracy /
0.94 precision / 0.89 coverage on the official LitQA2_FullText test split.

Usage:
    python services/literature/scripts/run_litqa2_real.py [n] [seed] [concurrency]
"""
from __future__ import annotations

import asyncio
import json
import random
import sys
import time
from pathlib import Path

# Windows consoles default to cp1252, which chokes on scientific-notation
# unicode (Greek letters, etc.) that shows up constantly in question text.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "packages" / "propab-core"))

from services.literature.app.config import settings
from services.literature.app.evaluator.astabench import record_score
from services.literature.app.evaluator.litqa2_live import (
    answer_one_question,
    load_litqa2_sample,
    score_results,
)
from services.literature.app.pipeline import LiteraturePipeline

BIOLOGY_LITERATURE_PROFILE = {
    "seed_papers": [],
    "search_terms": [],
    "source_priorities": ["pubmed", "europepmc", "biorxiv", "semantic_scholar", "crossref"],
    "classification_codes": {"mesh": []},
    "open_problem_sources": [],
    "tabulation_sources": [],
    "canonical_surveys": [],
    "novelty_criteria": "",
}


async def main(n: int = 20, seed: int = 0, concurrency: int = 4) -> None:
    print(f"Loading {n} real LitQA2 questions (seed={seed}) from futurehouse/lab-bench...")
    print(f"answer_model={settings.answer_model} (reformulation/judge on {settings.llm_model}), concurrency={concurrency}")
    questions = await load_litqa2_sample(n=n, seed=seed)
    print(f"Loaded {len(questions)} questions.")

    pipeline = await LiteraturePipeline.create(settings)
    results: list[dict] = []
    results_lock = asyncio.Lock()
    sem = asyncio.Semaphore(concurrency)
    total = len(questions)

    def _write_partial() -> None:
        # Best-effort progress dump: a transient write failure (OneDrive sync
        # locks the file with [Errno 22] under frequent writes) must NOT kill a
        # 30-minute run mid-flight — it's called after every question under the
        # results lock, and an unhandled OSError here propagates out of the
        # gather and aborts every in-flight question. Write to a temp file then
        # atomically replace, and swallow any OSError (the final write at the
        # end, plus the recorded score, are what actually matter).
        partial = {"subtask": "litqa2_live_partial", "n_cases": len(results), **score_results(results)}
        payload = json.dumps(
            {**partial, "per_case": [{k: v for k, v in r.items() if k != "choices"} for r in results]}, indent=2
        )
        dest = REPO_ROOT / "artifacts" / "litqa2_live_results.json"
        try:
            tmp = dest.with_suffix(".json.tmp")
            tmp.write_text(payload, encoding="utf-8")
            tmp.replace(dest)
        except OSError:
            pass

    async def _run_one(idx: int, q: dict) -> None:
        # Each question gets its own rng, seeded deterministically from the
        # question id, so concurrent execution doesn't make choice-shuffling
        # order-dependent (and the run stays reproducible).
        q_rng = random.Random(f"{seed}:{q['id']}")
        t0 = time.monotonic()
        try:
            async with sem:
                result = await asyncio.wait_for(
                    answer_one_question(
                        pipeline,
                        question=q["question"],
                        ideal=q["ideal"],
                        distractors=q["distractors"],
                        profile=BIOLOGY_LITERATURE_PROFILE,
                        domain_id="litqa2_eval",
                        rng=q_rng,
                        google_api_key=settings.google_api_key,
                        llm_model=settings.llm_model,
                        answer_model=settings.answer_model,
                        depth="standard",
                    ),
                    # 2/30 questions hit the 300s wall (live multi-round retrieval + a
                    # 180s answer call) and were recorded guaranteed-wrong with no answer
                    # — a pure harness loss. A more generous wall lets slow-but-finite
                    # retrieval finish and produce a real answer; it never penalises a
                    # question already under budget. (Iteration-2: a fast no-retrieval
                    # fallback on timeout + stronger retrieval recall is the bigger lever.)
                    timeout=480.0,
                )
        except Exception as exc:  # noqa: BLE001 — one bad question shouldn't kill the run
            result = {
                "question": q["question"], "is_correct": False, "is_sure": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
        result["id"] = q["id"]
        result["sources"] = q["sources"]
        result["key_passage"] = q.get("key_passage")  # gold answer-supporting text (diagnostic only)
        elapsed = time.monotonic() - t0
        if result.get("error"):
            status = "ERROR"
        elif result.get("is_correct"):
            status = "OK"
        elif not result.get("is_sure"):
            status = "ABSTAIN"
        else:
            status = "WRONG"
        n_ev = result.get("n_evidence", "-")
        async with results_lock:
            results.append(result)
            done = len(results)
            print(f"[{done}/{total}] {status:>7} ({elapsed:5.1f}s, n_evidence={n_ev}) {q['question'][:66]}")
            if result.get("error"):
                print(f"           -> {result['error'][:150]}")
            _write_partial()

    try:
        await asyncio.gather(*(_run_one(i, q) for i, q in enumerate(questions)))
    finally:
        await pipeline.aclose()

    scores = score_results(results)
    print("\n=== LitQA2 (real data, live retrieval + Gemini answering) ===")
    print(json.dumps(scores, indent=2))

    out = {
        "subtask": "litqa2_live",
        "dataset": "futurehouse/lab-bench:LitQA2 (real, public, non-gated)",
        "n_cases": scores["n"],
        "accuracy": scores["accuracy"],
        "coverage": scores["coverage"],
        "precision": scores["precision"],
        "per_case": [
            {k: v for k, v in r.items() if k not in ("choices",)} for r in results
        ],
    }
    record_score(out, artifacts_dir=str(REPO_ROOT / "artifacts"))

    # Same OneDrive-lock tolerance as _write_partial: the score is already
    # persisted by record_score above, so a failed final dump must not crash the
    # run with a traceback that obscures the printed result.
    results_path = REPO_ROOT / "artifacts" / "litqa2_live_results.json"
    try:
        tmp = results_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(out, indent=2), encoding="utf-8")
        tmp.replace(results_path)
        print(f"\nWrote {results_path}")
    except OSError as exc:
        print(f"\n(final results dump skipped: {exc}; score already recorded)")


if __name__ == "__main__":
    n_arg = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    seed_arg = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    conc_arg = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    asyncio.run(main(n_arg, seed_arg, conc_arg))
