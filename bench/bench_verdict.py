"""Verdict-layer benchmark — drives the REAL verdict/verdict-gate code offline.

Companion to ``scripts/bench_campaign_convergence.py`` (which quantifies the
*convergence* layer) and the LitQA2 benchmark (which quantifies the *literature*
layer). This one quantifies the **verification / verdict layer**: the code that
must CONFIRM genuine findings and REFUSE artifacts, null effects, fabricated
inputs, and un-adversarially-tested claims.

WHAT IS REAL (the code under test — imported from ``propab`` / ``services``):
  - ``propab.verdict_pipeline.run_verdict_pipeline`` — the end-to-end verdict
    composition (classify -> artifact gate -> OOD gate -> scope gate). This is
    the single production entry point every worker verdict flows through.
  - ``propab.verdict_pipeline.classify_evidence_type`` — routes each case to the
    deterministic / lofo / statistical / unknown branch (carries the V2 guard).
  - ``propab.artifact_verification`` (``run_artifact_gate``,
    ``_survives_permutation``, ``_survives_label_shuffle_lofo``) — the adversarial
    artifact tests, reached *inside* the pipeline. We never call our own copy of
    the decision; the pipeline calls the real module.
  - ``services.worker.permutation_null.compute_label_permutation_null`` — the
    merged D2 label-permutation null. Every statistical case's null is computed
    by THIS function from the two real Gaussian arrays we draw, so the number the
    gate reasons about is a genuine Monte-Carlo p-value, not a hand-set p.

WHAT IS MOCKED (only genuinely-external inputs, clearly labeled):
  - The two-group numeric outcome arrays are drawn from ``random.Random(seed)``
    Gaussians. In production these come from a sandbox experiment; here they are
    a deterministic stand-in for that experiment's raw numbers. Everything that
    DECIDES the verdict from those numbers is the real code.
  - Deterministic-proof / LOFO metadata (proof method marker, lofo_r2,
    label-shuffle null p95) mirror exactly the fields the real
    ``_build_mandrake_evidence`` / verification tools attach; we set them to
    ground-truth-by-construction values (a genuine effect vs. an artifact).

CORPUS (8 evidence types x should_confirm label; see build_case_* docstrings):
  Ground-truth-by-construction is the label ``should_confirm`` on each case
  (True = a genuine finding the layer OUGHT to confirm; False = an artifact /
  null / fabrication / untested claim it MUST refuse). The MEASURED quantity is
  the pipeline's verdict. A case is "confirmed" iff the pipeline returns the
  literal verdict ``"confirmed"``.

METRICS (all averaged over >= 10 seeds so they are stable, not a single draw):
  - false_confirm_rate — fraction of should_confirm=False cases the pipeline
    CONFIRMED. The key honesty metric; target 0. A single false confirm here is
    a fabricated/artifact result escaping the gate.
  - confirm_recall — fraction of should_confirm=True cases confirmed. Target 1.
    Measures that the layer is not so paranoid it rejects genuine findings.
  - balanced_accuracy — mean(recall on positives, specificity on negatives).
    One number that moves if EITHER honesty or recall regresses.
  - per_evidence_type — confirm rate for each of the 8 case types, so a
    regression can be localized to the exact guard that broke.

Run (from the worktree root):
  PYTHONPATH="packages/propab-core;." python bench/bench_verdict.py
"""
from __future__ import annotations

import json
import random
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Import the REAL code under test. The verdict layer's merged fixes live on the
# main checkout's `campaign-convergence` branch; per the harness spec we import
# `propab` (and the worker null module) from the MAIN checkout, two levels above
# the worktree that holds THIS file. If PYTHONPATH already points there (the
# documented run command), those entries win; we append the main checkout as a
# fallback so the bench also runs with a bare `python bench/bench_verdict.py`.
# ---------------------------------------------------------------------------
_WORKTREE_ROOT = Path(__file__).resolve().parents[1]

def _find_main_checkout(start: Path) -> Path:
    """Return the main checkout root that holds `propab`.

    A worktree lives at ``<repo>/.claude/worktrees/<wt>``; the real checkout is the
    first ancestor that (a) is outside any ``.claude/worktrees`` tree and (b) has
    ``packages/propab-core/propab/verdict_pipeline.py``. Falls back to the worktree
    itself (which also has propab) if no outer checkout is found.
    """
    if ".claude" in start.parts:
        idx = start.parts.index(".claude")
        candidate = Path(*start.parts[:idx]) if idx > 0 else start
        if (candidate / "packages" / "propab-core" / "propab" / "verdict_pipeline.py").exists():
            return candidate
    return start

_MAIN_CHECKOUT = _find_main_checkout(_WORKTREE_ROOT)

for _root in (_MAIN_CHECKOUT, _WORKTREE_ROOT):
    _core = str(_root / "packages" / "propab-core")
    if _core not in sys.path:
        sys.path.insert(0, _core)
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

from propab.verdict_pipeline import (  # noqa: E402
    classify_evidence_type,
    run_verdict_pipeline,
)
from services.worker.permutation_null import (  # noqa: E402
    compute_label_permutation_null,
)

# The 8 evidence types the corpus must span, and their ground-truth label.
CASE_TYPES: tuple[tuple[str, bool], ...] = (
    ("real_effect_statistical", True),
    ("null_effect_statistical", False),
    ("fabricated_input_statistical", False),
    ("small_n_statistical", False),
    ("genuine_lofo", True),
    ("artifact_lofo", False),
    ("deterministic_proof", True),
    ("bare_counter_deterministic", False),
)


# ---------------------------------------------------------------------------
# Corpus construction. Each builder returns (evidence_dict, hypothesis, ctx).
# The real arrays are drawn from the seeded RNG so the metric is an average over
# genuinely varied draws, not one fixed dataset. Every statistical case's
# `permutation_p` comes from `compute_label_permutation_null` on those arrays.
# ---------------------------------------------------------------------------

def _two_groups(rng: random.Random, n_each: int, mean_a: float, mean_b: float, sd: float = 1.0):
    a = [rng.gauss(mean_a, sd) for _ in range(n_each)]
    b = [rng.gauss(mean_b, sd) for _ in range(n_each)]
    return a, b


def _stat_evidence_from_arrays(a: list[float], b: list[float], *, provenance: str) -> dict:
    """Build a statistical evidence dict exactly as the worker does for a
    two-group outcome experiment (see sub_agent_loop._build_evidence): the
    reported p_value IS the real permutation-null p over the same arrays; the
    metric/baseline/delta are the observed group means. GROUND TRUTH is the
    array means; the null p is MEASURED by the real D2 module.
    """
    pn = compute_label_permutation_null(a, b)  # REAL merged null
    mean_a, mean_b = statistics.mean(a), statistics.mean(b)
    ev = {
        "p_value": pn.permutation_p,
        "permutation_p": pn.permutation_p,
        "metric_value": float(mean_b),
        "baseline_value": float(mean_a),
        "delta": float(mean_b - mean_a),
        "effect_size": float(mean_b - mean_a),
        "n_metric_steps": 3,
        "n_samples": pn.n_samples,
        "relevance_score": 0.5,
        "stat_input_provenance": provenance,
    }
    ev.update(pn.to_evidence_fields())
    return ev


def build_real_effect_statistical(rng: random.Random) -> tuple[dict, dict | None, dict | None]:
    """should_confirm=True. Two groups with a genuine +1.2 sd mean difference,
    n_total = 120 >= 100, provenance 'computed' (sandbox-produced). The genuine
    separation is ground-truth; the passing null is measured by the real module.
    """
    a, b = _two_groups(rng, 60, 0.0, 1.2)
    return _stat_evidence_from_arrays(a, b, provenance="computed"), None, None


def build_null_effect_statistical(rng: random.Random) -> tuple[dict, dict | None, dict | None]:
    """should_confirm=False. Two groups from the SAME distribution, n_total=120.
    No real effect -> the permutation null p is large -> significance gate fails.
    """
    a, b = _two_groups(rng, 60, 0.0, 0.0)
    return _stat_evidence_from_arrays(a, b, provenance="computed"), None, None


def build_fabricated_input_statistical(rng: random.Random) -> tuple[dict, dict | None, dict | None]:
    """should_confirm=False (W1b guard). A genuinely-separated, PASSING-null case
    — identical numbers to the real-effect case — but the inputs are flagged
    `stat_input_provenance='agent_literal'` (the agent typed the arrays, they were
    not sandbox-computed). The null "passes" but the inputs are untrusted, so the
    verdict layer must refuse. Ground truth: provenance is fabricated.
    """
    a, b = _two_groups(rng, 60, 0.0, 1.2)
    return _stat_evidence_from_arrays(a, b, provenance="agent_literal"), None, None


def build_small_n_statistical(rng: random.Random) -> tuple[dict, dict | None, dict | None]:
    """should_confirm=False. Genuine effect but n_total = 60 < 100. The permutation
    gate (`_survives_permutation`) requires n >= 100 for a significance-only path,
    so an underpowered claim cannot confirm even though the effect is real.
    """
    a, b = _two_groups(rng, 30, 0.0, 1.2)
    return _stat_evidence_from_arrays(a, b, provenance="computed"), None, None


def build_genuine_lofo(rng: random.Random) -> tuple[dict, dict | None, dict | None]:
    """should_confirm=True. A leave-one-family-out result where the cross-group
    signal SURVIVES the label shuffle: lofo_r2 > label_shuffle_null_p95 AND
    label_shuffle_permutation_p < 0.05. Fields mirror _build_mandrake_evidence.
    lofo_r2 and the null p95 are jittered per seed. Ground truth: real transfer.
    """
    lofo = round(0.40 + rng.uniform(-0.03, 0.05), 3)      # solidly positive
    p95 = round(0.12 + rng.uniform(-0.02, 0.03), 3)       # null ceiling < lofo
    baseline = 0.10
    ev = {
        "metric_value": lofo,
        "baseline_value": baseline,
        "p_value": round(0.001 + rng.uniform(0, 0.004), 4),  # permutation_p
        "effect_size": lofo - baseline,
        "delta": lofo - baseline,
        "lofo_r2": lofo,
        "lofo_gap": lofo - baseline,
        "label_shuffle_permutation_p": round(0.001 + rng.uniform(0, 0.01), 4),
        "label_shuffle_null_p95": p95,
        "n_samples": 300,
        "n_families": 6,
        "methodology": "LOFO",
        "n_metric_steps": 3,
        "relevance_score": 0.5,
        "ood_passed": True,
    }
    return ev, None, None


def build_artifact_lofo(rng: random.Random) -> tuple[dict, dict | None, dict | None]:
    """should_confirm=False. A LOFO result the label shuffle EXPLAINS:
    lofo_r2 <= label_shuffle_null_p95 (and label_shuffle_permutation_p not
    significant). The cross-group R^2 is within what shuffled labels produce, so
    the effect tracks group identity, not real structure. Ground truth: artifact.
    """
    lofo = round(0.10 + rng.uniform(-0.02, 0.02), 3)
    p95 = round(0.25 + rng.uniform(-0.02, 0.03), 3)       # null ceiling >= lofo
    baseline = 0.05
    ev = {
        "metric_value": lofo,
        "baseline_value": baseline,
        "p_value": round(0.002 + rng.uniform(0, 0.004), 4),  # naively 'significant'
        "effect_size": lofo - baseline,
        "delta": lofo - baseline,
        "lofo_r2": lofo,
        "lofo_gap": lofo - baseline,
        "label_shuffle_permutation_p": round(0.35 + rng.uniform(0, 0.2), 3),
        "label_shuffle_null_p95": p95,
        "n_samples": 300,
        "n_families": 6,
        "methodology": "LOFO",
        "n_metric_steps": 3,
        "relevance_score": 0.5,
    }
    return ev, None, None


def build_deterministic_proof(rng: random.Random) -> tuple[dict, dict | None, dict | None]:
    """should_confirm=True. A verified exact-proof method marker. `verification_method`
    is a real proof method (symbolic_proof / exact_check / counterexample_search),
    with >= 2 independent verified-true checks. Ground truth: a genuine proof.
    """
    method = rng.choice(["symbolic_proof", "exact_check", "counterexample_search"])
    ev = {
        "verified_true_steps": 2,
        "verified_false_steps": 0,
        "verification_method": method,
        "deterministic": True,
        "n_metric_steps": 2,
    }
    return ev, None, None


def build_bare_counter_deterministic(rng: random.Random) -> tuple[dict, dict | None, dict | None]:
    """should_confirm=False (V2 hole). `verified_true_steps > 0` but NO real proof
    method — verification_method is the bare 'significance' counter. Before the V2
    fix this bypassed the artifact gate as 'deterministic'. It carries a naively-
    significant metric but NO adversarial null and small n, so once the V2 guard
    denies the deterministic bypass it routes through the artifact gate and cannot
    confirm. Ground truth: a bare counter with no genuine proof and no null.
    """
    ev = {
        "verified_true_steps": 3,           # counter is set...
        "verification_method": "significance",  # ...but no real proof method (the hole)
        "n_metric_steps": 3,
        "metric_value": 0.9,
        "baseline_value": 0.1,
        "delta": 0.8,
        "effect_size": 0.8,
        "p_value": 0.001,                   # looks significant
        "relevance_score": 0.5,
        "n_samples": 40,                    # no adversarial null, underpowered
    }
    return ev, None, None


_BUILDERS = {
    "real_effect_statistical": build_real_effect_statistical,
    "null_effect_statistical": build_null_effect_statistical,
    "fabricated_input_statistical": build_fabricated_input_statistical,
    "small_n_statistical": build_small_n_statistical,
    "genuine_lofo": build_genuine_lofo,
    "artifact_lofo": build_artifact_lofo,
    "deterministic_proof": build_deterministic_proof,
    "bare_counter_deterministic": build_bare_counter_deterministic,
}


# ---------------------------------------------------------------------------
# Drive the REAL pipeline on the whole corpus for one seed.
# ---------------------------------------------------------------------------

def run_corpus_once(seed: int, *, pipeline=run_verdict_pipeline) -> list[dict]:
    """Build every case for this seed, run each through the real verdict pipeline,
    and record whether it CONFIRMED vs. its ground-truth should_confirm label.
    `pipeline` is injectable only so the sanity check can point at a broken variant.
    """
    rng = random.Random(seed)
    rows: list[dict] = []
    for case_type, should_confirm in CASE_TYPES:
        evidence, hypothesis, ctx = _BUILDERS[case_type](rng)
        classified = classify_evidence_type(dict(evidence))  # measured routing
        verdict, confidence, reason = pipeline(evidence, hypothesis, ctx)
        rows.append({
            "case_type": case_type,
            "should_confirm": should_confirm,      # GROUND TRUTH (by construction)
            "confirmed": verdict == "confirmed",   # MEASURED
            "verdict": verdict,
            "confidence": round(float(confidence), 3),
            "classified_as": classified,
            "reason": reason,
        })
    return rows


# ---------------------------------------------------------------------------
# Metrics.
# ---------------------------------------------------------------------------

def _metrics_from_rows(all_rows: list[list[dict]]) -> dict:
    """Aggregate over all seeds. Rates are computed over the flattened case set
    (n_seeds x 8 cases) so each metric is an average over varied draws.
    """
    flat = [r for rows in all_rows for r in rows]
    pos = [r for r in flat if r["should_confirm"]]
    neg = [r for r in flat if not r["should_confirm"]]

    false_confirms = sum(1 for r in neg if r["confirmed"])
    true_confirms = sum(1 for r in pos if r["confirmed"])

    false_confirm_rate = false_confirms / len(neg) if neg else 0.0
    confirm_recall = true_confirms / len(pos) if pos else 0.0
    specificity = 1.0 - false_confirm_rate
    balanced_accuracy = (confirm_recall + specificity) / 2.0

    per_type: dict[str, dict] = {}
    for case_type, should_confirm in CASE_TYPES:
        rows = [r for r in flat if r["case_type"] == case_type]
        n = len(rows)
        confirmed = sum(1 for r in rows if r["confirmed"])
        per_type[case_type] = {
            "should_confirm": should_confirm,
            "confirm_rate": round(confirmed / n, 4) if n else 0.0,
            "correct": (confirmed == n) if should_confirm else (confirmed == 0),
            "n": n,
        }

    return {
        "false_confirm_rate": round(false_confirm_rate, 4),
        "confirm_recall": round(confirm_recall, 4),
        "balanced_accuracy": round(balanced_accuracy, 4),
        "specificity": round(specificity, 4),
        "n_positive_cases": len(pos),
        "n_negative_cases": len(neg),
        "per_evidence_type": per_type,
    }


def _git_sha() -> str:
    for root in (_MAIN_CHECKOUT, _WORKTREE_ROOT):
        try:
            out = subprocess.run(
                ["git", "-C", str(root), "rev-parse", "HEAD"],
                capture_output=True, text=True, check=True,
            )
            return out.stdout.strip()
        except Exception:
            continue
    return "unknown"


# ---------------------------------------------------------------------------
# Sanity check: the metric MUST move. Point the pipeline at a broken artifact
# module (force `_survives_permutation` and `_survives_label_shuffle_lofo` to
# report survived=True) and confirm false_confirm_rate jumps. Then revert.
# ---------------------------------------------------------------------------

def sanity_check_metric_moves(n_seeds: int) -> dict:
    """Monkeypatch the real artifact tests to always 'survive' and re-measure.
    If the harness is wired to the real gate, false_confirm_rate must rise well
    above the honest baseline (artifacts/nulls now sail through the gate).
    """
    import propab.artifact_verification as av

    orig_perm = av._survives_permutation
    orig_lofo = av._survives_label_shuffle_lofo

    def _always_survive_perm(ctx, exp):
        v = orig_perm(ctx, exp)
        v.survived = True
        return v

    def _always_survive_lofo(exp):
        v = orig_lofo(exp)
        v.survived = True
        return v

    baseline_rows = [run_corpus_once(s) for s in range(n_seeds)]
    baseline = _metrics_from_rows(baseline_rows)["false_confirm_rate"]

    av._survives_permutation = _always_survive_perm
    av._survives_label_shuffle_lofo = _always_survive_lofo
    try:
        broken_rows = [run_corpus_once(s) for s in range(n_seeds)]
        broken = _metrics_from_rows(broken_rows)["false_confirm_rate"]
    finally:
        av._survives_permutation = orig_perm
        av._survives_label_shuffle_lofo = orig_lofo

    return {
        "baseline_false_confirm_rate": baseline,
        "broken_false_confirm_rate": broken,
        "metric_moved": broken > baseline,
    }


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

N_SEEDS = 12


def main() -> None:
    all_rows = [run_corpus_once(s) for s in range(N_SEEDS)]
    metrics = _metrics_from_rows(all_rows)

    config = {
        "n_seeds": N_SEEDS,
        "n_case_types": len(CASE_TYPES),
        "cases_per_seed": len(CASE_TYPES),
        "permutation_null": "services.worker.permutation_null.compute_label_permutation_null",
        "entry_point": "propab.verdict_pipeline.run_verdict_pipeline",
        "main_checkout": str(_MAIN_CHECKOUT),
    }

    out = {
        "metrics": metrics,
        "n_seeds": N_SEEDS,
        "config": config,
        "git_sha": _git_sha(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    artifact_path = _WORKTREE_ROOT / "artifacts" / "bench" / "verdict_baseline.json"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")

    # ---- human-readable report ------------------------------------------------
    print(f"Verdict-layer benchmark (real verdict_pipeline + artifact gate + D2 null), {N_SEEDS} seeds:")
    print(f"  false-confirm rate            : {metrics['false_confirm_rate']}   "
          "(fraction of artifact/null/fabricated cases wrongly CONFIRMED; LOWER is better, 0 ideal)")
    print(f"  confirm recall                : {metrics['confirm_recall']}   "
          "(fraction of genuine findings CONFIRMED; higher is better, 1 ideal)")
    print(f"  balanced accuracy             : {metrics['balanced_accuracy']}   "
          "(mean of recall-on-genuine and specificity-on-artifacts)")
    print(f"  specificity                   : {metrics['specificity']}   "
          "(fraction of should-refuse cases correctly NOT confirmed)")
    print("  per-evidence-type confirm rate (want 1.0 for should_confirm=True, 0.0 for False):")
    for case_type, should_confirm in CASE_TYPES:
        pt = metrics["per_evidence_type"][case_type]
        flag = "OK " if pt["correct"] else "XX "
        want = "confirm" if should_confirm else "refuse "
        print(f"    [{flag}] {case_type:<32} want={want} rate={pt['confirm_rate']}")

    sanity = sanity_check_metric_moves(N_SEEDS)
    print("  metric-moves sanity check (force artifact tests to always survive):")
    print(f"    baseline false-confirm={sanity['baseline_false_confirm_rate']} -> "
          f"broken false-confirm={sanity['broken_false_confirm_rate']} "
          f"(moved={sanity['metric_moved']})")

    print(f"  wrote {artifact_path}")


if __name__ == "__main__":
    main()
