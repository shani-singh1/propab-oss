"""Evidence-binding / citation-integrity benchmark — drives the REAL binding code.

Like ``scripts/bench_campaign_convergence.py`` drives the real convergence path,
this drives the REAL production evidence-binding path that decides whether a
citation is accepted as evidence for a belief:

    propab.evidence_binding.filter_node_citations(...)   (the prod entry point)
        -> binding_check -> _structured_overlap / biology tag path

Nothing that DECIDES the metric is reimplemented here. The only inputs we author
are a LABELED CORPUS of (belief statement, candidate node) pairs — the thing a
harness must provide — where every label is ground-truth *by construction*:

  genuine-supporter  : a node whose structured fields (verdict/metric/claim
                       scope/mechanism-or-feature id/salient terms) genuinely
                       bear on the belief.                       -> should BIND
  irrelevant         : an unrelated node.                        -> should REJECT
  fabricated-overlap : a node that superficially echoes the belief's phrasing
                       but shares NO real mechanism/scope/salient subject terms.
                                                                  -> should REJECT
  cross-domain       : a GENUINE supporter, but for a different subject/domain.
                                                                  -> should REJECT

The corpus deliberately spans biology/mandrake, math (Sidon / cap-set), physics,
econometrics, and a generic domain, because the historical pre-A4 failure was
~0 recall outside biology. Proving high recall in the non-biology domains is how
this harness demonstrates A4 made binding domain-general.

Metrics (all offline, deterministic per seed):
  binding_precision : of citations that BOUND, fraction that are genuine
                      supporters. KEY INTEGRITY METRIC (target 1.0): a
                      false-accept IS the citation-integrity failure.
  binding_recall    : of genuine supporters, fraction that BOUND (target high;
                      don't starve belief formation).
  per_domain        : precision & recall per domain — proves A4 domain-generality
                      (non-biology recall must be high, not ~0).

Run (from the worktree root):
    PYTHONPATH="packages/propab-core;." python bench/bench_binding.py

The bench self-locates the A4-merged ``propab`` (the module that actually
contains ``_structured_overlap``). If the checkout on ``PYTHONPATH`` predates
A4, it falls back to the sibling main checkout and ASSERTS the A4 code is loaded,
so the harness can never silently measure the wrong version of core.
"""
from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ── Locate the REAL, A4-merged propab (importing core is correct; we don't edit it) ──
_HERE = Path(__file__).resolve()
_WORKTREE_ROOT = _HERE.parents[1]


def _load_evidence_binding():
    """Import propab.evidence_binding, ensuring it is the A4 (domain-general) version.

    Order:
      1. Whatever is already importable (honors the documented PYTHONPATH).
      2. The worktree's own packages/propab-core.
      3. The sibling *main* checkout's packages/propab-core (the task says
         importing propab from the main checkout is correct).
    The first candidate whose evidence_binding defines ``_structured_overlap``
    (the A4 acceptance path) wins. We assert it loaded, so a pre-A4 core cannot
    be measured by mistake.
    """
    candidates: list[Path | None] = [None]  # already-on-path first
    candidates.append(_WORKTREE_ROOT / "packages" / "propab-core")
    # Walk up looking for a sibling checkout that is NOT inside a worktree tree.
    # Worktrees live under <main>/.claude/worktrees/<id>/ — main is 3 parents up.
    for up in (_WORKTREE_ROOT.parents[2:3] or []):
        candidates.append(up / "packages" / "propab-core")

    last_err: Exception | None = None
    for cand in candidates:
        try:
            if cand is not None:
                p = str(cand)
                if p not in sys.path:
                    sys.path.insert(0, p)
            # Drop any earlier partial import so the new path can win.
            for mod in [m for m in sys.modules if m == "propab" or m.startswith("propab.")]:
                del sys.modules[mod]
            import propab.evidence_binding as eb  # noqa: PLC0415
            if hasattr(eb, "_structured_overlap"):
                return eb
            last_err = RuntimeError(
                f"propab.evidence_binding at {eb.__file__} is pre-A4 "
                "(no _structured_overlap); trying next checkout",
            )
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            continue
    raise RuntimeError(
        "Could not import an A4-merged propab.evidence_binding "
        f"(with _structured_overlap). Last error: {last_err}",
    )


eb = _load_evidence_binding()
filter_node_citations = eb.filter_node_citations
BindingMetrics = eb.BindingMetrics


# ── Labeled corpus ────────────────────────────────────────────────────────────
# Each case is (belief_statement, candidate_node, label). Labels are ground-truth
# BY CONSTRUCTION (documented per case below), not measured. "bind" = the layer
# SHOULD accept (genuine supporter); "reject" = the layer SHOULD refuse.
GENUINE = "genuine"            # should BIND
IRRELEVANT = "irrelevant"      # should REJECT
FABRICATED = "fabricated"      # should REJECT
CROSS_DOMAIN = "cross_domain"  # should REJECT

_SHOULD_BIND = {GENUINE}


def _case(domain: str, label: str, belief: str, node: dict[str, Any]) -> dict[str, Any]:
    return {"domain": domain, "label": label, "belief": belief, "node": node}


def build_corpus() -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []

    # ── biology / mandrake ────────────────────────────────────────────────────
    # This is the ONE domain the pre-A4 biology tag regexes already covered; it
    # anchors the biology path (LOFO / within-family test targets).
    bio_belief = (
        "Cross-family LOFO generalization collapses while within-family signal "
        "survives for mandrake retention-time prediction.\n"
        "Population: mandrake enzyme families\nDistribution: leave-one-family-out\n"
        "Claimed generalization: predictive across families\n"
        "Expected failure modes: family identity confound\nOOD test: LOFO holdout"
    )
    cases += [
        _case("biology", GENUINE, bio_belief, {
            # genuine: same LOFO/within-family subject, label-shuffle null, family confound
            "text": ("Leave-one-family-out cross-family evaluation shows LOFO r2 near zero; "
                     "label-shuffle permutation null confirms family identity explains the signal.\n"
                     "Population: mandrake families\nDistribution: cross-family LOFO\n"
                     "Claimed generalization: transfer across families\n"
                     "Expected failure modes: within-family leakage\nOOD test: leave-one-family-out"),
            "finding": {"claim": "cross-family LOFO collapses", "metric_name": "lofo_r2"},
        }),
        _case("biology", IRRELEVANT, bio_belief, {
            "text": "The gift shop restocked postcards and umbrellas before the holiday weekend.",
            "finding": {"claim": "shop restocked inventory", "metric_name": "stock_count"},
        }),
        _case("biology", FABRICATED, bio_belief, {
            # echoes 'family' / 'one' but about tourists — no LOFO/within-family subject
            "text": "A family of tourists left one suitcase behind at the harbor cafe near the pier.",
            "finding": {"claim": "suitcase left behind", "metric_name": "item_count"},
        }),
        _case("biology", CROSS_DOMAIN, bio_belief, {
            # genuine within-family finding, but for perovskite CRYSTALS, not mandrake enzymes
            "text": ("Within-family variance of crystal dielectric constant is high across perovskite groups.\n"
                     "Population: perovskite crystal families\nDistribution: within-group\n"
                     "Claimed generalization: dielectric predictable within group\n"
                     "Expected failure modes: group confound\nOOD test: within-group split"),
            "finding": {"claim": "within-family dielectric variance high", "metric_name": "dielectric"},
        }),
    ]

    # ── math (Sidon / cap-set) ────────────────────────────────────────────────
    math_belief = (
        "Greedy Sidon sequence density F(n)/sqrt(n) crosses below 0.60 within n in [10000,50000].\n"
        "Population: greedy Sidon sequences\nDistribution: n in 10000..50000\n"
        "Claimed generalization: crossing persists across the dense n grid\n"
        "Expected failure modes: nonmonotonic dip near boundary\nOOD test: dense n grid holdout"
    )
    cases += [
        _case("math", GENUINE, math_belief, {
            "text": ("Numeric sweep confirms greedy Sidon density F(n)/sqrt(n) dips below 0.60 near n=32000.\n"
                     "Population: greedy Sidon sequences\nDistribution: n in 30000..35000\n"
                     "Claimed generalization: crossing region localized\n"
                     "Expected failure modes: sampling gap\nOOD test: dense n grid holdout"),
            "finding": {"claim": "greedy Sidon density crosses below 0.60", "metric_name": "F_over_sqrt_n"},
        }),
        _case("math", IRRELEVANT, math_belief, {
            "text": "Quarterly rainfall totals in the coastal basin were above the seasonal average.",
            "finding": {"claim": "rainfall above average", "metric_name": "rainfall_mm"},
        }),
        _case("math", FABRICATED, math_belief, {
            # superficial echo of the belief's cadence ("... within ...") but a wholly
            # different subject: shares NO Sidon/density/greedy/crossing subject term.
            "text": "The pastry chef plated the dessert within minutes as the dinner service began.",
            "finding": {"claim": "dessert plated on time", "metric_name": "plating_seconds"},
        }),
        _case("math", CROSS_DOMAIN, math_belief, {
            # genuine density-crossing finding, but for CAP-SETS in F_3^n, a different object
            "text": ("Exhaustive search shows cap-set upper density in F_3^n falls below the 2.756^n "
                     "scaling threshold for small n."),
            "finding": {"claim": "cap-set density below scaling bound", "metric_name": "cap_density"},
        }),
    ]

    # ── physics ───────────────────────────────────────────────────────────────
    phys_belief = (
        "Turbulent boundary-layer drag reduction from riblets scales with riblet spacing "
        "at moderate Reynolds number.\n"
        "Population: turbulent boundary layers over riblet surfaces\nDistribution: Reynolds 1e4..1e5\n"
        "Claimed generalization: drag reduction holds across riblet spacing\n"
        "Expected failure modes: viscous sublayer breakdown\nOOD test: hold out one Reynolds decade"
    )
    cases += [
        _case("physics", GENUINE, phys_belief, {
            "text": ("Wind-tunnel measurement: riblet spacing of 15 wall units yields 8% turbulent "
                     "boundary-layer drag reduction at Reynolds 5e4.\n"
                     "Population: riblet turbulent boundary layers\nDistribution: Reynolds 5e4\n"
                     "Claimed generalization: drag reduction across spacing\n"
                     "Expected failure modes: sublayer breakdown\nOOD test: hold out one Reynolds decade"),
            "finding": {"claim": "riblet spacing reduces turbulent drag", "metric_name": "drag_reduction_pct"},
        }),
        _case("physics", IRRELEVANT, phys_belief, {
            "text": "Quarterly bond yields ticked up after the central bank's Thursday statement.",
            "finding": {"claim": "yields ticked up", "metric_name": "yield_bps"},
        }),
        _case("physics", FABRICATED, phys_belief, {
            # 'boundary' / 'moderate' surface echo, museum subject — no riblet/drag/turbulence
            "text": "The museum lecture on baroque architecture drew a large audience despite the rain.",
            "finding": {"claim": "lecture drew audience", "metric_name": "attendance"},
        }),
        _case("physics", CROSS_DOMAIN, phys_belief, {
            # genuine 'X scales with spacing' finding, but for superconducting cuprates
            "text": ("Superconducting energy gap in cuprate films scales with dopant carrier spacing "
                     "near optimal doping."),
            "finding": {"claim": "superconducting gap scales with carrier spacing", "metric_name": "energy_gap"},
        }),
    ]

    # ── econometrics ──────────────────────────────────────────────────────────
    econ_belief = (
        "Minimum wage increases reduce teen employment elasticity in low-density labor markets.\n"
        "Population: teen workers in low-density counties\nDistribution: 2010-2019 county panel\n"
        "Claimed generalization: elasticity negative across low-density counties\n"
        "Expected failure modes: monopsony offset\nOOD test: hold out one state"
    )
    cases += [
        _case("econometrics", GENUINE, econ_belief, {
            "text": ("County panel regression finds minimum wage elasticity of teen employment is -0.12 "
                     "in low-density counties.\n"
                     "Population: teen workers in low-density counties\nDistribution: 2012-2018 county panel\n"
                     "Claimed generalization: negative elasticity across counties\n"
                     "Expected failure modes: monopsony offset\nOOD test: hold out one state"),
            "finding": {"claim": "teen employment elasticity negative under minimum wage", "metric_name": "elasticity"},
        }),
        _case("econometrics", IRRELEVANT, econ_belief, {
            "text": "Enzyme turnover number scales with substrate concentration below saturation in vitro.",
            "finding": {"claim": "turnover scales with substrate", "metric_name": "kcat"},
        }),
        _case("econometrics", FABRICATED, econ_belief, {
            # choir subject — no wage/employment/elasticity subject at all
            "text": "The community choir rehearsed a new arrangement before the autumn charity concert.",
            "finding": {"claim": "choir rehearsed arrangement", "metric_name": "rehearsals"},
        }),
        _case("econometrics", CROSS_DOMAIN, econ_belief, {
            # genuine demand-elasticity finding, but for cigarette taxes / smoking, not teen wages
            "text": ("Cigarette excise taxes reduce adult smoking prevalence with a demand elasticity of -0.4 "
                     "in high-income states."),
            "finding": {"claim": "cigarette tax reduces smoking prevalence", "metric_name": "elasticity"},
        }),
    ]

    # ── generic domain ────────────────────────────────────────────────────────
    gen_belief = (
        "Adding spaced-repetition review to onboarding increases 30-day feature-retention "
        "for new SaaS users.\n"
        "Population: new SaaS users in onboarding\nDistribution: cohorts Q1-Q3\n"
        "Claimed generalization: retention lift across cohorts\n"
        "Expected failure modes: novelty decay\nOOD test: hold out one cohort"
    )
    cases += [
        _case("generic", GENUINE, gen_belief, {
            "text": ("A/B test: spaced-repetition review during onboarding lifts 30-day feature-retention "
                     "by 6 points for new SaaS users.\n"
                     "Population: new SaaS users in onboarding\nDistribution: Q2 cohort\n"
                     "Claimed generalization: retention lift across cohorts\n"
                     "Expected failure modes: novelty decay\nOOD test: hold out one cohort"),
            "finding": {"claim": "spaced-repetition lifts feature-retention", "metric_name": "retention_rate"},
        }),
        _case("generic", IRRELEVANT, gen_belief, {
            "text": "The alloy's yield strength dropped after prolonged annealing above the recrystallization point.",
            "finding": {"claim": "yield strength dropped", "metric_name": "yield_strength"},
        }),
        _case("generic", FABRICATED, gen_belief, {
            # bike-lane subject — no onboarding/retention/spaced-repetition subject
            "text": "The city council approved a new bike lane along the riverfront promenade this quarter.",
            "finding": {"claim": "bike lane approved", "metric_name": "lane_length_km"},
        }),
        _case("generic", CROSS_DOMAIN, gen_belief, {
            # genuine retention-lift finding, but for a fitness app / gym members, different subject
            "text": ("Adding streak badges to a mobile fitness app increases 30-day workout adherence "
                     "for gym members."),
            "finding": {"claim": "streak badges lift workout adherence", "metric_name": "adherence_rate"},
        }),
    ]

    return cases


# ── Metric computation — drives the REAL filter_node_citations ─────────────────
def _bound(case: dict[str, Any]) -> bool:
    """Return True iff the REAL production binder ACCEPTS this case's citation.

    Uses filter_node_citations — the same entry point the orchestrator calls to
    keep only node ids that mechanically match a belief statement.
    """
    node_id = "cand"
    kept = filter_node_citations(case["belief"], [node_id], {node_id: case["node"]})
    return node_id in kept


def evaluate(cases: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute precision/recall overall + per domain against ground-truth labels."""
    domains = sorted({c["domain"] for c in cases})
    per_domain: dict[str, dict[str, Any]] = {}

    overall_tp = overall_fp = overall_fn = 0
    overall_genuine = overall_bound = 0

    for dom in domains:
        dc = [c for c in cases if c["domain"] == dom]
        tp = fp = fn = 0            # tp: genuine & bound; fp: reject-class & bound; fn: genuine & not bound
        n_genuine = n_bound = 0
        for c in dc:
            should_bind = c["label"] in _SHOULD_BIND
            did_bind = _bound(c)
            if should_bind:
                n_genuine += 1
            if did_bind:
                n_bound += 1
            if should_bind and did_bind:
                tp += 1
            elif (not should_bind) and did_bind:
                fp += 1
            elif should_bind and not did_bind:
                fn += 1
        precision = tp / (tp + fp) if (tp + fp) else 1.0  # no binds -> vacuously precise
        recall = tp / (tp + fn) if (tp + fn) else 1.0
        per_domain[dom] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "n_genuine": n_genuine,
            "n_bound": n_bound,
            "false_accepts": fp,
        }
        overall_tp += tp
        overall_fp += fp
        overall_fn += fn
        overall_genuine += n_genuine
        overall_bound += n_bound

    binding_precision = overall_tp / (overall_tp + overall_fp) if (overall_tp + overall_fp) else 1.0
    binding_recall = overall_tp / (overall_tp + overall_fn) if (overall_tp + overall_fn) else 1.0

    non_bio = [d for d in domains if d != "biology"]
    non_bio_recall = (
        sum(per_domain[d]["recall"] for d in non_bio) / len(non_bio) if non_bio else 0.0
    )

    return {
        "binding_precision": round(binding_precision, 4),
        "binding_recall": round(binding_recall, 4),
        "per_domain": per_domain,
        "non_biology_mean_recall": round(non_bio_recall, 4),
        "n_cases": len(cases),
        "n_genuine": overall_genuine,
        "n_bound": overall_bound,
        "false_accepts": overall_fp,
    }


def run(*, seed: int = 0) -> dict[str, Any]:
    """One deterministic evaluation. The seed shuffles corpus order only: the
    binder is order-independent, so the metric MUST be identical across seeds —
    that invariance is itself a (cheap) correctness check on the harness."""
    cases = build_corpus()
    random.Random(seed).shuffle(cases)
    return evaluate(cases)


# ── Sanity check: force the accept path and confirm the metric MOVES ───────────
def sanity_always_accept() -> dict[str, Any]:
    """Monkeypatch the REAL binding_check to always accept, and confirm
    binding_precision drops (a false-accept is exactly the integrity failure).
    Reverted before returning. This proves the metric is actually driven by the
    binder's decision and would catch a regression to the pre-A4 always-accept
    behavior (the 94.5% irrelevant-citation era)."""
    original = eb.binding_check
    try:
        eb.binding_check = lambda citing, cited: eb.BindingResult(True, "forced_accept")  # type: ignore[assignment]
        forced = run(seed=0)
    finally:
        eb.binding_check = original  # revert
    real = run(seed=0)
    return {
        "real_precision": real["binding_precision"],
        "forced_accept_precision": forced["binding_precision"],
        "precision_dropped": forced["binding_precision"] < real["binding_precision"],
        "forced_false_accepts": forced["false_accepts"],
    }


def _git_sha(cwd: Path | None = None) -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=str(cwd or _WORKTREE_ROOT),
            capture_output=True, text=True, timeout=30,
        )
        return out.stdout.strip() or "unknown"
    except Exception:  # noqa: BLE001
        return "unknown"


def _binder_source_sha() -> str:
    """HEAD sha of the checkout that actually supplied the measured binder.

    The A4 code may live in a different checkout than the one running the bench
    (e.g. the main checkout), so we record its sha too for honest provenance.
    """
    return _git_sha(Path(eb.__file__).resolve().parent)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=5, help="number of deterministic seeds to average")
    parser.add_argument(
        "--out", type=Path,
        default=_WORKTREE_ROOT / "artifacts" / "bench" / "binding_baseline.json",
    )
    args = parser.parse_args()

    seeds = list(range(args.seeds))
    rows = [run(seed=s) for s in seeds]

    def avg(key: str) -> float:
        return round(sum(r[key] for r in rows) / len(rows), 4)

    domains = sorted(rows[0]["per_domain"].keys())

    def avg_dom(dom: str, key: str) -> float:
        return round(sum(r["per_domain"][dom][key] for r in rows) / len(rows), 4)

    metrics: dict[str, Any] = {
        "binding_precision": avg("binding_precision"),
        "binding_recall": avg("binding_recall"),
        "non_biology_mean_recall": avg("non_biology_mean_recall"),
        "n_cases": rows[0]["n_cases"],
        "n_genuine": rows[0]["n_genuine"],
        "false_accepts": avg("false_accepts"),
        "per_domain": {
            dom: {"precision": avg_dom(dom, "precision"), "recall": avg_dom(dom, "recall")}
            for dom in domains
        },
    }

    sanity = sanity_always_accept()

    # ── Human-readable report ─────────────────────────────────────────────────
    print(f"Evidence-binding / citation-integrity benchmark "
          f"(real filter_node_citations), {len(rows)} seeds, "
          f"{metrics['n_cases']} labeled (belief,node) pairs:")
    print(f"  binding_precision      : {metrics['binding_precision']}   "
          "(of citations that BOUND, fraction genuine; target 1.0 — a false-accept IS the integrity failure)")
    print(f"  binding_recall         : {metrics['binding_recall']}   "
          "(of genuine supporters, fraction that BOUND; target high — don't starve belief formation)")
    print(f"  non_biology_mean_recall: {metrics['non_biology_mean_recall']}   "
          "(A4 domain-generality: pre-A4 this was ~0; must be high now)")
    print(f"  false_accepts (mean)   : {metrics['false_accepts']}   "
          "(reject-class citations the binder wrongly ACCEPTED)")
    print("  per_domain (precision / recall) — proves recall is high in EVERY domain, not just biology:")
    for dom in domains:
        d = metrics["per_domain"][dom]
        tag = "" if dom == "biology" else "  <- non-biology: A4 must keep recall high here"
        print(f"      {dom:14s}  precision={d['precision']:<6}  recall={d['recall']:<6}{tag}")
    print("  metric-moves sanity check (force accept path -> precision must DROP):")
    print(f"      real precision={sanity['real_precision']}  "
          f"forced-accept precision={sanity['forced_accept_precision']}  "
          f"dropped={sanity['precision_dropped']}  "
          f"(forced false-accepts={sanity['forced_false_accepts']})")

    payload = {
        "metrics": metrics,
        "n_seeds": len(seeds),
        "config": {
            "seeds": seeds,
            "domains": domains,
            "classes_per_domain": [GENUINE, IRRELEVANT, FABRICATED, CROSS_DOMAIN],
            "driver": "propab.evidence_binding.filter_node_citations",
            "evidence_binding_module": eb.__file__,
            "binder_source_git_sha": _binder_source_sha(),
            "a4_structured_overlap_present": hasattr(eb, "_structured_overlap"),
            "labels": "ground-truth by construction; bind/reject decision measured by the real binder",
        },
        "sanity_metric_moves": sanity,
        "git_sha": _git_sha(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
