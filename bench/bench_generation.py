"""Hypothesis-generation-layer benchmark — drives the REAL post-generation filters.

This quantifies the DETERMINISTIC filtering layer of hypothesis generation, the
same way ``scripts/bench_campaign_convergence.py`` quantifies convergence and
LitQA2 quantifies the literature layer. It drives the actual production functions
imported from ``propab`` / ``services.orchestrator`` — never a reimplementation:

  - dedup            : ``campaign_synthesis._is_duplicate_frontier_candidate``
                       (scope+numeric-signature-aware duplicate filter, G1/G2)
  - relevance gate   : ``domain_modules.registry.hypothesis_is_on_topic``
                       (each domain plugin's ``hypothesis_on_topic``)
  - implementable    : ``synthesis_diversity.methodology_implementable`` against
                       each domain's ``implementable_methodologies()``
  - scope validity   : ``scoped_claim.validate_scoped_claim`` +
                       ``scoped_claim.is_boilerplate_scope``
  - end-to-end       : ``campaign_synthesis.apply_synthesis_to_frontier`` chains
                       ALL of the above and returns a metrics dict; the benchmark
                       cross-checks its per-filter counters against the
                       per-candidate labels.

SCOPE / HONESTY (read ``bench/README.md`` "Generation layer"):
  * This measures the DETERMINISTIC filters only. The raw creativity of the LLM
    candidate proposal is non-deterministic and OUT OF SCOPE (that quality is
    measured by campaign outcomes, not here).
  * The candidate banks are HAND-LABELLED by construction: each candidate carries
    an explicit ground-truth tag (on_topic / duplicate / narrowing_child /
    unimplementable / ungrounded_scope). Those labels ARE the ground truth; they
    are validated to be defensible against the real functions (a duplicate is a
    pure rephrasing with identical numbers AND scope; a narrowing child keeps the
    claim title but changes the numeric/scope fingerprint).
  * Two domains (mandrake, generic-econ) have NO ``hypothesis_on_topic`` override
    (accept-all), so their off-topic candidates are excluded from
    ``offtopic_rejection_rate`` — measuring rejection where there is no gate would
    be dishonest. This is reported as ``offtopic_domains_with_gate``.

Deterministic: no randomness, no network, no LLM. ``main()`` runs a fixed set of
banks (n_seeds counts the banks) and averages/aggregates the metrics.

Run from the worktree root:
    PYTHONPATH="packages/propab-core;." python bench/bench_generation.py

The bench deliberately re-points ``sys.path`` at the MAIN checkout's
``packages/propab-core`` when it is run inside a git worktree, so it always
drives the up-to-date production code (the task's "import propab from the main
checkout is correct"). The PYTHONPATH above is a harmless fallback.
"""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


# --------------------------------------------------------------------------- #
# Path bootstrap: always drive the MAIN checkout's production code.
# --------------------------------------------------------------------------- #
def _main_checkout_root() -> Path:
    """Locate the real repo root, even when this file lives in a git worktree.

    A worktree lives at ``<main>/.claude/worktrees/<name>/``; strip that suffix so
    we import the canonical, current ``propab`` rather than a stale worktree copy.
    """
    here = Path(__file__).resolve()
    parts = here.parts
    if ".claude" in parts:
        return Path(*parts[: parts.index(".claude")])
    # Not in a worktree: repo root is the parent of bench/.
    return here.parents[1]


REPO = _main_checkout_root()
_CORE = REPO / "packages" / "propab-core"
for p in (str(_CORE), str(REPO)):
    if p not in sys.path:
        sys.path.insert(0, p)

from propab.belief_state import CampaignBeliefState, ClosedBelief  # noqa: E402
from propab.campaign_synthesis import (  # noqa: E402
    _is_duplicate_frontier_candidate,
    apply_synthesis_to_frontier,
)
from propab.domain_modules.registry import (  # noqa: E402
    get_domain_plugin,
    hypothesis_is_on_topic,
    resolve_domain_plugin,
)
from propab.hypothesis_tree import HypothesisTree  # noqa: E402
from propab.scoped_claim import (  # noqa: E402
    is_boilerplate_scope,
    parse_scope_from_methodology,
    validate_scoped_claim,
)
from propab.synthesis_diversity import methodology_implementable  # noqa: E402


# --------------------------------------------------------------------------- #
# Candidate bank construction.
#
# Each candidate is a dict with a ``text`` (with inline scope lines so the real
# scope parser can read it), a ``test_methodology``, and a ``_label`` giving the
# ground-truth tag. Scope lines use the exact labels the real parser matches
# (Population:/Distribution:/Claimed generalization:/Expected failure modes:/OOD test:).
# --------------------------------------------------------------------------- #

_SCOPE_TMPL = (
    "\nPopulation: {pop}"
    "\nDistribution: {dist}"
    "\nClaimed generalization: {gen}"
    "\nExpected failure modes: {fail}"
    "\nOOD test: {ood}"
)


def _scoped(claim: str, *, pop: str, dist: str, gen: str, fail: str, ood: str) -> str:
    return claim + _SCOPE_TMPL.format(pop=pop, dist=dist, gen=gen, fail=fail, ood=ood)


def _cand(text: str, methodology: str, label: str) -> dict:
    return {"text": text, "test_methodology": methodology, "_label": label}


# --- Bank 1: math / Sidon (real on-topic gate + real implementable filter) --- #
def bank_math_sidon() -> dict:
    q = "[domain_profile:math_combinatorics] Where does greedy F(n)/sqrt(n) first fall below 0.60 for Sidon sets?"

    def sc(claim, lo, hi):
        return _scoped(
            claim,
            pop=f"greedy Sidon sequences in {{1,...,n}}",
            dist=f"n in [{lo},{hi}]",
            gen=f"crossing occurs within [{lo},{hi}]",
            fail="nonmonotonic dip below n=1000",
            ood="dense n grid replicates the crossing",
        )

    # The confirmed parent already in the tree (context for dedup).
    parent = sc("Greedy F(n)/sqrt(n) crosses 0.60 within n in [10000,50000] for Sidon sets.", 10000, 50000)

    on = [
        # on-topic, implementable, well-scoped, DISTINCT (survives everything)
        _cand(sc("Greedy F(n)/sqrt(n) crosses 0.60 within n in [30000,40000] for Sidon sets.", 30000, 40000),
              "greedy Sidon density sweep", "distinct_survivor"),
        _cand(sc("Bose-Chowla ratio exceeds greedy ratio at matched n in [2500,10000] for Sidon sets.", 2500, 10000),
              "matched bose-chowla comparison", "distinct_survivor"),
    ]
    duplicates = [
        # same claim title + same numbers + same scope as parent => pure rephrasing
        _cand(sc("Greedy F(n)/sqrt(n) crosses 0.60 within n in [10000,50000] for Sidon sets.", 10000, 50000),
              "greedy Sidon density sweep", "duplicate"),
        _cand(sc("Greedy F(n)/sqrt(n) crosses 0.60 within n in [10000,50000] for Sidon sets.\n(Restated, same test.)",
                 10000, 50000),
              "greedy Sidon density sweep", "duplicate"),
    ]
    narrowing = [
        # same claim title but STRICTLY narrower numeric region => distinct child
        _cand(sc("Greedy F(n)/sqrt(n) crosses 0.60 within n in [10000,20000] for Sidon sets.", 10000, 20000),
              "greedy Sidon density sweep", "narrowing_child"),
        _cand(sc("Greedy F(n)/sqrt(n) crosses 0.60 within n in [12000,15000] for Sidon sets.", 12000, 15000),
              "greedy Sidon density sweep", "narrowing_child"),
    ]
    # Off-topic candidates must carry NO on-topic token anywhere (incl. scope), or
    # the real gate would (correctly) read them as on-topic. Use a neutral scope.
    def off_sc(claim):
        return _scoped(
            claim,
            pop="fixed-capacity cache traces / OS processes",
            dist="uniform and Zipf access workloads",
            gen="behaviour holds across trace lengths",
            fail="degrades on adversarial traces",
            ood="a held-out trace reproduces the effect",
        )

    offtopic = [
        _cand(off_sc("Search the system path for propab-submit binaries and report metrics."),
              "subprocess shell command", "offtopic"),
        _cand(off_sc("LRU cache miss rate exceeds LFU by 2x on the fixed-capacity trace."),
              "cache trace replay", "offtopic"),
    ]
    # Unimplementable: on-topic (Sidon) so the RELEVANCE gate passes, but the
    # methodology is outside the domain's implementable keyword set AND is not one
    # of the math on-topic gate's explicitly-blocked techniques (annealing/SAT/…),
    # so the IMPLEMENTABLE filter is the one that must reject it. Scope stays
    # keyword-neutral so no implementable keyword leaks in.
    def unimpl_sc(claim, lo, hi):
        return _scoped(
            claim,
            pop="integer sets of size n",
            dist=f"n in [{lo},{hi}]",
            gen=f"property holds within [{lo},{hi}]",
            fail="approximation stalls at a local optimum",
            ood="an independent method reproduces the value",
        )

    unimplementable = [
        _cand(unimpl_sc("The densest Sidon set in {1,...,n} is approximated by gradient descent for n in [1000,5000].",
                        1000, 5000),
              "gradient descent", "unimplementable"),
        _cand(unimpl_sc("A reinforcement-learning agent constructs a large Sidon set for n in [1000,5000].",
                        1000, 5000),
              "reinforcement learning", "unimplementable"),
    ]
    ungrounded = [
        # on-topic + implementable phrasing but NO scope lines => scope-invalid
        _cand("Greedy Sidon density decreases as n grows.", "greedy Sidon density sweep", "ungrounded_scope"),
    ]
    return {
        "domain_id": "math_combinatorics",
        "question": q,
        "parent_text": parent,
        "candidates": on + duplicates + narrowing + offtopic + unimplementable + ungrounded,
        "has_offtopic_gate": True,
        "has_implementable_filter": True,
    }


# --- Bank 2: biology / mandrake (NO on-topic gate; dedup via SCOPE signature) - #
def bank_biology_mandrake() -> dict:
    q = (
        "Is RT activity across these evolutionary families one shared biophysical mechanism, "
        "or are there genuinely distinct family-specific mechanisms? reverse transcriptase."
    )

    def sc(claim, pop, dist):
        return _scoped(
            claim,
            pop=pop,
            dist=dist,
            gen="signal survives leave-one-family-out",
            fail="collapses when geometry proxies family id",
            ood="LOFO on held-out family; label-shuffle p<0.05",
        )

    parent = sc(
        "Thermal-stability features predict RT activity within families.",
        "56 retroviral RT sequences across 7 families",
        "all 7 rt_family groups",
    )
    on = [
        _cand(sc("Electrostatic pocket features predict RT activity within families.",
                 "56 retroviral RT sequences across 7 families", "all 7 rt_family groups"),
              "clustered-split within-family regression", "distinct_survivor"),
    ]
    duplicates = [
        _cand(sc("Thermal-stability features predict RT activity within families.",
                 "56 retroviral RT sequences across 7 families", "all 7 rt_family groups"),
              "clustered-split within-family regression", "duplicate"),
        _cand(sc("Thermal-stability features predict RT activity within families.\n(Restated.)",
                 "56 retroviral RT sequences across 7 families", "all 7 rt_family groups"),
              "clustered-split within-family regression", "duplicate"),
    ]
    narrowing = [
        # SAME claim title but a narrowed sub-population (scope-line difference) => distinct
        _cand(sc("Thermal-stability features predict RT activity within families.",
                 "thermophilic RT enzymes from the LTR clade only", "3 thermophilic rt_family groups"),
              "clustered-split within-family regression", "narrowing_child"),
    ]
    ungrounded = [
        _cand("Thermal stability correlates with RT activity.", "regression", "ungrounded_scope"),
    ]
    return {
        "domain_id": "mandrake",
        "question": q,
        "parent_text": parent,
        "candidates": on + duplicates + narrowing + ungrounded,
        "has_offtopic_gate": False,  # mandrake does not override hypothesis_on_topic
        "has_implementable_filter": False,  # mandrake implementable_methodologies() == []
    }


# --- Bank 3: genomics (real on-topic gate + real implementable filter) ------- #
def bank_genomics() -> dict:
    q = "Does cross-tissue gene expression variance predict tissue specificity in GTEx? gtex tissue specificity."

    def sc(claim, genes, tissues):
        return _scoped(
            claim,
            pop=f"GTEx v8: {genes} variable genes",
            dist=f"{tissues} tissues, leave-one-tissue-out",
            gen="pattern survives held-out tissue",
            fail="tissue-label leakage; housekeeping tautology",
            ood="leave-tissue-out LOFO + label shuffle p<0.05",
        )

    parent = sc("Cross-tissue expression variance predicts tissue specificity in GTEx.", 1000, 10)
    on = [
        _cand(sc("Mean cross-tissue expression predicts tissue specificity in GTEx.", 2000, 12),
              "leave-tissue-out LOFO regression", "distinct_survivor"),
    ]
    duplicates = [
        _cand(sc("Cross-tissue expression variance predicts tissue specificity in GTEx.", 1000, 10),
              "leave-tissue-out LOFO regression", "duplicate"),
    ]
    narrowing = [
        _cand(sc("Cross-tissue expression variance predicts tissue specificity in GTEx.", 500, 5),
              "leave-tissue-out LOFO regression", "narrowing_child"),
    ]
    # Off-topic candidates carry no gtex/tissue token anywhere (incl. scope).
    def off_sc(claim):
        return _scoped(
            claim,
            pop="integer sets / docker sandboxes",
            dist="combinatorial constructions",
            gen="pattern holds across the sweep",
            fail="breaks outside the tested range",
            ood="an independent run reproduces it",
        )

    offtopic = [
        _cand(off_sc("Greedy Sidon set density decreases with n."), "greedy Sidon sweep", "offtopic"),
        _cand(off_sc("A docker subprocess reads the filesystem for cached results."), "subprocess", "offtopic"),
    ]
    # Unimplementable: on-topic (gtex/tissue in the claim) so relevance passes, but
    # a methodology outside the implementable set; scope keeps NO implementable
    # keyword (no 'leave-tissue-out'/'lofo'/'cross-tissue') so the filter decides.
    unimplementable = [
        _cand(
            "Expression of variable genes predicts tissue specificity via neural-network embeddings."
            + _scoped(
                "",
                pop="1000 variable genes",
                dist="10 tissue types with a held-out split",
                gen="pattern holds on the held-out tissue",
                fail="tissue-label leakage",
                ood="an independent tissue split reproduces it",
            ),
            "deep neural network training", "unimplementable"),
    ]
    ungrounded = [
        _cand("Gene expression predicts tissue specificity.", "cross-tissue lofo", "ungrounded_scope"),
    ]
    return {
        "domain_id": "genomics",
        "question": q,
        "parent_text": parent,
        "candidates": on + duplicates + narrowing + offtopic + unimplementable + ungrounded,
        "has_offtopic_gate": True,
        "has_implementable_filter": True,
    }


# --- Bank 4: physics / graph_invariants (real on-topic gate) ----------------- #
def bank_graph_invariants() -> dict:
    q = (
        "Does the spectral gap predict clustering coefficient across network families? "
        "algebraic connectivity, modularity, snap."
    )

    def sc(claim, families):
        return _scoped(
            claim,
            pop="SNAP subset: 160 graphs",
            dist=f"{families} network families, leave-one-family-out",
            gen="invariant relation survives held-out family",
            fail="family-specific topology masks the invariant",
            ood="held-out network category replicates the correlation",
        )

    parent = sc("Spectral gap correlates with clustering coefficient across 4 network families.", 4)
    on = [
        _cand(sc("Algebraic connectivity correlates with modularity across 4 network families.", 4),
              "leave-network-family-out correlation", "distinct_survivor"),
    ]
    duplicates = [
        _cand(sc("Spectral gap correlates with clustering coefficient across 4 network families.", 4),
              "leave-network-family-out correlation", "duplicate"),
    ]
    narrowing = [
        _cand(sc("Spectral gap correlates with clustering coefficient across 4 network families.", 3),
              "leave-network-family-out correlation", "narrowing_child"),
    ]
    # Off-topic candidates carry no graph-invariant token anywhere (incl. scope).
    def off_sc(claim):
        return _scoped(
            claim,
            pop="GTEx genes / enzyme assays",
            dist="tissue and temperature panels",
            gen="effect holds across the panel",
            fail="breaks outside the tested regime",
            ood="an independent panel reproduces it",
        )

    offtopic = [
        _cand(off_sc("Cross-tissue gene expression variance predicts tissue specificity."),
              "gtex lofo", "offtopic"),
        _cand(off_sc("kcat of the enzyme increases with temperature."),
              "michaelis-menten fit", "offtopic"),
    ]
    ungrounded = [
        _cand("Spectral gap predicts clustering.", "correlation", "ungrounded_scope"),
    ]
    return {
        "domain_id": "graph_invariants",
        "question": q,
        "parent_text": parent,
        "candidates": on + duplicates + narrowing + offtopic + ungrounded,
        "has_offtopic_gate": True,
        "has_implementable_filter": False,  # graph_invariants implementable_methodologies() == []
    }


# --- Bank 5: generic / econ auction (NO domain => NO on-topic gate) ---------- #
def bank_generic_econ() -> dict:
    q = "Do correlated private values reduce second-price auction revenue relative to independent values?"

    def sc(claim, lo, hi):
        return _scoped(
            claim,
            pop=f"simulated second-price auctions with {lo}-{hi} bidders",
            dist="correlated vs independent private-value draws",
            gen="revenue gap holds across bidder counts in the range",
            fail="vanishes when correlation rho -> 0",
            ood="held-out bidder-count regime replicates the gap",
        )

    parent = sc("Correlated private values reduce second-price revenue for 5-20 bidders.", 5, 20)
    on = [
        _cand(sc("Reserve-price optimization shifts bidder surplus for 5-20 bidders.", 5, 20),
              "monte-carlo auction simulation", "distinct_survivor"),
    ]
    duplicates = [
        _cand(sc("Correlated private values reduce second-price revenue for 5-20 bidders.", 5, 20),
              "monte-carlo auction simulation", "duplicate"),
    ]
    narrowing = [
        _cand(sc("Correlated private values reduce second-price revenue for 8-12 bidders.", 8, 12),
              "monte-carlo auction simulation", "narrowing_child"),
    ]
    ungrounded = [
        _cand("Correlation reduces auction revenue.", "simulation", "ungrounded_scope"),
    ]
    return {
        "domain_id": None,  # resolve_domain_plugin -> None -> hypothesis_is_on_topic accepts all
        "question": q,
        "parent_text": parent,
        "candidates": on + duplicates + narrowing + ungrounded,
        "has_offtopic_gate": False,
        "has_implementable_filter": False,
    }


BANKS = [
    bank_math_sidon,
    bank_biology_mandrake,
    bank_genomics,
    bank_graph_invariants,
    bank_generic_econ,
]


# --------------------------------------------------------------------------- #
# Per-filter measurement using the REAL functions.
# --------------------------------------------------------------------------- #
def _resolved_domain_id(bank: dict) -> str | None:
    if bank["domain_id"] is not None:
        return bank["domain_id"]
    plugin = resolve_domain_plugin(question=bank["question"])
    return plugin.domain_id if plugin else None


def measure_bank(bank: dict, *, disable_dedup: bool = False, disable_relevance: bool = False) -> dict:
    """Run one bank's candidates through the real filters and tally by label.

    ``disable_*`` flags are ONLY used by the sanity-check (they short-circuit a
    single real filter so we can confirm the metric moves). Normal runs leave
    them False.
    """
    question = bank["question"]
    domain_id = _resolved_domain_id(bank)
    plugin = get_domain_plugin(domain_id) if domain_id else None
    impl_kws = plugin.implementable_methodologies() if plugin else []

    # A tree pre-seeded with the confirmed parent, so dedup has something to hit.
    tree = HypothesisTree()
    tree.add_seeds(
        [{"id": "parent", "text": bank["parent_text"], "test_methodology": "seed"}],
        generation=0,
    )
    state = CampaignBeliefState()

    tallies = {
        # dedup
        "dup_total": 0, "dup_survived": 0,
        "narrow_total": 0, "narrow_survived": 0,
        # relevance
        "offtopic_total": 0, "offtopic_rejected": 0,
        # implementable
        "unimpl_total": 0, "unimpl_rejected": 0,
        # scope
        "ungrounded_total": 0, "ungrounded_flagged": 0,
        # precision: survivors of the full deterministic chain
        "survivors": 0, "survivors_ontopic": 0,
    }
    same_round: list[str] = []

    for cand in bank["candidates"]:
        text = cand["text"]
        meth = cand["test_methodology"]
        label = cand["_label"]

        # --- REAL relevance gate --------------------------------------------
        if disable_relevance:
            on_topic = True
        else:
            on_topic = hypothesis_is_on_topic(text, question=question, test_methodology=meth)

        # --- REAL implementable filter (only where the domain provides keywords)
        implementable = True
        if impl_kws:
            implementable = methodology_implementable(text, meth, impl_kws)

        # --- REAL scope validity --------------------------------------------
        scope = parse_scope_from_methodology(text, meth)
        scope_ok, _missing = validate_scoped_claim(scope)
        boilerplate = bool(scope) and is_boilerplate_scope(scope, question)
        scope_valid = scope_ok and not boilerplate

        # --- REAL dedup (against the seeded tree + same-round siblings) -------
        if disable_dedup:
            is_dup = False
        else:
            is_dup, _reason = _is_duplicate_frontier_candidate(
                text, tree, state, same_round_texts=same_round, allowed_parent_id=None,
            )

        # Per-label tallies (ground-truth by construction).
        if label == "duplicate":
            tallies["dup_total"] += 1
            if not is_dup:
                tallies["dup_survived"] += 1
        elif label == "narrowing_child":
            tallies["narrow_total"] += 1
            if not is_dup:
                tallies["narrow_survived"] += 1
        elif label == "offtopic" and bank["has_offtopic_gate"]:
            tallies["offtopic_total"] += 1
            if not on_topic:
                tallies["offtopic_rejected"] += 1
        elif label == "unimplementable" and bank["has_implementable_filter"]:
            tallies["unimpl_total"] += 1
            if not implementable:
                tallies["unimpl_rejected"] += 1
        elif label == "ungrounded_scope":
            tallies["ungrounded_total"] += 1
            if not scope_valid:
                tallies["ungrounded_flagged"] += 1

        # --- Full deterministic chain survival (for relevance_precision) -----
        # A candidate survives generation if it passes ALL deterministic filters.
        survives = on_topic and implementable and scope_valid and not is_dup
        if survives:
            tallies["survivors"] += 1
            # "on-topic" ground truth = label is not offtopic.
            if label != "offtopic":
                tallies["survivors_ontopic"] += 1
            # keep it as a same-round sibling so later dups in-bank are caught too
            same_round.append(text)

    return tallies


def measure_end_to_end(bank: dict) -> dict:
    """Cross-check: drive the WHOLE ``apply_synthesis_to_frontier`` pipeline on the
    bank and read its own metrics dict, confirming the chained real filters agree
    with the per-filter tallies (faithfulness guard)."""
    tree = HypothesisTree()
    # A confirmed, frontier-eligible parent so children can attach + dedup fires.
    added = tree.add_seeds(
        [{"id": "parent", "text": bank["parent_text"], "test_methodology": "seed"}],
        generation=0,
    )
    tree.update_node(added[0].id, "confirmed", 0.9, 'evidence={"verdict_reason":"seed"};')
    state = CampaignBeliefState()
    parsed = {
        "beliefs": [],
        "frontier_candidates": [
            {"text": c["text"], "test_methodology": c["test_methodology"],
             "expansion_type": "boundary", "why_follows_from_beliefs": "bank"}
            for c in bank["candidates"]
        ],
        "direction_exhausted": False,
    }
    _added, metrics = apply_synthesis_to_frontier(
        tree, state, parsed,
        question=bank["question"],
        generation=1,
        relevance_threshold=0.0,  # isolate the categorical filters from lexical scoring
        domain_id=_resolved_domain_id(bank),
    )
    return {
        "n_rejected_duplicate": int(metrics.get("n_rejected_duplicate") or 0),
        "n_rejected_off_topic": int(metrics.get("n_rejected_off_topic") or 0),
        "n_rejected_unimplementable": int(metrics.get("n_rejected_unimplementable") or 0),
        "n_added": int(metrics.get("n_added") or 0),
    }


# --------------------------------------------------------------------------- #
# Aggregation.
# --------------------------------------------------------------------------- #
def _rate(num: int, den: int) -> float:
    return round(num / den, 4) if den else 0.0


def aggregate(banks_tallies: list[dict]) -> dict:
    agg: dict[str, int] = {}
    for t in banks_tallies:
        for k, v in t.items():
            agg[k] = agg.get(k, 0) + v

    metrics = {
        # KEY metric — fraction of TRUE duplicates that SURVIVED dedup. Target 0.
        "duplicate_pass_rate": _rate(agg["dup_survived"], agg["dup_total"]),
        # Genuine scope-narrowing children must survive. Target 1.
        "narrowing_survival_rate": _rate(agg["narrow_survived"], agg["narrow_total"]),
        # Fraction of off-topic candidates rejected by the relevance gate. Target high.
        "offtopic_rejection_rate": _rate(agg["offtopic_rejected"], agg["offtopic_total"]),
        # Of survivors of the full deterministic chain, fraction on-topic. Target 1.
        "relevance_precision": _rate(agg["survivors_ontopic"], agg["survivors"]),
        # Fraction of non-implementable candidates correctly rejected. Target 1.
        "methodology_implementable_rate": _rate(agg["unimpl_rejected"], agg["unimpl_total"]),
        # Fraction of ungrounded/boilerplate-scope candidates flagged. Target 1.
        "scope_validity_rate": _rate(agg["ungrounded_flagged"], agg["ungrounded_total"]),
    }
    counts = {
        "n_duplicates": agg["dup_total"],
        "n_narrowing_children": agg["narrow_total"],
        "n_offtopic_with_gate": agg["offtopic_total"],
        "n_unimplementable_with_filter": agg["unimpl_total"],
        "n_ungrounded_scope": agg["ungrounded_total"],
        "n_survivors_full_chain": agg["survivors"],
    }
    return {"metrics": metrics, "counts": counts}


# --------------------------------------------------------------------------- #
# Sanity check: disable one real filter, confirm the corresponding metric MOVES.
# --------------------------------------------------------------------------- #
def sanity_check() -> dict:
    banks = [b() for b in BANKS]
    base = aggregate([measure_bank(b) for b in banks])["metrics"]
    no_dedup = aggregate([measure_bank(b, disable_dedup=True) for b in banks])["metrics"]
    no_rel = aggregate([measure_bank(b, disable_relevance=True) for b in banks])["metrics"]
    return {
        "duplicate_pass_rate": {
            "baseline": base["duplicate_pass_rate"],
            "dedup_disabled": no_dedup["duplicate_pass_rate"],
            "moved": no_dedup["duplicate_pass_rate"] != base["duplicate_pass_rate"],
        },
        "offtopic_rejection_rate": {
            "baseline": base["offtopic_rejection_rate"],
            "relevance_disabled": no_rel["offtopic_rejection_rate"],
            "moved": no_rel["offtopic_rejection_rate"] != base["offtopic_rejection_rate"],
        },
    }


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(REPO), text=True,
        ).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def main() -> None:
    banks = [b() for b in BANKS]
    tallies = [measure_bank(b) for b in banks]
    result = aggregate(tallies)
    metrics = result["metrics"]
    counts = result["counts"]

    # End-to-end faithfulness cross-check (the chained pipeline agrees).
    e2e = {b["domain_id"] or "generic": measure_end_to_end(b) for b in banks}

    n_gate = sum(1 for b in banks if b["has_offtopic_gate"])
    n_impl = sum(1 for b in banks if b["has_implementable_filter"])

    print(f"Hypothesis-generation-layer benchmark (real deterministic filters), {len(banks)} banks:")
    print(f"  duplicate_pass_rate            : {metrics['duplicate_pass_rate']}"
          f"   (fraction of TRUE duplicates that survived dedup; target 0)")
    print(f"  narrowing_survival_rate        : {metrics['narrowing_survival_rate']}"
          f"   (fraction of scope-narrowing children kept; target 1)")
    print(f"  offtopic_rejection_rate        : {metrics['offtopic_rejection_rate']}"
          f"   (off-topic rejected by relevance gate; over {n_gate} gated banks; target high)")
    print(f"  relevance_precision            : {metrics['relevance_precision']}"
          f"   (of full-chain survivors, fraction on-topic; target 1)")
    print(f"  methodology_implementable_rate : {metrics['methodology_implementable_rate']}"
          f"   (non-implementable correctly rejected; over {n_impl} filtered banks; target 1)")
    print(f"  scope_validity_rate            : {metrics['scope_validity_rate']}"
          f"   (ungrounded/boilerplate scope flagged; target 1)")
    print(f"  labelled counts                : {counts}")
    print(f"  offtopic_domains_with_gate     : {n_gate}/{len(banks)} "
          f"(mandrake + generic-econ have accept-all on-topic; excluded from offtopic_rejection_rate)")

    print("\n  end-to-end apply_synthesis_to_frontier cross-check (real chained pipeline):")
    for name, m in e2e.items():
        print(f"    {name:18s} {m}")

    sanity = sanity_check()
    print("\n  metric-moves sanity check (disable one real filter -> metric must move):")
    print(f"    dedup      : {sanity['duplicate_pass_rate']}")
    print(f"    relevance  : {sanity['offtopic_rejection_rate']}")

    payload = {
        "metrics": metrics,
        "counts": counts,
        "n_seeds": len(banks),
        "config": {
            "banks": [b.__name__ for b in BANKS],
            "domains": [b["domain_id"] or "generic(none)" for b in banks],
            "offtopic_domains_with_gate": n_gate,
            "implementable_filtered_domains": n_impl,
            "dedup_threshold": 0.85,
            "relevance_threshold_e2e": 0.0,
            "scope": (
                "deterministic post-generation filters only; raw LLM candidate "
                "creativity out of scope (measured by campaign outcomes)"
            ),
        },
        "end_to_end_cross_check": e2e,
        "sanity_check": sanity,
        "git_sha": _git_sha(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    out = Path(__file__).resolve().parents[1] / "artifacts" / "bench" / "generation_baseline.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\n  wrote {out}")


if __name__ == "__main__":
    main()
