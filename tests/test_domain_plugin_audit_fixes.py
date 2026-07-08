"""Regression tests for latent domain-plugin bugs found in the audit.

Two independent fixes, each with a fail-before / pass-after guard:

1. network_diffusion routing under-scored itself. It was the only content-routed
   plugin without a ``match_score`` override, so it scored a bare 1.0 while
   graph_invariants scored its marker count. A question dense in diffusion
   vocabulary that ALSO carried >=2 static graph-invariant markers therefore
   misrouted to graph_invariants (score 2 > 1) on registration order rather than
   to the diffusion domain that actually owns it.

2. graph_invariants confirmed on a bare correlation threshold, IGNORING the
   within-family shuffle null it computes. A spurious pair that clears the
   threshold but sits inside the null distribution still emitted
   verified_true_steps=1 / discovery_worthy=True — a fail-open every sibling
   statistical domain (genomics/enzyme/network_diffusion) closes by requiring the
   null.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from propab.domain_modules.graph_invariants.adapter import GraphInvariantsAdapter, GraphInvariantSpec
from propab.domain_modules.graph_invariants.plugin import GraphInvariantsPlugin
from propab.domain_modules.graph_invariants.verifier import (
    _family_correlation,
    _label_shuffle_null,
    run_graph_invariant_check,
)
from propab.domain_modules.network_diffusion.plugin import NetworkDiffusionPlugin
from propab.domain_modules.registry import resolve_domain_plugin


# ── Fix 1: network_diffusion vs graph_invariants routing collision ────────────

# A genuinely diffusion question that also happens to name two static graph
# invariants ("spectral gap", "clustering coefficient").
_COLLISION_Q = (
    "Does the spectral gap and clustering coefficient of a network predict SIR "
    "epidemic outbreak size and cascade spreading?"
)


def test_network_diffusion_has_marker_counting_match_score():
    """network_diffusion must score by diffusion-marker count, not a bare 1.0,
    so it can outrank a colliding graph question."""
    n = NetworkDiffusionPlugin()
    g = GraphInvariantsPlugin()
    assert n.matches(question=_COLLISION_Q)
    assert g.matches(question=_COLLISION_Q)
    # More diffusion markers than graph markers in this question.
    assert n.match_score(question=_COLLISION_Q) > g.match_score(question=_COLLISION_Q)


def test_diffusion_heavy_collision_routes_to_network_diffusion():
    """The exact fail-before case: 2 graph markers + several diffusion markers.
    With the old bare-1.0 score, graph_invariants (score 2) stole this; now the
    diffusion-marker count wins."""
    plugin = resolve_domain_plugin(question=_COLLISION_Q)
    assert plugin is not None and plugin.domain_id == "network_diffusion"


def test_pure_graph_invariant_question_still_routes_to_graph_invariants():
    """No regression: a static-invariant question with no diffusion vocabulary
    still resolves to graph_invariants."""
    plugin = resolve_domain_plugin(
        question="Does spectral gap correlate with clustering coefficient across "
        "network families using modularity?"
    )
    assert plugin is not None and plugin.domain_id == "graph_invariants"


# ── Fix 2: graph_invariants verdict must be gated by the shuffle null ─────────

def _spurious_frame() -> pd.DataFrame:
    """A frame whose held family passes the bare correlation threshold
    (held_corr>0.05, train_corr>0.15) but does NOT survive the label-shuffle null.
    Deterministic (fixed seed)."""
    rng = np.random.default_rng(0)
    tsrc = rng.normal(size=30)
    ttgt = tsrc + 0.05 * rng.normal(size=30)  # near-perfect train correlation
    hsrc = rng.normal(size=9)
    htgt = 0.5 * hsrc + 1.6 * rng.normal(size=9)  # weak, noise-dominated held pair
    return pd.DataFrame(
        {
            "network_family": ["train"] * 30 + ["held"] * 9,
            "src": list(tsrc) + list(hsrc),
            "tgt": list(ttgt) + list(htgt),
        }
    )


def test_spurious_pair_precondition_bare_threshold_would_confirm():
    """Sanity: the constructed pair clears the OLD bare threshold (so the fix is
    what changes the verdict, not the data)."""
    df = _spurious_frame()
    train_corr = _family_correlation(df, "train", "src", "tgt")
    held_corr = _family_correlation(df, "held", "src", "tgt")
    assert train_corr > 0.15 and held_corr > 0.05
    # ...but it does NOT survive the null.
    p95, perm_p = _label_shuffle_null(df, "held", "src", "tgt", abs(held_corr))
    assert not (abs(held_corr) > p95 and perm_p < 0.05)


def test_graph_verdict_fails_closed_when_null_not_survived(monkeypatch):
    """FAIL-BEFORE / PASS-AFTER: a pair that clears the bare threshold but sits
    inside the shuffle null must NOT be confirmed. Before the fix this emitted
    verified_true_steps=1 / discovery_worthy=True."""
    df = _spurious_frame()
    monkeypatch.setattr(GraphInvariantsAdapter, "load_frame", lambda self: df)
    r = run_graph_invariant_check(
        GraphInvariantSpec(
            source_invariant="src",
            target_invariant="tgt",
            held_out_family="held",
            claim_type="correlation_positive",
        )
    )
    assert r["verified_true_steps"] == 0
    assert r["discovery_worthy"] is False
    assert r["trivial_rediscovery"] is True
    # Still emits the lofo/null triple so the pipeline routes it as "lofo".
    assert "label_shuffle_permutation_p" in r and r.get("lofo_r2") is not None


def test_graph_verdict_confirms_genuine_pair_that_survives_null(monkeypatch):
    """A genuine held pair that both clears the threshold AND survives the null is
    still confirmed — the fix does not over-reject."""
    rng = np.random.default_rng(3)
    tsrc = rng.normal(size=30)
    ttgt = tsrc + 0.05 * rng.normal(size=30)
    hsrc = rng.normal(size=30)
    htgt = hsrc + 0.05 * rng.normal(size=30)  # strong, real held correlation
    df = pd.DataFrame(
        {
            "network_family": ["train"] * 30 + ["held"] * 30,
            "src": list(tsrc) + list(hsrc),
            "tgt": list(ttgt) + list(htgt),
        }
    )
    monkeypatch.setattr(GraphInvariantsAdapter, "load_frame", lambda self: df)
    r = run_graph_invariant_check(
        GraphInvariantSpec(
            source_invariant="src",
            target_invariant="tgt",
            held_out_family="held",
            claim_type="correlation_positive",
        )
    )
    assert r["verified_true_steps"] == 1
    assert r["discovery_worthy"] is True
    assert r["label_shuffle_permutation_p"] < 0.05
