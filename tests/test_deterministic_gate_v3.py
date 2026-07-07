"""V3: close the `deterministic` gate-bypass hole + graph_invariants real null.

Before V3, `classify_evidence_type` labelled ANY result with a verified_true
counter and a non-`{None,"","significance"}` method name as "deterministic",
which bypasses the artifact gate. graph_invariants exploited this by emitting
`verification_method="cross_network_lofo"` with `verified_true_steps=1` — a
statistical correlation on synthetic data got a free confirm with NO adversarial
null. V3 requires an EXPLICIT proof method (or `deterministic=True`) to earn the
gate-bypass; everything else falls to "unknown"/"lofo"/"statistical" and is gated.

graph_invariants now also emits a real label-shuffle null in the lofo shape, so a
genuine correlation is confirmed only when it survives the null and a spurious one
is downgraded.
"""
from __future__ import annotations

import numpy as np
import pytest

from propab.domain_modules.graph_invariants.adapter import real_graph_data_available
from propab.verdict_pipeline import artifact_gate_stage, classify_evidence_type

# Skip the graph-verifier test cleanly (not ERROR) when the git-ignored SNAP data is
# absent (CI / fresh checkout before scripts/fetch_graph_datasets.py runs).
_NEEDS_GRAPH_DATA = pytest.mark.skipif(
    not real_graph_data_available(),
    reason="real SNAP graph data not cached; run scripts/fetch_graph_datasets.py",
)


# ---- V3 classifier: the gate-bypass hole is closed -------------------------

def test_bare_counter_with_nonproof_method_is_not_deterministic():
    # The exact shape graph_invariants used to exploit — no null, no proof method.
    ev = {"verified_true_steps": 1, "verified_false_steps": 0,
          "verification_method": "cross_network_lofo"}
    assert classify_evidence_type(ev) == "unknown"  # was "deterministic" (free confirm)


def test_unknown_confirm_is_gated_to_inconclusive():
    ev = {"verified_true_steps": 1, "verified_false_steps": 0,
          "verification_method": "cross_network_lofo"}
    verdict, _conf, _reason = artifact_gate_stage(dict(ev), "confirmed", 0.9, "prelim")
    assert verdict == "inconclusive"


def test_explicit_proof_method_still_deterministic():
    ev = {"verified_true_steps": 1, "verified_false_steps": 0,
          "verification_method": "combinatorial_computation"}
    assert classify_evidence_type(ev) == "deterministic"


def test_explicit_deterministic_flag_still_deterministic():
    assert classify_evidence_type({"deterministic": True, "verified_true_steps": 1}) == "deterministic"


def test_deterministic_confirm_passes_gate_unchanged():
    ev = {"deterministic": True, "verified_true_steps": 1,
          "verification_method": "combinatorial_computation"}
    verdict, _conf, _reason = artifact_gate_stage(dict(ev), "confirmed", 0.95, "prelim")
    assert verdict == "confirmed"


# ---- graph_invariants: real adversarial null in lofo shape -----------------

@_NEEDS_GRAPH_DATA
def test_graph_verifier_emits_lofo_null_and_routes_lofo():
    from propab.domain_modules.graph_invariants.adapter import GraphInvariantSpec
    from propab.domain_modules.graph_invariants.verifier import run_graph_invariant_check

    r = run_graph_invariant_check(
        GraphInvariantSpec(
            source_invariant="clustering_coefficient",
            target_invariant="modularity",
            held_out_family=None,
            claim_type="correlation_positive",
        )
    )
    # Emits the lofo-shaped null the artifact gate reads → routed as "lofo", gated.
    assert "label_shuffle_null_p95" in r and "label_shuffle_permutation_p" in r
    assert r.get("lofo_r2") is not None
    assert classify_evidence_type(r) == "lofo"


def test_graph_null_fails_closed_on_no_relationship():
    # A frame with NO src->tgt relationship must not survive the null: the observed
    # |corr| should sit inside the shuffle-null distribution (perm_p not tiny).
    import pandas as pd

    from propab.domain_modules.graph_invariants.verifier import _label_shuffle_null

    rng = np.random.default_rng(1)
    df = pd.DataFrame({
        "network_family": ["fam"] * 40,
        "src": rng.normal(size=40),
        "tgt": rng.normal(size=40),  # independent of src
    })
    observed = abs(float(np.corrcoef(df["src"], df["tgt"])[0, 1]))
    out = _label_shuffle_null(df, "fam", "src", "tgt", observed)
    assert out is not None
    p95, perm_p = out
    # No real relationship → observed does not clear the null p95 with a tiny p.
    assert not (observed > p95 and perm_p < 0.05)
