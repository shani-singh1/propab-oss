"""
Tests for the network-diffusion domain plugin.

These exercise REAL SNAP graphs (``data/v1_candidates/ca-GrQc.txt.gz`` and
``email-Eu-core.txt.gz``). The data dir is gitignored and lives in the main
worktree, so :func:`_real_data_dir` locates it (this worktree, or the repo root
via git-common-dir) and the whole module is skipped when it is unavailable —
never silently passing on missing data.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from propab import config
from propab.domain_modules.network_diffusion.adapter import (
    STRUCTURAL_FEATURES,
    DiffusionSpec,
    sample_subgraphs,
    structural_features,
)
from propab.domain_modules.network_diffusion.plugin import NetworkDiffusionPlugin
from propab.domain_modules.network_diffusion.routing_inspector import inspect_corpus
from propab.domain_modules.network_diffusion.simulator import simulate
from propab.domain_modules.network_diffusion.verifier import (
    _spearman,
    classify_diffusion_verdict,
    run_diffusion_experiment,
)
from propab.domain_modules.registry import get_domain_plugin, resolve_domain_plugin

REQUIRED_FILES = ("ca-GrQc.txt.gz", "email-Eu-core.txt.gz")


def _real_data_dir() -> Path | None:
    """Locate the directory holding v1_candidates SNAP edge lists, or None."""
    candidates: list[Path] = []
    here = Path(__file__).resolve()
    # 1. this worktree's own data/
    candidates.append(here.parents[1] / "data")
    # 2. the main worktree via the shared git dir (worktrees share .git)
    git_dir = here.parents[1] / ".git"
    if git_dir.is_file():
        text = git_dir.read_text(encoding="utf-8", errors="replace")
        if text.startswith("gitdir:"):
            common = Path(text.split(":", 1)[1].strip())
            # gitdir points at .../.git/worktrees/<name>; the main repo is its
            # great-grandparent (../../.. -> repo root), which holds data/.
            main_root = common.parents[2] if len(common.parents) >= 3 else None
            if main_root is not None:
                candidates.append(main_root / "data")
    elif git_dir.is_dir():
        candidates.append(git_dir.parent / "data")
    for base in candidates:
        vc = base / "v1_candidates"
        if all((vc / f).is_file() for f in REQUIRED_FILES):
            return base
    return None


_DATA_DIR = _real_data_dir()
pytestmark = pytest.mark.skipif(
    _DATA_DIR is None,
    reason="SNAP edge lists (ca-GrQc, email-Eu-core) not found under data/v1_candidates",
)


@pytest.fixture
def real_data(monkeypatch):
    monkeypatch.setattr(config.settings, "propab_data_dir", str(_DATA_DIR))


# --- registration & contract -----------------------------------------------
def test_network_diffusion_registered():
    assert get_domain_plugin("network_diffusion") is not None


def test_available_features_nonempty():
    feats = NetworkDiffusionPlugin().available_features()
    assert feats and set(feats) == set(STRUCTURAL_FEATURES)


def test_uses_real_data_not_synthetic():
    assert NetworkDiffusionPlugin().uses_synthetic_data() is False


def test_confirmation_criteria_requires_holdout_and_null():
    crit = NetworkDiffusionPlugin().confirmation_criteria()
    assert crit["requires_holdout"] is True
    assert crit["holdout_type"] == "leave_network_family_out"
    assert crit["null_test"]


# --- routing discrimination -------------------------------------------------
def test_matches_contagion_question():
    p = NetworkDiffusionPlugin()
    assert p.matches(question="Does degree heterogeneity increase SIR epidemic outbreak size in a network?")
    assert p.matches(question="independent cascade adoption on a social graph")


def test_does_not_match_static_graph_invariant_question():
    p = NetworkDiffusionPlugin()
    # Static invariant question (no diffusion) must not be claimed by this domain.
    assert not p.matches(question="Does spectral gap correlate with clustering coefficient across graph families?")


def test_routing_prefers_diffusion_over_graph_invariants():
    got = resolve_domain_plugin(
        question="Does degree heterogeneity increase SIR epidemic outbreak size across networks?"
    )
    assert got is not None and got.domain_id == "network_diffusion"


def test_hypothesis_on_topic_rejects_other_domains():
    p = NetworkDiffusionPlugin()
    assert p.hypothesis_on_topic("SIR contagion spreading on a network")
    assert not p.hypothesis_on_topic("kcat of enzyme across EC classes")


def test_routing_corpus_no_mismatches():
    report = inspect_corpus()
    assert report["routing_mismatches"] == 0
    assert report["total"] >= 15


# --- real data / simulator sanity ------------------------------------------
def test_preflight_loads_real_graphs(real_data):
    r = NetworkDiffusionPlugin().preflight()
    assert r.passed, r.reason
    assert r.details.get("collaboration_nodes", 0) > 1000
    assert r.details.get("email_nodes", 0) > 500


def test_subgraphs_are_real_and_simulation_runs(real_data):
    rng = np.random.default_rng(0)
    subs = sample_subgraphs("collaboration", n_samples=3, target_size=90, rng=rng)
    assert subs, "expected at least one real subgraph"
    sub = subs[0]
    assert sub.n_nodes >= 25 and sub.n_edges >= 30
    feats = structural_features(sub)
    assert set(feats) == set(STRUCTURAL_FEATURES)
    frac = simulate(sub, simulator="sir", outcome="final_size", beta=0.2, gamma=0.5, n_runs=5, rng=rng)
    assert 0.0 <= frac <= 1.0


# --- the actual verifier: confirm / refute + null discipline ----------------
def test_epidemic_threshold_law_confirmed(real_data):
    """The canonical <k^2>/<k> -> outbreak law confirms and survives the null."""
    spec = DiffusionSpec(structural_feature="k2_over_k1", outcome="final_size", simulator="sir")
    res = run_diffusion_experiment(spec)
    assert res["uses_synthetic_data"] is False
    assert res["cross_family_replicates"] is True
    assert res["robust_to_simulator"] is True
    # Emits the exact lofo/null triple the artifact gate reads.
    assert res["lofo_r2"] > res["label_shuffle_null_p95"]
    assert res["label_shuffle_permutation_p"] < 0.05
    assert res["verified_true_steps"] == 1
    verdict, _rationale, conf = classify_diffusion_verdict("", res)
    assert verdict == "confirmed"
    assert conf > 0.8


def test_null_rejects_broken_signal(real_data):
    """
    A structural feature that does NOT consistently drive diffusion must not be
    confirmed — the null/replication guard must reject it. degree_gini flips sign
    across the two real families, so it cannot be 'confirmed'.
    """
    spec = DiffusionSpec(structural_feature="degree_gini", outcome="final_size", simulator="sir")
    res = run_diffusion_experiment(spec)
    verdict, _r, _c = classify_diffusion_verdict("", res)
    assert verdict != "confirmed", res["by_family"]
    assert res["verified_true_steps"] == 0


def test_perm_p_zero_is_not_treated_as_failure():
    """
    Regression: perm_p == 0.0 (null never beat the observed correlation) is the
    STRONGEST evidence and must not be swallowed by a falsy ``or`` default.
    """
    res = {
        "held_out_correlation": 0.74,
        "label_shuffle_permutation_p": 0.0,
        "cross_family_replicates": True,
        "robust_to_simulator": True,
        "verified_true_steps": 1,
        "simulator": "sir",
    }
    verdict, rationale, _c = classify_diffusion_verdict("", res)
    assert verdict == "confirmed", rationale
    assert "shuffle p=0.000" in rationale


# --- pure-unit checks (no data) --------------------------------------------
def test_spearman_matches_known_values():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert _spearman(x, x) == pytest.approx(1.0)
    assert _spearman(x, x[::-1]) == pytest.approx(-1.0)


def test_from_hypothesis_defaults_to_epidemic_moment():
    spec = DiffusionSpec.from_hypothesis({"text": "does degree heterogeneity drive contagion?"})
    assert spec.structural_feature == "k2_over_k1"
    spec2 = DiffusionSpec.from_hypothesis({"text": "independent cascade adoption vs degree gini inequality"})
    assert spec2.simulator == "cascade"
    assert spec2.structural_feature == "degree_gini"
