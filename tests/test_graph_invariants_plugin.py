"""Tests for graph invariants plugin."""
from __future__ import annotations

import itertools

import numpy as np
import pytest

from propab import config
from propab.domain_modules.graph_invariants.adapter import (
    KNOWN_INVARIANTS,
    GraphInvariantNotIdentified,
    GraphInvariantSpec,
    _synthetic_frame,
)
from propab.domain_modules.graph_invariants.plugin import GraphInvariantsPlugin
from propab.domain_modules.graph_invariants.routing_inspector import inspect_corpus, inspect_routing
from propab.domain_modules.registry import get_domain_plugin


@pytest.fixture
def tmp_data(monkeypatch, tmp_path):
    monkeypatch.setattr(config.settings, "propab_data_dir", str(tmp_path))


def test_graph_plugin_registered():
    assert get_domain_plugin("graph_invariants") is not None


def test_graph_preflight(tmp_data):
    r = GraphInvariantsPlugin().preflight()
    assert r.passed, r.reason


def test_graph_routing_corpus(tmp_data):
    report = inspect_corpus()
    assert report["routing_mismatches"] == 0
    assert report["total"] >= 20


# ── DOM4: from_hypothesis refuses when NO invariant is identified ─────────────

def test_from_hypothesis_refuses_non_graph_text():
    """Off-topic text names no graph invariant → raise, so the caller can map it
    to inconclusive instead of silently defaulting to spectral_gap->clustering."""
    for text in (
        "Sidon set density conjecture for n=100",
        "gene expression predicts kcat in enzymes",
        "some unrelated statement about the weather today",
    ):
        with pytest.raises(GraphInvariantNotIdentified):
            GraphInvariantSpec.from_hypothesis({"text": text})


def test_from_hypothesis_no_longer_silently_defaults():
    """The old code returned spectral_gap->clustering_coefficient for keyword-free
    text; that silent default must be gone."""
    with pytest.raises(GraphInvariantNotIdentified):
        GraphInvariantSpec.from_hypothesis({"text": "nothing to see here"})


def test_from_hypothesis_two_named_invariants():
    spec = GraphInvariantSpec.from_hypothesis(
        {"text": "modularity tracks clustering coefficient across families"}
    )
    assert spec.source_invariant == "modularity"
    assert spec.target_invariant == "clustering_coefficient"


def test_from_hypothesis_single_named_invariant_pairs_distinct():
    """A single-invariant claim resolves (never a self-correlation)."""
    spec = GraphInvariantSpec.from_hypothesis(
        {"text": "spectral gap exceeds 0.1 for all network families"}
    )
    assert spec.source_invariant == "spectral_gap"
    assert spec.target_invariant != spec.source_invariant
    assert spec.target_invariant in KNOWN_INVARIANTS


def test_inspect_routing_off_topic_is_not_ok():
    report = inspect_routing({"id": "x", "statement": "Sidon set density for n=100"})
    assert report["routing_ok"] is False
    assert report["resolved_invariants"] == []


# ── DOM2b: modularity is not a deterministic function of clustering ───────────

def _corr(a: np.ndarray, b: np.ndarray) -> float:
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def test_modularity_not_deterministic_function_of_clustering(tmp_data):
    """DOM2b: modularity was ``0.25*clustering + 0.1*(avg_deg/n)`` — a closed-form
    function of clustering (|corr| == 1 within any family). The real Newman
    modularity must decouple from clustering: no family shows a near-perfect
    correlation."""
    df = _synthetic_frame()
    for fam in df["network_family"].unique():
        sub = df[df["network_family"] == fam]
        r = _corr(
            sub["modularity"].to_numpy(float),
            sub["clustering_coefficient"].to_numpy(float),
        )
        assert abs(r) < 0.99, f"{fam}: modularity~clustering r={r}"


def test_no_exposed_invariant_pair_is_deterministic(tmp_data):
    """No exposed-invariant pair may be a deterministic function of another:
    within EVERY network family, |corr| must stay clear of 1 for all pairs.
    Also guards the old spectral_gap == algebraic_connectivity duplicate."""
    df = _synthetic_frame()
    invariants = list(KNOWN_INVARIANTS)
    for a, b in itertools.combinations(invariants, 2):
        for fam in df["network_family"].unique():
            sub = df[df["network_family"] == fam]
            r = _corr(sub[a].to_numpy(float), sub[b].to_numpy(float))
            assert abs(r) < 0.99, f"{a}~{b} on {fam}: |corr|={abs(r):.4f} (deterministic?)"


def test_spectral_gap_and_algebraic_connectivity_are_distinct(tmp_data):
    df = _synthetic_frame()
    assert not np.allclose(
        df["spectral_gap"].to_numpy(float),
        df["algebraic_connectivity"].to_numpy(float),
    )


# ── DOM4: plugin on-topic gate rejects cross-domain leakage ───────────────────

def test_plugin_hypothesis_on_topic_gate():
    p = GraphInvariantsPlugin()
    assert p.hypothesis_on_topic("Sidon set density for n=100") is False
    assert p.hypothesis_on_topic("gene expression predicts kcat") is False
    assert (
        p.hypothesis_on_topic("spectral gap correlates with clustering coefficient")
        is True
    )
