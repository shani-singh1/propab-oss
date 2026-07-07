"""Tests for graph invariants plugin (REAL SNAP networks)."""
from __future__ import annotations

import itertools

import numpy as np
import pytest

from propab import config
from propab.domain_modules.graph_invariants.adapter import (
    GRAPH_FAMILIES,
    KNOWN_INVARIANTS,
    REAL_NETWORKS,
    GraphInvariantNotIdentified,
    GraphInvariantSpec,
    GraphInvariantsAdapter,
    _real_frame,
    _source_dir,
    real_graph_data_available,
)

# Skip real-data tests cleanly (not ERROR) when the git-ignored SNAP data is absent —
# e.g. on CI / a fresh checkout before scripts/fetch_graph_datasets.py has run.
_NEEDS_GRAPH_DATA = pytest.mark.skipif(
    not real_graph_data_available(),
    reason="real SNAP graph data not cached; run scripts/fetch_graph_datasets.py",
)
from propab.domain_modules.graph_invariants.plugin import GraphInvariantsPlugin
from propab.domain_modules.graph_invariants.routing_inspector import inspect_corpus, inspect_routing
from propab.domain_modules.graph_invariants.verifier import (
    _family_correlation,
    _label_shuffle_null,
)
from propab.domain_modules.registry import get_domain_plugin


@pytest.fixture
def tmp_data(monkeypatch, tmp_path):
    monkeypatch.setattr(config.settings, "propab_data_dir", str(tmp_path))


@pytest.fixture(scope="module")
def real_frame():
    """Build the REAL-network invariant frame once and share it across tests.

    ``_real_frame`` reads the cached SNAP edge lists directly (independent of
    ``propab_data_dir``), so it is safe to build without the tmp_data fixture.
    Module-scoped because the snowball sampling + eigendecomposition is a few
    seconds of work we don't want to repeat per test.
    """
    if not real_graph_data_available():
        pytest.skip("real SNAP graph data not cached; run scripts/fetch_graph_datasets.py")
    return _real_frame()


def test_graph_plugin_registered():
    assert get_domain_plugin("graph_invariants") is not None


@_NEEDS_GRAPH_DATA
def test_graph_preflight(tmp_data):
    r = GraphInvariantsPlugin().preflight()
    assert r.passed, r.reason


@_NEEDS_GRAPH_DATA
def test_graph_routing_corpus(tmp_data):
    report = inspect_corpus()
    assert report["routing_mismatches"] == 0
    assert report["total"] >= 20


# ── Real data: the SNAP edge lists are present and load into real networks ────

@_NEEDS_GRAPH_DATA
def test_real_snap_files_present_on_disk():
    """Every configured real network file must exist on disk — no fabrication."""
    src = _source_dir()
    for _family, filename, _desc, _url in REAL_NETWORKS:
        assert (src / filename).is_file(), f"missing real SNAP file: {filename}"


@_NEEDS_GRAPH_DATA
def test_plugin_reports_real_data():
    """uses_synthetic_data() must be False now that real graphs are wired."""
    assert GraphInvariantsPlugin().uses_synthetic_data() is False


def test_real_frame_has_three_real_families(real_frame):
    fams = sorted(real_frame["network_family"].unique())
    # THREE distinct real topology classes: collaboration (co-authorship),
    # communication (email), and infrastructure (road / near-planar spatial mesh).
    assert (
        fams
        == sorted(GRAPH_FAMILIES)
        == ["collaboration", "communication", "infrastructure"]
    )
    # Each family contributes multiple real subgraph instances for the LOFO.
    for fam in fams:
        assert (real_frame["network_family"] == fam).sum() >= 20


@_NEEDS_GRAPH_DATA
def test_real_frame_meta_records_real_provenance(tmp_data):
    """The adapter cache meta must say synthetic=False / provenance=real and cite
    the SNAP sources — never relabel real data or leave a stale synthetic flag."""
    import json

    adapter = GraphInvariantsAdapter()
    adapter.ensure_cache()
    meta_path = adapter.ensure_cache().parent / "snap_subset_v1.meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta["synthetic"] is False
    assert meta["data_provenance"] == "real"
    urls = {s["url"] for s in meta["sources"]}
    assert any("snap.stanford.edu" in u for u in urls)


def test_invariants_computed_and_finite(real_frame):
    """All six invariants are computed on the real subgraphs and are finite,
    non-degenerate (each varies across the sampled real subgraphs)."""
    for inv in KNOWN_INVARIANTS:
        col = real_frame[inv].to_numpy(float)
        assert np.all(np.isfinite(col)), f"{inv} has non-finite values"
        assert np.std(col) > 0.0, f"{inv} is constant across real subgraphs"


# ── V3 null: spurious pair does NOT survive; genuine pair does ────────────────

def test_genuine_invariant_pair_survives_null(real_frame):
    """clustering_coefficient -> avg_degree is a genuine structural relationship that
    replicates across ALL THREE real families: the held-out |corr| must exceed the
    label-shuffle null's p95 (and be permutation-significant) in EACH held-out family
    (collaboration, communication, AND the near-planar power-grid/infrastructure
    network). Denser induced subgraphs pack in both a higher average degree and more
    closed triangles (higher transitivity), and — unlike spectral_gap->avg_degree,
    whose coupling collapses on the sparse grid subgraphs — this coupling survives the
    LOFO on every one of the three distinct topology classes with a consistent
    positive sign."""
    src, tgt = "clustering_coefficient", "avg_degree"
    for held in GRAPH_FAMILIES:
        held_corr = _family_correlation(real_frame, held, src, tgt)
        null = _label_shuffle_null(real_frame, held, src, tgt, abs(held_corr))
        assert null is not None, f"null could not be built for {held}"
        p95, perm_p = null
        assert abs(held_corr) > p95, (
            f"{src}->{tgt} on {held}: |r|={abs(held_corr):.3f} did not clear "
            f"null p95={p95:.3f}"
        )
        assert perm_p < 0.05, f"{src}->{tgt} on {held}: perm_p={perm_p:.3f} not significant"


def test_spurious_invariant_pair_does_not_survive_null(real_frame):
    """diameter -> modularity is a spurious cross-family pair on these real networks:
    a textbook-plausible "compact graphs are less modular" link that does NOT
    replicate across the three real topology classes. It must FAIL the label-shuffle
    null (|corr| below p95, or non-significant perm_p) in EVERY family — including the
    power-grid/infrastructure family — so a non-relationship cannot be waved through
    as a confirmed discovery. This is exactly the kind of real-data refutation the
    cross-family LOFO exists to catch, and adding the third (infrastructure) family
    only makes the null harder to spuriously beat."""
    src, tgt = "diameter", "modularity"
    survived_any = False
    for held in GRAPH_FAMILIES:
        held_corr = _family_correlation(real_frame, held, src, tgt)
        null = _label_shuffle_null(real_frame, held, src, tgt, abs(held_corr))
        assert null is not None
        p95, perm_p = null
        if abs(held_corr) > p95 and perm_p < 0.05:
            survived_any = True
    assert not survived_any, (
        "spurious diameter->modularity pair unexpectedly survived the null in a "
        "real family — the adversarial null is not discriminating"
    )


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


def test_from_hypothesis_resolves_real_held_out_family():
    """Naming a real family in the text pins it as the held-out family."""
    spec = GraphInvariantSpec.from_hypothesis(
        {"text": "spectral gap tracks clustering on the held-out communication family"}
    )
    assert spec.held_out_family == "communication"


def test_inspect_routing_off_topic_is_not_ok():
    report = inspect_routing({"id": "x", "statement": "Sidon set density for n=100"})
    assert report["routing_ok"] is False
    assert report["resolved_invariants"] == []


# ── DOM2b: modularity is not a deterministic function of clustering ───────────

def _corr(a: np.ndarray, b: np.ndarray) -> float:
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def test_modularity_not_deterministic_function_of_clustering(real_frame):
    """DOM2b: modularity was ``0.25*clustering + 0.1*(avg_deg/n)`` — a closed-form
    function of clustering (|corr| == 1 within any family). The real Newman
    modularity must decouple from clustering: no real family shows a near-perfect
    correlation."""
    for fam in real_frame["network_family"].unique():
        sub = real_frame[real_frame["network_family"] == fam]
        r = _corr(
            sub["modularity"].to_numpy(float),
            sub["clustering_coefficient"].to_numpy(float),
        )
        assert abs(r) < 0.99, f"{fam}: modularity~clustering r={r}"


def test_no_exposed_invariant_pair_is_deterministic(real_frame):
    """No exposed-invariant pair may be a deterministic function of another:
    within EVERY real network family, |corr| must stay clear of 1 for all pairs.
    Also guards the old spectral_gap == algebraic_connectivity duplicate."""
    invariants = list(KNOWN_INVARIANTS)
    for a, b in itertools.combinations(invariants, 2):
        for fam in real_frame["network_family"].unique():
            sub = real_frame[real_frame["network_family"] == fam]
            r = _corr(sub[a].to_numpy(float), sub[b].to_numpy(float))
            assert abs(r) < 0.99, f"{a}~{b} on {fam}: |corr|={abs(r):.4f} (deterministic?)"


def test_spectral_gap_and_algebraic_connectivity_are_distinct(real_frame):
    assert not np.allclose(
        real_frame["spectral_gap"].to_numpy(float),
        real_frame["algebraic_connectivity"].to_numpy(float),
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
