"""Genomics plugin tests (CI discovery under tests/)."""
from __future__ import annotations

import pytest

from propab import config
from propab.domain_modules.genomics import adapter as genomics_adapter
from propab.domain_modules.genomics.plugin import GenomicsPlugin
from propab.domain_modules.genomics.routing_inspector import inspect_corpus
from propab.domain_modules.registry import get_domain_plugin


@pytest.fixture
def tmp_genomics_data(monkeypatch, tmp_path):
    """Isolated data dir with the network fetch stubbed to fail fast.

    Preflight/routing exercise the load + LOFO logic, not the GTEx download, so
    we force the deterministic synthetic-fallback path. This keeps the test
    offline-safe and fast in CI instead of hanging on a live GTEx fetch (the
    real-data assertions live in the ``@requires_real_data`` tests, which skip
    cleanly when the real cache is absent).
    """
    monkeypatch.setattr(config.settings, "propab_data_dir", str(tmp_path))
    monkeypatch.setattr(
        genomics_adapter,
        "_fetch_gtex",
        lambda *a, **k: (_ for _ in ()).throw(OSError("network disabled in test")),
    )


def test_genomics_plugin_registered():
    assert get_domain_plugin("genomics") is not None


def test_genomics_preflight(tmp_genomics_data):
    r = GenomicsPlugin().preflight()
    assert r.passed, r.reason


def test_genomics_routing_corpus(tmp_genomics_data):
    report = inspect_corpus()
    assert report["routing_mismatches"] == 0
    assert report["total"] >= 20
