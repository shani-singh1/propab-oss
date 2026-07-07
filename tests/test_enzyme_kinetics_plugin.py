"""Tests for enzyme kinetics plugin."""
from __future__ import annotations

import pytest

from propab import config
from propab.domain_modules.enzyme_kinetics import adapter as enzyme_adapter
from propab.domain_modules.enzyme_kinetics.plugin import EnzymeKineticsPlugin
from propab.domain_modules.enzyme_kinetics.routing_inspector import inspect_corpus
from propab.domain_modules.registry import get_domain_plugin


@pytest.fixture
def tmp_data(monkeypatch, tmp_path):
    """Isolated data dir with the network fetch stubbed to fail fast.

    Preflight/routing exercise the load + LOFO logic, not the DLKcat download, so
    we force the deterministic synthetic-fallback path. This keeps the test
    offline-safe and fast in CI instead of hanging on a live DLKcat fetch (the
    real-data assertions live in the ``@requires_real_data`` tests, which skip
    cleanly when the real cache is absent).
    """
    monkeypatch.setattr(config.settings, "propab_data_dir", str(tmp_path))
    monkeypatch.setattr(
        enzyme_adapter,
        "_fetch_dlkcat",
        lambda *a, **k: (_ for _ in ()).throw(OSError("network disabled in test")),
    )


def test_enzyme_plugin_registered():
    assert get_domain_plugin("enzyme_kinetics") is not None


def test_enzyme_preflight(tmp_data):
    r = EnzymeKineticsPlugin().preflight()
    assert r.passed, r.reason


def test_enzyme_routing_corpus(tmp_data):
    report = inspect_corpus()
    assert report["routing_mismatches"] == 0
    assert report["total"] >= 20
