"""Tests for genomics domain plugin."""
from __future__ import annotations

import pytest

from propab import config
from propab.domain_modules.genomics.plugin import GenomicsPlugin
from propab.domain_modules.genomics.routing_inspector import inspect_corpus
from propab.domain_modules.registry import get_domain_plugin


@pytest.fixture
def tmp_genomics_data(monkeypatch, tmp_path):
    monkeypatch.setattr(config.settings, "propab_data_dir", str(tmp_path))


def test_genomics_plugin_registered():
    plugin = get_domain_plugin("genomics")
    assert plugin is not None
    assert plugin.domain_id == "genomics"


def test_preflight_passes(tmp_genomics_data):
    result = GenomicsPlugin().preflight()
    assert result.passed, result.reason


def test_run_verification_returns_lofo_r2(tmp_genomics_data):
    plugin = GenomicsPlugin()
    result = plugin.run_verification({"text": "cross-tissue gene expression LOFO"})
    assert "lofo_r2" in result
    assert "label_shuffle_null_p" in result


def test_routing_corpus_zero_mismatches(tmp_genomics_data):
    report = inspect_corpus()
    assert report["routing_mismatches"] == 0
    assert report["total"] >= 20
