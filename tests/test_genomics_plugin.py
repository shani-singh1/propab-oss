"""Genomics plugin tests (CI discovery under tests/)."""
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
    assert get_domain_plugin("genomics") is not None


def test_genomics_preflight(tmp_genomics_data):
    r = GenomicsPlugin().preflight()
    assert r.passed, r.reason


def test_genomics_routing_corpus(tmp_genomics_data):
    report = inspect_corpus()
    assert report["routing_mismatches"] == 0
    assert report["total"] >= 20
