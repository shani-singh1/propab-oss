"""Tests for enzyme kinetics plugin."""
from __future__ import annotations

import pytest

from propab import config
from propab.domain_modules.enzyme_kinetics.plugin import EnzymeKineticsPlugin
from propab.domain_modules.enzyme_kinetics.routing_inspector import inspect_corpus
from propab.domain_modules.registry import get_domain_plugin


@pytest.fixture
def tmp_data(monkeypatch, tmp_path):
    monkeypatch.setattr(config.settings, "propab_data_dir", str(tmp_path))


def test_enzyme_plugin_registered():
    assert get_domain_plugin("enzyme_kinetics") is not None


def test_enzyme_preflight(tmp_data):
    r = EnzymeKineticsPlugin().preflight()
    assert r.passed, r.reason


def test_enzyme_routing_corpus(tmp_data):
    report = inspect_corpus()
    assert report["routing_mismatches"] == 0
    assert report["total"] >= 20
