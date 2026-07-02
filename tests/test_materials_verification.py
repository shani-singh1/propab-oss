"""Tests for materials LOFO verification (matbench dielectric)."""
from pathlib import Path

import pytest

from propab.domain_adapters.materials_adapter import (
    MaterialsAdapter,
    MaterialsExperimentSpec,
    classify_materials_verdict,
    is_materials_campaign,
    resolve_materials_features,
)
from propab.tools.registry import ToolRegistry
from services.worker.domain_router import domain_from_profile_tag, _keyword_fallback_domain

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "v1_candidates" / "matbench_dielectric.json.gz"


@pytest.mark.skipif(not CACHE.is_file(), reason="matbench cache missing — run evaluate_v1_domain_candidates.py")
def test_materials_lofo_experiment_runs():
    result = MaterialsAdapter().run_experiment(
        MaterialsExperimentSpec(
            feature_subset=["n_sites", "n_elements", "mean_Z"],
            methodology="LOFO",
        )
    )
    assert result["n_samples"] >= 200
    assert result.get("lofo_r2") is not None
    assert result.get("label_shuffle_null_p95") is not None
    assert "family_leakage_confirmed" in result


@pytest.mark.skipif(not CACHE.is_file(), reason="matbench cache missing")
def test_materials_verification_tool():
    r = ToolRegistry().call(
        "materials_verification",
        {"feature_subset": ["n_sites", "n_elements", "mean_Z"]},
    )
    assert r.success
    assert r.output.get("lofo_r2") is not None
    assert r.output.get("label_shuffle_null_p95") is not None


def test_is_materials_campaign():
    assert is_materials_campaign(question="[domain_profile:materials] discover dielectric")


def test_domain_profile_tag_routing():
    assert domain_from_profile_tag("[domain_profile:materials] foo") == "materials"
    assert _keyword_fallback_domain("LOFO cross-family holdout on matbench") == "materials"
    assert _keyword_fallback_domain("LOFO on rt_family t70_raw") == "mandrake"


def test_resolve_materials_features_magpie_alias():
    feats = resolve_materials_features("magpie descriptors predict dielectric")
    assert "n_sites" in feats
    assert "n_elements" in feats


def test_resolve_materials_features_electronic_alias():
    feats = resolve_materials_features("takahashi mean atomic mass and ionicity predict dielectric")
    assert "mean_atomic_mass" in feats
    assert "mean_ionicity" in feats or "mass_density" in feats


def test_known_features_count():
    from propab.domain_adapters.materials_adapter import _KNOWN_FEATURES
    assert len(_KNOWN_FEATURES) >= 8


@pytest.mark.skipif(not CACHE.is_file(), reason="matbench cache missing")
def test_materials_extended_features_in_frame():
    df = MaterialsAdapter().load_frame()
    for col in ("mean_atomic_mass", "mass_density", "mean_ionicity", "mean_coordination"):
        assert col in df.columns
        assert df[col].notna().all()


@pytest.mark.skipif(not CACHE.is_file(), reason="matbench cache missing")
def test_materials_real_crystal_system_families():
    df = MaterialsAdapter().load_frame()
    counts = df["crystal_system"].value_counts()
    assert len(counts) >= 5
    assert counts.min() >= 50
    assert "0" not in counts.index  # not qcut quintiles


def test_resolve_materials_features_bandgap_alias():
    feats = resolve_materials_features("Penn model bandgap predicts dielectric")
    assert "mp_bandgap" in feats


def test_classify_materials_with_leakage_flag():
    result = {
        "mean_r2": 0.2,
        "family_baseline_r2": 0.1,
        "lofo_gap": 0.05,
        "permutation_p": 0.01,
        "family_leakage_confirmed": True,
        "label_shuffle_null_p95": 0.05,
        "label_shuffle_permutation_p": 0.02,
    }
    v, _, _ = classify_materials_verdict("cross-crystal-system generalization", result)
    assert v == "confirmed"
