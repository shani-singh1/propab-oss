"""Routing inspector and combinatorics routing fixes (fixes.md)."""
from __future__ import annotations

import pytest

from propab.domain_modules.math_combinatorics.constructors import (
    extract_claim_text,
    is_cap_set_hypothesis,
    is_sidon_hypothesis,
    parse_interval_claim,
)
from propab.domain_modules.math_combinatorics.routing_inspector import inspect_routing
from propab.domain_modules.math_combinatorics.verifier import run_combinatorics_experiment
from propab.hypothesis_tree import HypothesisTree
from propab.research_quality import compute_claim_dedup_key


def test_extract_claim_text_strips_scope_boilerplate() -> None:
    full = (
        "For greedy Sidon, F(n)/sqrt(n) falls below 0.90 for n >= 1000.\n"
        "Population: Cap-set best-known table for dimensions d=9\n"
        "Distribution: Barabási–Albert graphs"
    )
    claim = extract_claim_text(full, full_text=full)
    assert "cap-set" not in claim.lower() or "greedy" in claim.lower()
    assert "barab" not in claim.lower()
    assert "0.90" in claim


def test_sidon_with_cap_scope_routes_to_sidon_not_cap_set() -> None:
    hyp = {
        "text": (
            "For n in {500, 1000, 2000}, greedy Sidon ratio F(n)/sqrt(n) stays below 0.95.\n"
            "Population: Cap-set best-known table lookup\n"
            "Distribution: CLP ratio sweep dims 3-7"
        ),
        "test_methodology": "greedy multi-n Sidon sweep",
    }
    result = inspect_routing(hyp)
    assert result["routing_ok"] is True
    assert result["resolved_verifier"].startswith("sidon")
    assert result["actual_metric_name"] == "sidon_ratio_to_sqrt_n"
    assert result["actual_metric_name"] != "cap_set_clp_ratio"


def test_cap_set_claim_routes_to_cap_set() -> None:
    hyp = {
        "statement": "In F_3^n for n in {3,4,5,6,7}, CLP ratios decrease monotonically below 2.25",
        "test_methodology": "best-known cap-set table",
    }
    result = inspect_routing(hyp)
    assert result["routing_ok"] is True
    assert result["resolved_verifier"] == "cap_set_sweep"
    assert result["actual_metric_name"] == "cap_set_clp_ratio"


def test_n_range_not_parsed_as_ratio_band() -> None:
    stmt = "Greedy Sidon ratio falls in n ∈ [500, 719] with F(n)/sqrt(n) below 0.90"
    assert parse_interval_claim(stmt) is None


def test_ratio_band_still_parsed() -> None:
    assert parse_interval_claim("F(n)/sqrt(n) in [0.90, 0.95]") == (0.90, 0.95)


def test_ac60fcda_style_sidon_threshold_hypothesis() -> None:
    """Regression: campaign 4 misrouted this class to cap_set_clp_ratio."""
    hyp = {
        "text": (
            "For greedy Sidon in n ∈ [500, 719], F(n)/sqrt(n) falls below 0.90.\n"
            "Population: N=300–5000 node graphs"
        ),
        "test_methodology": "greedy Sidon search",
    }
    ev = run_combinatorics_experiment(hyp, ["sidon_set_density"])
    assert ev.get("metric_name") == "sidon_ratio_to_sqrt_n"
    assert ev.get("metric_name") != "cap_set_clp_ratio"


def test_confirmed_claim_dedup_not_evidence_hash() -> None:
    tree = HypothesisTree()
    key_a = compute_claim_dedup_key("Claim A about Sidon thresholds")
    key_b = compute_claim_dedup_key("Claim B about different n band")
    assert tree.register_confirmed_claim(key_a) is True
    assert tree.register_confirmed_claim(key_a) is False
    assert tree.register_confirmed_claim(key_b) is True


def test_ac60fcda_c8_vs_c7_cap_set_routing() -> None:
    """Regression: F_3^8 vs F_3^7 comparison must sweep CLP ratios, not single-point density."""
    hyp = {
        "text": (
            "In F_3^8, the maximum cap set size |A_max(8)| is bounded by 480, "
            "yielding a growth rate c_8 = |A_max(8)|^(1/8) < 2.16, which is strictly "
            "less than the c_7 ~ 2.25 achieved in F_3^7."
        ),
        "test_methodology": "best-known cap-set table",
    }
    result = inspect_routing(hyp)
    assert result["routing_ok"] is True
    assert result["resolved_verifier"] == "cap_set_sweep"
    assert result["actual_metric_name"] == "cap_set_clp_ratio"
    assert 7 in result["cap_dims"]
    assert 8 in result["cap_dims"]


def test_is_cap_set_false_on_scope_only_clp() -> None:
    full = (
        "F(n)/sqrt(n) below 0.95 for n=5000\n"
        "Population: Cap-set CLP ratio table dims 3-7"
    )
    assert is_cap_set_hypothesis(full, full_text=full) is False
    assert is_sidon_hypothesis(full, full_text=full) is True
