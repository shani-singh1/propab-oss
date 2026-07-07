"""DISC2: a known-value table lookup is an honest REDISCOVERY, not a discovery.

Two layers are covered:
  1. ``apply_claim_validation`` (math_combinatorics) demotes a supported claim to
     ``trivial_rediscovery=True, discovery_worthy=False`` whenever the evidence
     came from a best-known lookup table -- while a genuinely-computed supported
     claim stays ``discovery_worthy=True``.
  2. The domain-general paper/findings path never counts such a finding as a
     novel discovery: it is labelled "rediscovery (known value)" and split out of
     the headline ``discoveries`` count.
"""
from __future__ import annotations

from propab.domain_modules.math_combinatorics.constructors import apply_claim_validation
from propab.paper_narrative import (
    REDISCOVERY_LABEL,
    _is_rediscovery,
    findings_table,
)


# ── layer 1: apply_claim_validation ──────────────────────────────────────────

def _cap_lookup_result_toplevel() -> dict:
    """Single cap-set experiment result: construction_source at the top level."""
    return {
        "metric_name": "cap_set_density",
        "metric_value": 112 / 729,
        "n": 6,
        "cap_set_size": 112,
        "field_size": 729,
        "construction_source": "best_known_table",
        "verified_true_steps": 0,
        "verified_false_steps": 0,
        "discovery_worthy": False,
        "trivial_rediscovery": True,
        "notes": "Best-known cap-set size in F_3^6 is 112 (literature table).",
    }


def _cap_lookup_result_sweep() -> dict:
    """Sweep result whose entries carry construction_source=best_known_table."""
    return {
        "metric_name": "cap_set_density",
        "metric_value": 0.9,
        "sweep": [
            {"n": 6, "cap_set_size": 112, "construction_source": "best_known_table"},
        ],
        "verified_true_steps": 0,
        "verified_false_steps": 0,
        "discovery_worthy": True,
        "trivial_rediscovery": False,
    }


def _cap_computed_result() -> dict:
    """A genuinely-computed cap set (not a table lookup)."""
    return {
        "metric_name": "cap_set_density",
        "metric_value": 20 / 81,
        "n": 4,
        "cap_set_size": 20,
        "construction_source": "computed",
        "verified_true_steps": 0,
        "verified_false_steps": 0,
    }


def test_toplevel_table_lookup_supported_becomes_rediscovery() -> None:
    stmt = "A cap set in F_3^6 has size at least 100"
    out = apply_claim_validation(_cap_lookup_result_toplevel(), stmt)
    assert out.get("claim_supported") is True  # arithmetic verdict still recorded
    assert out.get("discovery_worthy") is False
    assert out.get("trivial_rediscovery") is True
    assert "rediscovery" in str(out.get("notes")).lower()


def test_sweep_table_lookup_supported_becomes_rediscovery() -> None:
    stmt = "A cap set in F_3^6 has size at least 100"
    out = apply_claim_validation(_cap_lookup_result_sweep(), stmt)
    assert out.get("claim_supported") is True
    assert out.get("discovery_worthy") is False
    assert out.get("trivial_rediscovery") is True
    assert "best-known table" in str(out.get("notes")).lower()


def test_computed_supported_stays_discovery() -> None:
    stmt = "A cap set in F_3^4 has size at least 15"
    out = apply_claim_validation(_cap_computed_result(), stmt)
    assert out.get("claim_supported") is True
    assert out.get("discovery_worthy") is True
    assert out.get("trivial_rediscovery") is False
    assert "rediscovery" not in str(out.get("notes")).lower()


def test_input_contract_intact_result_dict_unmutated() -> None:
    """The orchestrator replays real evidence dicts; the input must not be mutated
    and the output must remain a plain result dict with the same statement input."""
    res = _cap_lookup_result_toplevel()
    snapshot = dict(res)
    out = apply_claim_validation(res, "A cap set in F_3^6 has size at least 100")
    assert res == snapshot  # input dict untouched (function copies)
    assert isinstance(out, dict)


# ── layer 2: domain-general paper/findings demotion ──────────────────────────

def _finding(fid: str, *, rediscovery: bool) -> dict:
    """A minimal paper-facing finding entry as produced by _enrich_finding_row."""
    stats = {"metric_value": 112, "trivial_rediscovery": rediscovery}
    if rediscovery:
        stats["discovery_worthy"] = False
    return {
        "id": fid,
        "text": f"Cap set in F_3^6 has size at least 100 ({fid})",
        "key_finding": f"Cap set in F_3^6 has size at least 100 ({fid})",
        "confidence": 0.99,
        "stats": stats,
        "claim_type": "existence",
        "replication_level": "single",
        "verification_method": "combinatorial_computation",
        "trivial_rediscovery": rediscovery,
    }


def test_is_rediscovery_detects_flag() -> None:
    assert _is_rediscovery(_finding("r", rediscovery=True)) is True
    assert _is_rediscovery(_finding("g", rediscovery=False)) is False


def test_findings_table_labels_rediscovery_not_novel() -> None:
    findings = {
        "confirmed": [
            _finding("REDISC", rediscovery=True),
            _finding("NOVEL", rediscovery=False),
        ],
        "refuted": [],
        "inconclusive": [],
    }
    tex = findings_table(findings)
    assert REDISCOVERY_LABEL in tex
    # The rediscovery row is labelled; the genuine one is not.
    for line in tex.splitlines():
        if "REDISC" in line:
            assert REDISCOVERY_LABEL in line
        if "NOVEL" in line:
            assert REDISCOVERY_LABEL not in line


def test_compile_counts_exclude_rediscoveries_from_discoveries() -> None:
    """The headline discovery count subtracts known-value rediscoveries."""
    from propab.paper_compiler import _finding_is_rediscovery

    confirmed = [
        _finding("R1", rediscovery=True),
        _finding("R2", rediscovery=True),
        _finding("G1", rediscovery=False),
    ]
    rediscoveries = sum(1 for f in confirmed if _finding_is_rediscovery(f))
    assert rediscoveries == 2
    discoveries = len(confirmed) - rediscoveries
    assert discoveries == 1
