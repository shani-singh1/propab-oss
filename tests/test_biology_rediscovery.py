"""Rediscovery-rejection references for the biology domains (v1 checklist item 2).

Each biology plugin exposes ``known_value_check(claim, evidence)`` that consults
its own ``literature_profile()`` reference anchors and flags a claim reproducing
a KNOWN biology fact as a rediscovery (``trivial_rediscovery=True`` /
``discovery_worthy=False``) — the same flags the domain-general paper path
(``propab.paper_narrative._is_rediscovery``) reads to demote a finding out of the
headline discovery count. A genuinely novel claim must NOT be flagged.

References exercised:
  * genomics — canonical housekeeping-gene set (Eisenberg & Levanon 2013) +
    tau~0.5 tissue-specificity threshold (Kryuchkova-Mostacci & Robinson-Rechavi 2017).
  * enzyme_kinetics — Bar-Even 2011 catalytic-efficiency ceilings (median kcat/Km
    ~1e5; diffusion limit 1e8-1e9 M^-1 s^-1).
  * mandrake — Pfam RVT clan CL0027 family structure + Xiong-Eickbush conserved
    RT motifs.
"""
from __future__ import annotations

from propab.domain_modules.enzyme_kinetics.plugin import EnzymeKineticsPlugin
from propab.domain_modules.genomics.plugin import GenomicsPlugin
from propab.domain_modules.mandrake.plugin import MandrakePlugin
from propab.paper_narrative import _is_rediscovery


def _as_finding(verdict: dict) -> dict:
    """Wrap a known_value_check verdict as a paper-facing finding for _is_rediscovery."""
    return {"text": "claim", "stats": dict(verdict), **verdict}


# ── genomics ─────────────────────────────────────────────────────────────────

def test_genomics_housekeeping_gene_is_rediscovery():
    v = GenomicsPlugin().known_value_check(
        "ACTB is a housekeeping gene with constitutive cross-tissue expression."
    )
    assert v is not None
    assert v["trivial_rediscovery"] is True
    assert v["discovery_worthy"] is False
    assert "ACTB" in str(v["rediscovery_identifier"]) or "ACTB" in v["notes"]
    assert _is_rediscovery(_as_finding(v)) is True


def test_genomics_housekeeping_by_ensembl_id_is_rediscovery():
    # GAPDH = ENSG00000111640 — matched by Ensembl accession too.
    v = GenomicsPlugin().known_value_check(
        "The gene ENSG00000111640 shows constitutive, ubiquitously expressed levels."
    )
    assert v is not None and v["trivial_rediscovery"] is True


def test_genomics_tau_threshold_restatement_is_rediscovery():
    v = GenomicsPlugin().known_value_check(
        "Genes split into housekeeping vs tissue-specific at a tau index of ~0.5."
    )
    assert v is not None and v["trivial_rediscovery"] is True


def test_genomics_novel_claim_not_flagged():
    # A specific, non-tabulated cross-tissue relationship — not a known lookup.
    v = GenomicsPlugin().known_value_check(
        "Expression variance predicts leave-one-tissue-out generalization for "
        "immune-response genes surviving the tissue-shuffle null."
    )
    assert v is None


# ── enzyme_kinetics ──────────────────────────────────────────────────────────

def test_enzyme_diffusion_limit_violation_is_known_ceiling():
    v = EnzymeKineticsPlugin().known_value_check(
        "This enzyme achieves a catalytic efficiency kcat/Km of 5e9 M^-1 s^-1."
    )
    assert v is not None
    assert v["trivial_rediscovery"] is True
    assert "ceiling" in v["notes"].lower() or "diffusion" in v["notes"].lower()
    assert _is_rediscovery(_as_finding(v)) is True


def test_enzyme_median_efficiency_restatement_is_rediscovery():
    v = EnzymeKineticsPlugin().known_value_check(
        "The median enzyme catalytic efficiency (kcat/Km) is about 1e5 M^-1 s^-1."
    )
    assert v is not None and v["trivial_rediscovery"] is True


def test_enzyme_novel_claim_not_flagged():
    # A within-diffusion-limit, non-median cross-EC relationship is not a lookup.
    v = EnzymeKineticsPlugin().known_value_check(
        "Aromatic residue fraction predicts log_kcat across held-out EC classes "
        "with a catalytic efficiency near 3e6 M^-1 s^-1."
    )
    assert v is None


# ── mandrake ─────────────────────────────────────────────────────────────────

def test_mandrake_pfam_family_assignment_is_rediscovery():
    v = MandrakePlugin().known_value_check(
        "These reverse transcriptase sequences belong to the Pfam RVT clan CL0027."
    )
    assert v is not None
    assert v["trivial_rediscovery"] is True
    assert _is_rediscovery(_as_finding(v)) is True


def test_mandrake_conserved_motifs_restatement_is_rediscovery():
    v = MandrakePlugin().known_value_check(
        "RT domains share seven conserved catalytic motifs across families."
    )
    assert v is not None and v["trivial_rediscovery"] is True


def test_mandrake_novel_claim_not_flagged():
    # A within-family biophysical predictive signal surviving LOFO is the novelty
    # bar — not a family/motif lookup.
    v = MandrakePlugin().known_value_check(
        "Thumb-domain surface net charge predicts RT activity within the LTR "
        "family under a low-identity clustered split."
    )
    assert v is None
