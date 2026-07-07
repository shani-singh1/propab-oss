"""Known-value rediscovery check for genomics claims.

A campaign that "confirms" an already-established biology fact is a REDISCOVERY,
not a discovery. This module compares a hypothesis claim (+ optional evidence)
against the real reference anchors tabulated in ``GenomicsPlugin.literature_profile()``:

  * the canonical housekeeping-gene set (Eisenberg & Levanon 2013) — a claim that
    a listed gene is housekeeping / constitutively expressed / low-specificity is
    a lookup;
  * the tau tissue-specificity threshold (~0.5) benchmarked by
    Kryuchkova-Mostacci & Robinson-Rechavi 2017 — a claim that just restates the
    housekeeping-vs-tissue-specific split at tau~0.5 is a lookup.

The returned dict carries the ``trivial_rediscovery`` / ``discovery_worthy``
flags that ``propab.paper_narrative._is_rediscovery`` already reads, so a flagged
finding is labelled "rediscovery (known value)" in every paper section and never
counted as a novel discovery. The check is deliberately conservative: it only
fires on an explicit, unambiguous restatement of a tabulated fact.
"""
from __future__ import annotations

import re
from typing import Any


def _housekeeping_ids(profile: dict[str, Any]) -> tuple[set[str], set[str]]:
    """(HGNC symbols, Ensembl accessions) from the housekeeping tabulation source."""
    symbols: set[str] = set()
    ensembl: set[str] = set()
    for tab in profile.get("tabulation_sources", []) or []:
        for sym in tab.get("housekeeping_genes", []) or []:
            symbols.add(str(sym).upper())
        for acc in tab.get("housekeeping_gene_ensembl", []) or []:
            ensembl.add(str(acc).upper())
    return symbols, ensembl


def _tau_threshold(profile: dict[str, Any]) -> float | None:
    for tab in profile.get("tabulation_sources", []) or []:
        thr = tab.get("housekeeping_threshold_tau")
        if thr is not None:
            try:
                return float(thr)
            except (TypeError, ValueError):
                return None
    return None


_HOUSEKEEPING_CLAIM_RE = re.compile(
    r"housekeeping|constitutive|constitutively expressed|broadly expressed|"
    r"low tissue.?specific|low\s*tau|ubiquitously expressed",
    re.IGNORECASE,
)


def check_rediscovery(
    claim_text: str,
    evidence: dict[str, Any] | None,
    profile: dict[str, Any],
) -> dict[str, Any] | None:
    """Return a rediscovery verdict dict, or None if the claim is not a known value.

    The verdict dict is safe to merge onto a finding/evidence dict; it sets
    ``trivial_rediscovery=True`` + ``discovery_worthy=False`` so downstream paper
    logic demotes it out of the headline discovery count.
    """
    text = (claim_text or "").lower()
    evidence = evidence or {}
    symbols, ensembl = _housekeeping_ids(profile)

    # 1) A named canonical housekeeping gene asserted to be housekeeping / low-tau.
    tokens = set(re.findall(r"[A-Za-z0-9]+", claim_text or ""))
    tokens_upper = {t.upper() for t in tokens}
    hit_symbol = sorted(tokens_upper & symbols)
    # Ensembl ids are matched case-insensitively (versions already stripped upstream).
    hit_ensembl = sorted({t for t in tokens_upper if t in ensembl})
    named_hits = hit_symbol + hit_ensembl
    asserts_housekeeping = bool(_HOUSEKEEPING_CLAIM_RE.search(text))
    if named_hits and asserts_housekeeping:
        gene = named_hits[0]
        return {
            "trivial_rediscovery": True,
            "discovery_worthy": False,
            "rediscovery_source": "Eisenberg & Levanon 2013 housekeeping gene set",
            "rediscovery_identifier": gene,
            "notes": (
                f"Rediscovery (known value): {gene} is a canonical housekeeping gene "
                "(Eisenberg & Levanon 2013); its constitutive/low-tissue-specificity "
                "expression is an established lookup, not a novel finding."
            ),
        }

    # 2) A restatement of the tau~0.5 housekeeping-vs-specific split.
    tau_thr = _tau_threshold(profile)
    mentions_tau = ("tau" in text or "tissue-specificity" in text or "tissue specificity" in text)
    mentions_split = asserts_housekeeping or ("tissue-specific" in text or "tissue specific" in text)
    if tau_thr is not None and mentions_tau and mentions_split:
        # If the claim cites a threshold value, only fire when it matches ~0.5.
        num = re.search(r"tau\s*[~≈=<>]{0,2}\s*(0?\.\d+)", text)
        cites_thr = num is not None
        near_thr = cites_thr and abs(float(num.group(1)) - tau_thr) <= 0.1
        if (not cites_thr) or near_thr:
            return {
                "trivial_rediscovery": True,
                "discovery_worthy": False,
                "rediscovery_source": (
                    "Kryuchkova-Mostacci & Robinson-Rechavi 2017 tau benchmark"
                ),
                "rediscovery_identifier": f"tau~{tau_thr}",
                "notes": (
                    f"Rediscovery (known value): the housekeeping-vs-tissue-specific "
                    f"split at tau~{tau_thr} is the benchmarked reference threshold "
                    "(Kryuchkova-Mostacci & Robinson-Rechavi 2017), not a novel result."
                ),
            }

    return None
