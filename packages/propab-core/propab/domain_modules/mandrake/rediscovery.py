"""Known-value rediscovery check for mandrake (RT-family) claims.

Compares a hypothesis claim against the real RT-phylogenetics anchors tabulated
in ``MandrakePlugin.literature_profile()``:

  * the Pfam RVT clan CL0027 (PF00078 / PF07727 / PF13456) — the curated RT-domain
    family tabulation. A claim that merely assigns RT sequences to their known
    Pfam family, or restates the RVT-clan family structure, is a lookup;
  * the Xiong & Eickbush 1990 anchors — the seven conserved RT catalytic motifs
    over a ~240-aa domain, and the characteristically LOW (< ~25%) cross-class RT
    sequence identity. A "novel" signal that is really the known motif
    conservation, or the known family/phylogeny structure, is a rediscovery.

The returned dict carries the ``trivial_rediscovery`` / ``discovery_worthy``
flags that ``propab.paper_narrative._is_rediscovery`` already reads.
"""
from __future__ import annotations

import re
from typing import Any


def _pfam_identifiers(profile: dict[str, Any]) -> set[str]:
    ids: set[str] = set()
    for tab in profile.get("tabulation_sources", []) or []:
        if "RVT" in tab.get("name", "") or "Pfam" in tab.get("name", ""):
            for ident in tab.get("identifiers", []) or []:
                ids.add(str(ident).upper())
    return ids


def _motif_count(profile: dict[str, Any]) -> int | None:
    for tab in profile.get("tabulation_sources", []) or []:
        n = tab.get("conserved_motifs")
        if n is not None:
            try:
                return int(n)
            except (TypeError, ValueError):
                return None
    return None


_FAMILY_ASSIGNMENT_RE = re.compile(
    r"belongs? to|is a member of|assign(?:ed|s)? to|classif|falls? (?:in|into)|"
    r"family structure|phylogen",
    re.IGNORECASE,
)
_MOTIF_CLAIM_RE = re.compile(
    r"seven conserved|7 conserved|conserved (?:rt )?motif|catalytic motif|"
    r"yxdd motif|palm motif",
    re.IGNORECASE,
)


def check_rediscovery(
    claim_text: str,
    evidence: dict[str, Any] | None,
    profile: dict[str, Any],
) -> dict[str, Any] | None:
    """Return a rediscovery/known verdict dict, or None if the claim is not a known value."""
    text = (claim_text or "").lower()
    pfam_ids = _pfam_identifiers(profile)

    tokens = {t.upper() for t in re.findall(r"[A-Za-z0-9]+", claim_text or "")}
    named_pfam = sorted(tokens & pfam_ids)

    # 1) Explicit Pfam RVT-clan family identifier named, or a plain restatement of
    #    the RT phylogenetic family structure — a curated-tabulation lookup.
    mentions_rvt_clan = "rvt" in text or "cl0027" in text or "rvt clan" in text
    restates_family = bool(_FAMILY_ASSIGNMENT_RE.search(text)) and (
        "rt" in text or "reverse transcriptase" in text or "retro" in text
    )
    if named_pfam or mentions_rvt_clan or restates_family:
        ident = named_pfam[0] if named_pfam else ("CL0027" if mentions_rvt_clan else "RVT_clan")
        return {
            "trivial_rediscovery": True,
            "discovery_worthy": False,
            "rediscovery_source": "Pfam RVT clan CL0027 (Mistry et al. 2021)",
            "rediscovery_identifier": ident,
            "notes": (
                f"Rediscovery (known value): the RT-family assignment / RVT-clan "
                f"structure ({ident}) is curated in Pfam CL0027 "
                "(PF00078/PF07727/PF13456); a lookup, not a novel finding."
            ),
        }

    # 2) A restatement of the Xiong-Eickbush seven-conserved-motifs anchor.
    n_motifs = _motif_count(profile)
    if _MOTIF_CLAIM_RE.search(text):
        return {
            "trivial_rediscovery": True,
            "discovery_worthy": False,
            "rediscovery_source": "Xiong & Eickbush 1990 conserved RT motifs",
            "rediscovery_identifier": (
                f"RT_{n_motifs}_motifs" if n_motifs else "RT_conserved_motifs"
            ),
            "notes": (
                "Rediscovery (known value): the "
                f"{'seven ' if n_motifs == 7 else ''}conserved RT catalytic motifs are "
                "the established Xiong & Eickbush (1990) reference, not a novel finding."
            ),
        }

    return None
