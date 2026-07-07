"""Known-value rediscovery check for enzyme-kinetics claims.

Compares a hypothesis claim (+ optional evidence) against the real
catalytic-efficiency anchors tabulated in
``EnzymeKineticsPlugin.literature_profile()`` (Bar-Even et al. 2011):

  * median kcat/Km ~ 1e5 M^-1 s^-1, median kcat ~ 10 s^-1, median Km ~ 1e-4 M;
  * the diffusion limit 1e8 - 1e9 M^-1 s^-1 as a hard physical ceiling.

Two rediscovery/known outcomes fire:
  1. A claim that merely restates one of the Bar-Even median anchors as a
     "finding" is a rediscovery (known value).
  2. A claim of a catalytic efficiency ABOVE the diffusion limit is not novel —
     it is physically impossible and flagged as a known-ceiling violation.

The returned dict carries the ``trivial_rediscovery`` / ``discovery_worthy``
flags that ``propab.paper_narrative._is_rediscovery`` already reads.
"""
from __future__ import annotations

import re
from typing import Any


def _bar_even_anchors(profile: dict[str, Any]) -> dict[str, Any]:
    for tab in profile.get("tabulation_sources", []) or []:
        if "median_kcat_km_M_inv_s_inv" in tab or "diffusion_limit_kcat_km_M_inv_s_inv" in tab:
            return tab
    return {}


def _parse_kcat_km(text: str) -> float | None:
    """Pull a kcat/Km value (M^-1 s^-1) from claim text, e.g. '1e8', '3.2 x 10^8'."""
    low = text.lower()
    if "kcat/km" not in low and "catalytic efficiency" not in low and "specificity constant" not in low:
        return None
    # scientific notation: 1e8, 1.2e9, 3.2 x 10^8, 3.2*10^8
    m = re.search(r"(\d+(?:\.\d+)?)\s*(?:[x*]\s*10\s*\^?|e)\s*([+-]?\d+)", low)
    if m:
        try:
            return float(m.group(1)) * (10.0 ** int(m.group(2)))
        except (TypeError, ValueError):
            return None
    return None


def check_rediscovery(
    claim_text: str,
    evidence: dict[str, Any] | None,
    profile: dict[str, Any],
) -> dict[str, Any] | None:
    """Return a rediscovery/known verdict dict, or None if the claim is not a known value."""
    text = (claim_text or "").lower()
    anchors = _bar_even_anchors(profile)
    if not anchors:
        return None

    limit = anchors.get("diffusion_limit_kcat_km_M_inv_s_inv") or [1e8, 1e9]
    try:
        hard_ceiling = float(limit[1])
    except (TypeError, ValueError, IndexError):
        hard_ceiling = 1e9
    median_kk = float(anchors.get("median_kcat_km_M_inv_s_inv", 1e5))

    # 1) Catalytic efficiency claimed above the diffusion limit — a known-ceiling
    #    violation, not a novel super-efficient enzyme.
    value = _parse_kcat_km(text)
    if value is not None and value > hard_ceiling:
        return {
            "trivial_rediscovery": True,
            "discovery_worthy": False,
            "rediscovery_source": "Bar-Even et al. 2011 diffusion-limit ceiling",
            "rediscovery_identifier": "kcat_km_ceiling",
            "notes": (
                f"Rediscovery/known-ceiling: claimed kcat/Km {value:.2g} exceeds the "
                f"diffusion limit ceiling ~{hard_ceiling:.0g} M^-1 s^-1 "
                "(Bar-Even et al. 2011); not a novel finding."
            ),
        }

    # 2) A restatement of the Bar-Even median catalytic-efficiency anchor.
    mentions_median = "median" in text or "typical" in text or "average" in text
    mentions_efficiency = (
        "kcat/km" in text or "catalytic efficiency" in text or "specificity constant" in text
    )
    if mentions_median and mentions_efficiency:
        if value is None or abs(value - median_kk) / median_kk <= 0.5:
            return {
                "trivial_rediscovery": True,
                "discovery_worthy": False,
                "rediscovery_source": "Bar-Even et al. 2011 median catalytic efficiency",
                "rediscovery_identifier": "median_kcat_km",
                "notes": (
                    f"Rediscovery (known value): the median enzyme catalytic efficiency "
                    f"kcat/Km ~{median_kk:.0g} M^-1 s^-1 is the Bar-Even 2011 reference "
                    "anchor, not a novel result."
                ),
            }

    return None
