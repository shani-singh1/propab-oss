"""Domain profile registry and resolution."""
from __future__ import annotations

import re
from typing import Any

from propab.artifact_verification import EvidenceContext
from propab.domain_profiles.base import DomainProfile
from propab.domain_profiles.enzyme_kinetics import ENZYME_KINETICS_PROFILE
from propab.domain_profiles.graph_invariants import GRAPH_INVARIANTS_PROFILE
from propab.domain_profiles.materials import MATERIALS_PROFILE
from propab.domain_profiles.math_combinatorics import MATH_COMBINATORICS_PROFILE

_PROFILES: tuple[DomainProfile, ...] = (
    ENZYME_KINETICS_PROFILE,
    MATERIALS_PROFILE,
    GRAPH_INVARIANTS_PROFILE,
    MATH_COMBINATORICS_PROFILE,
)

_BY_ID: dict[str, DomainProfile] = {p.profile_id: p for p in _PROFILES}


def all_profiles() -> tuple[DomainProfile, ...]:
    return _PROFILES


def get_profile(profile_id: str) -> DomainProfile | None:
    return _BY_ID.get(profile_id)


def resolve_domain_profile(
    ctx: EvidenceContext,
    *,
    question: str = "",
    payload: dict[str, Any] | None = None,
) -> DomainProfile | None:
    """
    Pick a domain profile from explicit tags, evidence bucket, or question text.

    Campaign launchers should include ``[domain_profile:enzyme_kinetics]`` in the
    question or set ``domain_profile`` on the launch payload.
    """
    payload = payload or {}
    explicit = str(payload.get("domain_profile") or payload.get("domain_profile_id") or "").strip()
    if explicit:
        return _BY_ID.get(explicit)

    q = question or ctx.hypothesis_text or ""
    m = re.search(r"\[domain_profile:([a-z0-9_]+)\]", q, re.I)
    if m:
        return _BY_ID.get(m.group(1).lower())

    bucket = str(ctx.domain_bucket or payload.get("domain_bucket") or "").strip()
    if bucket in _BY_ID:
        return _BY_ID[bucket]

    for profile in _PROFILES:
        if profile.matches_question(q):
            return profile

    return None
