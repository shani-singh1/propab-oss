"""Audit confirmed campaign findings — classify, archive, flag gold (fixes.md)."""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any
from uuid import uuid4

from propab.domain_adapters.mandrake_adapter import resolve_mandrake_features

# fixes.md feature-family taxonomy
FEATURE_FAMILIES: dict[str, tuple[str, ...]] = {
    "thermal": ("t40_raw", "t45_raw", "t50_raw", "t55_raw", "t60_raw", "t65_raw", "t70_raw", "t75_raw", "t80_raw", "thermophilicity"),
    "geometry": ("triad_best_rmsd", "d1_d2_dist", "d2_d3_dist", "ramachandran", "g_factor", "yxdd"),
    "electrostatics": ("native_net_charge", "isoelectric", "salt_bridge", "pocket_hbond", "thumb_surface"),
    "fold_similarity": ("foldseek", "tm_score", "lddt"),
    "surface": ("camsol", "sasa", "hydrophobic"),
    "motif": ("dgr_motif", "qg_motif", "sp_motif", "motif"),
    "mixed": (),
}

DISPOSITION_ARCHIVE = "archive"  # obvious / already known
DISPOSITION_REJECT = "reject"  # family surrogate
DISPOSITION_GOLD = "gold"  # counterintuitive — pursue
DISPOSITION_KEEP = "keep"  # valid discrimination, not yet gold


@dataclass
class FindingAudit:
    hypothesis_id: str
    text: str
    feature_families: list[str]
    primary_family: str
    lofo_r2: float | None
    lofo_gap: float | None
    is_null: bool
    is_discrimination: bool
    disposition: str
    rationale: str
    features: list[str] = field(default_factory=list)
    compare_features: list[str] = field(default_factory=list)
    id: str = field(default_factory=lambda: str(uuid4()))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _parse_evidence(evidence_summary: str) -> dict[str, Any]:
    if not evidence_summary:
        return {}
    m = re.search(r"evidence=(\{.*?\});", evidence_summary)
    if not m:
        return {}
    try:
        obj = json.loads(m.group(1))
        return obj if isinstance(obj, dict) else {}
    except json.JSONDecodeError:
        return {}


def _extract_features(evidence_summary: str, text: str) -> list[str]:
    ev = _parse_evidence(evidence_summary)
    m = re.search(r"features=(\[.*?\])", evidence_summary or "")
    if m:
        try:
            feats = json.loads(m.group(1).replace("'", '"'))
            if isinstance(feats, list):
                return [str(f) for f in feats]
        except json.JSONDecodeError:
            pass
    return resolve_mandrake_features(text)


def _family_for_feature(name: str) -> str:
    low = name.lower()
    for fam, keys in FEATURE_FAMILIES.items():
        if fam == "mixed":
            continue
        if any(k in low for k in keys):
            return fam
    return "other"


def _families_for_features(features: list[str]) -> list[str]:
    seen: list[str] = []
    for f in features:
        fam = _family_for_feature(f)
        if fam not in seen:
            seen.append(fam)
    return seen or ["unknown"]


def _is_null_hypothesis(text: str) -> bool:
    core = re.split(r"\s*\(Question:", (text or "").strip(), maxsplit=1)[0].lower()
    return "null hypothesis" in core or "no falsifiable pattern" in core


def _is_discrimination_hypothesis(text: str) -> bool:
    low = (text or "").lower()
    return any(m in low for m in (
        "more robust cross-family",
        "than catalytic geometry",
        "than electrostatic",
        "than global structural",
        "than foldseek",
        "beats",
        "versus",
        " vs ",
        "collapse under lofo",
    ))


def classify_finding(
    *,
    hypothesis_id: str,
    text: str,
    evidence_summary: str = "",
    features: list[str] | None = None,
) -> FindingAudit:
    """Apply fixes.md triage: obvious / family surrogate / gold."""
    feats = list(features or _extract_features(evidence_summary, text))
    ev = _parse_evidence(evidence_summary)
    lofo = ev.get("metric_value")
    gap = ev.get("effect_size") or ev.get("lofo_gap")
    if lofo is not None:
        lofo = float(lofo)
    if gap is not None:
        gap = float(gap)

    families = _families_for_features(feats)
    primary = families[0] if len(families) == 1 else "mixed"
    if len(families) > 1:
        primary = "mixed"

    is_null = _is_null_hypothesis(text)
    is_disc = _is_discrimination_hypothesis(text)

    disposition = DISPOSITION_KEEP
    rationale = "Standard confirmed finding."

    if is_null:
        if lofo is not None and lofo < 0 and gap is not None and gap >= 0.35:
            disposition = DISPOSITION_REJECT
            rationale = (
                "Null 'confirmed' with negative LOFO and large gap — signal is group-specific "
                "(family surrogate), not cross-group biology."
            )
        else:
            disposition = DISPOSITION_ARCHIVE
            rationale = "Null calibration — not a discovery claim."

    elif is_disc and primary == "mixed":
        thermal_feats = [f for f in feats if _family_for_feature(f) == "thermal"]
        other_fams = [f for f in feats if _family_for_feature(f) != "thermal"]
        if thermal_feats and other_fams:
            if lofo is not None and lofo > 0.05:
                disposition = DISPOSITION_GOLD
                rationale = (
                    f"Positive LOFO ({lofo:.3f}) on thermal vs "
                    f"{_families_for_features(other_fams)} — unexpected cross-family signal."
                )
            elif lofo is not None and lofo < -0.15:
                other_family_names = _families_for_features(other_fams)
                disposition = DISPOSITION_ARCHIVE
                rationale = (
                    "Discrimination confirms comparator collapses under LOFO — expected "
                    f"({', '.join(other_family_names) or 'comparator'} tracks family). "
                    "Thermal 'wins' by being less negative, not cross-group."
                )
            else:
                disposition = DISPOSITION_KEEP
                rationale = "Valid mechanism discrimination but LOFO still weak/negative."
        else:
            disposition = DISPOSITION_KEEP
            rationale = "Mixed-feature discrimination test."

    elif primary == "thermal" and len(families) == 1:
        disposition = DISPOSITION_ARCHIVE
        rationale = "Thermal-only confirmed — likely obvious thermostability axis."

    elif lofo is not None and lofo > 0.05:
        disposition = DISPOSITION_GOLD
        rationale = f"Positive LOFO ({lofo:.3f}) — survives group removal."

    compare = [f for f in feats if _family_for_feature(f) != "thermal"] if "thermal" in families else []

    return FindingAudit(
        hypothesis_id=hypothesis_id,
        text=text.strip(),
        feature_families=families,
        primary_family=primary,
        lofo_r2=lofo,
        lofo_gap=gap,
        is_null=is_null,
        is_discrimination=is_disc,
        disposition=disposition,
        rationale=rationale,
        features=feats,
        compare_features=compare,
    )


def audit_confirmed_findings(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize a batch of confirmed hypothesis rows."""
    audits = [
        classify_finding(
            hypothesis_id=str(r.get("id") or r.get("hypothesis_id") or ""),
            text=str(r.get("text") or r.get("key_finding") or ""),
            evidence_summary=str(r.get("evidence_summary") or ""),
            features=r.get("feature_subset"),
        )
        for r in rows
    ]
    by_disposition: dict[str, int] = {}
    by_family: dict[str, int] = {}
    for a in audits:
        by_disposition[a.disposition] = by_disposition.get(a.disposition, 0) + 1
        by_family[a.primary_family] = by_family.get(a.primary_family, 0) + 1

    gold = [a for a in audits if a.disposition == DISPOSITION_GOLD]
    fake_diversity = by_family.get("thermal", 0) >= max(1, len(audits) * 0.75)

    return {
        "n_confirmed": len(audits),
        "by_disposition": by_disposition,
        "by_primary_family": by_family,
        "fake_diversity": fake_diversity,
        "gold_findings": [g.to_dict() for g in gold],
        "findings": [a.to_dict() for a in audits],
        "recommendation": (
            "No gold findings — confirmations are family surrogates or expected collapses. "
            "Use competing-mechanism discrimination (A vs B), not single-mechanism verification."
            if not gold
            else f"{len(gold)} gold finding(s) — prioritize discriminating experiments among competing mechanisms."
        ),
    }
