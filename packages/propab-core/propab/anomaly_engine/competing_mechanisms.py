"""Competing mechanism sets — multiple explanations per anomaly (fixes.md)."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any
from uuid import uuid4

from propab.anomaly_engine.anomaly_objects import AnomalyObject
from propab.anomaly_engine.mechanism_objects import MechanismObject
from propab.domain_adapters.mandrake_adapter import infer_biology_theme


@dataclass
class CompetingMechanismSet:
    """
    One anomaly → multiple competing mechanisms → discriminating experiments.

    Hypotheses should distinguish mechanism_a from mechanism_b, not merely verify one.
    """

    anomaly_id: str
    feature_subset: list[str]
    bucket: str
    mechanisms: list[MechanismObject]
    discrimination_pairs: list[dict[str, Any]]
    id: str = field(default_factory=lambda: str(uuid4()))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _mechanism_from_anomaly(
    anomaly: AnomalyObject,
    *,
    label: str,
    explanation: str,
    confidence: float,
) -> MechanismObject:
    return MechanismObject(
        explanation=explanation,
        candidate_features=list(anomaly.feature_subset),
        supporting_anomalies=["|".join(anomaly.feature_subset)],
        assumptions_challenged=[label],
        confidence=confidence,
    )


def _survivor_mechanisms(anomaly: AnomalyObject) -> list[MechanismObject]:
    feats = anomaly.feature_subset
    lofo = float(anomaly.observed_score)
    return [
        _mechanism_from_anomaly(
            anomaly,
            label="cross_group_biophysics",
            explanation=(
                f"Features {feats} encode a conserved biophysical constraint (LOFO={lofo:.3f}) "
                "that partially survives group removal."
            ),
            confidence=0.72,
        ),
        _mechanism_from_anomaly(
            anomaly,
            label="latent_family_proxy",
            explanation=(
                f"Features {feats} correlate with RT activity only because they proxy "
                "unmeasured family-specific chemistry — LOFO should degrade with finer family splits."
            ),
            confidence=0.55,
        ),
        _mechanism_from_anomaly(
            anomaly,
            label="assay_artifact",
            explanation=(
                f"Features {feats} track measurement batch or expression-system artifacts "
                "rather than intrinsic RT function."
            ),
            confidence=0.35,
        ),
    ]


def _collapse_mechanisms(anomaly: AnomalyObject) -> list[MechanismObject]:
    feats = anomaly.feature_subset
    gap = float(anomaly.metadata.get("lofo_gap", 0))
    theme = infer_biology_theme(" ".join(feats), feats)
    return [
        _mechanism_from_anomaly(
            anomaly,
            label="family_leakage",
            explanation=(
                f"Features {feats} ({theme}) predict within families (gap={gap:.2f}) but collapse under LOFO — "
                "classic evolutionary-family leakage."
            ),
            confidence=0.85,
        ),
        _mechanism_from_anomaly(
            anomaly,
            label="fold_family_coupling",
            explanation=(
                f"Structural similarity features {feats} track fold family identity; "
                "LOFO collapse exposes confounded architecture."
            ),
            confidence=0.70,
        ),
        _mechanism_from_anomaly(
            anomaly,
            label="small_n_artifact",
            explanation=(
                f"Within-family R² for {feats} is inflated by small per-family sample size; "
                "LOFO collapse is a statistical artifact."
            ),
            confidence=0.40,
        ),
    ]


def _neighbor_disagreement_mechanisms(anomaly: AnomalyObject) -> list[MechanismObject]:
    feats = anomaly.feature_subset
    partner = anomaly.metadata.get("disagreement_partner") or []
    pg = anomaly.metadata.get("primary_group", "?")
    cg = anomaly.metadata.get("partner_group", "?")
    return [
        _mechanism_from_anomaly(
            anomaly,
            label=f"{pg}_driver",
            explanation=f"Group {pg} features {feats} drive RT activity via a real biophysical axis.",
            confidence=0.65,
        ),
        MechanismObject(
            explanation=f"Group {cg} features {partner} capture opposing family-specific signal.",
            candidate_features=list(partner) if isinstance(partner, list) else [],
            supporting_anomalies=["|".join(partner) if isinstance(partner, list) else str(partner)],
            assumptions_challenged=[f"{cg}_driver"],
            confidence=0.65,
        ),
        _mechanism_from_anomaly(
            anomaly,
            label="measurement_confound",
            explanation="Opposing signals arise from incompatible measurement scales across feature groups.",
            confidence=0.35,
        ),
    ]


def _discrimination_pair(
    primary_feats: list[str],
    compare_feats: list[str],
    *,
    question: str,
) -> dict[str, Any]:
    return {
        "question": question,
        "primary_features": primary_feats,
        "compare_features": compare_feats,
        "methodology": "LOFO",
        "tool": "mandrake_verification",
        "expected_if_a": "primary LOFO > compare LOFO by >= 0.05",
        "expected_if_b": "compare LOFO >= primary LOFO",
    }


def build_competing_sets(
    anomalies: list[AnomalyObject],
    *,
    max_sets: int = 5,
) -> list[CompetingMechanismSet]:
    """Build competing mechanism sets from stratified anomalies (fixes.md P4)."""
    if not anomalies:
        return []

    by_bucket: dict[str, list[AnomalyObject]] = {}
    for a in anomalies:
        b = str(a.metadata.get("bucket") or "unknown")
        by_bucket.setdefault(b, []).append(a)

    order = ("survivor", "collapse", "neighbor_disagreement", "cross_family", "threshold", "outlier")
    picked: list[AnomalyObject] = []
    for b in order:
        for a in by_bucket.get(b, [])[:2]:
            if a not in picked:
                picked.append(a)
        if len(picked) >= max_sets:
            break
    if not picked:
        picked = anomalies[:max_sets]

    sets: list[CompetingMechanismSet] = []
    for anomaly in picked[:max_sets]:
        bucket = str(anomaly.metadata.get("bucket") or "unknown")
        if bucket == "collapse":
            mechs = _collapse_mechanisms(anomaly)
        elif bucket == "neighbor_disagreement":
            mechs = _neighbor_disagreement_mechanisms(anomaly)
        else:
            mechs = _survivor_mechanisms(anomaly)

        pairs: list[dict[str, Any]] = []
        if len(mechs) >= 2:
            a_feats = list(mechs[0].candidate_features)
            b_feats = list(mechs[1].candidate_features) or a_feats
            if a_feats != b_feats:
                pairs.append(_discrimination_pair(
                    a_feats, b_feats,
                    question=f"Does mechanism A ({mechs[0].assumptions_challenged[0]}) beat B ({mechs[1].assumptions_challenged[0]}) under LOFO?",
                ))
        # Cross-theme discrimination from anomaly feature group vs partner
        partner = anomaly.metadata.get("disagreement_partner")
        if isinstance(partner, list) and partner:
            pairs.append(_discrimination_pair(
                list(anomaly.feature_subset),
                partner,
                question="Which feature group retains LOFO — primary or partner?",
            ))

        sets.append(CompetingMechanismSet(
            anomaly_id=anomaly.id,
            feature_subset=list(anomaly.feature_subset),
            bucket=bucket,
            mechanisms=mechs[:3],
            discrimination_pairs=pairs[:2],
        ))
    return sets
