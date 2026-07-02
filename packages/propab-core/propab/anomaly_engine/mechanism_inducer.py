"""Mechanism inducer — LLM explains anomalies (never generates hypotheses)."""

from __future__ import annotations



import json

import logging

import re

from collections import defaultdict

from typing import Any



from propab.anomaly_engine.anomaly_objects import AnomalyObject

from propab.anomaly_engine.detector_config import ANOMALY_BUCKETS

from propab.anomaly_engine.mechanism_objects import MechanismObject
from propab.evidence_binding import BindingMetrics, filter_mechanism_anomalies



logger = logging.getLogger(__name__)



_BUCKET_LABELS = {

    "survivor": "Survivors — LOFO signal that survives group removal",

    "collapse": "Collapses — high within-family R² but LOFO degradation (family leakage)",

    "cross_family": "Cross-family — per-family LOFO varies (not uniform across groups)",

    "threshold": "Threshold effects — moderate within/LOFO tradeoffs",

    "symmetry_break": "Symmetry breaks — neighboring subsets flip LOFO sign",

    "outlier": "Outliers — extreme surprise or gap scores",

    "neighbor_disagreement": "Neighbor disagreements — cross-group opposing predictions",

}





def _group_by_bucket(anomalies: list[AnomalyObject]) -> dict[str, list[AnomalyObject]]:

    buckets: dict[str, list[AnomalyObject]] = defaultdict(list)

    for a in anomalies:

        bucket = str(a.metadata.get("bucket") or "unknown")

        buckets[bucket].append(a)

    return buckets





def _build_inducer_prompt(

    anomalies: list[AnomalyObject],

    *,

    question: str,

    domain_context: str = "",

) -> str:

    by_bucket = _group_by_bucket(anomalies)

    sections: list[str] = []

    for bucket in ANOMALY_BUCKETS:

        items = by_bucket.get(bucket) or []

        if not items:

            continue

        label = _BUCKET_LABELS.get(bucket, bucket)

        blob = [a.to_dict() for a in items[:4]]

        sections.append(f"### {label}\n{json.dumps(blob, indent=2)}")

    if not sections:

        sections.append(json.dumps([a.to_dict() for a in anomalies[:12]], indent=2))



    anomaly_block = "\n\n".join(sections)

    context_block = f"\nDomain context:\n{domain_context.strip()}\n" if domain_context.strip() else ""

    return f"""You are a scientific mechanism analyst. Do NOT generate research hypotheses.



Research question:

{question}

{context_block}

Observed anomalies from systematic feature-subset sweeps, stratified by anomaly class

(leave-one-group-out evaluation; simple linear / ridge / random-forest models):



{anomaly_block}



Task: Explain possible mechanisms spanning COMPETING anomaly classes above — not one

dominant theme. Each mechanism should draw from a different bucket when possible:

survivors (cross-group signal), collapses (family leakage), cross-family variation,

threshold effects, and neighbor disagreements (e.g. geometry predicts active while

fold similarity predicts inactive).



Return JSON ONLY with key "mechanisms" — an array of 3–5 objects, each with:

- explanation (string)

- candidate_features (list of feature names)

- supporting_anomalies (list of feature_subset keys joined with "|")

- assumptions_challenged (list of strings)

- confidence (float 0–1)

- anomaly_bucket (string: one of {list(ANOMALY_BUCKETS)})



Do NOT output hypotheses or experiment plans.

"""





def _parse_mechanisms_json(raw: str) -> list[dict[str, Any]]:

    text = (raw or "").strip()

    if text.startswith("```"):

        text = re.sub(r"^```(?:json)?\s*", "", text)

        text = re.sub(r"\s*```$", "", text)

    try:

        data = json.loads(text)

    except json.JSONDecodeError:

        m = re.search(r"\{[\s\S]*\}", text)

        if not m:

            return []

        try:

            data = json.loads(m.group(0))

        except json.JSONDecodeError:

            return []

    items = data.get("mechanisms") if isinstance(data, dict) else data

    return items if isinstance(items, list) else []





def _pick_from_buckets(by_bucket: dict[str, list[AnomalyObject]], limit: int) -> list[AnomalyObject]:

    """Round-robin across anomaly buckets for mechanism diversity (fixes.md P5)."""

    order = [b for b in ANOMALY_BUCKETS if by_bucket.get(b)]

    picked: list[AnomalyObject] = []

    while len(picked) < limit and order:

        added = False

        for bucket in order:

            pool = by_bucket.get(bucket) or []

            if pool:

                picked.append(pool.pop(0))

                added = True

                if len(picked) >= limit:

                    break

        if not added:

            break

    return picked





def _deterministic_mechanisms(anomalies: list[AnomalyObject]) -> list[MechanismObject]:

    """Offline fallback: one mechanism per anomaly bucket when available."""

    if not anomalies:

        return []



    by_bucket = _group_by_bucket(anomalies)

    picked = _pick_from_buckets(by_bucket, 5)

    if not picked:

        picked = anomalies[:5]



    out: list[MechanismObject] = []

    for a in picked:

        feats = list(a.feature_subset)

        gap = float(a.metadata.get("lofo_gap", 0))

        bucket = str(a.metadata.get("bucket") or "")

        partner = a.metadata.get("disagreement_partner")



        if bucket == "neighbor_disagreement" and partner:

            expl = (

                f"Cross-group disagreement: {feats} (group={a.metadata.get('primary_group')}, "

                f"LOFO={a.observed_score:.3f}) vs partner {partner} "

                f"(group={a.metadata.get('partner_group')}, LOFO={a.metadata.get('partner_lofo'):.3f}, "

                f"gap={a.metadata.get('partner_lofo_gap'):.2f}) — opposing predictive signals."

            )

            challenged = ["a single biophysical theme explains all RT activity patterns"]

            conf = 0.78

        elif bucket in {"survivor", "cross_family"} or a.anomaly_type in {"family_violation", "threshold_effect", "cluster_separation"}:

            expl = (

                f"Features {feats} retain relative LOFO signal "

                f"(LOFO={a.observed_score:.3f} vs group baseline {a.expected_score:.3f}, "

                f"bucket={bucket}), suggesting cross-group predictive structure."

            )

            challenged = ["group identity fully determines the target"]

            conf = min(0.85, 0.4 + max(0.0, a.observed_score + 0.5))

        elif bucket == "collapse" or a.anomaly_type in {"prediction_failure", "outlier"}:

            expl = (

                f"Features {feats} predict strongly within groups (within R²={a.metadata.get('within_family_r2', 'n/a')}) "

                f"but degrade under leave-one-group-out (LOFO={a.observed_score:.3f}, gap={gap:.2f}) — "

                "consistent with family-specific rather than cross-group signal."

            )

            challenged = ["group identity does not fully explain this feature subset"]

            conf = min(0.9, 0.5 + gap * 0.25)

        else:

            expl = (

                f"Features {feats} show {bucket or a.anomaly_type} pattern "

                f"(LOFO={a.observed_score:.3f}, surprise={a.surprise_score:.3f})."

            )

            challenged = ["group-agnostic prediction from features"]

            conf = 0.55



        out.append(MechanismObject(

            explanation=expl,

            candidate_features=feats,

            supporting_anomalies=["|".join(feats)],

            assumptions_challenged=challenged,

            confidence=round(max(0.35, conf), 3),

        ))

    return out


def _anomalies_by_key(anomalies: list[AnomalyObject]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for a in anomalies:
        key = "|".join(a.feature_subset)
        out[key] = a.to_dict()
    return out


def _apply_binding_to_mechanisms(
    mechanisms: list[MechanismObject],
    anomalies: list[AnomalyObject],
) -> tuple[list[MechanismObject], BindingMetrics]:
    by_key = _anomalies_by_key(anomalies)
    metrics = BindingMetrics()
    bound: list[MechanismObject] = []
    for m in mechanisms:
        d = m.to_dict()
        accepted = filter_mechanism_anomalies(d, by_key, metrics=metrics)
        bound.append(MechanismObject(
            explanation=m.explanation,
            candidate_features=list(m.candidate_features),
            supporting_anomalies=accepted,
            assumptions_challenged=list(m.assumptions_challenged),
            confidence=m.confidence,
            id=m.id,
        ))
    return bound, metrics


async def induce_mechanisms_llm(

    anomalies: list[AnomalyObject],

    *,

    llm: Any,

    session_id: str,

    question: str,

    domain_context: str = "",

) -> list[MechanismObject]:

    if not anomalies:

        return []

    prompt = _build_inducer_prompt(anomalies, question=question, domain_context=domain_context)

    try:

        raw = await llm.call(

            prompt=prompt,

            purpose="mechanism_inducer",

            session_id=session_id,

        )

        rows = _parse_mechanisms_json(raw)

    except Exception as exc:  # noqa: BLE001

        logger.warning("Mechanism inducer LLM failed (%s); using deterministic fallback.", exc)

        return _apply_binding_to_mechanisms(_deterministic_mechanisms(anomalies), anomalies)[0]



    if not rows:

        return _apply_binding_to_mechanisms(_deterministic_mechanisms(anomalies), anomalies)[0]



    out: list[MechanismObject] = []

    for row in rows[:5]:

        if not isinstance(row, dict):

            continue

        out.append(MechanismObject(

            explanation=str(row.get("explanation") or "")[:2000],

            candidate_features=[str(x) for x in (row.get("candidate_features") or [])][:12],

            supporting_anomalies=[str(x) for x in (row.get("supporting_anomalies") or [])][:8],

            assumptions_challenged=[str(x) for x in (row.get("assumptions_challenged") or [])][:6],

            confidence=float(row.get("confidence") or 0.5),

        ))

    raw = _deterministic_mechanisms(anomalies) if not out else out
    return _apply_binding_to_mechanisms(raw, anomalies)[0]





def induce_mechanisms_sync(anomalies: list[AnomalyObject]) -> list[MechanismObject]:

    if not anomalies:

        return [

            MechanismObject(

                explanation="No strong LOFO-surviving subsets in this sweep.",

                candidate_features=[],

                supporting_anomalies=[],

                assumptions_challenged=["group-agnostic prediction from features"],

                confidence=0.4,

            )

        ]

    return _apply_binding_to_mechanisms(_deterministic_mechanisms(anomalies), anomalies)[0]


