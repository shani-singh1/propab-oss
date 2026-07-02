"""Anomaly-driven seed hypothesis generation (Phase 7 integration)."""
from __future__ import annotations

import json
from pathlib import Path

from propab.anomaly_engine.artifacts import read_competing_mechanisms, read_mechanisms
from propab.anomaly_engine.competing_mechanisms import CompetingMechanismSet
from propab.anomaly_engine.mechanism_objects import MechanismObject
from propab.domain_adapters.mandrake_adapter import infer_biology_theme, resolve_mandrake_features
from propab.llm import LLMClient
from propab.types import EventType
from propab.events import EventEmitter
from services.orchestrator.schemas import Prior, RankedHypothesis
from services.orchestrator.hypotheses import _parse_hypothesis_json, _ensure_null_hypothesis
from services.orchestrator.intake import ParsedQuestion


def _mechanism_context(mechanisms: list[MechanismObject]) -> str:
    lines = ["Mechanisms explaining experimentally observed anomalies:"]
    for m in mechanisms:
        theme = infer_biology_theme(m.explanation, m.candidate_features)
        lines.append(f"- [{m.confidence:.2f}] ({theme}) {m.explanation[:400]}")
        lines.append(f"  Features: {', '.join(m.candidate_features)}")
        lines.append(f"  Challenges: {'; '.join(m.assumptions_challenged)}")
    return "\n".join(lines)


def _diverse_mechanisms(mechanisms: list[MechanismObject], limit: int = 5) -> list[MechanismObject]:
    """Round-robin by biology theme for mechanism diversity (fixes.md P6)."""
    buckets: dict[str, list[MechanismObject]] = {}
    for m in mechanisms:
        theme = infer_biology_theme(m.explanation, m.candidate_features)
        buckets.setdefault(theme, []).append(m)
    order = sorted(buckets.keys(), key=lambda t: (-len(buckets[t]), t))
    picked: list[MechanismObject] = []
    while len(picked) < limit:
        added = False
        for theme in order:
            pool = buckets.get(theme) or []
            if pool:
                picked.append(pool.pop(0))
                added = True
                if len(picked) >= limit:
                    break
        if not added:
            break
    return picked or mechanisms[:limit]


def _mandrake_plan_json(features: list[str], *, mechanism_id: str | None = None, compare: list[str] | None = None) -> str:
    return json.dumps({
        "tool": "mandrake_verification",
        "methodology": "LOFO",
        "features": features,
        "model": "ridge",
        "compare_features": compare or [],
        "mechanism_id": mechanism_id,
    })


def _competing_mechanism_context(sets: list[CompetingMechanismSet]) -> str:
    lines = ["Competing mechanism sets (each anomaly has multiple explanations — hypotheses must DISCRIMINATE):"]
    for i, s in enumerate(sets[:5], 1):
        lines.append(f"\n## Anomaly set {i} (bucket={s.bucket}, features={s.feature_subset})")
        for j, m in enumerate(s.mechanisms[:3]):
            label = (m.assumptions_challenged or ["?"])[0]
            lines.append(f"  Mechanism {chr(65 + j)} [{label}]: {m.explanation[:280]}")
            lines.append(f"    features: {', '.join(m.candidate_features)}")
        for p in s.discrimination_pairs[:2]:
            lines.append(
                f"  Discriminating experiment: {p.get('question')} "
                f"primary={p.get('primary_features')} vs compare={p.get('compare_features')}"
            )
    return "\n".join(lines)


def _discrimination_hypothesis_fallback(
    sets: list[CompetingMechanismSet],
    idx: int,
) -> tuple[str, list[str], list[str], str | None]:
    """Deterministic A-vs-B hypothesis from competing set."""
    if not sets:
        return "Discriminate competing mechanisms under LOFO", [], [], None
    s = sets[idx % len(sets)]
    pair = s.discrimination_pairs[0] if s.discrimination_pairs else {}
    primary = list(pair.get("primary_features") or s.feature_subset or [])
    compare = list(pair.get("compare_features") or [])
    if len(s.mechanisms) >= 2:
        a_label = (s.mechanisms[0].assumptions_challenged or ["A"])[0]
        b_label = (s.mechanisms[1].assumptions_challenged or ["B"])[0]
        text = (
            f"Mechanism {a_label} (features {primary}) retains higher LOFO R² than "
            f"mechanism {b_label} (features {compare or s.mechanisms[1].candidate_features}) "
            f"under leave-one-family-out — discriminating cross-group biophysics from family leakage."
        )
        return text, primary, compare or list(s.mechanisms[1].candidate_features), s.id
    return f"Verify anomaly bucket {s.bucket} for {s.feature_subset}", primary, compare, s.id


def _build_anomaly_seed_prompt(
    parsed: ParsedQuestion,
    prior: Prior,
    mechanisms: list[MechanismObject],
    max_hypotheses: int,
    prior_round_findings: str = "",
    competing_sets: list[CompetingMechanismSet] | None = None,
) -> str:
    sets = competing_sets or []
    mech_block = _competing_mechanism_context(sets) if sets else _mechanism_context(mechanisms)
    prior_block = (
        f"\nResults from previous research rounds:\n{prior_round_findings}\n"
        if prior_round_findings.strip()
        else ""
    )
    return f"""You are a research hypothesis generator for mechanism discrimination on Mandrake RT data.

Research question: {parsed.text}

{mech_block}

Prior established facts:
{json.dumps(prior.established_facts)}

Prior open gaps:
{json.dumps(prior.open_gaps)}

Prior dead ends (do not repeat these):
{json.dumps(prior.dead_ends)}
{prior_block}

These mechanisms explain experimentally observed anomalies from systematic LOFO feature sweeps
(stratified by anomaly bucket: survivors, collapses, cross-family, threshold, neighbor disagreements).

Generate exactly {max_hypotheses} hypotheses whose purpose is to DISTINGUISH competing mechanisms (A vs B vs C),
NOT merely verify a single mechanism.

Each hypothesis must pit one mechanism against another using compare_features in test_methodology.

Requirements:
- Each hypothesis must be specific, falsifiable, and testable via mandrake_verification on /app/mandrake-data.
- test_methodology MUST be JSON: {{"tool":"mandrake_verification","methodology":"LOFO","features":[...],"model":"ridge","compare_features":[...]}}
- compare_features MUST be non-empty for discrimination hypotheses (mechanism A vs B).
- Use real feature names from candidate_features (e.g. t70_raw, triad_best_rmsd, foldseek_best_TM).
- Pit mechanisms against each other (thermal vs geometry vs foldseek collapse).
- Include mechanism_id referencing the supporting mechanism when possible.
- One hypothesis may be a null (group identity fully explains residual signal).

Return JSON array only. Each item: {{id, text, test_methodology, gap_reference, expected_result, feature_subset, mechanism_id}}
"""


def load_competing_mechanisms_from_artifacts(artifacts_dir: Path | str) -> list[CompetingMechanismSet]:
    path = resolve_artifacts_dir(artifacts_dir) / "competing_mechanisms.json"
    if not path.is_file():
        return []
    return read_competing_mechanisms(path)


def load_mechanisms_from_artifacts(artifacts_dir: Path | str) -> list[MechanismObject]:
    path = resolve_artifacts_dir(artifacts_dir) / "mechanism_objects.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing mechanism artifacts: {path}")
    return read_mechanisms(path)


def resolve_artifacts_dir(artifacts_dir: Path | str) -> Path:
    """Resolve relative artifact dirs for local runs and Docker (/app/artifacts)."""
    raw = Path(artifacts_dir)
    if raw.is_absolute() and (raw / "mechanism_objects.json").is_file():
        return raw

    candidates: list[Path] = []
    if raw.is_absolute():
        candidates.append(raw)
    else:
        candidates.extend([
            Path.cwd() / raw,
            Path("/app") / raw,
            Path(__file__).resolve().parents[2] / raw,
        ])
        try:
            from propab.config import settings
            data_root = Path(settings.propab_data_dir or "")
            if str(data_root):
                candidates.append(data_root / raw)
        except Exception:  # noqa: BLE001
            pass

    for candidate in candidates:
        if (candidate / "mechanism_objects.json").is_file():
            return candidate
    return raw if raw.is_absolute() else Path.cwd() / raw


async def generate_anomaly_seed_hypotheses(
    parsed: ParsedQuestion,
    prior: Prior,
    mechanisms: list[MechanismObject],
    max_hypotheses: int,
    llm: LLMClient,
    session_id: str,
    emitter: EventEmitter,
    *,
    prior_round_findings: str = "",
    artifacts_dir: str | Path = "artifacts",
) -> list[RankedHypothesis]:
    diverse = _diverse_mechanisms(mechanisms, limit=5)
    competing = load_competing_mechanisms_from_artifacts(artifacts_dir)
    prompt = _build_anomaly_seed_prompt(
        parsed, prior, diverse, max_hypotheses, prior_round_findings,
        competing_sets=competing,
    )
    raw = await llm.call(prompt=prompt, purpose="anomaly_seed_generation", session_id=session_id)
    generated = _parse_hypothesis_json(raw)
    if not generated:
        raw_retry = await llm.call(
            prompt=prompt + f"\n\nReturn ONLY a JSON array of exactly {max_hypotheses} objects.",
            purpose="anomaly_seed_generation_retry",
            session_id=session_id,
        )
        generated = _parse_hypothesis_json(raw_retry)

    await emitter.emit(
        session_id=session_id,
        event_type=EventType.HYPO_GENERATED,
        step="hypothesis.anomaly_seed",
        payload={
            "source": "anomaly",
            "n_mechanisms": len(diverse),
            "n_competing_sets": len(competing),
            "hypotheses": generated if isinstance(generated, list) else [],
        },
    )

    hypotheses: list[RankedHypothesis] = []
    gen_list = generated if isinstance(generated, list) else []
    for idx in range(max_hypotheses):
        rank = idx + 1
        entry = gen_list[idx] if idx < len(gen_list) and isinstance(gen_list[idx], dict) else {}
        mech = diverse[idx % len(diverse)] if diverse else None
        compare: list[str] = list(entry.get("compare_features") or [])
        if competing and not compare:
            _, _, compare, set_id = _discrimination_hypothesis_fallback(competing, idx)
            if set_id and not entry.get("mechanism_id"):
                entry["mechanism_id"] = set_id
        feats = list(entry.get("feature_subset") or (mech.candidate_features if mech else []) or resolve_mandrake_features(str(entry.get("text") or "")))
        if not feats and mech:
            feats = list(mech.candidate_features)
        if not entry.get("text") and competing:
            text, feats, compare, set_id = _discrimination_hypothesis_fallback(competing, idx)
            if set_id:
                entry["mechanism_id"] = set_id
        else:
            text = str(entry.get("text") or (
                f"Mechanism {rank}: {mech.explanation[:120]}" if mech else f"Discriminate competing mechanisms (variant {rank})"
            ))
        mech_id = entry.get("mechanism_id") or (getattr(mech, "id", None) if mech else None)
        methodology = str(entry.get("test_methodology") or _mandrake_plan_json(
            feats, mechanism_id=str(mech_id) if mech_id else None, compare=compare or None,
        ))
        theme = infer_biology_theme(text, feats)
        hypotheses.append(RankedHypothesis(
            id=str(entry.get("id", f"anomaly_h{rank}")),
            text=text,
            test_methodology=methodology,
            scores={"composite": round(max(0.2, 1.0 - idx * 0.08), 3), "question_relevance": 0.85},
            rank=rank,
        ))
        # Attach extra fields via scores dict hack — campaign_loop seed_dicts need extension
        hypotheses[-1].scores["theme_id"] = theme  # type: ignore[assignment]
        hypotheses[-1].scores["feature_subset"] = feats  # type: ignore[assignment]
        if mech_id:
            hypotheses[-1].scores["mechanism_id"] = str(mech_id)  # type: ignore[assignment]
    return _ensure_null_hypothesis(hypotheses, parsed.text)
