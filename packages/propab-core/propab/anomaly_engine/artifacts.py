"""Artifact I/O for anomaly engine deliverables."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from propab.anomaly_engine.anomaly_objects import AnomalyObject
from propab.anomaly_engine.competing_mechanisms import CompetingMechanismSet
from propab.anomaly_engine.mechanism_objects import MechanismObject
from propab.anomaly_engine.sweep_engine import SweepResult


def _pandas():
    import pandas as pd
    return pd


def write_sweep_parquet(results: list[SweepResult], path: Path) -> Path:
    from propab.anomaly_engine.sweep_engine import sweep_results_to_dataframe

    path.parent.mkdir(parents=True, exist_ok=True)
    df = sweep_results_to_dataframe(results)
    df.to_parquet(path, index=False)
    return path


def read_sweep_parquet(path: Path) -> list[SweepResult]:
    from propab.anomaly_engine.sweep_engine import dataframe_to_sweep_results

    df = _pandas().read_parquet(path)
    return dataframe_to_sweep_results(df)


def write_anomalies(objects: list[AnomalyObject], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [o.to_dict() for o in objects]
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def read_anomalies(path: Path) -> list[AnomalyObject]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return [AnomalyObject.from_dict(d) for d in data if isinstance(d, dict)]


def write_mechanisms(objects: list[MechanismObject], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [o.to_dict() for o in objects]
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def read_mechanisms(path: Path) -> list[MechanismObject]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return [MechanismObject.from_dict(d) for d in data if isinstance(d, dict)]


def write_competing_mechanisms(objects: list[CompetingMechanismSet], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([o.to_dict() for o in objects], indent=2), encoding="utf-8")
    return path


def read_competing_mechanisms(path: Path) -> list[CompetingMechanismSet]:
    data = json.loads(path.read_text(encoding="utf-8"))
    out: list[CompetingMechanismSet] = []
    for d in data if isinstance(data, list) else []:
        if not isinstance(d, dict):
            continue
        mechs = [MechanismObject.from_dict(m) for m in (d.get("mechanisms") or []) if isinstance(m, dict)]
        out.append(CompetingMechanismSet(
            id=str(d.get("id") or ""),
            anomaly_id=str(d.get("anomaly_id") or ""),
            feature_subset=list(d.get("feature_subset") or []),
            bucket=str(d.get("bucket") or ""),
            mechanisms=mechs,
            discrimination_pairs=list(d.get("discrimination_pairs") or []),
        ))
    return out


def write_mechanism_report(
    *,
    path: Path,
    question: str,
    anomalies: list[AnomalyObject],
    mechanisms: list[MechanismObject],
    summary: dict[str, Any],
) -> Path:
    lines = [
        "# Anomaly Engine — Mechanism Report",
        "",
        f"**Question:** {question}",
        "",
        "## Summary",
        "",
        f"- Anomalies detected: {summary.get('n_anomalies', len(anomalies))}",
        f"- Mean surprise score: {summary.get('mean_surprise', 0):.4f}",
        f"- Max surprise score: {summary.get('max_surprise', 0):.4f}",
        "",
        "## Top Anomalies",
        "",
    ]
    for a in anomalies[:15]:
        lines.extend([
            f"### `{a.anomaly_type}` — `{', '.join(a.feature_subset)}`",
            "",
            f"- **Surprise:** {a.surprise_score:.4f} (LOFO={a.observed_score:.4f}, "
            f"family baseline={a.expected_score:.4f})",
            f"- **Within-family R²:** {a.metadata.get('within_family_r2', 'n/a')}",
            "",
        ])
    lines.extend(["## Induced Mechanisms", ""])
    for m in mechanisms:
        lines.extend([
            f"### Mechanism (confidence {m.confidence:.2f})",
            "",
            m.explanation,
            "",
            f"- **Features:** {', '.join(m.candidate_features)}",
            f"- **Challenges:** {'; '.join(m.assumptions_challenged)}",
            "",
        ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
