"""Generic anomaly pipeline: sweep → detect → induce."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from propab.anomaly_engine.anomaly_detector import detect_anomalies, summarize_anomalies
from propab.anomaly_engine.detector_config import DetectorConfig
from propab.anomaly_engine.artifacts import (
    write_anomalies,
    write_competing_mechanisms,
    write_mechanism_report,
    write_mechanisms,
    write_sweep_parquet,
)
from propab.anomaly_engine.competing_mechanisms import build_competing_sets
from propab.anomaly_engine.mechanism_inducer import induce_mechanisms_sync
from propab.anomaly_engine.sweep_engine import SweepConfig, run_sweep


async def run_anomaly_pipeline(
    df: pd.DataFrame,
    sweep_config: SweepConfig,
    *,
    out_dir: Path,
    question: str,
    detector_config: DetectorConfig | None = None,
    domain_context: str = "",
    use_llm: bool = False,
    llm: Any = None,
    session_id: str = "anomaly-pipeline",
) -> dict[str, Any]:
    """
    Domain-agnostic: pass any tabular dataframe + SweepConfig.
    Writes sweep/anomaly/mechanism artifacts (not paper.pdf).
    """
    sweep_results = run_sweep(df, sweep_config)
    anomalies = detect_anomalies(sweep_results, detector_config)
    summary = summarize_anomalies(anomalies)

    if use_llm and llm is not None:
        from propab.anomaly_engine.mechanism_inducer import induce_mechanisms_llm
        mechanisms = await induce_mechanisms_llm(
            anomalies,
            llm=llm,
            session_id=session_id,
            question=question,
            domain_context=domain_context,
        )
    else:
        mechanisms = induce_mechanisms_sync(anomalies)

    competing = build_competing_sets(anomalies)

    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "sweep_results": write_sweep_parquet(sweep_results, out_dir / "sweep_results.parquet"),
        "anomaly_objects": write_anomalies(anomalies, out_dir / "anomaly_objects.json"),
        "mechanism_objects": write_mechanisms(mechanisms, out_dir / "mechanism_objects.json"),
        "competing_mechanisms": write_competing_mechanisms(competing, out_dir / "competing_mechanisms.json"),
        "mechanism_report": write_mechanism_report(
            path=out_dir / "mechanism_report.md",
            question=question,
            anomalies=anomalies,
            mechanisms=mechanisms,
            summary=summary,
        ),
    }
    return {
        "n_sweep_experiments": len(sweep_results),
        "n_anomalies": len(anomalies),
        "n_mechanisms": len(mechanisms),
        "n_competing_sets": len(competing),
        "summary": summary,
        "paths": {k: str(v) for k, v in paths.items()},
    }
