#!/usr/bin/env python3
"""
Run anomaly engine pipeline (fixes.md Phases 1–6).

Core engine is domain-agnostic; this script uses the Mandrake demo adapter.

Outputs:
  artifacts/sweep_results.parquet
  artifacts/anomaly_objects.json
  artifacts/mechanism_objects.json
  artifacts/mechanism_report.md
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "services" / "orchestrator"))
sys.path.insert(0, str(ROOT / "packages" / "propab-core"))

from demo.mandrake.domain import (
    MANDRAKE_QUESTION,
    MECHANISM_DOMAIN_CONTEXT,
    load_frame,
    mandrake_detector_config,
    mandrake_sweep_config,
    repo_data_dir,
)
from propab.anomaly_engine.pipeline import run_anomaly_pipeline


def main() -> int:
    parser = argparse.ArgumentParser(description="Run anomaly pipeline (Mandrake demo adapter)")
    parser.add_argument("--data-dir", default=str(repo_data_dir(ROOT)))
    parser.add_argument("--out-dir", default=str(ROOT / "artifacts"))
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--max-subset-size", type=int, default=3)
    parser.add_argument("--use-llm", action="store_true", help="Use LLM for mechanism induction")
    parser.add_argument(
        "--models",
        default="LinearRegression,Ridge",
        help="Comma-separated model names (omit RandomForest for speed)",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not (data_dir / "handcrafted_features.csv").is_file():
        print(f"Missing Mandrake data at {data_dir}", file=sys.stderr)
        return 1

    from propab.anomaly_engine.sweep_engine import SweepConfig  # noqa: E402

    df = load_frame(data_dir)
    base_sweep = mandrake_sweep_config(max_subset_size=args.max_subset_size)
    sweep_cfg = SweepConfig(
        target_column=base_sweep.target_column,
        family_column=base_sweep.family_column,
        feature_columns=base_sweep.feature_columns,
        feature_groups=base_sweep.feature_groups,
        max_subset_size=base_sweep.max_subset_size,
        exclude_columns=base_sweep.exclude_columns,
        model_names=tuple(m.strip() for m in args.models.split(",") if m.strip()),
    )
    detector_cfg = mandrake_detector_config(top_k=args.top_k)

    llm = None
    if args.use_llm:
        from propab.config import settings
        from propab.db import create_engine, create_session_factory
        from propab.events import EventEmitter
        from propab.llm import LLMClient

        engine = create_engine(settings.database_url)
        sf = create_session_factory(engine)
        emitter = EventEmitter(session_factory=sf)
        llm = LLMClient(
            provider=settings.llm_provider,
            model=settings.llm_model,
            api_key=settings.llm_api_secret,
            emitter=emitter,
            session_factory=sf,
        )

    result = asyncio.run(
        run_anomaly_pipeline(
            df,
            sweep_cfg,
            out_dir=Path(args.out_dir),
            question=MANDRAKE_QUESTION,
            detector_config=detector_cfg,
            domain_context=MECHANISM_DOMAIN_CONTEXT,
            use_llm=args.use_llm and llm is not None,
            llm=llm,
        )
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
