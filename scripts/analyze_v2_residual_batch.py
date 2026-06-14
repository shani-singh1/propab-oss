#!/usr/bin/env python3
"""
Analyze V2 residual batch (fixes.md) — four learning questions.

Q1: Did identical predictions disappear?
Q2: Are trajectory residuals smaller than V1 theme_entropy?
Q3: Do residuals become directional (bias)?
Q4: Does analyst change predictions from residual history?
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean, pstdev

ROOT = Path(__file__).resolve().parents[1]

DYNAMICS_KEYS = (
    "start_H",
    "growth_rate",
    "saturation_H",
    "cross_H_1_5_at_tested",
    "cross_H_2_0_at_tested",
)


def _signs(values: list[float]) -> list[int]:
    return [1 if v > 0 else (-1 if v < 0 else 0) for v in values]


def _directional(values: list[float]) -> bool:
    nz = [s for s in _signs(values) if s != 0]
    return len(set(nz)) <= 1 if nz else False


def _unique_predictions(preds: list[dict]) -> int:
    keys = ("start_H", "growth_rate", "saturation_H", "cross_H_1_5_at_tested", "cross_H_2_0_at_tested")
    sigs = []
    for p in preds:
        sigs.append(tuple(round(float(p.get(k) or 0), 4) for k in keys))
    return len(set(sigs))


def _load_v1_baseline(path: Path) -> dict | None:
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data.get("residuals") or []
    ent_res = [float((r.get("residual") or {}).get("theme_entropy", 0)) for r in rows]
    if not ent_res:
        return None
    return {
        "n": len(ent_res),
        "theme_entropy_residuals": ent_res,
        "mean_abs": round(mean(abs(x) for x in ent_res), 4),
        "directional": _directional(ent_res),
    }


def analyze(residuals: list[dict], *, v1_path: Path | None) -> dict:
    eval_preds = [r.get("prediction") or {} for r in residuals]
    eval_res = [r.get("residual") or {} for r in residuals]
    next_preds = [r.get("next_candidate_prediction") or {} for r in residuals]

    v2_mode = any(
        any(k in (r.get("prediction") or {}) for k in DYNAMICS_KEYS)
        for r in residuals
    )

    dynamics_residuals: dict[str, list[float]] = {k: [] for k in DYNAMICS_KEYS}
    for res in eval_res:
        for k in DYNAMICS_KEYS:
            if k in res:
                dynamics_residuals[k].append(float(res[k]))

    abs_by_key = {
        k: round(mean(abs(v) for v in vals), 4) if vals else None
        for k, vals in dynamics_residuals.items()
    }
    directional_by_key = {
        k: _directional(vals) for k, vals in dynamics_residuals.items() if vals
    }

    q1_eval = {
        "n_runs": len(eval_preds),
        "unique_v2_prediction_signatures": _unique_predictions(eval_preds),
        "predictions_identical": _unique_predictions(eval_preds) <= 1,
        "sample_predictions": eval_preds[:3],
    }
    q1_next = {
        "unique_next_candidate_signatures": _unique_predictions(next_preds) if any(next_preds) else 0,
        "next_predictions_identical": _unique_predictions(next_preds) <= 1 if any(next_preds) else None,
        "sample_next": next_preds[:3],
    }

    v1 = _load_v1_baseline(v1_path) if v1_path else None
    v2_mean_abs = None
    flat = [abs(v) for vals in dynamics_residuals.values() for v in vals]
    if flat:
        v2_mean_abs = round(mean(flat), 4)

    q2 = {
        "v2_mean_abs_residual": v2_mean_abs,
        "v2_abs_by_metric": abs_by_key,
        "v1_baseline": v1,
        "trajectory_residuals_smaller_than_v1": (
            v2_mean_abs is not None and v1 is not None and v2_mean_abs < v1["mean_abs"]
        ),
    }

    q3 = {
        "directional_by_metric": directional_by_key,
        "any_directional_bias": any(directional_by_key.values()),
        "residual_vectors": {k: [round(v, 4) for v in vals] for k, vals in dynamics_residuals.items()},
    }

    growth_preds = [float(p.get("growth_rate") or 0) for p in next_preds if p]
    sat_preds = [float(p.get("saturation_H") or 0) for p in next_preds if p]
    q4 = {
        "next_candidate_growth_rates": growth_preds,
        "next_candidate_saturation_H": sat_preds,
        "growth_rate_trend": (
            "decreasing" if len(growth_preds) >= 2 and growth_preds[-1] < growth_preds[0] else
            "increasing" if len(growth_preds) >= 2 and growth_preds[-1] > growth_preds[0] else
            "flat_or_insufficient"
        ),
        "saturation_trend": (
            "decreasing" if len(sat_preds) >= 2 and sat_preds[-1] < sat_preds[0] else
            "increasing" if len(sat_preds) >= 2 and sat_preds[-1] > sat_preds[0] else
            "flat_or_insufficient"
        ),
        "learning_signal": (
            len(growth_preds) >= 2
            and (growth_preds[-1] < growth_preds[0] or sat_preds[-1] < sat_preds[0])
            and q1_next["unique_next_candidate_signatures"] > 1
        ),
    }

    return {
        "v2_eval_mode": v2_mode,
        "questions": {
            "Q1_identical_predictions_gone": {
                "eval_bound_candidate": q1_eval,
                "next_candidates_from_analyst": q1_next,
                "verdict": (
                    "PASS — predictions vary"
                    if (not q1_eval["predictions_identical"] or not q1_next.get("next_predictions_identical", True))
                    else "FAIL — still uniform"
                ),
            },
            "Q2_trajectory_residuals_smaller": q2,
            "Q3_directional_residuals": q3,
            "Q4_analyst_learns_from_history": q4,
        },
    }


async def _enrich_from_db(rows: list[dict]) -> list[dict]:
    """Attach next-candidate V2 predictions from lifetime.ingested payloads."""
    import asyncio
    from sqlalchemy import text
    from propab.config import settings
    from propab.db import create_engine, create_session_factory

    eng = create_engine(settings.database_url)
    sf = create_session_factory(eng)
    out = []
    async with sf() as db:
        for row in rows:
            cid = row["campaign_id"]
            payload = (
                await db.execute(
                    text(
                        """
                        SELECT payload_json FROM events
                        WHERE session_id = CAST(:id AS uuid)
                          AND step = 'lifetime.ingested'
                        ORDER BY created_at DESC LIMIT 1
                        """
                    ),
                    {"id": cid},
                )
            ).scalar_one_or_none()
            enriched = dict(row)
            if payload:
                p = payload if isinstance(payload, dict) else json.loads(payload)
                cand_id = p.get("candidate_policy_id")
                if cand_id:
                    from propab.policy_store import PolicyStore
                    store = PolicyStore.load()
                    cand = store.get_policy(cand_id)
                    if cand:
                        enriched["next_candidate_prediction"] = cand.predicted_effects.to_dict()
                enriched["entropy_eval_mode"] = (p.get("evaluation") or {}).get("entropy_eval_mode")
                enriched["observed_trajectory"] = (p.get("evaluation") or {}).get("observed_trajectory")
            out.append(enriched)
    await eng.dispose()
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--infile",
        default=str(ROOT / "artifacts" / "policy_residuals_v2.json"),
    )
    parser.add_argument(
        "--v1-baseline",
        default=str(ROOT / "artifacts" / "policy_residuals.json"),
    )
    parser.add_argument("--no-db-enrich", action="store_true")
    args = parser.parse_args()

    path = Path(args.infile)
    if not path.is_file():
        print(f"Missing {path}", file=sys.stderr)
        return 1

    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data.get("residuals") or []
    if not rows:
        print("No residuals", file=sys.stderr)
        return 1

    if not args.no_db_enrich:
        import asyncio
        rows = asyncio.run(_enrich_from_db(rows))

    report = analyze(rows, v1_path=Path(args.v1_baseline))
    out_path = path.with_name("v2_batch_analysis.json")
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
