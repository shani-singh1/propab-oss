"""Anomaly detector — per-bucket selection with feature-group diversity caps."""
from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np

from propab.anomaly_engine.anomaly_objects import AnomalyObject
from propab.anomaly_engine.detector_config import ANOMALY_BUCKETS, DetectorConfig, FalsifiableExpectation
from propab.anomaly_engine.sweep_engine import SweepResult


def _lofo_gap(row: SweepResult) -> float:
    return float(row.metadata.get("lofo_gap", row.within_family_r2 - row.leave_one_family_out_r2))


def _matches_expectation(subset: list[str], exp: FalsifiableExpectation) -> bool:
    text = " ".join(subset).lower()
    return any(s.lower() in text for s in exp.feature_substrings)


def _classify_anomaly(row: SweepResult, cfg: DetectorConfig) -> str:
    gap = _lofo_gap(row)
    if row.surprise_score >= cfg.min_surprise and row.leave_one_family_out_r2 >= cfg.lofo_survival:
        if row.surprise_score >= cfg.lofo_survival or row.leave_one_family_out_r2 >= 0.0:
            return "family_violation"
    if gap >= cfg.collapse_gap and row.within_family_r2 >= cfg.min_within_for_collapse:
        return "prediction_failure"
    if gap < cfg.collapse_gap and row.within_family_r2 >= 0.15 and row.surprise_score > cfg.min_surprise:
        return "family_violation"
    if row.surprise_score >= 0.05 and row.global_r2 >= 0.5:
        return "cluster_separation"
    if row.surprise_score <= -0.5 or gap >= 0.8:
        return "outlier"
    return "threshold_effect"


def _neighboring_subsets(
    subset: list[str],
    all_subsets: list[list[str]],
    *,
    limit: int = 4,
) -> list[list[str]]:
    sset = set(subset)
    neighbors: list[list[str]] = []
    for other in all_subsets:
        if other == subset:
            continue
        if len(set(other) & sset) >= max(1, len(sset) - 1):
            neighbors.append(other)
        if len(neighbors) >= limit:
            break
    return neighbors


def _parse_subset(raw: list[str] | str) -> list[str]:
    if isinstance(raw, list):
        return raw
    text = str(raw).strip()
    if "|" in text:
        return [p.strip() for p in text.split("|") if p.strip()]
    return [text] if text else []


def _subset_key(subset: list[str] | str) -> tuple[str, ...]:
    return tuple(sorted(_parse_subset(subset)))


def _feature_group_map(groups: dict[str, list[str]]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for group, cols in groups.items():
        for c in cols:
            mapping[c.lower()] = group
    return mapping


def _dominant_group(subset: list[str], group_map: dict[str, str]) -> str:
    if not group_map:
        return "unknown"
    counts: Counter[str] = Counter()
    for f in subset:
        counts[group_map.get(f.lower(), "other")] += 1
    return counts.most_common(1)[0][0] if counts else "unknown"


def _group_ok(
    group_counts: Counter[str],
    grp: str,
    cfg: DetectorConfig,
) -> bool:
    """Enforce fixes.md P2: no feature group exceeds max_group_fraction of top_k."""
    return (group_counts[grp] + 1) <= cfg.group_cap()


def _bucket_score(bucket: str, row: SweepResult, by_key: dict[tuple[str, ...], SweepResult]) -> float:
    if bucket == "collapse":
        return _collapse_score(row)
    if bucket == "cross_family":
        return _cross_family_score(row)
    if bucket == "symmetry_break":
        return _symmetry_break_score(row, by_key)
    if bucket == "outlier":
        return abs(row.surprise_score) + _lofo_gap(row)
    if bucket == "threshold":
        return abs(row.surprise_score)
    return row.leave_one_family_out_r2


def _sort_for_diversity(
    rows: list[SweepResult],
    group_counts: Counter[str],
    group_map: dict[str, str],
    score_fn,
) -> list[SweepResult]:
    """Prefer candidates from underrepresented feature groups."""
    return sorted(
        rows,
        key=lambda r: (
            group_counts[_dominant_group(_parse_subset(r.feature_subset), group_map)],
            -score_fn(r),
        ),
    )


def _cross_family_score(row: SweepResult) -> float:
    pf = row.metadata.get("per_family_lofo") or {}
    if not isinstance(pf, dict) or len(pf) < 2:
        return 0.0
    vals = [float(v) for v in pf.values()]
    std = float(np.std(vals))
    pos = sum(1 for v in vals if v > 0.05)
    neg = sum(1 for v in vals if v < -0.05)
    opposing = 1.0 if pos >= 1 and neg >= 1 else 0.0
    return std + opposing * 0.25


def _collapse_score(row: SweepResult) -> float:
    """High within-family + LOFO collapse (fixes.md P3)."""
    gap = _lofo_gap(row)
    return gap * max(0.0, row.within_family_r2)


def _symmetry_break_score(row: SweepResult, by_key: dict[tuple[str, ...], SweepResult]) -> float:
    subset = _parse_subset(row.feature_subset)
    best = 0.0
    for neighbor in _neighboring_subsets(subset, [list(k) for k in by_key.keys()], limit=8):
        other = by_key.get(_subset_key(neighbor))
        if other is None:
            continue
        if np.sign(row.leave_one_family_out_r2) != np.sign(other.leave_one_family_out_r2):
            delta = abs(row.leave_one_family_out_r2 - other.leave_one_family_out_r2)
            best = max(best, delta)
    return best


def _find_neighbor_disagreements(
    deduped: list[SweepResult],
    group_map: dict[str, str],
    cfg: DetectorConfig,
) -> list[tuple[float, SweepResult, dict[str, Any]]]:
    """Cross-group pairs: survivor vs collapse (fixes.md P4)."""
    survivors = [
        r for r in deduped
        if r.leave_one_family_out_r2 >= cfg.lofo_survival and r.within_family_r2 >= 0.08
    ]
    collapses = [
        r for r in deduped
        if _lofo_gap(r) >= cfg.collapse_gap * 0.85 and r.within_family_r2 >= cfg.min_within_for_collapse * 0.8
    ]
    pairs: list[tuple[float, SweepResult, dict[str, Any]]] = []
    for srow in survivors:
        sg = _dominant_group(_parse_subset(srow.feature_subset), group_map)
        for crow in collapses:
            cg = _dominant_group(_parse_subset(crow.feature_subset), group_map)
            if sg == cg or sg == "unknown" or cg == "unknown":
                continue
            score = srow.leave_one_family_out_r2 - crow.leave_one_family_out_r2 + _lofo_gap(crow)
            meta = {
                "disagreement_partner": _parse_subset(crow.feature_subset),
                "partner_group": cg,
                "partner_lofo": crow.leave_one_family_out_r2,
                "partner_within": crow.within_family_r2,
                "partner_lofo_gap": _lofo_gap(crow),
                "primary_group": sg,
            }
            pairs.append((score, srow, meta))
    return sorted(pairs, key=lambda x: x[0], reverse=True)


def _row_to_object(
    row: SweepResult,
    all_subsets: list[list[str]],
    cfg: DetectorConfig,
    *,
    bucket: str,
    extra_metadata: dict[str, Any] | None = None,
) -> AnomalyObject:
    subset = _parse_subset(row.feature_subset)
    meta = {
        "within_family_r2": row.within_family_r2,
        "global_r2": row.global_r2,
        "lofo_gap": _lofo_gap(row),
        "model_name": row.model_name,
        "family_baseline_r2": row.family_baseline_r2,
        "bucket": bucket,
        "feature_group": _dominant_group(subset, _feature_group_map(cfg.feature_groups or {})),
        "lofo_family_std": row.metadata.get("lofo_family_std"),
        "per_family_lofo": row.metadata.get("per_family_lofo"),
    }
    if extra_metadata:
        meta.update(extra_metadata)
    return AnomalyObject(
        feature_subset=subset,
        metric_name="leave_one_family_out_r2",
        expected_score=row.family_baseline_r2,
        observed_score=row.leave_one_family_out_r2,
        surprise_score=row.surprise_score,
        anomaly_type=_classify_anomaly(row, cfg),
        affected_families=list(row.metadata.get("affected_families") or []),
        neighboring_subsets=_neighboring_subsets(subset, all_subsets),
        metadata=meta,
    )


def _dedupe_results(
    results: list[SweepResult],
) -> tuple[list[SweepResult], list[list[str]], dict[tuple[str, ...], SweepResult]]:
    by_subset: dict[tuple[str, ...], SweepResult] = {}
    for r in results:
        subset = _parse_subset(r.feature_subset)
        key = tuple(sorted(subset))
        prev = by_subset.get(key)
        if prev is None or r.leave_one_family_out_r2 > prev.leave_one_family_out_r2:
            r.feature_subset = subset
            by_subset[key] = r

    deduped = list(by_subset.values())
    all_subsets = [list(k) for k in by_subset.keys()]
    for row in deduped:
        row.metadata["lofo_gap"] = round(row.within_family_r2 - row.leave_one_family_out_r2, 4)
    return deduped, all_subsets, by_subset


def _bucket_candidates(
    deduped: list[SweepResult],
    by_key: dict[tuple[str, ...], SweepResult],
    cfg: DetectorConfig,
) -> dict[str, list[SweepResult]]:
    survivors = sorted(
        deduped,
        key=lambda r: (r.leave_one_family_out_r2, r.surprise_score, r.within_family_r2),
        reverse=True,
    )
    survivors = [r for r in survivors if r.within_family_r2 >= 0.08]

    collapses = sorted(deduped, key=_collapse_score, reverse=True)
    collapses = [
        r for r in collapses
        if _lofo_gap(r) >= cfg.collapse_gap and r.within_family_r2 >= cfg.min_within_for_collapse
    ]

    cross_family = sorted(deduped, key=_cross_family_score, reverse=True)
    cross_family = [
        r for r in cross_family
        if float(r.metadata.get("lofo_family_std") or 0) >= cfg.cross_family_lofo_std
        or _cross_family_score(r) >= cfg.cross_family_lofo_std
    ]

    thresholds = [
        r for r in deduped
        if 0.08 <= r.within_family_r2 <= 0.45
        and 0.05 <= _lofo_gap(r) < cfg.collapse_gap
    ]
    thresholds = sorted(thresholds, key=lambda r: abs(r.surprise_score), reverse=True)
    if len(thresholds) < 3:
        thresholds = sorted(
            [r for r in deduped if _classify_anomaly(r, cfg) == "threshold_effect"],
            key=lambda r: abs(r.surprise_score),
            reverse=True,
        )

    symmetry = sorted(deduped, key=lambda r: _symmetry_break_score(r, by_key), reverse=True)
    symmetry = [r for r in symmetry if _symmetry_break_score(r, by_key) >= 0.12]

    outliers = sorted(
        deduped,
        key=lambda r: (abs(r.surprise_score), _lofo_gap(r)),
        reverse=True,
    )
    outliers = [
        r for r in outliers
        if r.surprise_score <= -0.35 or _lofo_gap(r) >= 0.75 or _classify_anomaly(r, cfg) == "outlier"
    ]

    for exp in cfg.expectations:
        exp_rows = [r for r in deduped if _matches_expectation(_parse_subset(r.feature_subset), exp)]
        for row in sorted(exp_rows, key=lambda r: -_lofo_gap(r))[:2]:
            if _lofo_gap(row) >= exp.min_lofo_gap and row not in collapses:
                collapses.insert(0, row)

    return {
        "survivor": survivors,
        "collapse": collapses,
        "cross_family": cross_family,
        "threshold": thresholds,
        "symmetry_break": symmetry,
        "outlier": outliers,
        "neighbor_disagreement": deduped,
    }


def detect_anomalies(
    results: list[SweepResult],
    config: DetectorConfig | None = None,
) -> list[AnomalyObject]:
    """
    Per-bucket top-k selection (fixes.md P0/P1) with feature-group caps (P2).
    Never global surprise backfill.
    """
    cfg = config or DetectorConfig()
    if not results:
        return []

    deduped, all_subsets, by_key = _dedupe_results(results)
    group_map = _feature_group_map(cfg.feature_groups or {})
    bucket_slots = cfg.resolved_bucket_slots()

    seen_keys: set[tuple[str, ...]] = set()
    group_counts: Counter[str] = Counter()
    selected: list[tuple[SweepResult, str, dict[str, Any] | None]] = []
    bucket_counts: Counter[str] = Counter()

    def _can_add(row: SweepResult, bucket: str) -> bool:
        if _subset_key(row.feature_subset) in seen_keys:
            return False
        if bucket_counts[bucket] >= bucket_slots.get(bucket, 0):
            return False
        grp = _dominant_group(_parse_subset(row.feature_subset), group_map)
        if group_map and not _group_ok(group_counts, grp, cfg):
            return False
        return True

    def _add(row: SweepResult, bucket: str, extra: dict[str, Any] | None = None) -> None:
        if not _can_add(row, bucket):
            return
        seen_keys.add(_subset_key(row.feature_subset))
        grp = _dominant_group(_parse_subset(row.feature_subset), group_map)
        group_counts[grp] += 1
        bucket_counts[bucket] += 1
        selected.append((row, bucket, extra))

    candidates = _bucket_candidates(deduped, by_key, cfg)

    for bucket in ANOMALY_BUCKETS:
        if bucket == "neighbor_disagreement":
            continue
        budget = bucket_slots.get(bucket, 0)
        if budget <= 0:
            continue
        for row in _sort_for_diversity(
            candidates.get(bucket, []),
            group_counts,
            group_map,
            lambda r, b=bucket: _bucket_score(b, r, by_key),
        ):
            _add(row, bucket)
            if bucket_counts[bucket] >= budget:
                break

    nd_budget = bucket_slots.get("neighbor_disagreement", 0)
    if nd_budget > 0:
        for score, srow, meta in _find_neighbor_disagreements(deduped, group_map, cfg):
            if bucket_counts["neighbor_disagreement"] >= nd_budget:
                break
            if _subset_key(srow.feature_subset) in seen_keys:
                continue
            grp = meta.get("primary_group") or _dominant_group(_parse_subset(srow.feature_subset), group_map)
            if group_map and not _group_ok(group_counts, grp, cfg):
                continue
            seen_keys.add(_subset_key(srow.feature_subset))
            group_counts[grp] += 1
            bucket_counts["neighbor_disagreement"] += 1
            extra = {**meta, "disagreement_score": round(score, 4)}
            selected.append((srow, "neighbor_disagreement", extra))

    objects = [
        _row_to_object(row, all_subsets, cfg, bucket=bucket, extra_metadata=extra)
        for row, bucket, extra in selected
    ]

    for obj in objects:
        bucket = obj.metadata.get("bucket", "")
        if bucket == "survivor":
            obj.anomaly_type = "family_violation"
        elif bucket == "collapse":
            obj.anomaly_type = "prediction_failure"
        elif bucket == "outlier":
            obj.anomaly_type = "outlier"
        elif bucket == "threshold" and obj.anomaly_type not in {"outlier", "prediction_failure"}:
            obj.anomaly_type = "threshold_effect"

    return objects[: cfg.top_k]


def summarize_anomalies(objects: list[AnomalyObject]) -> dict[str, Any]:
    by_type: dict[str, int] = {}
    by_bucket: dict[str, int] = {}
    by_group: dict[str, int] = {}
    for o in objects:
        by_type[o.anomaly_type] = by_type.get(o.anomaly_type, 0) + 1
        bucket = str(o.metadata.get("bucket") or "unknown")
        by_bucket[bucket] = by_bucket.get(bucket, 0) + 1
        grp = str(o.metadata.get("feature_group") or "unknown")
        by_group[grp] = by_group.get(grp, 0) + 1
    scores = [o.surprise_score for o in objects]
    lofo_scores = [o.observed_score for o in objects]
    max_group_frac = max(by_group.values()) / len(objects) if objects else 0.0
    return {
        "n_anomalies": len(objects),
        "by_type": by_type,
        "by_bucket": by_bucket,
        "by_feature_group": by_group,
        "max_group_fraction": round(max_group_frac, 3),
        "mean_surprise": float(np.mean(scores)) if scores else 0.0,
        "max_surprise": float(max(scores)) if scores else 0.0,
        "best_lofo": float(max(lofo_scores)) if lofo_scores else 0.0,
        "mean_lofo_gap": float(np.mean([o.metadata.get("lofo_gap", 0) for o in objects])) if objects else 0.0,
    }
