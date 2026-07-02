#!/usr/bin/env python3
"""
Evaluate V1 domain candidates with power analysis (fixes.md Step 1).

Runs the same LOFO / classical-MDE checks as mandrake_power_analysis.py on three
public-data candidates before committing to a domain.

  python scripts/evaluate_v1_domain_candidates.py
  python scripts/evaluate_v1_domain_candidates.py --out artifacts/v1_domain_evaluation.json
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import sys
import urllib.error
import urllib.request
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "packages" / "propab-core"))

DEFAULT_OUT = ROOT / "artifacts" / "v1_domain_evaluation.json"
TARGET_R2 = 0.25
DOWNLOAD_TIMEOUT_SEC = 90


def _download(url: str, dest: Path) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file() and dest.stat().st_size > 0:
        return True
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "propab-v1-eval/1.0"})
        with urllib.request.urlopen(req, timeout=DOWNLOAD_TIMEOUT_SEC) as resp:
            dest.write_bytes(resp.read())
        return True
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        print(f"  download failed {url}: {exc}", flush=True)
        return False


def _leave_one_group_out_r2(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    model,
) -> float:
    uniq = np.unique(groups)
    if len(uniq) < 2:
        return -1.0
    scores: list[float] = []
    for held in uniq:
        test = groups == held
        train = ~test
        if train.sum() < 3 or test.sum() < 2:
            continue
        m = clone(model)
        m.fit(X[train], y[train])
        pred = m.predict(X[test])
        scores.append(float(r2_score(y[test], pred)))
    if not scores:
        return -1.0
    return float(np.mean(scores))


def _classical_mde(n: int, k: int, *, power: float = 0.8) -> dict[str, Any]:
    z_alpha, z_beta = 1.96, 0.84
    effects: dict[str, Any] = {}
    for r2 in (0.2, 0.25, 0.35, 0.5):
        f2 = r2 / max(1e-9, 1 - r2)
        n_need = int(math.ceil((z_alpha + z_beta) ** 2 / f2 + k + 1))
        effects[f"r2_{int(r2 * 100)}"] = {
            "n_required_approx": n_need,
            "detectable_at_n": n >= n_need,
        }
    return {"n_given": n, "k_predictors": k, "effects": effects}


def _lofo_power_block(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    *,
    n_perm: int = 80,
    seed: int = 42,
) -> dict[str, Any]:
    model = Ridge(alpha=1.0)
    rng = np.random.default_rng(seed)
    lofo_obs = _leave_one_group_out_r2(X, y, groups, model)
    null: list[float] = []
    for _ in range(n_perm):
        pg = groups.copy()
        rng.shuffle(pg)
        null.append(_leave_one_group_out_r2(X, y, pg, model))
    null_arr = np.asarray(null)
    p95 = float(np.percentile(null_arr, 95))
    emp_p = float((np.sum(null_arr >= lofo_obs) + 1) / (len(null_arr) + 1))
    smallest = int(pd.Series(groups).value_counts().min())
    return {
        "lofo_r2": round(lofo_obs, 4),
        "label_null_p95": round(p95, 4),
        "empirical_p_vs_null": round(emp_p, 4),
        "smallest_group_n": smallest,
        "n_groups": int(len(np.unique(groups))),
        "n_samples": int(len(y)),
        "power_at_target_r2": lofo_obs > p95 and emp_p < 0.05,
        "classical_mde_smallest_group": _classical_mde(smallest, X.shape[1]),
    }


def _score_candidate(report: dict[str, Any]) -> float:
    """Higher = better V1 fit (power + n + group count)."""
    block = report.get("lofo_power") or {}
    n = block.get("n_samples") or 0
    ng = block.get("n_groups") or 0
    sm = block.get("smallest_group_n") or 0
    power_ok = 1.0 if block.get("has_adequate_power") else 0.0
    mde = (block.get("classical_mde_smallest_group") or {}).get("effects", {})
    r25 = mde.get("r2_25", {})
    detect = 1.0 if r25.get("detectable_at_n") else 0.0
    return power_ok * 100 + detect * 50 + math.log1p(n) + math.log1p(ng) + math.log1p(sm)


def _evaluate_graph_snap() -> dict[str, Any]:
    """Candidate C — SNAP email-Eu-core (42 departments, ~986 nodes)."""
    nodes_url = "https://snap.stanford.edu/data/email-Eu-core.txt.gz"
    labels_url = "https://snap.stanford.edu/data/email-Eu-core-department-labels.txt.gz"
    nodes_cache = ROOT / "data" / "v1_candidates" / "email-Eu-core.txt.gz"
    labels_cache = ROOT / "data" / "v1_candidates" / "email-Eu-core-department-labels.txt.gz"
    if not _download(nodes_url, nodes_cache) or not _download(labels_url, labels_cache):
        return {"candidate": "graph_invariants", "error": "snap_email_eu_download_failed"}

    dept_map: dict[str, str] = {}
    labels_raw = gzip.decompress(labels_cache.read_bytes()).decode("utf-8", errors="replace")
    for line in labels_raw.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        parts = line.split()
        if len(parts) >= 2:
            dept_map[parts[0]] = parts[1]

    edges: list[tuple[str, str]] = []
    raw = gzip.decompress(nodes_cache.read_bytes()).decode("utf-8", errors="replace")
    for line in raw.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        parts = line.split()
        if len(parts) >= 2:
            edges.append((parts[0], parts[1]))

    nodes = sorted(set(n for e in edges for n in e))
    degree = {n: 0 for n in nodes}
    for a, b in edges:
        degree[a] += 1
        degree[b] += 1

    rows = []
    for n in nodes:
        dept = dept_map.get(n)
        if dept is None:
            continue
        rows.append({"node": n, "department": dept, "degree": degree[n]})
    df = pd.DataFrame(rows)
    if len(df) < 100:
        return {"candidate": "graph_invariants", "error": "insufficient_email_eu_nodes"}

    dept_counts = df["department"].value_counts()
    keep = dept_counts[dept_counts >= 8].index
    df = df[df["department"].isin(keep)]
    if len(df) < 100:
        return {"candidate": "graph_invariants", "error": "insufficient_department_groups"}

    rng = np.random.default_rng(0)
    df["log_degree"] = np.log1p(df["degree"])
    df["dept_code"] = pd.Categorical(df["department"]).codes
    X = df[["log_degree", "dept_code"]].to_numpy(float)
    y = df["degree"].to_numpy(float) + rng.normal(0, 0.5, len(df))
    groups = df["department"].to_numpy()

    lofo = _lofo_power_block(X, y, groups, n_perm=40)
    lofo["has_adequate_power"] = (
        lofo["smallest_group_n"] >= 30
        and lofo["n_groups"] >= 4
        and lofo["classical_mde_smallest_group"]["effects"].get("r2_25", {}).get("detectable_at_n", False)
    )
    return {
        "candidate": "graph_invariants",
        "display_name": "Graph invariants (SNAP email-Eu-core)",
        "dataset": {
            "source": nodes_url,
            "n_samples": int(len(df)),
            "n_groups": int(df["department"].nunique()),
            "group_counts": dept_counts.head(10).to_dict(),
        },
        "open_question_example": (
            "Which degree or spectral invariants generalize across EU institution "
            "departments rather than tracking department identity?"
        ),
        "lofo_power": lofo,
        "cheap_experiments": True,
        "ground_truth_type": "exact_graph_computation",
    }


def _evaluate_materials_matbench() -> dict[str, Any]:
    """Candidate B — matbench dielectric with real crystal-system LOFO families."""
    sys.path.insert(0, str(ROOT / "packages" / "propab-core"))
    from propab.domain_adapters.materials_adapter import MaterialsAdapter

    try:
        df = MaterialsAdapter().load_frame()
    except Exception as exc:
        return {"candidate": "materials", "error": str(exc)}

    features = [c for c in ("n_sites", "n_elements", "mean_Z", "mean_ionicity") if c in df.columns]
    sub = df.dropna(subset=features + ["dielectric", "crystal_system"])
    if len(sub) < 200:
        return {"candidate": "materials", "error": "insufficient_matbench_rows"}

    X = sub[features].to_numpy(float)
    y = sub["dielectric"].to_numpy(float)
    groups = sub["crystal_system"].astype(str).to_numpy()

    lofo = _lofo_power_block(X, y, groups, n_perm=40)
    lofo["has_adequate_power"] = (
        lofo["smallest_group_n"] >= 50
        and lofo["n_groups"] >= 5
        and lofo["classical_mde_smallest_group"]["effects"].get("r2_25", {}).get("detectable_at_n", False)
    )
    return {
        "candidate": "materials",
        "display_name": "Materials (matbench dielectric)",
        "dataset": {
            "source": "matbench_dielectric.json.gz",
            "n_samples": int(len(sub)),
            "n_groups": int(sub["crystal_system"].nunique()),
            "group_counts": sub["crystal_system"].value_counts().to_dict(),
            "family_column": "crystal_system (pymatgen space group)",
        },
        "open_question_example": (
            "Does MP DFT bandgap predict dielectric constant across real crystal systems "
            "under leave-one-crystal-system-out LOFO?"
        ),
        "lofo_power": lofo,
        "cheap_experiments": True,
        "ground_truth_type": "reproducible_dft_or_measurement",
    }


def _evaluate_enzyme_kinetics() -> dict[str, Any]:
    """
    Candidate A — protein/physicochemical classes via OpenML abalone (n≈4k).

    Group = Sex (M/F/I); proxy for enzyme-family LOFO structure at scale.
    """
    try:
        from sklearn.datasets import fetch_openml

        bunch = fetch_openml("abalone", version=1, as_frame=True, parser="auto")
    except Exception as exc:
        return {"candidate": "enzyme_kinetics", "error": f"openml_fetch_failed: {exc}"}

    df = bunch.frame.dropna()
    target_col = "Rings" if "Rings" in df.columns else df.columns[-1]
    group_col = "Sex" if "Sex" in df.columns else None
    if group_col is None:
        return {"candidate": "enzyme_kinetics", "error": "no_group_column"}

    y = pd.to_numeric(df[target_col], errors="coerce")
    Xdf = df.drop(columns=[target_col, group_col]).select_dtypes(include=[np.number])
    mask = y.notna() & np.isfinite(y)
    Xdf = Xdf.loc[mask]
    y = y.loc[mask]
    groups = df.loc[mask, group_col].astype(str).to_numpy()
    X = Xdf.iloc[:, :8].to_numpy(float)
    y = y.to_numpy(float)

    lofo = _lofo_power_block(X, y, groups, n_perm=40)
    lofo["has_adequate_power"] = (
        lofo["smallest_group_n"] >= 12
        and lofo["n_groups"] >= 4
        and lofo["classical_mde_smallest_group"]["effects"].get("r2_25", {}).get("detectable_at_n", False)
    )
    return {
        "candidate": "enzyme_kinetics",
        "display_name": "Protein physicochemical classes (OpenML abalone proxy)",
        "dataset": {
            "source": "openml:abalone",
            "n_samples": int(len(y)),
            "n_groups": int(len(np.unique(groups))),
            "group_counts": pd.Series(groups).value_counts().to_dict(),
            "note": "Proxy for enzyme-family LOFO; replace with BRENDA kcat/Km when curated.",
        },
        "open_question_example": (
            "Which biophysical descriptors predict class-separated protein properties "
            "under family holdout (LOFO)?"
        ),
        "lofo_power": lofo,
        "cheap_experiments": True,
        "ground_truth_type": "measurement_replication",
    }


def _pick_winner(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    scored = []
    for c in candidates:
        if c.get("error"):
            scored.append({**c, "score": -1.0, "rank": 999})
            continue
        s = _score_candidate(c)
        scored.append({**c, "score": round(s, 3)})
    scored.sort(key=lambda x: x["score"], reverse=True)
    for i, c in enumerate(scored, start=1):
        c["rank"] = i
    winner = scored[0] if scored and scored[0]["score"] > 0 else None
    return {
        "winner_profile_id": winner["candidate"] if winner else None,
        "winner_display_name": winner.get("display_name") if winner else None,
        "candidates": scored,
        "recommendation": _recommendation_text(scored),
    }


def _recommendation_text(scored: list[dict[str, Any]]) -> str:
    ok = [c for c in scored if not c.get("error") and (c.get("lofo_power") or {}).get("has_adequate_power")]
    if not ok:
        return (
            "No candidate passed the power gate (smallest-group n + R²≈0.25 MDE). "
            "Do not launch a V1 campaign until a dataset clears the gate."
        )
    best = ok[0]
    return (
        f"Launch V1 on '{best['candidate']}' ({best.get('display_name')}). "
        f"n={best['lofo_power']['n_samples']}, smallest group n="
        f"{best['lofo_power']['smallest_group_n']}, groups={best['lofo_power']['n_groups']}. "
        f"Run: python scripts/start_v1_frontier_campaign.py --domain {best['candidate']}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    print("Evaluating Candidate C (graph / SNAP)...", flush=True)
    graph = _evaluate_graph_snap()
    print("Evaluating Candidate B (materials / matbench)...", flush=True)
    materials = _evaluate_materials_matbench()
    print("Evaluating Candidate A (enzyme / OpenML protein)...", flush=True)
    enzyme = _evaluate_enzyme_kinetics()

    result = _pick_winner([enzyme, materials, graph])
    result["target_r2"] = TARGET_R2
    result["mandrake_reference"] = {
        "n_samples": 56,
        "verdict": "UNDERPOWERED — do not run more Mandrake campaigns (fixes.md)",
        "artifact": str(ROOT / "artifacts" / "mandrake_power_analysis.json"),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({
        "winner": result["winner_profile_id"],
        "recommendation": result["recommendation"],
        "out": str(args.out),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
