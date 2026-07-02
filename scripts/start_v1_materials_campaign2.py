#!/usr/bin/env python3
"""
Launch V1 materials campaign 2: literature prior + closed structural subspace.

Preflight:
  python scripts/inspect_matbench_structure.py
  python scripts/smoke_materials_verification.py
  python scripts/build_v1_literature_prior.py --domain materials \\
    --campaign-prior-out artifacts/v1_literature_prior/materials_prior_for_campaign.json

Example:
  python scripts/start_v1_materials_campaign2.py --hours 8
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "packages" / "propab-core"))

from propab.belief_state import ClosedBelief  # noqa: E402
from propab.domain_adapters.materials_adapter import MaterialsAdapter, _KNOWN_FEATURES  # noqa: E402

DEFAULT_PRIOR = ROOT / "artifacts" / "v1_literature_prior" / "materials_prior_for_campaign.json"
DEFAULT_OUT = ROOT / "artifacts" / "v1_materials_campaign2_latest.json"

CAMPAIGN2_QUESTION = (
    "[domain_profile:materials] "
    "Discover which electronic, ionic, or compositional descriptors predict dielectric "
    "constant across crystal-system families on matbench dielectric. Prioritize "
    "mean_atomic_mass, mass_density, mean_ionicity, mean_coordination, "
    "std_principal_quantum_n, and mean_electronegativity over raw site counts. "
    "Require cross-family LOFO generalization that beats label-shuffle null p95."
)

CLOSED_BELIEFS = [
    ClosedBelief(
        statement=(
            "Structural size/composition proxies alone (n_sites, n_elements, mean_Z) "
            "do not generalize across crystal-system families for dielectric LOFO."
        ),
        reason=(
            "Campaign 1 (bcda0509): 145 hypotheses tested, 0 confirmed; only those "
            "three features recycled; best lofo_r2=0.08 vs label_shuffle_null_p95≈0.108."
        ),
    ),
]


def _preflight_features() -> tuple[bool, str]:
    n = len(_KNOWN_FEATURES)
    if n < 8:
        return False, f"Only {n} known features (need ≥8)"
    return True, f"{n} features: {', '.join(_KNOWN_FEATURES)}"


def _run_smoke() -> tuple[bool, str]:
    script = ROOT / "scripts" / "smoke_materials_verification.py"
    proc = subprocess.run([sys.executable, str(script)], capture_output=True, text=True)
    tail = (proc.stdout or proc.stderr or "")[-2000:]
    if proc.returncode != 0:
        return False, tail
    return True, tail


def _load_prior(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Missing prior artifact: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    for key in ("established_facts", "contested_claims", "open_gaps"):
        data.setdefault(key, [])
    return data


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api", default="http://localhost:8000")
    parser.add_argument("--hours", type=float, default=8.0)
    parser.add_argument("--prior", default=str(DEFAULT_PRIOR))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--skip-smoke", action="store_true")
    parser.add_argument("--skip-prior-build", action="store_true")
    args = parser.parse_args()

    ok, msg = _preflight_features()
    print(f"Feature preflight: {msg}")
    if not ok:
        print("BLOCKED", file=sys.stderr)
        return 1

    # Sanity: adapter loads extended frame
    df = MaterialsAdapter().load_frame()
    missing = [c for c in _KNOWN_FEATURES if c not in df.columns]
    if missing:
        print(f"BLOCKED: frame missing columns {missing}", file=sys.stderr)
        return 1
    print(f"Frame OK: {len(df)} rows")

    if not args.skip_smoke:
        print("Running smoke_materials_verification.py ...")
        smoke_ok, smoke_out = _run_smoke()
        print(smoke_out)
        if not smoke_ok:
            print("BLOCKED: smoke failed", file=sys.stderr)
            return 1

    prior_path = Path(args.prior)
    if not args.skip_prior_build and not prior_path.is_file():
        print("Building literature prior (may take a few minutes)...")
        build = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "build_v1_literature_prior.py"),
                "--domain",
                "materials",
                "--campaign-prior-out",
                str(prior_path),
            ],
            cwd=str(ROOT),
        )
        if build.returncode != 0:
            print("BLOCKED: literature prior build failed", file=sys.stderr)
            return 1

    try:
        prior = _load_prior(prior_path)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    contested = len(prior.get("contested_claims") or [])
    gaps = len(prior.get("open_gaps") or [])
    print(f"Prior loaded: contested={contested} gaps={gaps} method={prior.get('extraction_method')}")

    api = args.api.rstrip("/")
    body = json.dumps({
        "question": CAMPAIGN2_QUESTION,
        "compute_budget_hours": args.hours,
        "max_hypotheses": None,
        "policy_mode": "accepted",
        "literature_prior": prior,
        "closed_beliefs": [c.to_dict() for c in CLOSED_BELIEFS],
        "orchestrator_directive": (
            "Campaign 2: test literature-backed electronic/ionic descriptors; "
            "do not re-propose n_sites/n_elements/mean_Z-only hypotheses."
        ),
        "breakthrough_criteria": {
            "metric_name": "lofo_r2",
            "improvement_threshold": 0.05,
            "direction": "higher_is_better",
            "min_confidence": 0.85,
            "min_replications": 2,
            "min_confirmed_findings": 1,
        },
    }).encode("utf-8")
    req = Request(
        f"{api}/campaigns",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read())
    except HTTPError as exc:
        print(f"HTTP {exc.code}: {exc.read().decode('utf-8', errors='replace')[:2000]}", file=sys.stderr)
        return 1
    except URLError as exc:
        print(f"Cannot reach API at {api}: {exc}", file=sys.stderr)
        return 1

    cid = data["campaign_id"]
    print(f"Campaign created: {cid}")

    record = {
        "campaign_id": cid,
        "campaign_number": 2,
        "baseline_campaign_id": "bcda0509-6245-4347-a3b0-2e7a1a054e59",
        "domain_profile": "materials",
        "question": CAMPAIGN2_QUESTION,
        "compute_budget_hours": args.hours,
        "prior_path": str(prior_path),
        "closed_beliefs": [c.to_dict() for c in CLOSED_BELIEFS],
        "known_features": list(_KNOWN_FEATURES),
        "api": api,
        "stream_url": f"{api}/stream/{cid}",
        "state_url": f"{api}/campaigns/{cid}",
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps(record, indent=2))
    print(f"\nMonitor: python scripts/monitor_campaign.py --state-file {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
