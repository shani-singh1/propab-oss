#!/usr/bin/env python3
"""
Launch V1 materials campaign 3: real crystal systems + MP bandgap.

Preflight:
  python scripts/build_matbench_mp_bandgap_cache.py
  python scripts/materials_crystal_system_power.py
  python scripts/smoke_materials_verification.py

Example:
  python scripts/start_v1_materials_campaign3.py --hours 8
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
from propab.domain_adapters.materials_adapter import MaterialsAdapter  # noqa: E402
from propab.domain_adapters.materials_mp_bandgap import bandgap_cache_path  # noqa: E402

DEFAULT_PRIOR = ROOT / "artifacts" / "v1_literature_prior" / "materials_prior_for_campaign.json"
DEFAULT_OUT = ROOT / "artifacts" / "v1_materials_campaign3_latest.json"
POWER_PATH = ROOT / "artifacts" / "materials_crystal_system_power.json"

CAMPAIGN3_QUESTION = (
    "[domain_profile:materials] "
    "On matbench dielectric with real crystal-system families (from space group), "
    "test whether MP DFT bandgap and electronic descriptors predict dielectric constant "
    "under leave-one-crystal-system-out LOFO. Prioritize mp_bandgap per Penn model / "
    "Clausius-Mossotti literature. Require lofo_r2 beating label-shuffle null p95."
)

CLOSED_BELIEFS = [
    ClosedBelief(
        statement=(
            "Composition-only structural descriptors (n_sites, n_elements, mean_Z, "
            "mean_atomic_mass, mass_density, mean_ionicity, mean_electronegativity) "
            "do not generalize across crystal-system families on matbench dielectric."
        ),
        reason=(
            "Campaigns 1–2: 375+ hypotheses, 0 confirmed; best lofo_r2≈0.199 vs "
            "label_shuffle_null_p95≈0.232 on site-count-quintile families; campaign 2 "
            "exhausted composition-only feature sets."
        ),
    ),
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api", default="http://localhost:8000")
    parser.add_argument("--hours", type=float, default=8.0)
    parser.add_argument("--prior", default=str(DEFAULT_PRIOR))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--skip-power-check", action="store_true")
    parser.add_argument("--skip-smoke", action="store_true")
    args = parser.parse_args()

    adapter = MaterialsAdapter()
    bg_cache = bandgap_cache_path(adapter.data_dir)
    if not bg_cache.is_file():
        print(f"BLOCKED: run scripts/build_matbench_mp_bandgap_cache.py first ({bg_cache})", file=sys.stderr)
        return 1

    df = adapter.load_frame()
    if "mp_bandgap" not in df.columns or df["mp_bandgap"].notna().sum() < 100:
        print("BLOCKED: mp_bandgap coverage too low in frame", file=sys.stderr)
        return 1
    csys = df["crystal_system"].value_counts()
    print(f"Crystal systems ({len(csys)}): {csys.to_dict()}")

    if not args.skip_power_check:
        if not POWER_PATH.is_file():
            print("Running materials_crystal_system_power.py ...")
            proc = subprocess.run([sys.executable, str(ROOT / "scripts" / "materials_crystal_system_power.py")])
            if proc.returncode != 0:
                print("BLOCKED: power check failed", file=sys.stderr)
                return 1
        power = json.loads(POWER_PATH.read_text(encoding="utf-8"))
        if not power.get("lofo_power", {}).get("has_adequate_power"):
            print("BLOCKED: inadequate LOFO power for crystal-system families", file=sys.stderr)
            return 1
        print("Power gate: PASS")

    if not args.skip_smoke:
        proc = subprocess.run([sys.executable, str(ROOT / "scripts" / "smoke_materials_verification.py")])
        if proc.returncode != 0:
            print("BLOCKED: smoke failed", file=sys.stderr)
            return 1

    prior_path = Path(args.prior)
    if not prior_path.is_file():
        print(f"WARNING: no prior at {prior_path}, launching with empty prior", file=sys.stderr)
        prior: dict = {"established_facts": [], "contested_claims": [], "open_gaps": []}
    else:
        prior = json.loads(prior_path.read_text(encoding="utf-8"))

    api = args.api.rstrip("/")
    body = json.dumps({
        "question": CAMPAIGN3_QUESTION,
        "compute_budget_hours": args.hours,
        "policy_mode": "accepted",
        "literature_prior": prior,
        "closed_beliefs": [c.to_dict() for c in CLOSED_BELIEFS],
        "orchestrator_directive": (
            "Campaign 3: real crystal-system LOFO; test mp_bandgap and electronic "
            "structure features; do not re-test exhausted composition-only descriptors."
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
    req = Request(f"{api}/campaigns", data=body, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read())
    except HTTPError as exc:
        print(f"HTTP {exc.code}: {exc.read().decode('utf-8', errors='replace')[:2000]}", file=sys.stderr)
        return 1
    except URLError as exc:
        print(f"Cannot reach API: {exc}", file=sys.stderr)
        return 1

    cid = data["campaign_id"]
    record = {
        "campaign_id": cid,
        "campaign_number": 3,
        "domain_profile": "materials",
        "question": CAMPAIGN3_QUESTION,
        "compute_budget_hours": args.hours,
        "crystal_system_counts": csys.to_dict(),
        "mp_bandgap_coverage": float(df["mp_bandgap"].notna().mean()),
        "prior_path": str(prior_path),
        "api": api,
        "stream_url": f"{api}/stream/{cid}",
        "state_url": f"{api}/campaigns/{cid}",
    }
    out = Path(args.out)
    out.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps(record, indent=2))
    print(f"\nMonitor: python scripts/monitor_campaign.py --state-file {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
