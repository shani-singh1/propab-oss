#!/usr/bin/env python3
"""
Launch the V1 frontier campaign (fixes.md Step 4).

Prerequisites:
  1. python scripts/evaluate_v1_domain_candidates.py
  2. Propab stack running (docker compose up -d)
  3. Power gate passed for chosen domain

Example:
  python scripts/start_v1_frontier_campaign.py --domain enzyme_kinetics --hours 8
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "packages" / "propab-core"))

from propab.domain_profiles import get_profile  # noqa: E402

EVAL_PATH = ROOT / "artifacts" / "v1_domain_evaluation.json"
DEFAULT_OUT = ROOT / "artifacts" / "v1_frontier_campaign_latest.json"

_QUESTIONS: dict[str, str] = {
    "enzyme_kinetics": (
        "[domain_profile:enzyme_kinetics] "
        "Discover which biophysical or sequence-derived features predict a target "
        "protein property across enzyme-family boundaries. Use public protein/kinetics "
        "data with natural family groups. Require cross-family LOFO generalization — "
        "within-family fit alone is insufficient. State one falsifiable relationship "
        "that survives family-label permutation null and artifact gate."
    ),
    "materials": (
        "[domain_profile:materials] "
        "Discover which structural or compositional descriptors predict a materials "
        "property (e.g. dielectric constant, formation energy) across crystal-system "
        "or composition families. Use Materials Project / matbench-class data. "
        "Require cross-family holdout; report one relationship that generalizes."
    ),
    "graph_invariants": (
        "[domain_profile:graph_invariants] "
        "Discover a graph structural invariant or relationship that holds across SNAP "
        "network categories (not category metadata leakage). Use real network data "
        "with natural category groups. Prefer exact combinatorial or spectral checks "
        "where possible; empirical claims require cross-category LOFO."
    ),
    "math_combinatorics": (
        "[domain_profile:math_combinatorics] "
        "Investigate the following open questions in additive combinatorics through "
        "systematic computational search: "
        "(1) SIDON DENSITY: For n in {100, 200, 500, 1000, 2000, 5000, 10000}, compute "
        "the ratio F(n)/sqrt(n) where F(n) is the maximum Sidon set size. The known bound "
        "is F(n) <= sqrt(n) + O(n^{1/4}). Does the ratio converge? If so, to what constant? "
        "Is the constant below 1.0, suggesting the upper bound is not tight? "
        "(2) CAP SET GAP: For F_3^n with n in {3,4,5,6,7}, find the largest cap set "
        "achievable by any construction. The CLP upper bound is 2.756^n. The best known "
        "construction achieves 2.217^n. For each n, what is the actual maximum found, "
        "and does it suggest a tighter upper bound or better construction? "
        "(3) SIDON STRUCTURE: Among all maximum or near-maximum Sidon sets found for "
        "a given n, what structural properties do they share? Are they concentrated in "
        "intervals or spread? Is there a pattern in gaps between consecutive elements? "
        "Do NOT confirm a hypothesis if it merely recomputes a known result. A finding "
        "is only interesting if it reveals asymptotic behavior, the gap between bounds "
        "and constructions, or structural properties not characterized in the literature. "
        "Every experiment must use combinatorics verification only. Do NOT perform file "
        "system searches, PATH lookups, or OS interactions."
    ),
}


def _load_winner_domain() -> str | None:
    if not EVAL_PATH.is_file():
        return None
    try:
        data = json.loads(EVAL_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return data.get("winner_profile_id")


def _power_gate_ok(domain: str) -> tuple[bool, str]:
    # Deterministic math domains use plugin preflight at campaign launch, not LOFO power gate.
    if domain == "math_combinatorics":
        from propab.domain_modules.registry import get_domain_plugin

        plugin = get_domain_plugin(domain)
        if plugin is None:
            return False, f"domain plugin {domain} not registered"
        pf = plugin.preflight()
        if pf.passed:
            return True, f"preflight passed: {pf.reason}"
        return False, f"preflight failed: {pf.reason}"

    if not EVAL_PATH.is_file():
        return False, f"Missing {EVAL_PATH} — run evaluate_v1_domain_candidates.py first"
    data = json.loads(EVAL_PATH.read_text(encoding="utf-8"))
    for c in data.get("candidates") or []:
        if c.get("candidate") == domain:
            if c.get("error"):
                return False, str(c["error"])
            lp = c.get("lofo_power") or {}
            if lp.get("has_adequate_power"):
                return True, "power gate passed"
            return False, (
                f"power gate failed: smallest_group_n={lp.get('smallest_group_n')}, "
                f"groups={lp.get('n_groups')}"
            )
    return False, f"domain {domain} not found in evaluation artifact"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--domain",
        default=None,
        help="Domain profile id (default: winner from v1_domain_evaluation.json)",
    )
    parser.add_argument("--api", default="http://localhost:8000")
    parser.add_argument("--hours", type=float, default=8.0, help="Wall-clock budget (default 8h)")
    parser.add_argument(
        "--skip-power-check",
        action="store_true",
        help="Launch even if power analysis did not pass (not recommended)",
    )
    parser.add_argument(
        "--no-hypothesis-cap",
        action="store_true",
        help="Do not cap total hypotheses (discovery campaigns)",
    )
    parser.add_argument(
        "--min-confirmed-findings",
        type=int,
        default=None,
        help=(
            "Breakthrough threshold on confirmed count (default: 1 for verification, "
            "999 for math_combinatorics discovery mode)"
        ),
    )
    parser.add_argument(
        "--discovery-mode",
        action="store_true",
        help="Run until time budget or belief exhaustion only (no early breakthrough stop)",
    )
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args()

    domain = args.domain or _load_winner_domain()
    if not domain:
        print("No --domain and no winner in v1_domain_evaluation.json", file=sys.stderr)
        return 1

    profile = get_profile(domain)
    if profile is None:
        from propab.domain_modules.registry import get_domain_plugin

        if get_domain_plugin(domain) is None:
            print(f"Unknown domain profile: {domain}", file=sys.stderr)
            return 1
        profile_meta = {"profile_id": domain, "display_name": domain}
    else:
        profile_meta = profile.to_dict()

    if not args.skip_power_check:
        ok, reason = _power_gate_ok(domain)
        if not ok:
            print(f"Power gate blocked launch: {reason}", file=sys.stderr)
            print("Use --skip-power-check to override (fixes.md: not recommended).", file=sys.stderr)
            return 1

    question = _QUESTIONS.get(domain, f"[domain_profile:{domain}] Open discovery campaign.")
    api = args.api.rstrip("/")

    discovery_mode = args.discovery_mode or domain == "math_combinatorics"
    min_confirmed = args.min_confirmed_findings
    if min_confirmed is None:
        min_confirmed = 999 if discovery_mode else 1

    breakthrough = {
        "metric_name": "lofo_r2",
        "improvement_threshold": 0.05,
        "direction": "higher_is_better",
        "min_confidence": 0.85,
        "min_replications": 2,
        "min_confirmed_findings": min_confirmed,
    }
    if domain == "math_combinatorics":
        # Baseline stays 0 for verification campaigns. Use API-max thresholds so
        # is_breakthrough() never fires before time budget (conf=1.0, reps=20).
        breakthrough = {
            "metric_name": "sidon_ratio_to_sqrt_n",
            "improvement_threshold": 1.0,
            "direction": "higher_is_better",
            "min_confidence": 1.0,
            "min_replications": 20 if discovery_mode else 2,
            "min_confirmed_findings": None if discovery_mode else min_confirmed,
        }

    max_hypotheses = None if (args.no_hypothesis_cap or discovery_mode) else None

    payload: dict[str, object] = {
        "question": question,
        "compute_budget_hours": args.hours,
        "max_hypotheses": max_hypotheses,
        "domain_profile": domain,
        "policy_mode": "accepted",
        "breakthrough_criteria": breakthrough,
    }
    if domain == "math_combinatorics":
        payload["orchestrator_directive"] = (
            "Target open problems only. Do not treat single-point greedy computations "
            "at small n as discoveries. Hypotheses must involve multi-n sweeps (n≥500), "
            "asymptotic ratio F(n)/sqrt(n), cap-set CLP gap trends, or Sidon structure."
        )

    body = json.dumps(payload).encode("utf-8")

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
    record = {
        "campaign_id": cid,
        "domain_profile": domain,
        "domain_profile_meta": profile_meta,
        "question": question,
        "compute_budget_hours": args.hours,
        "max_hypotheses": max_hypotheses,
        "discovery_mode": discovery_mode,
        "min_confirmed_findings": min_confirmed,
        "stopping_condition": (
            "time_budget_or_belief_exhaustion"
            if discovery_mode
            else "three_consecutive_exhaustion_rounds (belief_state)"
        ),
        "api": api,
        "stream_url": f"{api}/stream/{cid}",
        "state_url": f"{api}/campaigns/{cid}",
        "v1_metric": "strong belief + supporting_nodes + artifact gate survivor",
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps(record, indent=2))
    print(f"\nMonitor: python scripts/monitor_campaign.py --state-file {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
