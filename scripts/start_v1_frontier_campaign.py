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
    "genomics": (
        "[domain_profile:genomics] "
        "Discover which gene expression patterns generalize across tissue types in a "
        "GTEx-style subset. Use leave-tissue-out holdout — within-tissue fit alone is "
        "insufficient. Open question: which non-housekeeping genes show partial "
        "cross-tissue conservation and what features (tau index, expression variance, "
        "CV) predict it? Require tissue-label shuffle null p<0.05 before confirm."
    ),
    "math_combinatorics": (
        "[domain_profile:math_combinatorics] "
        "Answer these tractable questions using ONLY greedy Sidon search, Bose-Chowla "
        "construction, cap-set best-known table lookup, and band/threshold claim "
        "verification. Do NOT propose Ruzsa, Singer, Poisson gap tests, Fourier analysis, "
        "SAT solvers, stochastic optimizers, or structural statistics the verifier cannot "
        "compute. "
        "ESTABLISHED BASELINE (from prior campaigns, do not re-test as novel): "
        "(A) Greedy Sidon: for n in {500,1000,2000,5000,10000}, F(n)/sqrt(n) is strictly "
        "below 0.95 (observed max 0.939 at n=500, descending to 0.67 at n=10000). "
        "(B) Cap sets F_3^n dims 3-7: CLP ratios decrease monotonically and stay well "
        "below 2.25. "
        "OPEN QUESTIONS TO TEST NOW (Q1 priority — campaigns 3-4): "
        "(1) Given F(n)/sqrt(n) crossed below 0.70 near n=8700 and reached 0.67 at n=10000, "
        "where does it first fall below 0.60? Run threshold sweeps for n in [10000, 50000]. "
        "Is the descent still strictly monotone at these scales? "
        "(2) AP-free greedy density at n in [5000, 50000] — does it show the same monotonic "
        "decrease pattern as Sidon? "
        "(3) Bose-Chowla vs greedy at matched n for q giving n>10000 — does BC still "
        "underperform by a measurable margin? "
        "Cap-set table lookups are LOWER priority unless tree diversity requires them. "
        "Prefer Sidon threshold and AP-free sweeps over cap-set CLP repeats. "
        "Every hypothesis must state a falsifiable numeric band, threshold, or comparison "
        "claim that the combinatorics verifier can check. Do NOT confirm without matching "
        "computed results to the stated claim. Do NOT perform file system or OS operations."
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
    if domain in ("math_combinatorics", "genomics"):
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
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preflight + routing check only; do not POST /campaigns",
    )
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

    if args.dry_run:
        from propab.domain_modules.registry import get_domain_plugin

        plugin = get_domain_plugin(domain)
        pf = plugin.preflight() if plugin else None
        routing_rc = 0
        routing_detail = "skipped"
        if domain == "genomics":
            from propab.domain_modules.genomics.routing_inspector import (
                ROUTING_CORPUS,
                inspect_corpus as inspect_genomics_corpus,
            )

            rep = inspect_genomics_corpus(ROUTING_CORPUS)
            routing_rc = 0 if rep["routing_mismatches"] == 0 else 1
            routing_detail = f"genomics {rep['routing_ok']}/{rep['total']}"
        elif domain == "math_combinatorics":
            routing_detail = "use full inspect_hypothesis_routing.py --corpus for combinatorics"
        report = {
            "dry_run": True,
            "domain": domain,
            "preflight_passed": bool(pf and pf.passed),
            "preflight_reason": pf.reason if pf else "no plugin",
            "preflight_details": pf.details if pf else {},
            "routing_inspector_rc": routing_rc,
            "routing_detail": routing_detail,
            "would_launch_hours": args.hours,
            "question_preview": _QUESTIONS.get(domain, "")[:200],
        }
        print(json.dumps(report, indent=2))
        if not report["preflight_passed"] or routing_rc != 0:
            return 1
        return 0

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
            "Q1 focus: greedy Sidon threshold sweeps n=10000..50000 (0.60 crossing), "
            "AP-free density sweeps, Bose-Chowla vs greedy at large matched n. "
            "Cap-set lookups secondary — avoid cap-set monoculture. "
            "Reject Ruzsa/Singer/Poisson/Fourier/stochastic/SAT at synthesis. "
            "Baseline: F(n)/sqrt(n) < 0.67 at n=10000; crossed 0.70 near n=8700. "
            "Generate falsifiable band/threshold/monotonicity claims only."
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
