#!/usr/bin/env python3
"""P1 preflight — verify deployed worker records scope/OOD/artifact gate fields (fixes.md)."""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "packages" / "propab-core"))

from services.orchestrator.seed_validation import PHASE2_CONTAGION_QUESTION

SYNTHETIC_TEXT = (
    "In Barabási-Albert graphs (N=800, m=3), spectral radius predicts SIS steady-state "
    "infection fraction better than mean degree alone.\n"
    "Population: synthetic BA networks with N=800 and m=3 edge attachment\n"
    "Distribution: i.i.d. BA draws with fixed N and m across replicates\n"
    "Claimed generalization: ranking holds on BA graphs with N=1200 held out of training sweep\n"
    "Expected failure modes: fails on Erdős-Rényi graphs or when infection threshold beta is extreme\n"
    "OOD test: evaluate predictor ranking on held-out N=1200 BA graphs not used in fitting\n"
    "Methodology: one statistical_significance call on two small metric lists, then stop."
)

REQUIRED_KEYS = ("scope_gate_result", "ood_passed", "artifact_gate")


def _parse_evidence(summary: str) -> dict:
    if not summary:
        return {}
    m = re.search(r"evidence=(\{.*?\});", summary)
    if not m:
        return {}
    try:
        return json.loads(m.group(1))
    except json.JSONDecodeError:
        return {}


def _seed_rows(conn, session_id: str, hypothesis_id: str) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO research_sessions (id, question, status, stage)
            VALUES (%s::uuid, %s, 'running', 'gates_preflight')
            ON CONFLICT (id) DO NOTHING
            """,
            (session_id, PHASE2_CONTAGION_QUESTION),
        )
        cur.execute(
            """
            INSERT INTO hypotheses (
                id, session_id, text, test_methodology, scores_json,
                rank, status, created_at
            ) VALUES (
                %s::uuid, %s::uuid, %s, %s, '{}'::jsonb, 1, 'pending', NOW()
            )
            ON CONFLICT (id) DO NOTHING
            """,
            (hypothesis_id, session_id, SYNTHETIC_TEXT, "statistical_significance smoke"),
        )
    conn.commit()


def main() -> int:
    parser = argparse.ArgumentParser(description="P1 gate preflight (fixes.md)")
    parser.add_argument("--broker", default="redis://localhost:6379/0")
    parser.add_argument("--timeout", type=int, default=900)
    args = parser.parse_args()

    try:
        import psycopg
        from celery import Celery
    except ImportError as exc:
        print(f"Missing dependency: {exc}", file=sys.stderr)
        return 1

    session_id, hypothesis_id = str(uuid4()), str(uuid4())
    dsn = "postgresql://propab:propab@localhost:5432/propab"

    with psycopg.connect(dsn) as conn:
        _seed_rows(conn, session_id, hypothesis_id)

    payload = {
        "session_id": session_id,
        "hypothesis_id": hypothesis_id,
        "hypothesis": {
            "id": "gates_preflight",
            "text": SYNTHETIC_TEXT,
            "test_methodology": "statistical_significance smoke",
            "scores": {},
            "rank": 1,
        },
        "baseline": {
            "metric_name": "final_outbreak_fraction",
            "metric_value": 0.875,
            "description": "preflight baseline",
            "lit_compare_safe": True,
        },
        "prior": {"established_facts": [], "contested_claims": [], "open_gaps": [], "dead_ends": [], "key_papers": []},
        "domain": "network_contagion",
        "question": PHASE2_CONTAGION_QUESTION,
        "agent_limits": {"max_steps": 8, "max_tool_calls": 4},
    }

    app = Celery(broker=args.broker, backend=args.broker)
    print("Dispatching synthetic hypothesis to deployed worker...", flush=True)
    t0 = time.time()
    async_result = app.send_task("propab.run_sub_agent", args=[payload])
    try:
        async_result.get(timeout=args.timeout)
    except Exception as exc:
        print(f"Worker task failed or timed out: {exc}", file=sys.stderr)
        return 1

    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT evidence_summary, verdict FROM hypotheses WHERE id=%s::uuid",
                (hypothesis_id,),
            )
            row = cur.fetchone()
            cur.execute("DELETE FROM tool_calls WHERE hypothesis_id=%s::uuid", (hypothesis_id,))
            cur.execute("DELETE FROM experiment_steps WHERE hypothesis_id=%s::uuid", (hypothesis_id,))
            cur.execute("DELETE FROM llm_calls WHERE session_id=%s::uuid", (session_id,))
            cur.execute("DELETE FROM events WHERE session_id=%s::uuid", (session_id,))
            cur.execute("DELETE FROM hypotheses WHERE id=%s::uuid", (hypothesis_id,))
            cur.execute("DELETE FROM research_sessions WHERE id=%s::uuid", (session_id,))
        conn.commit()

    if not row:
        print("STOP: hypothesis row missing after worker run.", file=sys.stderr)
        return 1

    evidence_summary, verdict = row
    ev = _parse_evidence(str(evidence_summary or ""))
    missing = [k for k in REQUIRED_KEYS if k not in ev]
    report = {
        "ok": len(missing) == 0,
        "verdict": verdict,
        "elapsed_sec": round(time.time() - t0, 1),
        "required_keys": list(REQUIRED_KEYS),
        "present": {k: k in ev for k in REQUIRED_KEYS},
        "missing": missing,
        "scope_gate_result": ev.get("scope_gate_result"),
        "ood_passed": ev.get("ood_passed"),
        "artifact_gate_verdict": (ev.get("artifact_gate") or {}).get("verdict"),
    }
    print(json.dumps(report, indent=2))
    if missing:
        print(f"STOP: gate fields missing in evidence: {missing}", file=sys.stderr)
        return 1
    print("P1 preflight PASSED — gates active in deployed worker.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
