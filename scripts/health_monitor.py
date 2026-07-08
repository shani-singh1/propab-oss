#!/usr/bin/env python3
"""health_monitor.py — standalone status/alert tool for the Propab stack.

Read-only. Checks three things and prints a compact dashboard:

  1. Docker services are Up (and healthy where a healthcheck is declared).
  2. The API answers GET /campaigns with HTTP 200.
  3. No STUCK campaigns: a campaign is `active` in the DB but has emitted no new
     event for the last N minutes (default 30) — i.e. the run silently died.

Exits NON-ZERO if anything is unhealthy, so it is usable from `watch`/cron:

    watch -n 60 'python scripts/health_monitor.py || echo ALERT'

It never writes to the DB or the stack. Missing dependencies (docker not on
PATH, DB unreachable, psycopg not installed) degrade to a reported FAIL rather
than a crash, so the tool itself stays reliable.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone

# Long-running services that must be Up.
LONG_RUNNING = ["api", "orchestrator", "worker", "literature", "postgres", "redis", "qdrant", "minio"]
# Subset that declare a healthcheck and must report "healthy".
HEALTHCHECKED = {"api", "orchestrator", "postgres", "redis", "qdrant", "minio"}

GREEN, RED, YELLOW, DIM, BOLD, RESET = "\033[32m", "\033[31m", "\033[33m", "\033[2m", "\033[1m", "\033[0m"


def _c(color: str, text: str) -> str:
    if not sys.stdout.isatty():
        return text
    return f"{color}{text}{RESET}"


# ── check result plumbing ────────────────────────────────────────────────────

class Report:
    def __init__(self) -> None:
        self.lines: list[tuple[str, str, str]] = []  # (status, label, detail)
        self.ok = True

    def add(self, status: str, label: str, detail: str = "") -> None:
        if status == "FAIL":
            self.ok = False
        self.lines.append((status, label, detail))

    def render(self) -> str:
        icon = {"OK": _c(GREEN, "OK  "), "FAIL": _c(RED, "FAIL"), "WARN": _c(YELLOW, "WARN")}
        out = []
        for status, label, detail in self.lines:
            tag = icon.get(status, status)
            line = f"  {tag}  {label}"
            if detail:
                line += _c(DIM, f"  — {detail}")
            out.append(line)
        return "\n".join(out)


# ── 1. docker services ───────────────────────────────────────────────────────

def check_docker(report: Report, compose_files: list[str]) -> None:
    if shutil.which("docker") is None:
        report.add("FAIL", "docker", "docker not found on PATH")
        return

    cmd = ["docker", "compose"]
    for f in compose_files:
        cmd += ["-f", f]
    cmd += ["ps", "--format", "json"]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    except (subprocess.SubprocessError, OSError) as exc:
        report.add("FAIL", "docker compose ps", str(exc))
        return
    if proc.returncode != 0:
        report.add("FAIL", "docker compose ps", (proc.stderr or "non-zero exit").strip()[:200])
        return

    # Output is either a JSON array or newline-delimited JSON objects.
    raw = proc.stdout.strip()
    entries: list[dict] = []
    if raw.startswith("["):
        try:
            entries = json.loads(raw)
        except json.JSONDecodeError:
            entries = []
    else:
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    by_service: dict[str, dict] = {}
    for e in entries:
        svc = e.get("Service") or e.get("service") or ""
        if svc:
            by_service[svc] = e

    for svc in LONG_RUNNING:
        e = by_service.get(svc)
        if e is None:
            report.add("FAIL", f"service {svc}", "not running (no container)")
            continue
        state = (e.get("State") or e.get("state") or "").lower()
        health = (e.get("Health") or e.get("health") or "").lower()
        if state != "running":
            report.add("FAIL", f"service {svc}", f"state={state or 'unknown'} (expected running)")
        elif svc in HEALTHCHECKED and health and health not in ("healthy",):
            report.add("FAIL", f"service {svc}", f"health={health} (expected healthy)")
        else:
            detail = f"running ({health})" if health else "running"
            report.add("OK", f"service {svc}", detail)


# ── 2. API ───────────────────────────────────────────────────────────────────

def _http_status(url: str, timeout: float = 10.0) -> tuple[int, str]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return resp.getcode(), ""
    except urllib.error.HTTPError as exc:
        return exc.code, ""
    except (urllib.error.URLError, OSError) as exc:
        return 0, str(getattr(exc, "reason", exc))


def check_api(report: Report, api_base: str) -> None:
    code, err = _http_status(f"{api_base}/health")
    if code == 200:
        report.add("OK", "api /health", "200")
    else:
        report.add("FAIL", "api /health", f"{code or 'no response'}{(' — ' + err) if err else ''}")

    code, err = _http_status(f"{api_base}/campaigns")
    if code == 200:
        report.add("OK", "api /campaigns", "200")
    else:
        report.add("FAIL", "api /campaigns", f"{code or 'no response'}{(' — ' + err) if err else ''}")


# ── 3. stuck campaigns (DB) ──────────────────────────────────────────────────

def _libpq_dsn(url: str) -> str:
    """Strip the SQLAlchemy '+driver' so psycopg/libpq accepts the URL."""
    for driver in ("+asyncpg", "+psycopg", "+psycopg2"):
        url = url.replace(driver, "")
    return url


def check_campaigns(report: Report, database_url: str, stuck_minutes: int) -> None:
    try:
        import psycopg  # lazy — tool still runs for docker/api checks without it
    except ImportError:
        report.add("FAIL", "campaigns", "psycopg not installed (pip install 'psycopg[binary]')")
        return

    dsn = _libpq_dsn(database_url)
    try:
        conn = psycopg.connect(dsn, connect_timeout=10)
    except Exception as exc:  # noqa: BLE001 — surface any connection failure as FAIL
        report.add("FAIL", "database", f"cannot connect: {str(exc).strip()[:160]}")
        return

    try:
        with conn, conn.cursor() as cur:
            # Active campaigns joined to their most recent event (events.session_id == campaign id).
            cur.execute(
                """
                SELECT c.id,
                       LEFT(c.question, 60) AS q,
                       c.started_at,
                       MAX(e.created_at) AS last_event,
                       EXTRACT(EPOCH FROM (NOW() - COALESCE(MAX(e.created_at), c.started_at))) AS idle_s
                FROM research_campaigns c
                LEFT JOIN events e ON e.session_id = c.id
                WHERE c.status = 'active'
                GROUP BY c.id, c.question, c.started_at
                ORDER BY idle_s DESC
                """
            )
            rows = cur.fetchall()
    except Exception as exc:  # noqa: BLE001
        report.add("FAIL", "campaigns query", str(exc).strip()[:160])
        conn.close()
        return
    finally:
        conn.close()

    if not rows:
        report.add("OK", "campaigns", "no active campaigns")
        return

    threshold_s = stuck_minutes * 60
    stuck = 0
    for cid, q, _started, last_event, idle_s in rows:
        idle_s = float(idle_s or 0)
        idle_min = int(idle_s // 60)
        label = f"campaign {str(cid)[:8]} ({q.strip()}…)"
        if idle_s > threshold_s:
            stuck += 1
            when = "no events ever" if last_event is None else f"idle {idle_min}m"
            report.add("FAIL", label, f"STUCK — active but {when} (> {stuck_minutes}m)")
        else:
            report.add("OK", label, f"active, idle {idle_min}m")
    if stuck:
        report.add("FAIL", "campaigns summary", f"{stuck}/{len(rows)} active campaigns STUCK")


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> int:
    default_db = (
        os.getenv("DATABASE_URL_SYNC")
        or os.getenv("DATABASE_URL")
        or "postgresql://propab:propab@localhost:5432/propab"
    )
    p = argparse.ArgumentParser(description="Propab stack health & stuck-campaign monitor (read-only).")
    p.add_argument("--api-base", default=os.getenv("API_BASE_URL", "http://localhost:8000"),
                   help="API base URL (default http://localhost:8000)")
    p.add_argument("--database-url", default=default_db,
                   help="Postgres URL (SQLAlchemy or libpq form); default from DATABASE_URL[_SYNC]")
    p.add_argument("--stuck-minutes", type=int, default=30,
                   help="Flag an active campaign STUCK after this many minutes with no new events")
    p.add_argument("-f", "--file", action="append", default=[], dest="compose_files",
                   help="docker compose -f file (repeatable); default: compose auto-detection")
    p.add_argument("--no-docker", action="store_true", help="skip docker service checks")
    p.add_argument("--no-api", action="store_true", help="skip API checks")
    p.add_argument("--no-db", action="store_true", help="skip stuck-campaign DB checks")
    p.add_argument("--json", action="store_true", help="emit machine-readable JSON instead of a dashboard")
    args = p.parse_args()

    report = Report()
    if not args.no_docker:
        check_docker(report, args.compose_files)
    if not args.no_api:
        check_api(report, args.api_base.rstrip("/"))
    if not args.no_db:
        check_campaigns(report, args.database_url, args.stuck_minutes)

    if args.json:
        print(json.dumps({
            "healthy": report.ok,
            "checked_at": datetime.now(timezone.utc).isoformat(),
            "checks": [{"status": s, "label": l, "detail": d} for s, l, d in report.lines],
        }, indent=2))
        return 0 if report.ok else 1

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
    print(_c(BOLD, f"Propab health monitor — {ts}"))
    print(report.render() or "  (no checks run)")
    verdict = _c(GREEN, "HEALTHY") if report.ok else _c(RED, "UNHEALTHY")
    print(_c(BOLD, f"\nStatus: {verdict}"))
    return 0 if report.ok else 1


if __name__ == "__main__":
    sys.exit(main())
