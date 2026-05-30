#!/usr/bin/env python3
"""Poll GET /campaigns/{id} until session completed or campaign terminal + paper; ledger checks."""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


def get_json(url: str, timeout: float = 60.0) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.loads(resp.read())


def _fetch_paper_with_retries(api: str, cid: str, attempts: int = 12, delay: float = 5.0) -> dict | None:
    """Paper may land shortly after campaign.budget_exhausted; retry 404."""
    paper_url = f"{api}/sessions/{cid}/paper"
    last_err: str | None = None
    for i in range(attempts):
        try:
            return get_json(paper_url, timeout=45.0)
        except urllib.error.HTTPError as exc:
            last_err = f"HTTP {exc.code}"
            if exc.code != 404:
                raise
        except Exception as exc:  # noqa: BLE001
            last_err = str(exc)
        time.sleep(delay)
    print(f"Paper still unavailable after retries: {last_err}", file=sys.stderr)
    return None


def _run_ledger_paper_review(tee, api: str, cid: str, blob: dict) -> int:
    summ = blob.get("summary") or {}
    camp = blob.get("campaign") or {}
    tree = camp.get("hypothesis_tree") or {}
    conf_ids = tree.get("confirmed") or []
    nodes = tree.get("nodes") or {}
    n_conf_nodes = sum(
        1
        for n in nodes.values()
        if isinstance(n, dict) and str(n.get("verdict")) == "confirmed"
    )
    tee(
        f"SYNC_CHECK: summary.total_confirmed={summ.get('total_confirmed')} "
        f"tree.confirmed_list_len={len(conf_ids)} nodes_verdict_confirmed={n_conf_nodes}"
    )
    if len(conf_ids) != int(summ.get("total_confirmed") or -1):
        tee("WARNING: summary total_confirmed differs from tree.confirmed length")

    paper_blob = _fetch_paper_with_retries(api, cid)
    if not paper_blob:
        tee("Paper not fetchable — check worker logs / MinIO")
        return 1
    payload = paper_blob.get("paper") or {}
    tee(f"PAPER_KEYS: {list(payload.keys())[:12]}")
    n_expect = int(summ.get("total_confirmed") or 0)
    abs_tex = str(payload.get("abstract_latex") or "")
    if n_expect > 0:
        if not abs_tex.strip():
            tee("ABSTRACT_LEDGER_MISSING: paper payload has no abstract_latex (rebuild API/worker)")
            return 1
        ok = False
        merged = abs_tex.replace(" ", "")
        if f"confirmed={n_expect}" in merged:
            ok = True
        elif re.search(rf"(?i)\b{n_expect}\s+were\s+supported\b", abs_tex):
            ok = True
        elif re.search(rf"(?i)\b{n_expect}\s+hypotheses?\s+were\s+supported\b", abs_tex):
            ok = True
        elif re.search(rf"(?i){n_expect}\s+confirmed", abs_tex):
            ok = True
        else:
            m = re.search(r"(?i)confirmed\s*=\s*(\d+)", abs_tex)
            if m and int(m.group(1)) == n_expect:
                ok = True
        if ok:
            tee(f"ABSTRACT_LEDGER_OK total_confirmed={n_expect} matches abstract")
        else:
            tee(
                f"ABSTRACT_LEDGER_MISMATCH expected confirmed count {n_expect} in abstract; "
                f"snippet={abs_tex[:400]!r}"
            )
            return 1
    # Host cannot reach internal MinIO URLs; use results_latex from API JSON for text search.
    res_tex = str(payload.get("results_latex") or "")
    if res_tex:
        for pat in (r"confirmed=(\d+)", r"confirmed\s*=\s*(\d+)"):
            m = re.search(pat, res_tex, re.I)
            if m:
                tee(f"RESULTS_TEX_CONFIRMED_MENTION: {m.group(1)}")
                break
        tee(f"RESULTS_TEX_LEN: {len(res_tex)}")
    tex_url = str(payload.get("tex_url") or "")
    if tex_url.startswith("http") and ("localhost" in tex_url or "127.0.0.1" in tex_url):
        try:
            with urllib.request.urlopen(tex_url, timeout=45) as r:
                tex = r.read().decode("utf-8", errors="replace")
            m = re.search(r"\\begin\{abstract\}([\s\S]*?)\\end\{abstract\}", tex)
            if m:
                body = m.group(1).strip()[:900]
                tee(f"ABSTRACT_SNIPPET: {body}")
        except Exception as exc:  # noqa: BLE001
            tee(f"tex_url fetch skipped/failed: {exc}")
    else:
        tee(
            "tex_url is internal (Docker MinIO) — use results_latex above or "
            "`docker compose exec api curl -s <tex_url>` from inside the network."
        )
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--api", default="http://localhost:8000")
    p.add_argument("--state-file", default="")
    p.add_argument("--campaign-id", default="")
    p.add_argument("--interval", type=float, default=45.0)
    p.add_argument("--max-seconds", type=int, default=1980, help="Default ~33 min")
    p.add_argument("--log", default="", help="Append lines here")
    args = p.parse_args()
    api = args.api.rstrip("/")
    cid = args.campaign_id.strip()
    if not cid and args.state_file:
        raw = Path(args.state_file).read_text(encoding="utf-8")
        cid = (json.loads(raw).get("campaign_id") or "").strip()
    if not cid:
        print("Need --campaign-id or --state-file with campaign_id", file=sys.stderr)
        return 2

    log_f = open(args.log, "a", encoding="utf-8") if args.log else None

    def tee(msg: str) -> None:
        line = f"{time.strftime('%Y-%m-%dT%H:%M:%S')} {msg}"
        print(line, flush=True)
        if log_f:
            log_f.write(line + "\n")
            log_f.flush()

    url = f"{api}/campaigns/{cid}"
    t0 = time.monotonic()
    last_conf = None
    terminal_campaign_statuses = frozenset({"budget_exhausted", "breakthrough"})
    try:
        while (time.monotonic() - t0) < args.max_seconds:
            try:
                blob = get_json(url)
            except urllib.error.HTTPError as exc:
                tee(f"HTTP {exc.code}")
                time.sleep(args.interval)
                continue
            except Exception as exc:  # noqa: BLE001
                tee(f"poll error: {exc}")
                time.sleep(args.interval)
                continue

            summ = blob.get("summary") or {}
            sess = blob.get("research_session") or {}
            conf = summ.get("total_confirmed")
            st = sess.get("status")
            cstat = summ.get("status")
            stage = sess.get("stage")
            tee(
                f"status={st} stage={stage} campaign={cstat} "
                f"confirmed={conf} tested={summ.get('total_hypotheses')} "
                f"elapsed={summ.get('elapsed_sec')} remaining={summ.get('remaining_sec')}"
            )
            last_conf = conf
            try:
                _rem_raw = summ.get("remaining_sec")
                remaining_sec = float(_rem_raw) if _rem_raw is not None else 0.0
            except (TypeError, ValueError):
                remaining_sec = 0.0
            if st == "completed":
                tee("Session completed — ledger vs summary + paper")
                return _run_ledger_paper_review(tee, api, cid, blob)

            if cstat in terminal_campaign_statuses and remaining_sec <= 0:
                tee(
                    f"Campaign terminal ({cstat}) with zero remaining — "
                    "waiting for paper + session.completed (or accepting paper-only success)"
                )
                paper_try = _fetch_paper_with_retries(api, cid, attempts=15, delay=4.0)
                if paper_try and st == "completed":
                    return _run_ledger_paper_review(tee, api, cid, blob)
                if paper_try:
                    tee(
                        "Paper is ready but research_sessions.status is not 'completed' yet — "
                        "running ledger review anyway (check API image has db_mark fix + rebuild)."
                    )
                    return _run_ledger_paper_review(tee, api, cid, blob)
                tee("Terminal campaign but paper not ready — keep polling")
            time.sleep(max(5.0, args.interval))
        tee("Max wall time reached without terminal completion")
        tee(f"Last observed confirmed={last_conf}")
        return 1
    finally:
        if log_f:
            log_f.close()


if __name__ == "__main__":
    raise SystemExit(main())
