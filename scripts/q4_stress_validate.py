"""Run stress test Q4 and validate paper Results vs API hypotheses/trace."""
from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request

BASE = "http://localhost:8000"
QUESTION = (
    "Compare SGD, Adam, AdamW, RMSProp, and Adagrad across five different loss surface "
    "geometries — rank them by convergence speed, final loss, and stability."
)


def http_json(method: str, path: str, body: dict | None = None, timeout: int = 120) -> dict:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        BASE + path,
        data=data,
        method=method,
        headers={"Content-Type": "application/json"} if body else {},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def main() -> None:
    sid = http_json(
        "POST",
        "/research",
        {
            "question": QUESTION,
            "config": {
                "max_hypotheses": 5,
                "paper_ttl_days": 30,
                "llm_model": "stress",
            },
        },
    )["session_id"]
    print("SESSION_ID", sid)

    state: dict = {}
    for i in range(200):
        state = http_json("GET", f"/sessions/{sid}", timeout=60)
        if i % 8 == 0:
            print(i * 5, "s", state.get("status"), state.get("stage"))
        if state.get("status") in ("completed", "failed"):
            break
        time.sleep(5)
    print("FINAL", state.get("status"), state.get("stage"), state.get("error"))

    if state.get("status") != "completed":
        raise SystemExit(1)

    paper = http_json("GET", f"/sessions/{sid}/paper")["paper"]
    hyps = http_json("GET", f"/sessions/{sid}/hypotheses")["hypotheses"]
    trace = http_json("GET", f"/sessions/{sid}/trace")["trace"]

    results = paper.get("results_latex") or ""
    print("trace_steps", len(trace), "hypotheses", len(hyps))
    print("verdicts_api", [h.get("verdict") for h in hyps])
    print("confidence_api", [h.get("confidence") for h in hyps])

    # Worker v2 evidence format (LaTeX may escape underscores)
    if "relevance" not in results.lower():
        print("VALIDATION_FAIL: results_latex missing relevance score text")
    else:
        print("VALIDATION_OK: results_latex mentions relevance score")

    if re.search(r"confidence 0\.62\)", results):
        print("WARN: old 0.62 confidence string still present in results")
    if re.search(r"confidence 0\.67\)", results):
        print("INFO: saw 0.67 confirmed confidence in results")

    inconc = sum(1 for h in hyps if h.get("verdict") == "inconclusive")
    confirmed = sum(1 for h in hyps if h.get("verdict") == "confirmed")
    print("verdict_counts confirmed=", confirmed, "inconclusive=", inconc)

    # Cross-check: each hypothesis id should appear in trace
    hid = {h["id"] for h in hyps}
    tids = {t.get("hypothesis_id") for t in trace}
    missing = hid - tids
    if missing:
        print("WARN: hypothesis ids missing from trace", missing)
    else:
        print("VALIDATION_OK: every hypothesis has trace rows")

    # Tool names per hypothesis (compact) — steps store tool in input_json
    by_h: dict[str, list[str]] = {}
    for t in trace:
        hid2 = t.get("hypothesis_id")
        if not hid2:
            continue
        label = "?"
        inj = t.get("input_json")
        if isinstance(inj, str):
            try:
                inj = json.loads(inj)
            except json.JSONDecodeError:
                inj = {}
        if isinstance(inj, dict) and inj.get("tool"):
            label = str(inj["tool"])
        elif t.get("step_type") == "code_exec":
            label = "code_exec"
        by_h.setdefault(str(hid2), []).append(label)
    print("tools_per_hypothesis_sample", {k: v[:8] for k, v in list(by_h.items())[:3]})

    print("results_snippet:\n", results[:1200].replace("\n", " ")[:1200])


if __name__ == "__main__":
    try:
        main()
    except urllib.error.HTTPError as e:
        print("HTTP", e.code, e.read().decode()[:500])
        raise
