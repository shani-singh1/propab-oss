"""
Submit all 5 research questions to Propab and track them.
Run: python scripts/run_5q.py
"""
from __future__ import annotations

import json
import sys
import time
import urllib.request
import urllib.error

API = "http://localhost:8000"

QUESTIONS = [
    {
        "id": "Q1",
        "question": "Does the choice of activation function significantly affect transformer training stability and convergence speed on sequence classification tasks?",
        "max_hypotheses": 4,
    },
    {
        "id": "Q2",
        "question": "Does batch normalization placement (pre-norm vs post-norm) affect gradient flow and convergence rate in MLPs trained on noisy synthetic data?",
        "max_hypotheses": 4,
    },
    {
        "id": "Q3",
        "question": "Does learning rate warmup improve final model quality beyond its effect on early training stability?",
        "max_hypotheses": 4,
    },
    {
        "id": "Q4",
        "question": "Compare SGD, Adam, AdamW, RMSProp, and Adagrad across five different loss surface geometries — rank them by convergence speed, final loss, and stability.",
        "max_hypotheses": 4,
    },
    {
        "id": "Q5",
        "question": "Does model width or model depth contribute more to parameter efficiency in MLPs trained on a fixed compute budget?",
        "max_hypotheses": 4,
    },
]


def post_json(url: str, data: dict) -> dict:
    body = json.dumps(data).encode()
    req = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())


def get_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=30) as resp:
        return json.loads(resp.read())


def submit_question(q: dict) -> str:
    resp = post_json(
        f"{API}/research",
        {
            "question": q["question"],
            "config": {
                "max_hypotheses": q["max_hypotheses"],
                "paper_ttl_days": 30,
            },
        },
    )
    session_id = resp["session_id"]
    print(f"  {q['id']} -> session_id={session_id}")
    return session_id


def main() -> None:
    print("=== Submitting 5 questions ===\n")
    sessions: list[dict] = []
    for q in QUESTIONS:
        print(f"Submitting {q['id']}: {q['question'][:80]}...")
        try:
            sid = submit_question(q)
            sessions.append({"id": q["id"], "session_id": sid, "status": "submitted"})
        except Exception as exc:
            print(f"  ERROR: {exc}")
            sessions.append({"id": q["id"], "session_id": None, "status": f"error: {exc}"})
        time.sleep(1)

    print(f"\n=== All submitted. Monitoring {len([s for s in sessions if s['session_id']])} sessions ===\n")

    deadline = time.monotonic() + 3600  # 1 hour max
    while True:
        if time.monotonic() > deadline:
            print("Monitor timeout reached (1h). Check /sessions manually.")
            break

        all_done = True
        print(f"\n--- Status check {time.strftime('%H:%M:%S')} ---")
        for s in sessions:
            if s["session_id"] is None:
                continue
            if s["status"] in ("completed", "failed"):
                print(f"  {s['id']}: {s['status']}")
                continue
            try:
                data = get_json(f"{API}/sessions/{s['session_id']}")
                status = data.get("status", "unknown")
                stage = data.get("stage", "")
                s["status"] = status
                print(f"  {s['id']}: status={status} stage={stage}")
                if status not in ("completed", "failed"):
                    all_done = False
            except Exception as exc:
                print(f"  {s['id']}: poll error={exc}")
                all_done = False

        if all_done:
            print("\n=== All sessions finished ===\n")
            break

        print("  (waiting 30s...)")
        time.sleep(30)

    # Final summary
    print("\n=== FINAL SUMMARY ===")
    for s in sessions:
        print(f"\n{s['id']} [{s['session_id']}]: {s['status']}")
        if s["session_id"] and s["status"] in ("completed", "failed"):
            try:
                data = get_json(f"{API}/sessions/{s['session_id']}")
                print(f"  stage={data.get('stage')}")
            except Exception:
                pass

    # Save session IDs for forensics
    with open("artifacts/run_5q_sessions.json", "w") as f:
        json.dump(sessions, f, indent=2)
    print("\nSession IDs saved to artifacts/run_5q_sessions.json")


if __name__ == "__main__":
    main()
