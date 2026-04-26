from __future__ import annotations

import json
import os
import subprocess
import time
import urllib.request
from urllib.error import HTTPError, URLError
from pathlib import Path
from typing import Any

BASE = "http://localhost:8000"
OUT_PATH = Path("artifacts") / "questions_audit_report.json"
QUESTIONS = [
    "Does the choice of activation function significantly affect transformer training stability and convergence speed on sequence classification tasks?",
    "Does batch normalization placement (pre-norm vs post-norm) affect gradient flow and convergence rate in MLPs trained on noisy synthetic data?",
    "Does learning rate warmup improve final model quality beyond its effect on early training stability?",
    "Compare SGD, Adam, AdamW, RMSProp, and Adagrad across five different loss surface geometries — rank them by convergence speed, final loss, and stability.",
    "Does model width or model depth contribute more to parameter efficiency in MLPs trained on a fixed compute budget?",
]


def http_json(url: str, method: str = "GET", body: dict[str, Any] | None = None, timeout: int = 120) -> dict[str, Any]:
    payload = json.dumps(body).encode("utf-8") if body is not None else None
    headers = {"Content-Type": "application/json"} if body is not None else {}
    last_exc: Exception | None = None
    for attempt in range(5):
        try:
            req = urllib.request.Request(url, data=payload, headers=headers, method=method)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except HTTPError as exc:
            # Retry transient upstream pressure / gateway events.
            if exc.code in (429, 500, 502, 503, 504) and attempt < 4:
                time.sleep(1.0 * (2**attempt))
                last_exc = exc
                continue
            raise
        except (URLError, ConnectionResetError, TimeoutError) as exc:
            if attempt < 4:
                time.sleep(1.0 * (2**attempt))
                last_exc = exc
                continue
            raise
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("http_json retry loop exhausted without response.")


def probe_tex_pdf(session_id: str) -> dict[str, Any]:
    py = f"""
from propab.storage import get_object_bytes
import fitz
sid='{session_id}'
base=f'sessions/{{sid}}/paper'
tex=get_object_bytes(object_name=f'{{base}}/main.tex')
pdf=get_object_bytes(object_name=f'{{base}}/main.pdf')
print('HAS_TEX', bool(tex), 'HAS_PDF', bool(pdf))
if tex:
    t=tex.decode('utf-8','ignore')
    a0=t.find('\\\\begin{{abstract}}')
    a1=t.find('\\\\end{{abstract}}')
    print('ABSTRACT_TEXT_START')
    print((t[a0:a1+14] if a0>=0 and a1>a0 else t[:1800]).replace('\\\\n',' '))
    print('ABSTRACT_TEXT_END')
if pdf:
    doc=fitz.open(stream=pdf, filetype='pdf')
    text=' '.join((doc.load_page(p).get_text('text') or '') for p in range(min(3, doc.page_count)))
    print('PDF_TEXT_START')
    print(text[:2500].replace('\\\\n',' '))
    print('PDF_TEXT_END')
"""
    run = subprocess.run(
        ["docker", "exec", "propab-oss-orchestrator-1", "python", "-c", py],
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    return {
        "exit_code": run.returncode,
        "stdout_tail": run.stdout[-12000:],
        "stderr_tail": run.stderr[-4000:],
    }


def main() -> None:
    limit = int(os.getenv("AUDIT_QUESTIONS_LIMIT", "5"))
    if limit < 1:
        limit = 1
    selected = QUESTIONS[: min(limit, len(QUESTIONS))]
    sessions: list[dict[str, Any]] = []

    for i, question in enumerate(selected, 1):
        created = http_json(
            f"{BASE}/research",
            method="POST",
            body={"question": question, "config": {"max_hypotheses": 5, "paper_ttl_days": 30, "llm_model": "stress"}},
        )
        sid = created["session_id"]
        print(f"[Q{i}] session={sid}")
        final_state: dict[str, Any] = {}
        for tick in range(240):
            final_state = http_json(f"{BASE}/sessions/{sid}", timeout=40)
            if tick % 10 == 0:
                print(f"[Q{i}] t={tick*5:>4}s status={final_state.get('status')} stage={final_state.get('stage')}")
            if final_state.get("status") in ("completed", "failed"):
                break
            time.sleep(5)

        entry: dict[str, Any] = {
            "q_index": i,
            "question": question,
            "session_id": sid,
            "status": final_state.get("status"),
            "stage": final_state.get("stage"),
            "error": final_state.get("error"),
        }
        try:
            hyps = http_json(f"{BASE}/sessions/{sid}/hypotheses")["hypotheses"]
            trace = http_json(f"{BASE}/sessions/{sid}/trace")["trace"]
            entry["hypothesis_count"] = len(hyps)
            entry["trace_steps"] = len(trace)
            entry["verdicts"] = [h.get("verdict") for h in hyps]
            entry["confidences"] = [h.get("confidence") for h in hyps]
            entry["evidence_samples"] = [str(h.get("evidence_summary") or "")[:280] for h in hyps[:3]]
            entry["trace_step_types"] = sorted({str(t.get("step_type") or "") for t in trace})
            entry["trace_tool_names"] = sorted(
                {
                    str((t.get("input_json") or {}).get("tool"))
                    for t in trace
                    if isinstance(t.get("input_json"), dict) and (t.get("input_json") or {}).get("tool")
                }
            )
        except Exception as exc:  # noqa: BLE001
            entry["trace_collection_error"] = str(exc)

        try:
            paper = http_json(f"{BASE}/sessions/{sid}/paper")["paper"]
            entry["paper_present"] = True
            entry["paper_keys"] = sorted(list(paper.keys()))
            entry["results_snippet"] = (paper.get("results_latex") or "")[:1200]
        except Exception as exc:  # noqa: BLE001
            entry["paper_present"] = False
            entry["paper_error"] = str(exc)

        entry["tex_pdf_probe"] = probe_tex_pdf(sid)
        sessions.append(entry)
        print(f"[Q{i}] final status={entry['status']} paper_present={entry.get('paper_present')}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps({"generated_at_epoch": time.time(), "sessions": sessions}, indent=2), encoding="utf-8")
    print(f"REPORT_PATH {OUT_PATH}")


if __name__ == "__main__":
    main()
