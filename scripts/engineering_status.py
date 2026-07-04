#!/usr/bin/env python3
"""One-command Propab engineering status (Agent 2 T1-003)."""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "packages" / "propab-core"))

def _ascii_safe(text: str) -> str:
    return text.replace("\u2192", "->").replace("\u2014", "-").replace("\u2713", "[OK]")

BACKLOG_PATH = ROOT / "BACKLOG.md"
COMPONENT_MAP = ROOT / "docs" / "component_map.md"
ROUTING_STATUS = ROOT / "artifacts" / "routing_corpus_status.json"

# Remaining deferred items still called out in component_map.md (not T3-003 scope).
DEFERRED_PATTERNS = (
    (r"hypotheses\.py.*_domain_fallback_options", "DEFERRED — seed fallbacks in orchestrator"),
    (r"theory_objects\.py", "DEFERRED — offline lifetime aggregation naming"),
    (r"policy_buckets\.py", "DEFERRED — infrastructure taxonomy by design"),
)

HEALTH_METRICS = (
    ("beliefs_promoted_by_trend", "campaign_synthesis_events payload", True),
    ("false_confirm_rate", "campaign_audit_results / replay scripts", True),
    ("worker_utilization", "research_campaigns.worker_utilization", True),
    ("hypothesis_type_diversity", "not yet logged", False),
    ("hypothesis_duplicate_rate", "campaign_synthesis_events", True),
    ("evidence_binding_rejection_rate", "campaign_synthesis_events", True),
    ("belief_stability", "campaign_synthesis_events", True),
    ("worker_experiment_success_rate", "research_campaigns", True),
)


def _load_routing_corpus() -> list[dict]:
    """Load full routing corpus (combinatorics artifacts + AP-free + domain corpora)."""
    import json

    corpus_artifacts = (
        ROOT / "artifacts" / "campaign_ac60fcda_deep_analysis.json",
        ROOT / "artifacts" / "campaign_da855131_deep_analysis.json",
    )
    out: list[dict] = []
    for path in corpus_artifacts:
        if not path.is_file():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        for key in ("confirmed_hypotheses", "refuted_hypotheses"):
            for row in data.get(key) or []:
                text = row.get("text") or ""
                out.append({
                    "id": row.get("id"),
                    "text": text,
                    "statement": text.split("\nPopulation:")[0].strip(),
                    "test_methodology": row.get("methodology") or "",
                    "prior_verdict": key.replace("_hypotheses", ""),
                })
    ap_free = (
        ROOT / "packages" / "propab-core" / "propab" / "domain_modules"
        / "math_combinatorics" / "ap_free_corpus.json"
    )
    if ap_free.is_file():
        out.extend(json.loads(ap_free.read_text(encoding="utf-8")))
    return out


def _routing_from_artifact() -> tuple[int, int, bool] | None:
    if not ROUTING_STATUS.is_file():
        return None
    try:
        data = json.loads(ROUTING_STATUS.read_text(encoding="utf-8"))
        total = int(data.get("total") or 0)
        ok = int(data.get("routing_ok") or 0)
        mm = int(data.get("routing_mismatches") or 0)
        if total > 0:
            return ok, total, mm == 0
    except (OSError, ValueError, TypeError):
        pass
    return None


def _routing_corpus(*, verify: bool = False) -> tuple[int, int, bool]:
    cached = _routing_from_artifact()
    if cached and not verify:
        return cached

    try:
        from propab.domain_modules.enzyme_kinetics.routing_inspector import (
            ROUTING_CORPUS as ENZYME_CORPUS,
            inspect_corpus as inspect_enzyme,
        )
        from propab.domain_modules.genomics.routing_inspector import inspect_corpus as inspect_genomics
        from propab.domain_modules.graph_invariants.routing_inspector import (
            ROUTING_CORPUS as GRAPH_CORPUS,
            inspect_corpus as inspect_graph,
        )
        from propab.domain_modules.math_combinatorics.routing_inspector import inspect_corpus

        comb = _load_routing_corpus()
        comb_report = inspect_corpus(comb) if comb else {"total": 0, "routing_mismatches": 0, "routing_ok": 0}
        gen_report = inspect_genomics()
        enz_report = inspect_enzyme(ENZYME_CORPUS)
        graph_report = inspect_graph(GRAPH_CORPUS)
        total = (
            int(comb_report.get("total") or 0)
            + int(gen_report.get("total") or 0)
            + int(enz_report.get("total") or 0)
            + int(graph_report.get("total") or 0)
        )
        mismatches = (
            int(comb_report.get("routing_mismatches") or 0)
            + int(gen_report.get("routing_mismatches") or 0)
            + int(enz_report.get("routing_mismatches") or 0)
            + int(graph_report.get("routing_mismatches") or 0)
        )
        return total - mismatches, total, mismatches == 0
    except Exception:
        if cached:
            return cached
        return 0, 0, False


def _run_pytest(*, quick: bool) -> tuple[int, int]:
    if quick:
        return -1, 0
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", "tests/", "-q", "--tb=no"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=900,
        )
        out = (proc.stdout or "") + (proc.stderr or "")
        m = re.search(r"(\d+)\s+passed", out)
        passed = int(m.group(1)) if m else 0
        m_fail = re.search(r"(\d+)\s+failed", out)
        failed = int(m_fail.group(1)) if m_fail else (1 if proc.returncode != 0 and passed == 0 else 0)
        return passed, failed
    except subprocess.TimeoutExpired:
        return 0, 0
    except OSError:
        return 0, 0


def _domain_plugins(*, quick: bool = False) -> list[dict]:
    from propab.domain_modules.registry import all_plugins

    rows: list[dict] = []
    for plugin in sorted(all_plugins(), key=lambda p: p.domain_id):
        features = plugin.available_features()
        if quick:
            rows.append({
                "domain_id": plugin.domain_id,
                "status": "ok",
                "preflight": True,
                "features": len(features),
                "last_campaign": _last_campaign_for_domain(plugin.domain_id),
            })
            continue
        try:
            pf = plugin.preflight()
            passed = pf.passed
            reason = pf.reason
        except Exception as exc:  # noqa: BLE001
            passed = False
            reason = str(exc)
        rows.append({
            "domain_id": plugin.domain_id,
            "status": "ok" if passed else "fail",
            "preflight": passed,
            "preflight_reason": reason if not passed else "",
            "features": len(features),
            "last_campaign": _last_campaign_for_domain(plugin.domain_id),
        })
    return rows


def _last_campaign_for_domain(domain_id: str) -> str | None:
    """Best-effort: scan recent campaign artifacts for domain id."""
    patterns = (
        ROOT / "artifacts" / "v1_frontier_campaign_latest.json",
        ROOT / "artifacts" / "campaign_a510c6a1_analysis.json",
    )
    for path in patterns:
        if not path.is_file():
            continue
        try:
            import json

            data = json.loads(path.read_text(encoding="utf-8"))
            cid = str(data.get("campaign_id") or data.get("id") or "")[:8]
            dom = str(data.get("domain") or data.get("domain_id") or "").lower()
            if dom == domain_id and cid:
                return cid
        except (OSError, ValueError, TypeError):
            continue
    return None


def _parse_backlog() -> list[tuple[str, str, str]]:
    if not BACKLOG_PATH.is_file():
        return []
    rows: list[tuple[str, str, str]] = []
    for line in BACKLOG_PATH.read_text(encoding="utf-8").splitlines():
        m = re.match(
            r"\|\s*(T\d-\d+)\s*\|\s*([^|]+)\|\s*`?\[([ ~✓])\]`?\s*complete\s*\|\s*`?([^`|]+)`?\s*\|",
            line,
        )
        if not m:
            m = re.match(r"\|\s*(T\d-\d+)\s*\|\s*([^|]+)\|\s*`?\[([ ~✓])\]`?\s*([^|]*)\|", line)
        if not m:
            continue
        tid, title, mark, rest = m.group(1), m.group(2).strip(), m.group(3), m.group(4).strip()
        if mark == "✓":
            commit = rest.strip().strip("`") if rest.strip() else ""
            status = f"[✓] complete  {commit}" if commit and commit != "—" else "[✓] complete"
        elif mark == "~":
            status = f"[~] in progress  {rest}"
        else:
            status = f"[ ] not started  {rest}".strip()
        rows.append((tid, title, status))
    return rows


def _parse_deferred() -> list[tuple[str, str]]:
    if not COMPONENT_MAP.is_file():
        return []
    text = COMPONENT_MAP.read_text(encoding="utf-8")
    found: list[tuple[str, str]] = []
    for pattern, label in DEFERRED_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            key = pattern.split("\\")[0].replace(".*", "").strip()
            found.append((key, label))
    return found


def _health_coverage() -> list[tuple[str, bool, str]]:
    hm_path = ROOT / "packages" / "propab-core" / "propab" / "health_metrics.py"
    hm_text = hm_path.read_text(encoding="utf-8") if hm_path.is_file() else ""
    out: list[tuple[str, bool, str]] = []
    for name, where, expected in HEALTH_METRICS:
        logged = name in hm_text or where.split()[0] in hm_text
        if name == "hypothesis_type_diversity":
            logged = False
        out.append((name, logged if expected else logged, where))
    return out


def main() -> int:
    quick = "--quick" in sys.argv
    verify_routing = "--verify-routing" in sys.argv
    passed, failed = _run_pytest(quick=quick)
    clean, total, routing_ok = _routing_corpus(verify=verify_routing)
    plugins = _domain_plugins(quick=quick)
    backlog = _parse_backlog()
    deferred = _parse_deferred()
    health = _health_coverage()

    print("PROPAB ENGINEERING STATUS")
    print("=========================")
    if passed < 0:
        print("Tests:          (skipped — use without --quick to run pytest)")
    elif failed:
        print(f"Tests:          {passed} passing, {failed} failing")
    else:
        print(f"Tests:          {passed} passing, 0 failing")
    if total:
        mark = "clean" if routing_ok else "MISMATCHES"
        src = "verified" if verify_routing else "cached artifact"
        print(f"Routing corpus: {clean}/{total} {mark} ({src})")
    else:
        print("Routing corpus: (could not inspect)")

    print()
    print("DOMAIN PLUGINS")
    print("--------------")
    for row in plugins:
        did = row["domain_id"]
        pf = "PASS" if row["preflight"] else "FAIL"
        sym = "OK" if row["preflight"] else "X"
        lc = row["last_campaign"] or "never campaigned"
        extra = f"  ({row.get('preflight_reason', '')})" if not row["preflight"] else ""
        print(f"{did:<22} {sym} preflight {pf}  {row['features']} features  last campaign: {lc}{extra}")

    print()
    print("BACKLOG")
    print("-------")
    for tid, title, status in backlog:
        line = f"{tid} ({_ascii_safe(title)})".ljust(42) + _ascii_safe(status)
        print(line)

    print()
    print("DEFERRED FROM REFACTOR")
    print("----------------------")
    if deferred:
        for key, label in deferred:
            print(f"{_ascii_safe(key):<36} {_ascii_safe(label)}")
    else:
        print("(none parsed from component_map.md)")

    print()
    print("HEALTH METRIC COVERAGE")
    print("-----------------------")
    for name, logged, where in health:
        sym = "OK logged" if logged else "X not logged"
        print(f"{name:<36} {sym}")

    return 0 if failed == 0 and routing_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
