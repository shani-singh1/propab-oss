#!/usr/bin/env python3
"""P0 — collect mechanisms/seeds/anomalies from last N campaigns."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import psycopg

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"


def _load_tree(raw) -> dict:
    if not raw:
        return {}
    return json.loads(raw) if isinstance(raw, str) else raw


def main() -> int:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    with psycopg.connect("postgresql://propab:propab@localhost:5432/propab") as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id::text, question, status, seed_source, anomaly_artifacts_dir,
                       hypothesis_tree_json, started_at
                FROM research_campaigns
                ORDER BY started_at DESC NULLS LAST
                LIMIT %s
                """,
                (n,),
            )
            campaigns = []
            for cid, question, status, seed_source, art_dir, tree_json, started in cur.fetchall():
                tree = _load_tree(tree_json)
                nodes = tree.get("nodes") or {}
                seeds = []
                mechanisms = []
                for nid, node in nodes.items():
                    if not isinstance(node, dict):
                        continue
                    if node.get("depth", 0) == 0:
                        seeds.append({
                            "node_id": nid,
                            "text": (node.get("text") or "")[:500],
                            "mechanism": node.get("mechanism"),
                            "mechanism_id": node.get("mechanism_id"),
                        })
                    mech = node.get("mechanism")
                    if mech:
                        mechanisms.append({
                            "node_id": nid,
                            "depth": node.get("depth"),
                            "mechanism": mech,
                            "text_snippet": (node.get("text") or "")[:200],
                        })
                ledger = tree.get("finding_ledger") or []
                for entry in ledger:
                    if isinstance(entry, dict):
                        for m in entry.get("mechanisms") or []:
                            mechanisms.append({"source": "finding_ledger", "mechanism": m, "claim": entry.get("claim", "")[:160]})

                anomalies = []
                art_path = Path(art_dir) if art_dir else None
                if art_path and not art_path.is_absolute():
                    art_path = ROOT / art_path
                for fname in ("anomaly_objects.json", "mechanism_objects.json"):
                    fp = (art_path / fname) if art_path else ART / fname
                    if fp.exists() and str(cid) in str(art_dir or ""):
                        pass
                # Global artifacts fallback for mandrake
                for fp in [ART / "anomaly_objects.json", ART / "mechanism_objects.json"]:
                    if fp.exists() and seed_source == "anomaly":
                        data = json.loads(fp.read_text(encoding="utf-8"))
                        if fname := fp.name:
                            if "anomaly" in fp.name:
                                anomalies.extend(data if isinstance(data, list) else data.get("anomalies", []))
                            else:
                                for mo in (data if isinstance(data, list) else []):
                                    mechanisms.append({"source": "artifact_mechanism_object", "mechanism": mo})

                campaigns.append({
                    "campaign_id": cid,
                    "question": question,
                    "status": status,
                    "seed_source": seed_source,
                    "started_at": str(started),
                    "n_tree_nodes": len(nodes),
                    "seeds": seeds,
                    "mechanisms_from_nodes": [m for m in mechanisms if m.get("source") != "artifact_mechanism_object"],
                    "artifact_anomalies": anomalies,
                    "artifact_mechanisms": [m for m in mechanisms if m.get("source") == "artifact_mechanism_object"],
                    "finding_ledger_len": len(ledger),
                })

    out = {"n_campaigns": len(campaigns), "campaigns": campaigns}
    (ART / "mechanism_audit_p0_raw.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({
        "campaigns": len(campaigns),
        "total_seeds": sum(len(c["seeds"]) for c in campaigns),
        "total_node_mechanisms": sum(len(c["mechanisms_from_nodes"]) for c in campaigns),
        "out": str(ART / "mechanism_audit_p0_raw.json"),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
