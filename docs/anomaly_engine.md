# Anomaly Engine v1 (Mandrake)

Upstream pipeline: **Sweep → Anomaly → Mechanism → Hypothesis → Verification → Paper**

## Quick start

```bash
# 1. Run offline pipeline (requires mandrake-data/ at repo root)
python scripts/run_anomaly_pipeline.py

# 2. Start anomaly-seeded campaign (1 h, max 50 hypotheses)
python scripts/start_mandrake_campaign.py
```

## Deliverables

| Artifact | Description |
|----------|-------------|
| `artifacts/sweep_results.parquet` | Feature-subset × model metrics |
| `artifacts/anomaly_objects.json` | Top 10–20 surprising subsets (LOFO survives) |
| `artifacts/mechanism_objects.json` | 3–5 induced mechanisms |
| `artifacts/mechanism_report.md` | Human-readable summary |

## Integration

Campaigns with `seed_source: "anomaly"` load `mechanism_objects.json` and generate
**mechanism-discrimination** hypotheses before the standard tree/verification loop.

The campaign loop, verification, policy, and paper paths are unchanged.

## Domain

Mandrake Retroviral Wall — predict RT activity (`pe_efficiency_pct`) independent of
`rt_family` using biophysical features (geometry, electrostatics, thermal, surface;
ESM features excluded).
