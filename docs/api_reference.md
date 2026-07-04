# Propab API reference

Interactive OpenAPI docs are served at **`http://localhost:8000/docs`** when the API
is running. This page describes the workflows operators and integrators use most often.

Base URL (local): `http://localhost:8000`

---

## Authentication

Local development has no API auth. Production deployments should place the API
behind a reverse proxy and restrict the orchestrator internal endpoint with
`ORCHESTRATOR_INTERNAL_TOKEN` (Bearer token on `/internal/*` calls from API → orchestrator).

---

## Workflow 1 — Launch a campaign

```http
POST /campaigns
Content-Type: application/json
```

**Example request**

```json
{
  "question": "Do housekeeping genes show cross-tissue LOFO R² above 0.15? [domain_profile:genomics]",
  "compute_budget_hours": 3.0,
  "policy_mode": "accepted",
  "max_hypotheses": 120
}
```

**Example response** (`200`)

```json
{
  "campaign_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "stream_url": "/stream/a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "status": "started"
}
```

**Errors**

| Code | Meaning |
|------|---------|
| `422` | Invalid body (question too short, bad `policy_mode`, etc.) |
| `502` | Orchestrator unreachable (`ORCHESTRATOR_URL` set but service down) |

The campaign loop runs in the **orchestrator** service. State is checkpointed to
Postgres after each batch; the API process can restart without killing the campaign.

---

## Workflow 2 — Monitor live

### SSE event stream

```http
GET /stream/{campaign_id}
Accept: text/event-stream
```

Each line is `data: {json}\n\n`. Event types include hypothesis lifecycle,
verification verdicts, synthesis rounds, and `paper.ready`.

### Poll snapshot

```http
GET /campaigns/{campaign_id}
```

Returns:

- `campaign` — full serialized tree, belief state, budget fields
- `summary` — tested / confirmed / refuted counts, elapsed time
- `event_counts_by_type` — quick sanity check that workers are emitting events

**Example headline fields**

```json
{
  "campaign_id": "...",
  "summary": {
    "total_tested": 42,
    "confirmed": 8,
    "refuted": 15,
    "inconclusive": 19,
    "elapsed_sec": 5400
  }
}
```

**Errors:** `404` if the campaign id does not exist.

### Terminal dashboard (optional)

```bash
python scripts/health_dashboard.py --once
python scripts/health_dashboard.py --interval 30
```

---

## Workflow 3 — Retrieve findings

### Campaign tree (preferred for active campaigns)

```http
GET /campaigns/{campaign_id}
```

Confirmed nodes live under `campaign.hypothesis_tree.nodes` with `verdict: "confirmed"`.
The finding ledger is in `campaign.hypothesis_tree.finding_ledger`.

### Session endpoints (legacy short sessions + paper)

| Endpoint | Purpose |
|----------|---------|
| `GET /sessions/{id}/hypotheses` | Ranked hypothesis list |
| `GET /sessions/{id}/events` | Full event log (`?limit=500` for tail) |
| `GET /sessions/{id}/paper` | Final paper payload when `paper.ready` fired |
| `GET /sessions/{id}/trace` | Experiment step trace |

**Errors:** `404` when session/paper not found.

---

## Workflow 4 — Resume a stopped campaign

1. Check readiness:

```http
GET /campaigns/{campaign_id}/resume-readiness
```

2. Resume with optional budget extension:

```http
POST /campaigns/{campaign_id}/resume
Content-Type: application/json

{
  "compute_budget_hours": 4.0,
  "clear_hypothesis_cap": true
}
```

**Example response**

```json
{
  "campaign_id": "...",
  "stream_url": "/stream/...",
  "status": "resumed"
}
```

**Errors**

| Code | Meaning |
|------|---------|
| `404` | Campaign not found |
| `409` | Campaign already `active` |
| `502` | Orchestrator unreachable |

Contrarian re-run (new belief direction, tree preserved):

```json
{
  "belief_reset": "contrarian",
  "orchestrator_directive": "Prior direction exhausted — test the opposite mechanism."
}
```

There is no dedicated “stop” HTTP endpoint. Campaigns stop when budget is exhausted,
breakthrough criteria fire, or the orchestrator sets a non-null `stop_reason`. To halt
early, restart the orchestrator after setting campaign status out of `active` in Postgres
(see [operator runbook](./operator_runbook.md)).

---

## Other endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Liveness + build metadata |
| `GET` | `/campaigns` | List all campaigns (newest first) |
| `POST` | `/research` | Short hypothesis session (not full campaign) |
| `GET` | `/tools` | Sub-agent tool specs |
| `GET` | `/tools/{domain}` | Tools filtered by domain cluster |

---

## CLI alternatives

```bash
# Dry-run domain preflight + routing (no LLM spend)
python scripts/start_v1_frontier_campaign.py --domain genomics --dry-run

# Compare two completed campaigns
python scripts/compare_campaigns.py --campaign-a <id> --campaign-b <id>
```

---

## Related docs

- [Adding a domain](./adding_a_domain.md)
- [Operator runbook](./operator_runbook.md)
- [ARCHITECTURE.md](../ARCHITECTURE.md)
