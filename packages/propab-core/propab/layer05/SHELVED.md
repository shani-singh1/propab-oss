# SHELVED — mostly unwired (2026-07-09)

`layer05` (offline replay, search/hybrid/ensemble simulators, offline policy eval,
calibration, fitness ledgers) is **almost entirely off the live campaign path**.

Traced live footprint (architecture audit): only a `SimulationFitnessLedger` load
+ `policy_analyst` (which is decorative — "the LLM never edits") + a small
`_policy_score_multiplier` nudge on `frontier_score`. The **simulator bulk**
(`simulate_search`, hybrid/ensemble, `replay_campaign_snapshots`, offline eval) is
consumed **only** by `operator_credit`, which itself has no consumers → a dead
subgraph.

**Decision (recorded in `docs/architecture/architecture-decision-register.md` I1):**
keep the code but treat the simulator/offline-eval bulk as **inactive/deferred**
until the orchestrator reasoning loop (redesign phase C3) needs learned policy.
The tiny live `_policy_score_multiplier` hook remains KEEP-WATCH.

Do not assume layer05 simulation affects live campaigns. Deferred-by-decision.
