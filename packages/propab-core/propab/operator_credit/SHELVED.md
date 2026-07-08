# SHELVED — unwired (2026-07-09)

`operator_credit` (the "telemetry moat" / per-operator credit, difference-rewards,
counterfactual replay) is **not wired into the live campaign path**. As of the
architecture audit it has **zero** non-self, non-test consumers anywhere in
`packages/` or `services/`.

**Decision (recorded in `docs/architecture/architecture-decision-register.md` I2):**
keep the code but treat it as **inactive** — do not assume it influences any
campaign. It is intentionally deferred until the orchestrator becomes a reasoning
agent (redesign phase C3), at which point per-operator/per-mechanism credit can
actually inform what the brain tries next. Only then does it become a real moat.

Do not add live dependencies on this package until that wiring is designed. If you
are auditing dead code: this is deferred-by-decision, not accidental.
