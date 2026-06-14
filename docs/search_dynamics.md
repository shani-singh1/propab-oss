# Search Dynamics — Theory Document

Propab's equivalent of statistical mechanics for automated research campaigns.
This document is **theory only** — no implementation (per fixes.md #8).

---

## 1. Why does entropy rise?

Theme entropy measures diversity of active research themes on the frontier.
Early campaigns start with concentrated seeds (low entropy). As hypotheses
are tested and expanded, the tree branches into boundary, mechanistic, and
generalization children — each potentially introducing new themes.

**Mechanism:** expansion operators (especially `confirmed_expand` and
`boundary` mutation) inject new theme vectors into the histogram faster
than closure operators consolidate them.

**Prediction:** entropy rises monotonically until saturation, unless
`closure_aware` branching dominates and prunes low-gain branches early.

---

## 2. Why do plateaus happen?

Plateaus occur when the frontier stops generating novel themes:

1. **Theme saturation** — one theme exceeds ~60% of tested nodes.
2. **Closure dominance** — confirmed/refuted ratio stabilizes; expansions
   produce retests rather than new directions.
3. **Operator lock-in** — the same retrieval + verification pair succeeds
   often enough that the bandit stops exploring alternatives.

**Signature:** `dH/dt → 0` for ≥5 consecutive snapshots while `tested` still
increases (activity without diversity gain).

---

## 3. Why do local minima occur?

Local minima in search are regions where:

- All pending hypotheses share the same parent theme.
- Verification operators consistently return `inconclusive`.
- Branching operator selects depth-first paths that do not escape the theme
  basin.

**Escape operators:** `contradiction` mutation after refutation,
`refuted_expand` decomposition, and hybrid retrieval that pulls cross-theme
evidence from historical campaigns.

---

## 4. Why do certain operators work?

Operators are effective when matched to search phase:

| Phase | Effective operators | Why |
|-------|---------------------|-----|
| Cold start | `breadth_first`, `bm25`, `numerical` | Explore wide, cheap verification |
| Growth | `closure_aware`, `hybrid`, `simulation` | Balance exploration with closure |
| Plateau | `contradiction`, `semantic`, `symbolic` | Inject novelty or deep verification |

Credit assignment should be **phase-conditioned**, not global.

---

## 5. Search phases

We model campaigns as three dynamical phases:

```
cold_start  →  growth  →  plateau
   |              |           |
 low H         rising H     flat H
 high pending  rising closure  high closure
```

Phase transitions are detectable from `SearchStateV3`:

- `uncertainty × (1 - closure)` high → cold start
- `diversity` rising, `saturation` mid → growth
- `saturation > 0.6`, `dH/dt ≈ 0` → plateau

---

## 6. Implications for operator evolution

1. **Do not optimize operators globally** — phase-aware priors beat flat priors.
2. **Counterfactual replay is the gradient** — remove operator X, measure
   reward drop; that IS the credit signal.
3. **Campaign families inherit phase transitions** — child campaigns that
   share a baseline start closer to growth phase; calibrate accordingly.
4. **OperatorBench should stratify by phase** — an operator that fails
   globally may dominate in cold start.

---

## 7. Open questions

- Is theme entropy the right order parameter, or should we use hypothesis
  tree branching factor?
- Can we predict plateau onset 3–5 snapshots early from operator history?
- When does adding campaigns to a family stop improving mega-level credits?

These questions should be answered from the `CampaignCorpus` and
`OperatorTraceLedger` before scaling to 50–100 campaigns.
