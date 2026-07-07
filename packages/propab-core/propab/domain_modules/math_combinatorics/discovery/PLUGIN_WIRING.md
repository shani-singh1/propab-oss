# Plugin wiring note — math_combinatorics discovery kernel

This subpackage is self-contained and does **not** edit `verifier.py`,
`plugin.py`, or `constructors.py`. It is the DISCOVERY KERNEL for OEIS **A396704**
(maximum size of a B_3 set in `{0,1}^n`). This note describes the intended hook so
the plugin owner can wire it after independently verifying the kernel.

## What the kernel exposes (stable entry points)

```python
from propab.domain_modules.math_combinatorics.discovery import (
    is_B3, certify_b3_record,          # paranoid, independent witness checks
    RECORDS, get_record, best_known,   # sourced best-known registry
    find_max_b3, find_b3_set,          # the smart finder (B&B + greedy + DLS)
    canonical_form,                    # hyperoctahedral canonicalization
)
```

- `is_B3(S)` — deterministic B_3 test on an explicit set (list of 0/1 tuples).
- `certify_b3_record(witness, published_best)` — the record gate; returns a dict
  with `certified` and per-check booleans. Independent of the finder.
- `find_max_b3(n, time_budget=...)` — returns `{n, size, set, method,
  proven_optimal, verified, ...}`, the `set` always re-checked by `is_B3`.
- `record_registry.RECORDS["A396704"]` — best-known values + status per `n`.

## Intended objective_spec hook

The engine routes computational campaigns through a deterministic `objective_spec`
(see `test_domain_objective_spec.py`). Add a B_3 objective analogous to the cap-set
path in `verifier.py`:

- **objective id:** `"b3_binary_cube"` (domain `math_combinatorics`).
- **detector:** statement mentions `B_3` / `B3` / "threefold sums distinct" /
  "A396704" / `{0,1}^n` with `n` parsed as for cap sets (`_extract_cap_dims`-style).
- **runner:** call `find_max_b3(n, time_budget)`; put `size`, `set` (witness),
  `method`, `proven_optimal`, and `elapsed` on the result. This mirrors
  `compute_cap_set` (real computed size + independent validity), never a table read.
- **verdict mapping:** compare `size` against `best_known("A396704", n)`:
  - `size < best_known` → `below_best_known`;
  - `size == best_known` and status `provisional_lower_bound` → `matches_best_known`
    (reproduced a search bound — honest, not a rediscovery of a proven value);
  - `size == best_known` and status `proven_optimal` → reproduced a proven value
    (flag like the existing `trivial_rediscovery` for proven table values);
  - `size > best_known` → candidate record → **route to certification** (below).

## Record-witness routing (the headline path)

When `find_max_b3` returns `size > best_known("A396704", n)` for a term whose
status is `provisional_lower_bound` or `open`:

1. Call `certify_b3_record(result, best_known("A396704", n))`.
2. Only if `certified` is True, surface a `discovery_worthy` finding **and emit the
   witness JSON** (`{"n": n, "claimed_size": size, "set": [...]}`) for INDEPENDENT
   re-verification. Do not let any code path assert "record" without a `certified`
   witness. `certify_b3_record` certifies a *lower-bound improvement only*, never
   optimality.

## Notes / guards

- Keep this domain-specific; core files stay domain-general (domain-independence
  rule). All B_3 logic lives here.
- No new heavy dependency: the finder is pure-Python ILS/DLS (no SAT/ILP solver was
  available in the environment; a solver-backed exact path can be added later
  behind an availability check, as the note in the research doc suggests).
- `known_witnesses.py` holds witnesses the finder itself produced (provenance +
  warm starts); a size-16 n=7 witness reproduces the *published* bound and is
  explicitly **not** a record.
