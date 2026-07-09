# Domain capabilities — repetitive research workflows → tools & skills

> Status: **DESIGN / BUILD PLAN** (2026-07-09). The general-agent redesign made the
> worker a general LLM agent that composes **tools** (deterministic callable code) and
> **skills** (methodology guidance). This doc catalogs the *repetitive* things a
> researcher does — cross-subfield, cross-question — and the reusable tool/skill each
> one warrants. It is the build plan the parallel subagents execute.

## 0. The bar (why a capability earns a slot)

Build a tool/skill **only when the activity recurs across many questions or a whole
workflow** — never for a single question (that is the plugin/vibe-computation trap we
dissolved). "Factor an integer", "align two sequences", "run a correct permutation
null", "check an identity" are done in thousands of investigations; "beat A396704 a(7)"
is not. If you can't name three unrelated questions a capability serves, it does not
belong here.

**Tool vs skill.**
- **Tool** = a `TOOL_SPEC` function in `packages/propab-core/propab/tools/<cluster>/`.
  Deterministic, sandbox-safe or worker-side, auto-registered by `ToolRegistry`.
  Does *computation* (factor, solve, align, enrich, certify).
- **Skill** = a `*.skill.md` in `packages/propab-core/propab/skills/<area>/`, chosen
  agentically. Does *methodology* (which null when, how to design the search, what
  counts as a proof). We already ship a strong `core/` set (experimental-design,
  confound-and-leakage-control, evidence-honesty, adversarial-test-design, …) — extend,
  don't duplicate.

**Non-negotiables (every tool inherits these).**
1. **Honesty by construction.** Anything that emits a *claim* must be independently
   re-verifiable. A search tool returns a witness the *certifier* re-checks; a stats
   tool carries the real null; a symbolic result is spot-checked numerically. Never a
   self-reported record. Mirror the B_3 pattern (`extremal_set_search` gated by
   `certify_b3_record`).
2. **Fail honestly.** Timeouts/unknowns are reported as `unknown`, never guessed. No
   fabricated p-values, no example-default masking a real input (the `_filter_params`
   footgun).
3. **Determinism + seeds** where results feed a record.
4. **Sandbox-safe or worker-side.** CPU-bound pure computation can run in the code
   sandbox; anything needing network (DB fetch, BLAST) is a *worker-side* tool with
   real timeouts + graceful degradation (see the literature-service pattern).
5. **Domain-general core.** These are reusable *capabilities*, not per-domain plugins;
   the agent selects from the full catalog by description.

**Inspiration, not copy.** For biology we take *inspiration* from
[biomni](https://github.com/snap-stanford/biomni) (a broad biomedical tool/DB toolbox)
and [K-Dense scientific-agent-skills](https://github.com/K-Dense-AI/scientific-agent-skills)
(agent skills) — the *breadth of workflows* they cover is the useful signal. We do NOT
import their code or architecture; every capability must fit our `TOOL_SPEC` +
certifier-gated + honesty-first model and our available libraries.

**Library reality (worker image, verified 2026-07-09).** Present: `sympy`, `numpy`,
`scipy`, `networkx`, `ortools`, `mpmath`, `sklearn`. **Missing** (must be added to the
worker requirements + image rebuilt before those tools work): `biopython (Bio)`,
`statsmodels`, `scanpy`, `galois`, `cvxpy`, `lifelines`. → **Math tools are buildable
today; biology tools have a dependency-provisioning step first.**

---

## 1. MATHEMATICS

Subfields served are noted per tool. The through-line: a mathematician **computes small
cases → spots a pattern → conjectures → searches for a proof or a counterexample →
verifies a certificate.** Each stage is a repetitive, tool-able activity.

### 1a. Symbolic computation (analysis · algebra · calculus · number theory)
| Tool | Workflow it serves | Lib | Honesty note |
|---|---|---|---|
| `symbolic_algebra` | simplify / expand / factor / solve equations & systems / differentiate / integrate / limits / series / substitute | sympy | pure; results are exact — spot-check numerically on random points when returning an identity |
| `symbolic_verify_identity` | *check a claimed equality/inequality/identity* holds (the honesty tool: independent verification of an agent's algebra) | sympy + numeric | returns proven/refuted/unknown; a symbolic `simplify(lhs-rhs)==0` PLUS random numeric sampling, never one alone |
| `exact_linear_algebra` | rank / kernel / determinant / inverse / eigen / char-poly / Smith & Jordan normal form over ℚ or 𝔽_p | sympy.Matrix | exact rational/finite-field arithmetic (no float error) |
| `polynomial_tools` | roots / factorization over ℚ,ℝ,ℂ,𝔽_p / resultants / discriminants / Gröbner bases | sympy | exact |

### 1b. Number theory & finite fields (number theory · cryptography · coding theory)
| Tool | Workflow | Lib | Honesty |
|---|---|---|---|
| `number_theory` | factor / isprime / nextprime / gcd,lcm / modular inverse & pow / CRT / totient / divisors / order / primitive roots / continued fractions / Jacobi | sympy.ntheory | deterministic; factorization witness is trivially re-multipliable (verify) |
| `finite_field_compute` | arithmetic, minimal/irreducible polynomials, subfield structure over GF(p^k) | sympy (galois if added) | exact |

### 1c. Combinatorics & graph theory (combinatorics · graph theory · discrete math)
| Tool | Workflow | Lib | Honesty |
|---|---|---|---|
| `combinatorial_enumeration` | exact count/generate of structured objects (subsets/partitions/permutations/compositions/set-partitions with a predicate); bijective checks | sympy / itertools | exact counts; large-n returns `unknown` not a truncated guess |
| `graph_invariants` | chromatic/clique/independence/domination number, connectivity, girth, diameter, spectrum, planarity, automorphisms, is-isomorphic | networkx | invariants are re-computable; NP-hard ones report method + optimality flag |
| `constraint_solve` | **general** decide/optimize over a declared CP/SAT model (booleans, linear/AllDifferent/table constraints, objective) — generalizes the B_3 `decide_b3_cpsat` to any combinatorial feasibility/optimization | ortools CP-SAT | SAT witness re-checked against the model; UNSAT is a proof only if the encoding is sound (agent must certify the encoding); honest `unknown` on timeout |
| `linear_optimization` | LP / MILP / assignment / flow | scipy.optimize / ortools | primal+dual / integrality certificate where available |

### 1d. Conjecturing & pattern discovery (number theory · combinatorics · experimental math)
| Tool | Workflow | Lib | Honesty |
|---|---|---|---|
| `sequence_oracle` | given first terms: identify (OEIS lookup, network), and/or guess a linear recurrence / rational generating function / closed form | sympy `find_linear_recurrence` + OEIS API | a *guessed* recurrence is a CONJECTURE — flagged as such; validated only by predicting held-out terms it wasn't fit on |
| `integer_relation` | given numeric constants, find an integer linear relation / closed form (PSLQ) | mpmath.pslq | a relation is a conjecture until re-derived at higher precision; report the precision + residual |
| `counterexample_search` | **general** framework: given a predicate over a described finite/parametrized space, search (exhaustive + randomized) for a counterexample to a universal claim | numpy/itertools | a counterexample is self-certifying (re-evaluate the predicate on it); "none found in N" is EVIDENCE, never a proof |

### 1e. Rigor / certificates (all subfields)
| Tool | Workflow | Lib | Honesty |
|---|---|---|---|
| `inequality_prover` | prove a polynomial inequality via sum-of-squares (SOS) certificate | sympy + SDP (cvxpy if added) | the SOS decomposition IS the certificate — re-expand to verify; else `unknown` |
| `certificate_check` | independent re-verification of a supplied witness/certificate for a stated property (the domain-general certifier hook) | pure | the honesty backstop; deterministic re-derivation from scratch |

### 1f. Mathematics **skills** (`skills/mathematics/`)
- `experiment-design-in-mathematics` — FIND vs PROVE; when SAT/ILP vs local search vs
  algebraic construction vs symbolic; compute-budget-aware escalation.
- `conjecturing-from-computation` — small cases → OEIS/gfun/PSLQ → held-out validation;
  never report a fitted formula as established.
- `symmetry-reduction-for-search-and-proof` — orbit/canonical-form pruning; the SOUND-
  reduction rule (an unsound symmetry break → false UNSAT → false "theorem": catastrophic).
- `counterexample-first-thinking` — try to break a conjecture before trying to prove it.
- `what-counts-as-a-proof` — computation-as-proof (exhaustive/exact) vs computation-as-
  evidence (sampled/heuristic); the line a "confirmed" verdict must respect.
- `certificate-first-verification` — always re-verify a witness with an independent
  checker before any claim (reuse `core/evidence-honesty`).

---

## 2. BIOLOGY

Through-line: a computational biologist **retrieves data → QC's & normalizes → runs a
statistical/ML analysis with the right controls → corrects for multiple testing →
cross-checks against orthogonal evidence & literature.** Breadth mirrors biomni/k-dense;
honesty discipline is ours (we already caught the fake-1.0 / leakage class).

> **Prerequisite (Batch B0):** add `biopython`, `statsmodels` (and optionally `scanpy`,
> `lifelines`, `pysam`) to the worker/sandbox requirements and rebuild the image. Until
> then only numpy/scipy-implementable bio-stats tools work. Keep heavy/network tools
> worker-side with timeouts.

### 2a. Sequence & structure (molecular biology · genomics · structural biology)
| Tool | Workflow | Lib | Honesty |
|---|---|---|---|
| `sequence_align` | pairwise + multiple sequence alignment; % identity; local/global | biopython | deterministic; report algorithm + params |
| `blast_search` | homology / similarity search against a DB (network, worker-side) | NCBI BLAST API | real timeout + graceful degrade; report E-values, never invent hits |
| `motif_scan` | find/score sequence motifs (PWM), restriction sites, ORFs | biopython/numpy | deterministic |
| `structure_analysis` | parse PDB/mmCIF, secondary structure, contacts, SASA, distances | biopython.PDB | from the real structure file only |
| `translate_annotate` | DNA↔protein translation, codon usage, GC content, feature extraction | biopython | pure |

### 2b. Retrieval (all biology — the "read the right database" workflow)
| Tool | Workflow | Lib | Honesty |
|---|---|---|---|
| `fetch_bio_entity` | fetch a record by ID from UniProt / NCBI Entrez / Ensembl / PDB / KEGG / Reactome / STRING | requests (worker-side) | cache + timeout + graceful degrade (literature-service pattern); returns provenance |
| `id_map` | cross-map identifiers (gene↔protein↔transcript↔pathway) | requests | provenance-tagged |

### 2c. Expression, variants, enrichment (genomics · transcriptomics · systems biology)
| Tool | Workflow | Lib | Honesty |
|---|---|---|---|
| `differential_expression` | group-vs-group DE with proper model + **multiple-testing correction** | scipy/statsmodels | reuse our leakage/null discipline; BH-FDR mandatory; effect size + CI, never bare p |
| `enrichment_analysis` | GO / KEGG / gene-set (hypergeometric + GSEA) over a gene list | scipy + gene-set files | background set explicit; FDR-corrected; guards a degenerate/empty background |
| `variant_annotation` | annotate variants (gene, consequence, frequency) | VEP/Ensembl (network) | provenance; unknown ≠ benign |
| `single_cell_analysis` | scRNA QC → normalize → cluster → marker genes | scanpy | report QC thresholds; markers are candidates, not conclusions |
| `phylogenetic_tree` | build + score a tree from aligned sequences | biopython.Phylo/dendropy | bootstrap support reported |

### 2d. Cross-domain statistics (biology · materials · any empirical domain)
> These are **general** and belong in `tools/statistics/` (some already exist:
> `statistical_significance`, `bootstrap_confidence`, `label_shuffle_null`).
| Tool | Workflow | Honesty |
|---|---|---|
| `multiple_testing_correction` | BH-FDR / Bonferroni / q-values over a p-vector | the missing guard behind most false biology findings — make it a first-class tool |
| `survival_analysis` | Kaplan-Meier + Cox (needs lifelines) | proportional-hazards check reported |
| `power_analysis` | required-n / achieved-power for a design | prevents over-claiming on underpowered data |

### 2e. Biology **skills** (`skills/biology/`) — biomni/k-dense-inspired, our rigor
- `experimental-design-in-biology` — batch effects, confounds, biological vs technical
  replicates, appropriate controls (extends `core/experimental-design-and-controls`).
- `choosing-the-analysis-for-the-data` — RNA-seq vs scRNA vs proteomics vs GWAS vs
  survival; the right test + null per data shape.
- `multiple-testing-and-effect-size-rigor` — FDR, effect size, avoiding p-hacking / the
  "significant but tiny" trap.
- `validating-a-biological-finding` — replication, orthogonal assays, held-out cohorts,
  leakage guards (our LOFO/within-group-shuffle null discipline).
- `reading-the-right-database` — which DB is authoritative for which entity/claim.

---

## 3. Build batches (for parallel subagents — disjoint file trees)

Each batch = one subagent. Each builds its tool files under the named cluster dir +
tests under `tests/tools/`; **edits only, no git commits** (parent verifies + commits).
Disjoint files → no races. Math batches ship immediately (libs present); Bio batch B0
must land before B1–B3.

| Batch | Deliverables | Cluster dir | Blocked on |
|---|---|---|---|
| **M1** symbolic core | `symbolic_algebra`, `symbolic_verify_identity`, `exact_linear_algebra`, `polynomial_tools` | `tools/mathematics/` | — |
| **M2** number theory & FF | `number_theory`, `finite_field_compute` | `tools/mathematics/` | — |
| **M3** combinatorics & graphs | `combinatorial_enumeration`, `graph_invariants` | `tools/mathematics/` | — |
| **M4** solve & optimize | `constraint_solve`, `linear_optimization` | `tools/mathematics/` | — |
| **M5** conjecturing | `sequence_oracle`, `integer_relation`, `counterexample_search` | `tools/mathematics/` | — |
| **M6** math skills | the 6 `skills/mathematics/*.skill.md` | `skills/mathematics/` | — |
| **S1** stats gap | `multiple_testing_correction`, `power_analysis` | `tools/statistics/` | — |
| **B0** bio deps | add biopython/statsmodels/(scanpy) to worker reqs; rebuild; smoke-import | requirements + Dockerfile | — |
| **B1** bio sequence | `sequence_align`, `motif_scan`, `translate_annotate`, `structure_analysis` | `tools/biology/` | B0 |
| **B2** bio retrieval | `fetch_bio_entity`, `id_map`, `blast_search` | `tools/biology/` | B0 |
| **B3** bio analysis | `differential_expression`, `enrichment_analysis` | `tools/biology/` | B0, S1 |
| **B4** bio skills | the 5 `skills/biology/*.skill.md` | `skills/biology/` | — |

## 4. Definition of done (every tool/skill)

- **Tool:** a `TOOL_SPEC` dict (name/domain/audience/description/params/output/example)
  + the function; auto-registers in `ToolRegistry` (verify it appears in
  `get_for("worker")`). **Tests** in `tests/tools/` that are *meaningful*: correct
  output on a known case, an honesty invariant (independent re-check / correct null /
  `unknown`-on-timeout / no example-default masking), and a fail-before/pass-after.
  Full suite stays green; additive only.
- **Skill:** a focused `*.skill.md` (when-to-use trigger + procedure + failure modes +
  "what would make this dishonest") consistent with the `core/` set; no code.
- **Never** merge a tool that can self-report a claim without an independent check, or
  that guesses on failure. The certifier/verification path is the point of the whole
  system — new tools must strengthen it, not route around it.
