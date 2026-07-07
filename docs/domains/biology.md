# Biology domains — honest capability statement (v1)

Propab ships three real-data biology domains: **genomics**, **enzyme_kinetics**,
and **mandrake**. Each runs a real leave-one-group-out (LOFO) generalization test
plus a label-shuffle permutation null through Propab's honesty gate. This document
states plainly what each domain's real data + verifier **can** and **cannot**
detect, the key limits, how to fetch data and run, and the honest bottom line.

> **Bottom line, stated up front.** On the datasets currently shipped, these
> domains more often produce a rigorous *"no signal"* (an honest refute or
> inconclusive) than a novel confirmed finding. That is by design: the
> label-shuffle null is conservative, the real biological signals here are subtle
> and confounded by group identity, and the honesty gate refuses to confirm a
> result that a shuffled-label control could reproduce. A rigorous negative is a
> feature, not a bug. See "What richer data would change" at the end.

---

## The shared verification contract

For every biology domain the verifier does the same three things:

1. **Group the rows** by a real biological partition (dominant tissue; top-level
   EC class; RT evolutionary family).
2. **Leave-one-group-out (LOFO):** fit a ridge model on all groups but one and
   score R² on the held-out group. This measures *generalization across groups*,
   not in-sample fit.
3. **Label-shuffle null:** re-run LOFO many times with the group labels permuted.
   A finding is only **confirmed** when the observed LOFO R² beats this null
   (permutation p < 0.05) — i.e. the signal genuinely depends on the group
   structure, not on the model memorizing rows.

A high in-sample R² that the shuffle null also reproduces is **refuted**, not
confirmed. This is what stops a within-row tautology (a target that is an
algebraic transform of its own features) from reading as a discovery.

Each domain also exposes `known_value_check(claim, evidence)` on its plugin: a
claim that merely restates an established biology fact (see the per-domain
references below) is flagged `trivial_rediscovery=True` and demoted out of the
headline discovery count by the domain-general paper path
(`propab.paper_narrative._is_rediscovery`).

---

## genomics — GTEx v8 cross-tissue expression

- **Real data:** a subset of the **GTEx v8 median gene-level TPM** atlas
  (`GTEx_Analysis_2017-06-05_v8 gene_median_tpm`). 1000 most-variable genes ×
  10 organ tissues, log2(median TPM + 1). Provenance:
  `packages/propab-core/propab/domain_modules/genomics/DATA_PROVENANCE.md`.
- **Grouping:** each gene by its dominant (max-expression) tissue.
- **Features:** mean expression, expression variance, coefficient of variation
  across tissues, Yanai tau tissue-specificity index.
- **CAN detect:** whether a gene-level expression feature predicts a gene-level
  target (tau, mean expression) in a **held-out tissue** better than a tissue-
  label-shuffle null.
- **CANNOT detect / known limits:**
  - Many gene-level summaries (tau, CV, variance) are **algebraic functions of
    the same expression row**. A model can "predict" one from the others with no
    cross-tissue biology — the shuffle null then correctly does **not** reject, so
    the verdict is refuted. This is the dominant outcome on this subset (see the
    live transcript: LOFO R²=1.0 with shuffle p=1.0 → refuted).
  - It is a *median-TPM summary* atlas, not per-sample data: no eQTLs, no
    individual-level variance, no isoform structure.
- **Rediscovery references (real known values):** the canonical housekeeping-gene
  set (ACTB, GAPDH, TUBB, B2M, …; Eisenberg & Levanon 2013) and the tau ≈ 0.5
  housekeeping-vs-tissue-specific threshold (Kryuchkova-Mostacci &
  Robinson-Rechavi 2017). A claim that a listed gene is housekeeping, or that the
  split sits at tau ≈ 0.5, is flagged as a rediscovery.

## enzyme_kinetics — DLKcat (BRENDA + SABIO-RK) kcat

- **Real data:** the **DLKcat** kcat compilation (Li et al., *Nature Catalysis*
  2022; SysBioChalmers/DLKcat), which aggregates real measured turnover numbers
  from BRENDA and SABIO-RK with EC number, organism, substrate and sequence.
  Balanced to ≤600 per EC class (~3.5k enzymes across EC1–EC6). Provenance:
  `packages/propab-core/propab/domain_modules/enzyme_kinetics/DATA_PROVENANCE.md`.
- **Grouping:** top-level EC class (EC1…EC6).
- **Features:** molecular weight, sequence length, GRAVY hydropathy, and residue-
  composition fractions (charged, hydrophobic, aromatic, polar, glycine, proline),
  all derived from the real protein sequence. Target: `log_kcat` (log10 of the
  real measured kcat).
- **CAN detect:** whether a coarse sequence/physicochemical feature predicts
  `log_kcat` across a **held-out EC class** beyond an EC-label-shuffle null.
- **CANNOT detect / known limits:**
  - Bulk sequence-composition features carry **little cross-EC signal** for kcat —
    turnover is set by active-site chemistry, cofactors and mechanism, not global
    amino-acid fractions. LOFO R² is typically ≤ 0 here (see the live transcript),
    so the honest verdict is refuted. This is a real biological fact, not a
    pipeline defect.
  - No 3D structure, active-site, cofactor or substrate-chemistry features; kcat
    only (no Km/kcat-Km modelling from structure).
- **Rediscovery references (real known values):** the Bar-Even et al. 2011
  catalytic-efficiency anchors — median kcat/Km ≈ 1e5 M⁻¹s⁻¹, median kcat ≈ 10 s⁻¹,
  and the diffusion-limit ceiling ≈ 1e8–1e9 M⁻¹s⁻¹. A claim restating the median,
  or claiming kcat/Km **above** the diffusion limit, is flagged (known value /
  known-ceiling violation).

## mandrake — retroviral RT-family LOFO

- **Real data:** **56 retroviral reverse-transcriptase sequences** across
  **7 evolutionary families** with handcrafted biophysical features (thermal-
  stability profile, catalytic-triad geometry, electrostatics, Foldseek fold
  similarity, surface properties, sequence motifs). Target: PE efficiency.
- **PRIVATE dataset.** `mandrake-data/` is git-ignored and has **no public URL** —
  it cannot be fabricated or auto-fetched. When it is absent the mandrake tests
  **skip cleanly** and `scripts/build_real_domain_datasets.py` reports it as
  absent (see below).
- **Grouping:** RT evolutionary family.
- **CAN detect:** whether a biophysical feature set predicts RT activity within
  families / survives leave-one-family-out, and whether an apparent signal
  collapses to a family surrogate (the finding-audit machinery flags "family
  surrogate" vs "gold" findings).
- **CANNOT detect / known limits — the headline limit is statistical power:**
  - **n ≈ 56 across 7 imbalanced families is very small.** With so few sequences
    the LOFO + permutation test has limited power to resolve a genuine family-
    specific signal from a sequence-redundancy artifact — for *any* method, human
    or machine (`scripts/mandrake_power_analysis.py`). Underpowered "no signal" is
    the expected, honest result.
  - Geometry / fold features can proxy family identity, so a naive cross-family
    signal is often a surrogate, not biology.
- **Rediscovery references (real known values):** the **Pfam RVT clan CL0027**
  (PF00078 / PF07727 / PF13456) RT-domain family tabulation (Mistry et al. 2021),
  and the **Xiong & Eickbush 1990** anchors — the seven conserved RT catalytic
  motifs over a ~240-aa domain and the characteristically low (< ~25%) cross-class
  RT sequence identity. A claim that merely assigns RT sequences to their known
  Pfam family, or restates the seven conserved motifs, is flagged as a
  rediscovery.

---

## How to fetch data and run

```bash
# 1. Build the real caches (auto-fetches GTEx + DLKcat; reports mandrake status).
#    Offline, the adapters keep an HONEST, clearly-labelled synthetic fallback —
#    dataset_is_synthetic() / the cache .meta.json says which is on disk.
PYTHONPATH="packages/propab-core;." python scripts/build_real_domain_datasets.py

# 2. Run the biology tests. With real data present they run in full; with data
#    absent or synthetic they SKIP cleanly (green-with-skips, never red).
pytest tests/test_genomics_honesty.py tests/test_genomics_plugin.py \
       tests/test_enzyme_kinetics_honesty.py tests/test_enzyme_kinetics_plugin.py \
       tests/test_mandrake_verification.py tests/test_biology_rediscovery.py -q -rs

# 3. Live end-to-end (needs real data cached + GOOGLE_API_KEY in .env):
#    real LLM hypothesis generation + real LOFO/shuffle-null verification.
PYTHONPATH="packages/propab-core;." python artifacts/run_biology_live_validation.py
# -> artifacts/biology_live_validation.md (+ .json)
```

- `real_data_cached()` on the genomics/enzyme adapters reports, **without any
  network call**, whether a real (non-synthetic) cache is on disk. The real-data
  tests use it to skip cleanly rather than download data in CI or vacuously pass
  on the synthetic frame.
- The real caches live under `data/{genomics,enzyme_kinetics}/` and are
  git-ignored; the adapters regenerate them on first use.

## Honest statement on current yield

Running the shipped datasets through the real pipeline **more often yields a
rigorous "no signal" than a novel confirmed finding.** The captured live
validation (`artifacts/biology_live_validation.md`) is a faithful example: a real
LLM proposed well-framed, non-trivial hypotheses for genomics and enzyme_kinetics,
and the real LOFO + shuffle null honestly **refuted all of them** — genomics
because the proposed relationships were within-row tautologies the shuffle null
(correctly) did not reject, enzyme_kinetics because bulk sequence composition
carries no real cross-EC kcat signal. That is the correct scientific outcome on
these data, and the gate's refusal to fabricate a confirmation is the point.

### What richer data would change

- **genomics:** per-sample (not median) GTEx expression, or paired genotype
  (eQTL) data, would give real cross-sample variance and non-tautological
  features that could carry genuine held-out-tissue signal.
- **enzyme_kinetics:** structural / active-site / cofactor features (beyond bulk
  composition) and Km alongside kcat would let the model reach the chemistry that
  actually sets turnover.
- **mandrake:** more RT sequences per family (raising n well above ~56) would give
  the LOFO + permutation test the statistical power to actually resolve family-
  specific signal from redundancy artifact — the single biggest lever here.
