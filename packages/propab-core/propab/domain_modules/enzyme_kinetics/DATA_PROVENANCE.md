# Enzyme kinetics dataset provenance

The enzyme kinetics adapter serves **real, measured enzyme turnover numbers
(kcat)** — not synthetic data. `EnzymeKineticsPlugin.uses_synthetic_data()`
returns `False` whenever the real dataset is on disk (it reads the
`synthetic` flag written into the cache `.meta.json`).

## Source

- **Dataset:** DLKcat kcat compilation (`Kcat_combination_0918.json`).
- **Origin:** Li, F., Yuan, L., Lu, H. et al. *Deep learning-based kcat
  prediction enables improved enzyme-constrained model reconstruction.*
  **Nature Catalysis** 5, 662–672 (2022). https://doi.org/10.1038/s41929-022-00798-z
- **Repository:** https://github.com/SysBioChalmers/DLKcat
  (`DeeplearningApproach/Data/database/Kcat_combination_0918.json`)
- **Fetched URL:** `https://raw.githubusercontent.com/SysBioChalmers/DLKcat/master/DeeplearningApproach/Data/database/Kcat_combination_0918.json`
- **Upstream data sources:** the DLKcat compilation aggregates real measured
  kcat values from **BRENDA** (https://www.brenda-enzymes.org, CC BY 4.0) and
  **SABIO-RK** (https://sabiork.h-its.org, CC BY 4.0).
- **Access date:** 2026-07-07.
- **License:** DLKcat repository is released under the MIT/academic terms of
  SysBioChalmers; the underlying kcat measurements originate from BRENDA and
  SABIO-RK (CC BY 4.0). Only a small, transformed subset is cached locally.

## What is cached (`data/enzyme_kinetics/brenda_subset_v1.csv`)

Each row is one real enzyme kcat measurement:

| column | meaning | provenance |
|---|---|---|
| `ec_class` | top-level EC class `EC1..EC6` | first digit of the real EC number |
| `ec_number` | full EC number | real (DLKcat) |
| `organism`, `substrate` | source organism / substrate | real (DLKcat) |
| `kcat`, `log_kcat` | measured kcat (s⁻¹) and log10 | **real measured target** |
| `sequence_length`, `molecular_weight` | from the real protein sequence | derived |
| `gravy_hydropathy`, `frac_*` | Kyte–Doolittle GRAVY + residue composition | derived from real sequence |

Rows are filtered to `Unit == s^(-1)`, `kcat > 0`, `sequence_length >= 20`,
deduplicated, and capped to ≤600 per EC class for a balanced, fast LOFO.

## Why the LOFO structure is unchanged

The verifier still does **leave-one-EC-class-out** ridge + **EC-label-shuffle
null**. Only the data feeding it changed from synthetic to real. On the real
data the cross-EC signal from sequence composition is honestly weak/absent
(the null is not rejected) — which is the correct, non-rediscovery outcome.
